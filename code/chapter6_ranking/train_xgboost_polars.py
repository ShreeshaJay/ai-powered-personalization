"""
Chapter 6: XGBoost Training with Polars Pipeline
=================================================
Memory-efficient training script for 16GB laptops.

Uses Polars for data loading and feature engineering, then trains XGBoost
for binary classification (listen completion prediction).

Usage:
------
# Quick local run (30 days, ~1.3M rows)
python train_xgboost_polars.py --train_days 30

# Larger experiment (90 days, ~4M rows)
python train_xgboost_polars.py --train_days 90

# Full dataset (300 days, ~50M rows) - requires more memory
python train_xgboost_polars.py

# With custom hyperparameters
python train_xgboost_polars.py --train_days 30 --n_estimators 200 --max_depth 8
"""

import argparse
import logging
import os
import time
from pathlib import Path
from datetime import datetime
import json

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    accuracy_score,
    f1_score,
    classification_report
)
import numpy as np

from utils.polars_pipeline import load_yambda_polars, PolarsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost ranking model with Polars pipeline'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir', type=str,
        default=None,
        help='Path to Yandex/flat data directory (or set YAMBDA_DATA_DIR env var)'
    )
    parser.add_argument(
        '--train_days', type=int, default=30,
        help='Training window in days (30=~5M rows, 300=~12M, 0 or negative=full ~50M)'
    )
    parser.add_argument(
        '--test_days', type=int, default=1,
        help='Test window in days (default: 1)'
    )
    
    # Model hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum tree depth')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate (eta)')
    parser.add_argument('--min_child_weight', type=int, default=1,
                        help='Minimum sum of instance weight in a child')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio of training instances')
    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='Subsample ratio of columns for each tree')
    parser.add_argument('--scale_pos_weight', type=float, default=1.0,
                        help='Balance weight for positive class')
    parser.add_argument('--objective', type=str, default='binary:logistic',
                        help='XGBoost objective function')
    parser.add_argument('--eval_metric', type=str, default='auc',
                        choices=['auc', 'aucpr', 'logloss'],
                        help='Evaluation metric (auc for balanced, aucpr for imbalanced)')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    
    # Output arguments
    parser.add_argument(
        '--output_dir', type=str, default='./outputs',
        help='Directory to save model and results'
    )
    parser.add_argument(
        '--experiment_name', type=str, default=None,
        help='Name for this experiment (default: auto-generated)'
    )
    
    return parser.parse_args()


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute comprehensive evaluation metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'auc_pr': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'positive_rate_true': y_true.mean(),
        'positive_rate_pred': y_pred.mean(),
    }
    
    return metrics


def print_feature_importance(model, feature_cols, top_n=15):
    """Print top feature importances."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    logger.info(f"\nTop {top_n} Feature Importances:")
    logger.info("-" * 50)
    for i in range(min(top_n, len(feature_cols))):
        idx = indices[i]
        logger.info(f"  {i+1:2d}. {feature_cols[idx]:30s} {importance[idx]:.4f}")


def resolve_data_dir(args_data_dir: str) -> str:
    """Resolve data directory from argument or environment variable."""
    # Priority: CLI arg > environment variable
    if args_data_dir:
        return args_data_dir
    
    env_data_dir = os.environ.get('YAMBDA_DATA_DIR')
    if env_data_dir:
        return env_data_dir
    
    # No data directory specified
    raise ValueError(
        "Data directory not specified. Please either:\n"
        "  1. Pass --data_dir /path/to/Yandex/flat\n"
        "  2. Set environment variable: export YAMBDA_DATA_DIR=/path/to/Yandex/flat\n"
        "\n"
        "The Yambda dataset can be downloaded from:\n"
        "  https://huggingface.co/datasets/yandex/yambda"
    )


def main():
    """Main training function."""
    args = parse_args()
    
    # Resolve data directory
    args.data_dir = resolve_data_dir(args.data_dir)
    
    # Handle train_days: 0 or negative means full dataset
    if args.train_days <= 0:
        args.train_days = None
        logger.info("Using FULL dataset (train_days=None)")
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        days_str = "full" if args.train_days is None else f"{args.train_days}d"
        args.experiment_name = f"xgb_polars_{days_str}_{timestamp}"
    
    logger.info("=" * 60)
    logger.info("XGBoost Training with Polars Pipeline")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Train window: {args.train_days or 'FULL'} days")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("\n[1/5] Loading data with Polars...")
    start_time = time.time()
    
    train_lf, test_lf = load_yambda_polars(
        args.data_dir,
        train_days=args.train_days,
        test_days=args.test_days,
        lazy=True
    )
    
    load_time = time.time() - start_time
    logger.info(f"Data scan complete in {load_time:.1f}s")
    
    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    logger.info("\n[2/5] Running feature pipeline...")
    start_time = time.time()
    
    pipeline = PolarsPipeline()
    train_df, feature_cols = pipeline.fit_transform(train_lf)
    test_df, _ = pipeline.transform(test_lf)
    
    feature_time = time.time() - start_time
    logger.info(f"Feature engineering complete in {feature_time:.1f}s")
    logger.info(f"Train samples: {train_df.height:,}")
    logger.info(f"Test samples: {test_df.height:,}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Convert to pandas for XGBoost
    logger.info("\nConverting to pandas...")
    X_train, y_train = pipeline.to_pandas(train_df, feature_cols)
    X_test, y_test = pipeline.to_pandas(test_df, feature_cols)
    
    # Free Polars memory
    del train_df, test_df, train_lf, test_lf
    
    # Log class distribution
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    logger.info(f"Train positive rate: {train_pos_rate:.1%}")
    logger.info(f"Test positive rate: {test_pos_rate:.1%}")
    
    # =========================================================================
    # Step 3: Train XGBoost
    # =========================================================================
    logger.info("\n[3/5] Training XGBoost...")
    start_time = time.time()
    
    # Configure model
    # Note: early_stopping_rounds is set in constructor for XGBoost 2.0+
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=args.scale_pos_weight,
        objective=args.objective,
        eval_metric=args.eval_metric,
        early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
        n_jobs=-1,
        random_state=42
    )
    
    logger.info(f"Model config: objective={args.objective}, eval_metric={args.eval_metric}")
    
    # Training with evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    best_iteration = model.best_iteration if model.best_iteration else args.n_estimators
    
    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time:.1f}s")
    logger.info(f"Best iteration: {best_iteration}")
    
    # =========================================================================
    # Step 4: Evaluate
    # =========================================================================
    logger.info("\n[4/5] Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    
    logger.info("\nTraining Metrics:")
    logger.info(f"  AUC-ROC: {train_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:  {train_metrics['auc_pr']:.4f}")
    logger.info(f"  F1:      {train_metrics['f1']:.4f}")
    
    logger.info("\nTest Metrics:")
    logger.info(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:  {test_metrics['auc_pr']:.4f}")
    logger.info(f"  F1:      {test_metrics['f1']:.4f}")
    
    # Feature importance
    print_feature_importance(model, feature_cols)
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    logger.info("\n[5/5] Saving results...")
    
    # Save model (use booster directly for compatibility)
    model_path = output_dir / 'model.json'
    model.get_booster().save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save pipeline
    pipeline_path = output_dir / 'pipeline'
    pipeline.save(str(pipeline_path))
    logger.info(f"Pipeline saved to {pipeline_path}")
    
    # Save metrics and config
    results = {
        'experiment_name': args.experiment_name,
        'config': {
            'train_days': args.train_days,
            'test_days': args.test_days,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'scale_pos_weight': args.scale_pos_weight,
            'objective': args.objective,
            'eval_metric': args.eval_metric,
            'early_stopping_rounds': args.early_stopping_rounds,
        },
        'data': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'train_positive_rate': float(train_pos_rate),
            'test_positive_rate': float(test_pos_rate),
        },
        'metrics': {
            'train': train_metrics,
            'test': test_metrics,
        },
        'timing': {
            'data_load_seconds': load_time,
            'feature_engineering_seconds': feature_time,
            'training_seconds': train_time,
            'best_iteration': best_iteration,
        }
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")
    logger.info(f"Total time: {load_time + feature_time + train_time:.1f}s")
    logger.info(f"Outputs: {output_dir}")
    logger.info("=" * 60)
    
    return results


if __name__ == '__main__':
    main()

