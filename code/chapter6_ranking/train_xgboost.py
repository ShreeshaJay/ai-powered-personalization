"""
Chapter 6: XGBoost Training Script
==================================
Train XGBoost ranker on Yambda dataset with Global Temporal Split.

Usage:
------
# Train with default settings
python train_xgboost.py

# Train with custom sample fraction (for testing)
python train_xgboost.py --sample_frac 0.01

# Train with custom hyperparameters
python train_xgboost.py --n_estimators 200 --max_depth 8 --learning_rate 0.05

Output:
-------
Models and metrics saved to outputs/models/{timestamp}/
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    GTSConfig,
    XGBoostConfig,
    TrainingConfig,
    ensure_directories
)
from data_loader import YambdaDataLoader, get_train_test_data
from models import XGBoostRanker, HistoricalFeatureBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost ranker on Yambda dataset'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(DATA_DIR),
        help='Path to Yambda flat data directory'
    )
    parser.add_argument(
        '--sample_frac',
        type=float,
        default=None,
        help='Fraction of data to sample (None for full data)'
    )
    
    # GTS arguments
    parser.add_argument(
        '--train_days',
        type=int,
        default=300,
        help='Number of days for training data'
    )
    parser.add_argument(
        '--gap_minutes',
        type=int,
        default=30,
        help='Gap between train and test in minutes'
    )
    parser.add_argument(
        '--test_days',
        type=int,
        default=1,
        help='Number of days for test data'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of boosting rounds'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=6,
        help='Maximum tree depth'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Subsample ratio of training instances'
    )
    parser.add_argument(
        '--colsample_bytree',
        type=float,
        default=0.8,
        help='Subsample ratio of columns'
    )
    parser.add_argument(
        '--scale_pos_weight',
        type=float,
        default=1.0,
        help='Weight for positive class (use n_neg/n_pos for imbalanced data)'
    )
    
    # Training arguments
    parser.add_argument(
        '--val_split_ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (from end of training data)'
    )
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=10,
        help='Rounds to stop after no improvement'
    )
    parser.add_argument(
        '--completion_threshold',
        type=int,
        default=50,
        help='Played ratio threshold for positive label'
    )
    parser.add_argument(
        '--use_historical_features',
        action='store_true',
        default=True,
        help='Compute and use historical aggregate features'
    )
    parser.add_argument(
        '--no_historical_features',
        action='store_true',
        help='Disable historical features (baseline mode)'
    )
    parser.add_argument(
        '--feature_source_ratio',
        type=float,
        default=0.8,
        help='Fraction of training data to use for computing historical features (default: 0.8). '
             'Prevents look-ahead bias by only using early data for feature computation.'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/models/{timestamp})'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name (used in output directory)'
    )
    
    return parser.parse_args()


def create_output_dir(args) -> Path:
    """Create output directory for this training run."""
    ensure_directories()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.experiment_name:
            dir_name = f"{timestamp}_{args.experiment_name}"
        else:
            dir_name = timestamp
        output_dir = MODELS_DIR / dir_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_temporal_val_split(
    train_df: pd.DataFrame,
    val_ratio: float = 0.1
) -> tuple:
    """
    Create temporal validation split from training data.
    
    Takes the last val_ratio fraction of training data as validation,
    maintaining temporal order (no data leakage).
    
    Args:
        train_df: Training DataFrame with 'timestamp' column
        val_ratio: Fraction of data for validation
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Sort by timestamp to ensure temporal order
    train_df = train_df.sort_values('timestamp')
    
    # Split point
    split_idx = int(len(train_df) * (1 - val_ratio))
    
    train_split = train_df.iloc[:split_idx].copy()
    val_split = train_df.iloc[split_idx:].copy()
    
    logger.info(f"Temporal split: {len(train_split):,} train, {len(val_split):,} validation")
    
    return train_split, val_split


def train(args):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("XGBoost Ranker Training")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = create_output_dir(args)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    config = vars(args).copy()
    config['output_dir'] = str(output_dir)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # =========================================================================
    # Load Data with Global Temporal Split
    # =========================================================================
    logger.info("\n--- Loading Data ---")
    
    gts_config = GTSConfig(
        train_days=args.train_days,
        gap_minutes=args.gap_minutes,
        test_days=args.test_days
    )
    
    train_df, test_df = get_train_test_data(
        data_dir=args.data_dir,
        task='listen_completion',
        gts_config=gts_config,
        sample_frac=args.sample_frac,
        as_pandas=True
    )
    
    logger.info(f"Loaded {len(train_df):,} training samples, {len(test_df):,} test samples")
    
    # Create label (listen completion)
    train_df['label'] = (train_df['played_ratio_pct'] >= args.completion_threshold).astype(int)
    test_df['label'] = (test_df['played_ratio_pct'] >= args.completion_threshold).astype(int)
    
    logger.info(f"Training positive rate: {train_df['label'].mean():.2%}")
    logger.info(f"Test positive rate: {test_df['label'].mean():.2%}")
    
    # =========================================================================
    # Create Temporal Validation Split
    # =========================================================================
    logger.info("\n--- Creating Validation Split ---")
    
    train_split, val_split = create_temporal_val_split(
        train_df, 
        val_ratio=args.val_split_ratio
    )
    
    # =========================================================================
    # Compute Historical Features (if enabled)
    # =========================================================================
    use_historical = args.use_historical_features and not args.no_historical_features
    historical_builder = None
    
    if use_historical:
        logger.info("\n--- Computing Historical Features ---")
        
        # IMPORTANT: To prevent data leakage within training set, we compute
        # historical features from only the FIRST portion of training data.
        # This ensures no training sample uses "future" information.
        #
        # Split: train_split → [feature_source (80%) | remaining (20%)]
        # Features computed from: feature_source only
        # Features applied to: ALL splits (train, val, test)
        
        feature_source_ratio = args.feature_source_ratio
        train_split_sorted = train_split.sort_values('timestamp')
        feature_source_idx = int(len(train_split_sorted) * feature_source_ratio)
        
        feature_source_df = train_split_sorted.iloc[:feature_source_idx].copy()
        
        logger.info(f"Computing features from first {feature_source_ratio:.0%} of training data "
                   f"({len(feature_source_df):,} samples) to prevent look-ahead bias")
        
        historical_builder = HistoricalFeatureBuilder()
        
        # Fit ONLY on early training data (no look-ahead)
        historical_builder.fit(feature_source_df)
        
        # Transform all splits using features from early data only
        train_split = historical_builder.transform(train_split)
        val_split = historical_builder.transform(val_split)
        test_df = historical_builder.transform(test_df)
        
        logger.info(f"Added {len(historical_builder.get_feature_names())} historical features")
        logger.info(f"Features: {historical_builder.get_feature_names()[:10]}...")
    else:
        logger.info("\n--- Skipping Historical Features (baseline mode) ---")
    
    # =========================================================================
    # Initialize and Train Model
    # =========================================================================
    logger.info("\n--- Training Model ---")
    
    model = XGBoostRanker(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=args.scale_pos_weight,
        early_stopping_rounds=args.early_stopping_rounds
    )
    
    # Train with validation set for early stopping
    model.fit(
        X=train_split,
        y=train_split['label'],
        eval_set=[(val_split, val_split['label'])],
        verbose=True
    )
    
    # =========================================================================
    # Evaluate
    # =========================================================================
    logger.info("\n--- Evaluation ---")
    
    # Validation metrics
    val_metrics = model.evaluate(val_split, val_split['label'])
    logger.info(f"Validation Metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Test metrics
    test_metrics = model.evaluate(test_df, test_df['label'])
    logger.info(f"Test Metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    logger.info("\nFeature Importance (Top 10):")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    logger.info("\n--- Saving Results ---")
    
    # Save model
    model.save(str(output_dir))
    
    # Save historical feature builder if used
    if historical_builder is not None:
        historical_builder.save(str(output_dir))
    
    # Save metrics
    metrics = {
        'validation': val_metrics,
        'test': test_metrics,
        'feature_importance': feature_importance.to_dict('records'),
        'historical_features_used': use_historical
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importance as CSV
    feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    logger.info(f"\nTraining complete! Results saved to: {output_dir}")
    logger.info("=" * 60)
    
    return {
        'output_dir': str(output_dir),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


if __name__ == '__main__':
    args = parse_args()
    train(args)

