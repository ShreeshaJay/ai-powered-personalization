"""
Chapter 6: XGBoost Learning-to-Rank Training
=============================================
Pairwise Learning-to-Rank using XGBRanker with LambdaMART objective.

This script implements pairwise LTR and compares it with pointwise classification:

**Pointwise** (train_xgboost_polars.py):
    - Treats each (user, item) pair independently
    - Predicts P(completion) for each item
    - Optimizes binary cross-entropy
    - Evaluates with AUC-ROC

**Pairwise** (this script):
    - Compares pairs of items within the same user/query group
    - Learns which item should rank higher
    - Optimizes pairwise ranking loss (LambdaMART)
    - Evaluates with NDCG@K

Usage:
------
# Quick local run (30 days)
python train_xgboost_ltr.py --train_days 30

# Compare with pointwise
python train_xgboost_polars.py --train_days 30  # Pointwise
python train_xgboost_ltr.py --train_days 30      # Pairwise

# Full dataset
python train_xgboost_ltr.py --train_days 0
"""

import argparse
import logging
import os
import time
from pathlib import Path
from datetime import datetime
import json

import xgboost as xgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

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
        description='Train XGBoost Learning-to-Rank model (pairwise)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to Yandex/flat data directory (or set YAMBDA_DATA_DIR env var)'
    )
    parser.add_argument(
        '--train_days', type=int, default=30,
        help='Training window in days (30=~5M rows, 0=full ~50M)'
    )
    parser.add_argument(
        '--test_days', type=int, default=1,
        help='Test window in days (default: 1)'
    )
    
    # LTR-specific arguments
    parser.add_argument(
        '--objective', type=str, default='rank:pairwise',
        choices=['rank:pairwise', 'rank:ndcg', 'rank:map'],
        help='LTR objective: pairwise (LambdaMART), ndcg (LambdaRank), map'
    )
    parser.add_argument(
        '--min_group_size', type=int, default=2,
        help='Minimum items per user/query group (filter users with fewer items)'
    )
    parser.add_argument(
        '--max_group_size', type=int, default=1000,
        help='Maximum items per user/query group (truncate large groups)'
    )
    
    # Model hyperparameters (similar to pointwise for fair comparison)
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


def create_query_groups(df: pl.DataFrame, group_col: str = 'uid') -> np.ndarray:
    """
    Create query groups for LTR from a DataFrame.
    
    XGBRanker expects a 1D array where each element is the SIZE of a query group.
    Items in the same group are compared pairwise during training.
    
    Design Decision: User-Level Grouping
    ------------------------------------
    We group by user_id rather than session_id because:
    1. Yambda dataset doesn't have explicit session identifiers
    2. Creating implicit sessions from timestamp gaps is unreliable without
       knowing if the data contains comprehensive clickstream history
    3. User-level grouping is simpler and provides more training pairs
    
    Limitation: Items from different sessions (e.g., morning commute vs evening)
    are compared as if they competed for attention, which may introduce noise.
    For production systems with session tracking, consider session-level grouping.
    
    Args:
        df: Polars DataFrame sorted by group_col
        group_col: Column to group by (default: user_id)
        
    Returns:
        numpy array of group sizes [n1, n2, n3, ...] where n_i is size of group i
        
    Example:
        df with uid: [1, 1, 1, 2, 2, 3, 3, 3, 3]
        returns: [3, 2, 4]  (user 1 has 3 items, user 2 has 2, user 3 has 4)
    """
    group_sizes = (
        df.group_by(group_col, maintain_order=True)
        .agg(pl.len().alias('group_size'))
        ['group_size']
        .to_numpy()
    )
    return group_sizes


def filter_and_prepare_groups(
    df: pl.DataFrame,
    min_group_size: int = 2,
    max_group_size: int = 1000,
    group_col: str = 'uid'
) -> pl.DataFrame:
    """
    Filter users and prepare data for LTR training.
    
    LTR requires:
    1. At least 2 items per user (to form pairs)
    2. Data sorted by user_id for correct group assignment
    3. Optional: truncate very large groups for efficiency
    
    Args:
        df: Input DataFrame
        min_group_size: Minimum items per user (users with fewer are dropped)
        max_group_size: Maximum items per user (excess items are dropped)
        group_col: Column to group by
        
    Returns:
        Filtered and sorted DataFrame
    """
    # Count items per user
    user_counts = df.group_by(group_col).agg(pl.len().alias('_count'))
    
    # Filter to users with at least min_group_size items
    valid_users_df = user_counts.filter(pl.col('_count') >= min_group_size)
    valid_users = valid_users_df[group_col]
    
    logger.info(f"Users before filter: {user_counts.height:,}")
    logger.info(f"Users with >= {min_group_size} items: {valid_users_df.height:,}")
    
    # Filter DataFrame to valid users
    df = df.filter(pl.col(group_col).is_in(valid_users))
    
    # Sort by user_id (required for XGBRanker group assignment)
    df = df.sort(group_col)
    
    # Truncate large groups if needed
    if max_group_size < float('inf'):
        df = (
            df.with_columns(
                pl.col(group_col).cum_count().over(group_col).alias('_rank_in_group')
            )
            .filter(pl.col('_rank_in_group') <= max_group_size)
            .drop('_rank_in_group')
        )
    
    logger.info(f"Rows after filtering: {df.height:,}")
    
    return df


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@K for ranking evaluation.
    
    Args:
        y_true: True relevance labels
        y_pred: Predicted scores
        groups: Array of group sizes
        k: Cutoff for NDCG computation
        
    Returns:
        Mean NDCG@K across all groups
    """
    ndcg_scores = []
    start_idx = 0
    
    for group_size in groups:
        end_idx = start_idx + group_size
        
        # Get predictions and labels for this group
        group_pred = y_pred[start_idx:end_idx]
        group_true = y_true[start_idx:end_idx]
        
        # Sort by predicted score (descending)
        sorted_indices = np.argsort(group_pred)[::-1]
        sorted_true = group_true[sorted_indices]
        
        # Compute DCG@K
        k_actual = min(k, len(sorted_true))
        dcg = 0.0
        for i in range(k_actual):
            dcg += (2 ** sorted_true[i] - 1) / np.log2(i + 2)
        
        # Compute ideal DCG (sort by true relevance)
        ideal_sorted = np.sort(group_true)[::-1]
        idcg = 0.0
        for i in range(k_actual):
            idcg += (2 ** ideal_sorted[i] - 1) / np.log2(i + 2)
        
        # Compute NDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(1.0)  # All items have same relevance
        
        start_idx = end_idx
    
    return np.mean(ndcg_scores)


def compute_map(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """
    Compute Mean Average Precision for ranking evaluation.
    
    Args:
        y_true: True binary relevance labels
        y_pred: Predicted scores
        groups: Array of group sizes
        
    Returns:
        Mean Average Precision across all groups
    """
    ap_scores = []
    start_idx = 0
    
    for group_size in groups:
        end_idx = start_idx + group_size
        
        # Get predictions and labels for this group
        group_pred = y_pred[start_idx:end_idx]
        group_true = y_true[start_idx:end_idx]
        
        # Sort by predicted score (descending)
        sorted_indices = np.argsort(group_pred)[::-1]
        sorted_true = group_true[sorted_indices]
        
        # Compute Average Precision
        n_relevant = sorted_true.sum()
        if n_relevant == 0:
            start_idx = end_idx
            continue
            
        precision_sum = 0.0
        relevant_count = 0
        for i, is_relevant in enumerate(sorted_true):
            if is_relevant:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        ap_scores.append(precision_sum / n_relevant)
        start_idx = end_idx
    
    return np.mean(ap_scores) if ap_scores else 0.0


def compute_mrr(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        y_true: True binary relevance labels
        y_pred: Predicted scores
        groups: Array of group sizes
        
    Returns:
        Mean Reciprocal Rank across all groups
    """
    rr_scores = []
    start_idx = 0
    
    for group_size in groups:
        end_idx = start_idx + group_size
        
        # Get predictions and labels for this group
        group_pred = y_pred[start_idx:end_idx]
        group_true = y_true[start_idx:end_idx]
        
        # Sort by predicted score (descending)
        sorted_indices = np.argsort(group_pred)[::-1]
        sorted_true = group_true[sorted_indices]
        
        # Find rank of first relevant item
        first_relevant_idx = np.where(sorted_true == 1)[0]
        if len(first_relevant_idx) > 0:
            rr_scores.append(1.0 / (first_relevant_idx[0] + 1))
        else:
            rr_scores.append(0.0)
        
        start_idx = end_idx
    
    return np.mean(rr_scores)


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
    if args_data_dir:
        return args_data_dir
    
    env_data_dir = os.environ.get('YAMBDA_DATA_DIR')
    if env_data_dir:
        return env_data_dir
    
    raise ValueError(
        "Data directory not specified. Please either:\n"
        "  1. Pass --data_dir /path/to/Yandex/flat\n"
        "  2. Set environment variable: export YAMBDA_DATA_DIR=/path/to/Yandex/flat"
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
        args.experiment_name = f"xgb_ltr_{days_str}_{timestamp}"
    
    logger.info("=" * 60)
    logger.info("XGBoost Learning-to-Rank Training (Pairwise)")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Train window: {args.train_days or 'FULL'} days")
    logger.info(f"LTR objective: {args.objective}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    total_start_time = time.time()
    
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
    logger.info(f"Train samples (before LTR filter): {train_df.height:,}")
    logger.info(f"Test samples (before LTR filter): {test_df.height:,}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # =========================================================================
    # Step 3: Prepare LTR Groups
    # =========================================================================
    logger.info("\n[3/5] Preparing query groups for LTR...")
    start_time = time.time()
    
    # Filter and prepare training data
    logger.info("Processing training data...")
    train_df = filter_and_prepare_groups(
        train_df,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        group_col='uid'
    )
    train_groups = create_query_groups(train_df, group_col='uid')
    logger.info(f"Train groups: {len(train_groups):,} users, {train_df.height:,} samples")
    logger.info(f"Avg items per user: {train_df.height / len(train_groups):.1f}")
    
    # Filter and prepare test data
    logger.info("Processing test data...")
    test_df = filter_and_prepare_groups(
        test_df,
        min_group_size=args.min_group_size,
        max_group_size=args.max_group_size,
        group_col='uid'
    )
    test_groups = create_query_groups(test_df, group_col='uid')
    logger.info(f"Test groups: {len(test_groups):,} users, {test_df.height:,} samples")
    
    group_time = time.time() - start_time
    logger.info(f"Group preparation complete in {group_time:.1f}s")
    
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
    # Step 4: Train XGBoost LTR
    # =========================================================================
    logger.info("\n[4/5] Training XGBoost LTR...")
    start_time = time.time()
    
    # Configure LTR model
    model = xgb.XGBRanker(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective=args.objective,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=args.early_stopping_rounds if args.early_stopping_rounds > 0 else None,
    )
    
    logger.info(f"Model config: objective={args.objective}")
    
    # Train with group information
    model.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_group=[train_groups, test_groups],
        verbose=10
    )
    
    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time:.1f}s")
    logger.info(f"Best iteration: {model.best_iteration}")
    
    # =========================================================================
    # Step 5: Evaluate Model
    # =========================================================================
    logger.info("\n[5/5] Evaluating model...")
    
    # Get predictions (ranking scores, not probabilities)
    train_scores = model.predict(X_train)
    test_scores = model.predict(X_test)
    
    # Compute ranking metrics
    logger.info("\n" + "=" * 60)
    logger.info("RANKING METRICS (LTR-specific)")
    logger.info("=" * 60)
    
    # Training metrics
    train_ndcg_5 = compute_ndcg(y_train.values, train_scores, train_groups, k=5)
    train_ndcg_10 = compute_ndcg(y_train.values, train_scores, train_groups, k=10)
    train_map = compute_map(y_train.values, train_scores, train_groups)
    train_mrr = compute_mrr(y_train.values, train_scores, train_groups)
    
    logger.info("\nTraining Metrics:")
    logger.info(f"  NDCG@5:  {train_ndcg_5:.4f}")
    logger.info(f"  NDCG@10: {train_ndcg_10:.4f}")
    logger.info(f"  MAP:     {train_map:.4f}")
    logger.info(f"  MRR:     {train_mrr:.4f}")
    
    # Test metrics
    test_ndcg_5 = compute_ndcg(y_test.values, test_scores, test_groups, k=5)
    test_ndcg_10 = compute_ndcg(y_test.values, test_scores, test_groups, k=10)
    test_map = compute_map(y_test.values, test_scores, test_groups)
    test_mrr = compute_mrr(y_test.values, test_scores, test_groups)
    
    logger.info("\nTest Metrics:")
    logger.info(f"  NDCG@5:  {test_ndcg_5:.4f}")
    logger.info(f"  NDCG@10: {test_ndcg_10:.4f}")
    logger.info(f"  MAP:     {test_map:.4f}")
    logger.info(f"  MRR:     {test_mrr:.4f}")
    
    # Also compute classification-style metrics for comparison with pointwise
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION METRICS (for comparison with pointwise)")
    logger.info("=" * 60)
    
    # Normalize scores to [0, 1] for AUC computation
    train_scores_norm = (train_scores - train_scores.min()) / (train_scores.max() - train_scores.min() + 1e-8)
    test_scores_norm = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
    
    train_auc = roc_auc_score(y_train, train_scores_norm)
    test_auc = roc_auc_score(y_test, test_scores_norm)
    train_ap = average_precision_score(y_train, train_scores_norm)
    test_ap = average_precision_score(y_test, test_scores_norm)
    
    logger.info("\nTraining Metrics:")
    logger.info(f"  AUC-ROC: {train_auc:.4f}")
    logger.info(f"  AUC-PR:  {train_ap:.4f}")
    
    logger.info("\nTest Metrics:")
    logger.info(f"  AUC-ROC: {test_auc:.4f}")
    logger.info(f"  AUC-PR:  {test_ap:.4f}")
    
    # Feature importance
    print_feature_importance(model, feature_cols)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    logger.info("\n[6/6] Saving results...")
    
    # Save model
    model_path = output_dir / 'model.json'
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save pipeline
    pipeline_path = output_dir / 'pipeline'
    pipeline.save(str(pipeline_path))
    logger.info(f"Pipeline saved to {pipeline_path}")
    
    # Save results
    total_time = time.time() - total_start_time
    results = {
        'experiment_name': args.experiment_name,
        'model_type': 'XGBRanker (pairwise LTR)',
        'objective': args.objective,
        'train_days': args.train_days,
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'train_groups': int(len(train_groups)),
        'test_groups': int(len(test_groups)),
        'features': len(feature_cols),
        'feature_names': feature_cols,
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'min_group_size': args.min_group_size,
            'max_group_size': args.max_group_size,
        },
        'ranking_metrics': {
            'train': {
                'ndcg_5': train_ndcg_5,
                'ndcg_10': train_ndcg_10,
                'map': train_map,
                'mrr': train_mrr,
            },
            'test': {
                'ndcg_5': test_ndcg_5,
                'ndcg_10': test_ndcg_10,
                'map': test_map,
                'mrr': test_mrr,
            }
        },
        'classification_metrics': {
            'train': {'auc_roc': train_auc, 'auc_pr': train_ap},
            'test': {'auc_roc': test_auc, 'auc_pr': test_ap}
        },
        'timing': {
            'load_time': load_time,
            'feature_time': feature_time,
            'group_time': group_time,
            'train_time': train_time,
            'total_time': total_time,
        },
        'best_iteration': model.best_iteration,
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: XGBRanker (pairwise LTR)")
    logger.info(f"Objective: {args.objective}")
    logger.info(f"Test NDCG@10: {test_ndcg_10:.4f}")
    logger.info(f"Test MAP: {test_map:.4f}")
    logger.info(f"Test AUC-ROC: {test_auc:.4f} (for comparison with pointwise)")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Outputs: {output_dir}")
    logger.info("=" * 60)
    
    # Print comparison guidance
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH POINTWISE")
    logger.info("=" * 60)
    logger.info("""
To compare with pointwise classification, run:
  python train_xgboost_polars.py --train_days {0}

Key differences:
  - Pointwise: Optimizes per-item prediction (AUC-focused)
  - Pairwise:  Optimizes relative ranking within users (NDCG-focused)

When to use each:
  - Pointwise: Click prediction, probability calibration needed
  - Pairwise:  Final ranking stage, position-aware metrics matter
""".format(args.train_days or 0))


if __name__ == '__main__':
    main()

