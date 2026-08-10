"""
Chapter 6: Inference Script
===========================
Generate predictions using trained XGBoost ranker.

Usage:
------
# Predict on test data
python predict.py --model_path outputs/models/20231215_120000

# Predict with custom input (CSV file)
python predict.py --model_path outputs/models/20231215_120000 --input_file data.csv

# Get top-k recommendations per user
python predict.py --model_path outputs/models/20231215_120000 --top_k 10

Output:
-------
Predictions saved to outputs/predictions/ or specified output path.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, PREDICTIONS_DIR, GTSConfig, ensure_directories
from data_loader import get_train_test_data
from models import XGBoostRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate predictions using trained XGBoost ranker'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Path to input CSV file (default: use test split from dataset)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(DATA_DIR),
        help='Path to Yambda flat data directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file path (default: outputs/predictions/predictions.parquet)'
    )
    parser.add_argument(
        '--sample_frac',
        type=float,
        default=None,
        help='Fraction of data to sample'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Return top-k items per user (for ranking)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100000,
        help='Batch size for prediction (memory management)'
    )
    
    return parser.parse_args()


def predict_batch(
    model: XGBoostRanker,
    df: pd.DataFrame,
    batch_size: int = 100000
) -> np.ndarray:
    """
    Generate predictions in batches to manage memory.
    
    Args:
        model: Trained XGBoostRanker
        df: Input DataFrame
        batch_size: Number of samples per batch
        
    Returns:
        Array of predicted probabilities
    """
    n_samples = len(df)
    predictions = np.zeros(n_samples)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = df.iloc[start_idx:end_idx]
        predictions[start_idx:end_idx] = model.predict_proba(batch)
        
        if n_samples > batch_size:
            logger.info(f"Processed {end_idx:,}/{n_samples:,} samples")
    
    return predictions


def rank_items_per_user(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_k: int
) -> pd.DataFrame:
    """
    Rank items per user and return top-k.
    
    Args:
        df: DataFrame with 'uid' and 'item_id' columns
        scores: Predicted scores
        top_k: Number of items to return per user
        
    Returns:
        DataFrame with top-k items per user
    """
    df = df.copy()
    df['score'] = scores
    
    # Rank within each user
    df['rank'] = df.groupby('uid')['score'].rank(ascending=False, method='first')
    
    # Filter to top-k
    top_k_df = df[df['rank'] <= top_k].copy()
    top_k_df = top_k_df.sort_values(['uid', 'rank'])
    
    logger.info(f"Generated top-{top_k} recommendations for {df['uid'].nunique():,} users")
    
    return top_k_df


def predict(args):
    """Main prediction function."""
    logger.info("=" * 60)
    logger.info("XGBoost Ranker Inference")
    logger.info("=" * 60)
    
    ensure_directories()
    
    # =========================================================================
    # Load Model
    # =========================================================================
    logger.info(f"\n--- Loading Model from {args.model_path} ---")
    
    model = XGBoostRanker.from_pretrained(args.model_path)
    logger.info(f"Model loaded. Features: {model.feature_columns}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\n--- Loading Data ---")
    
    if args.input_file:
        # Load from custom CSV
        logger.info(f"Loading from {args.input_file}")
        df = pd.read_csv(args.input_file)
    else:
        # Load test split from dataset
        _, df = get_train_test_data(
            data_dir=args.data_dir,
            task='listen_completion',
            gts_config=GTSConfig(),
            sample_frac=args.sample_frac,
            as_pandas=True
        )
    
    logger.info(f"Loaded {len(df):,} samples")
    
    # =========================================================================
    # Generate Predictions
    # =========================================================================
    logger.info("\n--- Generating Predictions ---")
    
    scores = predict_batch(model, df, batch_size=args.batch_size)
    
    logger.info(f"Score statistics:")
    logger.info(f"  Mean: {scores.mean():.4f}")
    logger.info(f"  Std:  {scores.std():.4f}")
    logger.info(f"  Min:  {scores.min():.4f}")
    logger.info(f"  Max:  {scores.max():.4f}")
    
    # =========================================================================
    # Optionally Rank Items
    # =========================================================================
    if args.top_k:
        logger.info(f"\n--- Ranking Top-{args.top_k} Items per User ---")
        output_df = rank_items_per_user(df, scores, args.top_k)
    else:
        output_df = df.copy()
        output_df['score'] = scores
    
    # =========================================================================
    # Save Predictions
    # =========================================================================
    logger.info("\n--- Saving Predictions ---")
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = PREDICTIONS_DIR / 'predictions.parquet'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet for efficiency
    if str(output_path).endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_parquet(output_path, index=False)
    
    logger.info(f"Predictions saved to: {output_path}")
    logger.info(f"Shape: {output_df.shape}")
    
    logger.info("=" * 60)
    
    return output_df


if __name__ == '__main__':
    args = parse_args()
    predict(args)

