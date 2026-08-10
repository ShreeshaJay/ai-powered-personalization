"""
Chapter 6: Model Evaluation Script
==================================
Evaluate trained XGBoost ranker on test data.

Usage:
------
# Evaluate a trained model
python evaluate.py --model_path outputs/models/20231215_120000

# Evaluate with custom data
python evaluate.py --model_path outputs/models/20231215_120000 --data_dir path/to/data

Output:
-------
Metrics printed to console and saved to model directory.
"""

import argparse
import logging
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, GTSConfig
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
        description='Evaluate XGBoost ranker on test data'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
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
        help='Fraction of test data to sample'
    )
    parser.add_argument(
        '--completion_threshold',
        type=int,
        default=50,
        help='Played ratio threshold for positive label'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate evaluation plots'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: model directory)'
    )
    
    return parser.parse_args()


def plot_roc_curve(y_true, y_proba, output_path: Path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label='XGBoost')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {output_path / 'roc_curve.png'}")


def plot_precision_recall_curve(y_true, y_proba, output_path: Path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'pr_curve.png', dpi=150)
    plt.close()
    logger.info(f"PR curve saved to {output_path / 'pr_curve.png'}")


def plot_score_distribution(y_true, y_proba, output_path: Path):
    """Plot score distribution for positive and negative classes."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(
        y_proba[y_true == 0],
        bins=50,
        alpha=0.5,
        label='Negative (incomplete)',
        color='red'
    )
    plt.hist(
        y_proba[y_true == 1],
        bins=50,
        alpha=0.5,
        label='Positive (complete)',
        color='green'
    )
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png', dpi=150)
    plt.close()
    logger.info(f"Score distribution saved to {output_path / 'score_distribution.png'}")


def evaluate(args):
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("XGBoost Ranker Evaluation")
    logger.info("=" * 60)
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_path
    
    # =========================================================================
    # Load Model
    # =========================================================================
    logger.info(f"\n--- Loading Model from {model_path} ---")
    
    model = XGBoostRanker.from_pretrained(str(model_path))
    logger.info(f"Model loaded. Features: {model.feature_columns}")
    
    # =========================================================================
    # Load Test Data
    # =========================================================================
    logger.info("\n--- Loading Test Data ---")
    
    # Load model config to get GTS parameters
    config_path = model_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        gts_config = GTSConfig(
            train_days=train_config.get('train_days', 300),
            gap_minutes=train_config.get('gap_minutes', 30),
            test_days=train_config.get('test_days', 1)
        )
    else:
        gts_config = GTSConfig()
    
    _, test_df = get_train_test_data(
        data_dir=args.data_dir,
        task='listen_completion',
        gts_config=gts_config,
        sample_frac=args.sample_frac,
        as_pandas=True
    )
    
    test_df['label'] = (test_df['played_ratio_pct'] >= args.completion_threshold).astype(int)
    logger.info(f"Loaded {len(test_df):,} test samples")
    logger.info(f"Positive rate: {test_df['label'].mean():.2%}")
    
    # =========================================================================
    # Generate Predictions
    # =========================================================================
    logger.info("\n--- Generating Predictions ---")
    
    y_proba = model.predict_proba(test_df)
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = test_df['label'].values
    
    # =========================================================================
    # Calculate Metrics
    # =========================================================================
    logger.info("\n--- Metrics ---")
    
    metrics = model.evaluate(test_df, test_df['label'])
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    # Classification report
    logger.info("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Incomplete', 'Complete']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    logger.info(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # =========================================================================
    # Generate Plots
    # =========================================================================
    if args.plot:
        logger.info("\n--- Generating Plots ---")
        plot_roc_curve(y_true, y_proba, output_dir)
        plot_precision_recall_curve(y_true, y_proba, output_dir)
        plot_score_distribution(y_true, y_proba, output_dir)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    eval_results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_true),
        'positive_rate': float(y_true.mean())
    }
    
    eval_path = output_dir / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {eval_path}")
    logger.info("=" * 60)
    
    return eval_results


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)

