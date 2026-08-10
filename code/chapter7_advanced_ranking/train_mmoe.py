"""
Chapter 7: MMoE Training Script for Yambda Dataset - Multi-Task Learning

This script trains a MMoE (Multi-gate Mixture-of-Experts) model for multi-task
learning on the Yambda dataset with two tasks:
    1. Listen Completion (primary): played_ratio_pct >= 50
    2. Like Prediction (secondary): Whether the user liked the item

Multi-task learning allows the model to:
- Share representations across related tasks
- Improve generalization through auxiliary tasks
- Handle multiple objectives simultaneously

Usage:
    # Quick run (30 days)
    python train_mmoe.py --train_days 30
    
    # Full dataset
    python train_mmoe.py --train_days 0
    
    # With custom hyperparameters
    python train_mmoe.py --train_days 30 --num_experts 6 --expert_dims 512,256

Author: Chapter 7 - Advanced Ranking Models
"""

import argparse
import logging
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl
from polars import col
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))

from models.mmoe import MMoE, MMoEConfig, MMoEDataset, MultiTaskLoss, collate_fn
from utils.polars_pipeline import load_yambda_polars, PolarsPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Configuration
# ============================================================================

SPARSE_FEATURES = ['uid', 'item_id', 'hour_of_day', 'day_of_week']

DENSE_FEATURES = [
    'is_organic', 'track_length_seconds',
    'user_total_listens', 'user_avg_completion', 'user_std_completion',
    'user_median_completion', 'user_unique_items', 'user_organic_ratio',
    'user_active_span', 'user_listen_rate',
    'item_total_plays', 'item_avg_completion', 'item_std_completion',
    'item_unique_listeners', 'item_organic_ratio', 'item_repeat_ratio',
    'has_listened_before', 'previous_listen_count',
]

# Multi-Task Configuration
# 4 tasks for comprehensive user engagement modeling:
#   1. engagement:  P(played_ratio_pct > 0)   - User started listening (binary)
#   2. completion:  P(played_ratio_pct >= 50) - User finished listening (binary)
#   3. play_ratio:  E[played_ratio_pct/100]   - Expected play percentage (regression)
#   4. like:        P(user liked item)        - Positive explicit feedback (binary)
#   5. dislike:     P(user disliked item)     - Negative explicit feedback (binary)
#
# Value function example:
#   value = w1*P(engagement) + w2*pred_ratio + w3*P(like) - w4*P(dislike)
#   expected_listen_time = P(engagement) * pred_ratio * track_length

TASK_NAMES = ['engagement', 'completion', 'play_ratio', 'like', 'dislike']
REGRESSION_TASKS = ['play_ratio']  # Tasks that use MSE loss instead of BCE
BINARY_TASKS = ['engagement', 'completion', 'like', 'dislike']
COMPLETION_THRESHOLD = 50
ENGAGEMENT_THRESHOLD = 0  # Any non-zero listening

# Time window for like/dislike labels (in timestamp units)
# Yambda timestamps are in 5-second bins, so 24 hours = 24 * 60 * 60 / 5 = 17,280 units
LIKE_WINDOW_HOURS = 24
TIMESTAMP_BIN_SECONDS = 5
LIKE_WINDOW_TS = (LIKE_WINDOW_HOURS * 60 * 60) // TIMESTAMP_BIN_SECONDS  # 17,280


# ============================================================================
# Multi-Task Data Loading
# ============================================================================

def _add_time_windowed_labels(
    listens_lf: pl.LazyFrame,
    events_lf: pl.LazyFrame,
    label_name: str,
    window_ts: int
) -> pl.LazyFrame:
    """
    Add binary labels based on time-windowed join with events (likes/dislikes).
    
    A listen is labeled as positive if the user performed the action (like/dislike)
    on the same item within [listen_ts, listen_ts + window_ts].
    
    This prevents temporal leakage by ensuring we only use future actions that
    could have been caused by this specific listen.
    
    Args:
        listens_lf: LazyFrame of listens with columns [uid, item_id, timestamp, ...]
        events_lf: LazyFrame of events (likes/dislikes) with columns [uid, item_id, timestamp]
        label_name: Name for the output label column (e.g., 'label_liked')
        window_ts: Maximum time difference (in timestamp units) for event to count
        
    Returns:
        LazyFrame with label column added
    """
    # Rename event timestamp to avoid collision
    events_with_ts = events_lf.select([
        col('uid'),
        col('item_id'),
        col('timestamp').alias('event_ts')
    ])
    
    # Join on (uid, item_id) - this creates multiple rows if user has multiple events for same item
    joined = listens_lf.join(
        events_with_ts,
        on=['uid', 'item_id'],
        how='left'
    )
    
    # Filter to events within window: listen_ts <= event_ts <= listen_ts + window
    # Then aggregate to get a single flag per listen
    result = joined.with_columns([
        # Check if event is within window (and event exists, i.e., not null)
        pl.when(
            col('event_ts').is_not_null() &
            (col('event_ts') >= col('timestamp')) &
            (col('event_ts') <= col('timestamp') + window_ts)
        ).then(1).otherwise(0).alias('_event_in_window')
    ]).group_by(
        # Group by all original columns to collapse multiple event matches
        [c for c in listens_lf.collect_schema().names()]
    ).agg([
        # If any event was in window, mark as positive
        pl.max('_event_in_window').alias(label_name)
    ]).with_columns([
        col(label_name).fill_null(0).cast(pl.Int8)
    ])
    
    return result


def load_multitask_data(data_dir: str, train_days: Optional[int] = None, test_days: int = 1, gap_seconds: int = 1800):
    """
    Load Yambda data with multi-task labels for 4 tasks:
    
    1. engagement:  played_ratio_pct > 0   (user started listening)
    2. completion:  played_ratio_pct >= 50 (user finished listening)
    3. like:        user liked the item within 24 hours of this listen
    4. dislike:     user disliked the item within 24 hours of this listen
    
    IMPORTANT: Like/dislike labels use a time-windowed join to prevent temporal leakage.
    A listen is only labeled as "liked" if the like action occurred AFTER the listen
    and within the configured time window (default: 24 hours).
    
    Returns LazyFrames with all label columns added.
    """
    data_path = Path(data_dir)
    likes_path = data_path / 'likes.parquet'
    dislikes_path = data_path / 'dislikes.parquet'
    
    logger.info(f"Loading listens from {data_path / 'listens.parquet'}")
    
    train_lf, test_lf = load_yambda_polars(
        data_dir=data_dir, train_days=train_days, test_days=test_days, gap_seconds=gap_seconds, lazy=True
    )
    
    # Add engagement label (played_ratio_pct > 0)
    # Note: In listens.parquet, all rows have some listening, so engagement is mostly 1
    # This becomes more meaningful when comparing against impression data
    train_lf = train_lf.with_columns([
        (col('played_ratio_pct') > 0).cast(pl.Int8).alias('label_engagement'),
        # Regression target: play_ratio as fraction [0, 1]
        (col('played_ratio_pct') / 100.0).clip(0, 1).alias('label_play_ratio')
    ])
    test_lf = test_lf.with_columns([
        (col('played_ratio_pct') > 0).cast(pl.Int8).alias('label_engagement'),
        # Regression target: play_ratio as fraction [0, 1]
        (col('played_ratio_pct') / 100.0).clip(0, 1).alias('label_play_ratio')
    ])
    
    # Add like labels with TIME-WINDOWED JOIN (prevents temporal leakage)
    if likes_path.exists():
        logger.info(f"Loading likes from {likes_path}")
        logger.info(f"  Using {LIKE_WINDOW_HOURS}-hour window for like attribution")
        likes_lf = pl.scan_parquet(likes_path)
        
        train_lf = _add_time_windowed_labels(train_lf, likes_lf, 'label_liked', LIKE_WINDOW_TS)
        test_lf = _add_time_windowed_labels(test_lf, likes_lf, 'label_liked', LIKE_WINDOW_TS)
    else:
        logger.warning(f"Likes file not found, creating dummy label_liked column")
        train_lf = train_lf.with_columns([pl.lit(0).cast(pl.Int8).alias('label_liked')])
        test_lf = test_lf.with_columns([pl.lit(0).cast(pl.Int8).alias('label_liked')])
    
    # Add dislike labels with TIME-WINDOWED JOIN (prevents temporal leakage)
    if dislikes_path.exists():
        logger.info(f"Loading dislikes from {dislikes_path}")
        logger.info(f"  Using {LIKE_WINDOW_HOURS}-hour window for dislike attribution")
        dislikes_lf = pl.scan_parquet(dislikes_path)
        
        train_lf = _add_time_windowed_labels(train_lf, dislikes_lf, 'label_disliked', LIKE_WINDOW_TS)
        test_lf = _add_time_windowed_labels(test_lf, dislikes_lf, 'label_disliked', LIKE_WINDOW_TS)
    else:
        logger.warning(f"Dislikes file not found, creating dummy label_disliked column")
        train_lf = train_lf.with_columns([pl.lit(0).cast(pl.Int8).alias('label_disliked')])
        test_lf = test_lf.with_columns([pl.lit(0).cast(pl.Int8).alias('label_disliked')])
    
    # Log label statistics
    logger.info("Computing label statistics...")
    train_df_sample = train_lf.head(100000).collect()
    logger.info("Estimated label statistics (from 100K sample):")
    logger.info(f"  engagement (binary): {train_df_sample['label_engagement'].mean():.2%}")
    logger.info(f"  play_ratio (regression): mean={train_df_sample['label_play_ratio'].mean():.3f}, std={train_df_sample['label_play_ratio'].std():.3f}")
    # Note: 'label' (completion) is added by PolarsPipeline later
    logger.info(f"  like (binary):       {train_df_sample['label_liked'].mean():.2%}")
    logger.info(f"  dislike (binary):    {train_df_sample['label_disliked'].mean():.2%}")
    
    return train_lf, test_lf


# ============================================================================
# Data Preparation
# ============================================================================

class MMoEFeatureProcessor:
    """Prepares features for MMoE model with multi-task labels."""
    
    def __init__(self, sparse_features: list, dense_features: list, task_names: list):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.task_names = task_names
        self.sparse_encoders: Dict[str, LabelEncoder] = {}
        self.dense_scaler: Optional[StandardScaler] = None
        self.vocab_sizes: Dict[str, int] = {}
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'MMoEFeatureProcessor':
        logger.info("Fitting feature processor...")
        for col_name in self.sparse_features:
            if col_name in df.columns:
                le = LabelEncoder()
                values = df[col_name].fillna(-1).astype(str)
                le.fit(values)
                self.sparse_encoders[col_name] = le
                self.vocab_sizes[col_name] = len(le.classes_) + 1
                logger.info(f"  {col_name}: vocab_size={self.vocab_sizes[col_name]:,}")
        
        if self.dense_features:
            dense_cols = [c for c in self.dense_features if c in df.columns]
            dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
            self.dense_scaler = StandardScaler()
            self.dense_scaler.fit(dense_data)
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        sparse_data = {}
        for col_name in self.sparse_features:
            if col_name in df.columns and col_name in self.sparse_encoders:
                values = df[col_name].fillna(-1).astype(str)
                le = self.sparse_encoders[col_name]
                encoded = np.zeros(len(values), dtype=np.int64)
                known_mask = values.isin(le.classes_)
                encoded[known_mask] = le.transform(values[known_mask]) + 1
                sparse_data[col_name] = encoded
        
        dense_cols = [c for c in self.dense_features if c in df.columns]
        dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
        if self.dense_scaler is not None:
            dense_data = self.dense_scaler.transform(dense_data)
        
        labels = {}
        # engagement: played_ratio_pct > 0 (binary)
        if 'label_engagement' in df.columns:
            labels['engagement'] = df['label_engagement'].values.astype(np.float32)
        # completion: played_ratio_pct >= 50 (binary, stored as 'label' by PolarsPipeline)
        if 'label' in df.columns:
            labels['completion'] = df['label'].values.astype(np.float32)
        # play_ratio: played_ratio_pct / 100 (regression, continuous [0, 1])
        if 'label_play_ratio' in df.columns:
            labels['play_ratio'] = df['label_play_ratio'].values.astype(np.float32)
        # like: (uid, item_id) in likes.parquet (binary)
        if 'label_liked' in df.columns:
            labels['like'] = df['label_liked'].values.astype(np.float32)
        # dislike: (uid, item_id) in dislikes.parquet (binary)
        if 'label_disliked' in df.columns:
            labels['dislike'] = df['label_disliked'].values.astype(np.float32)
        
        return sparse_data, dense_data, labels
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        self.fit(df)
        return self.transform(df)


# ============================================================================
# Artifact Export for Chapter 8: Value Functions
# ============================================================================

def export_inference_artifacts(
    feature_processor: 'MMoEFeatureProcessor',
    pipeline: 'PolarsPipeline',
    output_path: Path,
) -> None:
    """
    Export artifacts needed for inference in Chapter 8.
    
    This allows Chapter 8 to:
    1. Load raw data
    2. Transform features using saved encoders/scalers
    3. Load model and run inference
    4. Compute value functions on predictions
    """
    import pickle
    
    logger.info("\nExporting inference artifacts...")
    
    inference_dir = output_path / 'inference'
    inference_dir.mkdir(exist_ok=True)
    
    # 1. Save feature processor (encoders + scaler)
    processor_state = {
        'sparse_features': feature_processor.sparse_features,
        'dense_features': feature_processor.dense_features,
        'task_names': feature_processor.task_names,
        'vocab_sizes': feature_processor.vocab_sizes,
        'sparse_encoders': feature_processor.sparse_encoders,
        'dense_scaler': feature_processor.dense_scaler,
    }
    with open(inference_dir / 'feature_processor.pkl', 'wb') as f:
        pickle.dump(processor_state, f)
    logger.info(f"  Saved feature_processor.pkl")
    
    # 2. Save pipeline state (for polars feature engineering)
    pipeline.save(str(inference_dir / 'polars_pipeline'))
    logger.info(f"  Saved polars_pipeline/")
    
    # 3. Create inference example script
    inference_example = '''"""
Chapter 8: Loading MMoE Model for Inference

This script demonstrates how to load the trained MMoE model
and run inference on new data to compute value functions.

Usage:
    python inference_example.py --data_dir /path/to/yambda/flat
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.mmoe import MMoE, MMoEConfig
from utils.polars_pipeline import load_yambda_polars, PolarsPipeline


def load_inference_artifacts(model_dir: Path):
    """Load all artifacts needed for inference."""
    
    # 1. Load model checkpoint
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    model = MMoE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Load feature processor
    with open(model_dir / 'inference' / 'feature_processor.pkl', 'rb') as f:
        processor_state = pickle.load(f)
    
    # 3. Load polars pipeline
    pipeline = PolarsPipeline()
    pipeline.load(str(model_dir / 'inference' / 'polars_pipeline'))
    
    return model, config, processor_state, pipeline


def transform_for_inference(df: pd.DataFrame, processor_state: dict):
    """Transform pandas DataFrame for model input."""
    
    # Transform sparse features
    sparse_data = {}
    for col in processor_state['sparse_features']:
        if col in df.columns and col in processor_state['sparse_encoders']:
            values = df[col].fillna(-1).astype(str)
            le = processor_state['sparse_encoders'][col]
            
            encoded = np.zeros(len(values), dtype=np.int64)
            known_mask = values.isin(le.classes_)
            encoded[known_mask] = le.transform(values[known_mask]) + 1
            sparse_data[col] = torch.LongTensor(encoded)
    
    # Transform dense features
    dense_cols = [c for c in processor_state['dense_features'] if c in df.columns]
    dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
    if processor_state['dense_scaler'] is not None:
        dense_data = processor_state['dense_scaler'].transform(dense_data)
    dense_data = torch.FloatTensor(dense_data)
    
    return sparse_data, dense_data


def predict_with_value_function(
    model: MMoE,
    sparse_data: dict,
    dense_data: torch.Tensor,
    weights: dict = None,
) -> dict:
    """
    Run inference and compute value function.
    
    Args:
        model: Trained MMoE model
        sparse_data: Dict of sparse feature tensors
        dense_data: Dense feature tensor
        weights: Dict of task weights, e.g., {'completion': 1.0, 'like': 2.0}
    
    Returns:
        Dict with predictions and value scores
    """
    if weights is None:
        weights = {'completion': 1.0, 'like': 1.0}
    
    model.eval()
    with torch.no_grad():
        outputs = model(sparse_data, dense_data)
        probs = {name: torch.sigmoid(logits).numpy().flatten() 
                 for name, logits in outputs.items()}
    
    # Compute value function
    value = sum(weights.get(task, 1.0) * probs[task] for task in probs)
    
    return {
        'prob_completion': probs['completion'],
        'prob_like': probs['like'],
        'value': value,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to MMoE output directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Yambda flat data directory')
    parser.add_argument('--completion_weight', type=float, default=1.0)
    parser.add_argument('--like_weight', type=float, default=2.0)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    print("Loading model and artifacts...")
    model, config, processor_state, pipeline = load_inference_artifacts(model_dir)
    
    print("Loading test data...")
    _, test_lf = load_yambda_polars(args.data_dir, train_days=30, test_days=1)
    test_polars_df, _ = pipeline.transform(test_lf)
    test_df = test_polars_df.to_pandas()
    
    print(f"Running inference on {len(test_df):,} samples...")
    sparse_data, dense_data = transform_for_inference(test_df, processor_state)
    
    weights = {
        'completion': args.completion_weight,
        'like': args.like_weight,
    }
    
    results = predict_with_value_function(model, sparse_data, dense_data, weights)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'uid': test_df['uid'].values,
        'item_id': test_df['item_id'].values,
        'prob_completion': results['prob_completion'],
        'prob_like': results['prob_like'],
        'value': results['value'],
    })
    
    # Show top recommendations by value
    print("\\nTop 10 recommendations by value:")
    print(output_df.nlargest(10, 'value')[['uid', 'item_id', 'prob_completion', 'prob_like', 'value']])
    
    # Save results
    output_df.to_parquet(model_dir / 'inference_results.parquet', index=False)
    print(f"\\nSaved inference results to {model_dir / 'inference_results.parquet'}")
'''
    
    with open(inference_dir / 'inference_example.py', 'w') as f:
        f.write(inference_example)
    logger.info(f"  Saved inference_example.py")
    
    logger.info("  Inference artifacts export complete!")


def export_predictions_for_value_functions(
    model: MMoE,
    test_loader: DataLoader,
    test_df: pd.DataFrame,
    test_labels: Dict[str, np.ndarray],
    device: torch.device,
    output_path: Path,
) -> Dict[str, any]:
    """
    Export pre-computed predictions for quick analysis in Chapter 8.
    
    Note: For production-like inference, use the artifacts in inference/
    which allow loading the model and running inference on new data.
    
    This file provides pre-computed predictions for:
    - Quick iteration on value function weights
    - Calibration analysis
    - Offline evaluation
    """
    logger.info("\nExporting pre-computed predictions for analysis...")
    
    model.eval()
    
    # Collect predictions
    all_logits = {name: [] for name in model.task_names}
    all_probs = {name: [] for name in model.task_names}
    all_gate_weights = {name: [] for name in model.task_names}
    
    with torch.no_grad():
        for sparse_batch, dense_batch, _ in test_loader:
            sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
            if dense_batch is not None:
                dense_batch = dense_batch.to(device)
            
            # Forward pass to get logits
            outputs = model(sparse_batch, dense_batch)
            
            for task_name in model.task_names:
                logits = outputs[task_name]
                all_logits[task_name].append(logits.cpu().numpy())
                all_probs[task_name].append(torch.sigmoid(logits).cpu().numpy())
            
            # Extract gate weights (how each task weights the experts)
            # Recreate input for gate extraction
            embed_list = []
            for feat_name, indices in sparse_batch.items():
                embed = model.embeddings[feat_name](indices)
                embed_list.append(embed)
            sparse_concat = torch.cat(embed_list, dim=1)
            
            if dense_batch is not None:
                x = torch.cat([sparse_concat, dense_batch], dim=1)
            else:
                x = sparse_concat
            
            for task_idx, task_name in enumerate(model.task_names):
                gate_w = model.gates[task_idx](x).cpu().numpy()
                all_gate_weights[task_name].append(gate_w)
    
    # Concatenate all batches
    for task_name in model.task_names:
        all_logits[task_name] = np.concatenate(all_logits[task_name]).flatten()
        all_probs[task_name] = np.concatenate(all_probs[task_name]).flatten()
        all_gate_weights[task_name] = np.concatenate(all_gate_weights[task_name], axis=0)
    
    # 1. Export predictions DataFrame
    # Build predictions with all 4 task outputs
    predictions_data = {
        'uid': test_df['uid'].values,
        'item_id': test_df['item_id'].values,
        'timestamp': test_df['timestamp'].values if 'timestamp' in test_df.columns else np.arange(len(test_df)),
    }
    
    # Add labels and predictions for all tasks
    for task_name in model.task_names:
        if task_name in test_labels:
            predictions_data[f'label_{task_name}'] = test_labels[task_name]
        if task_name in all_probs:
            predictions_data[f'prob_{task_name}'] = all_probs[task_name]
        if task_name in all_logits:
            predictions_data[f'logit_{task_name}'] = all_logits[task_name]
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_parquet(output_path / 'predictions.parquet', index=False)
    logger.info(f"  Saved predictions.parquet ({len(predictions_df):,} samples)")
    logger.info(f"    Columns: {list(predictions_df.columns)}")
    
    # 2. Export gate weights (aggregated statistics)
    gate_weights_summary = {}
    for task_name in model.task_names:
        gate_w = all_gate_weights[task_name]
        gate_weights_summary[task_name] = {
            'mean_weights': gate_w.mean(axis=0).tolist(),
            'std_weights': gate_w.std(axis=0).tolist(),
            'expert_names': [f'expert_{i}' for i in range(gate_w.shape[1])],
        }
    
    # Save detailed gate weights for a sample (first 10000)
    sample_size = min(10000, len(test_df))
    gate_weights_df = pd.DataFrame({
        'sample_idx': np.arange(sample_size),
    })
    for task_name in model.task_names:
        gate_w = all_gate_weights[task_name][:sample_size]
        for exp_idx in range(gate_w.shape[1]):
            gate_weights_df[f'{task_name}_expert_{exp_idx}'] = gate_w[:, exp_idx]
    
    gate_weights_df.to_parquet(output_path / 'gate_weights_sample.parquet', index=False)
    logger.info(f"  Saved gate_weights_sample.parquet ({sample_size:,} samples)")
    
    # 3. Export calibration data
    calibration_data = {}
    for task_name in model.task_names:
        probs = all_probs[task_name]
        labels = test_labels[task_name]
        
        # Compute calibration curve
        try:
            fraction_positives, mean_predicted = calibration_curve(
                labels, probs, n_bins=10, strategy='uniform'
            )
            calibration_data[task_name] = {
                'fraction_positives': fraction_positives.tolist(),
                'mean_predicted': mean_predicted.tolist(),
                'n_samples': int(len(labels)),
                'positive_rate': float(labels.mean()),
            }
        except Exception as e:
            logger.warning(f"  Could not compute calibration for {task_name}: {e}")
            calibration_data[task_name] = {'error': str(e)}
    
    with open(output_path / 'calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=2)
    logger.info(f"  Saved calibration.json")
    
    # 4. Export value function configuration
    task_stats = {}
    for task_name in model.task_names:
        probs = all_probs[task_name]
        labels = test_labels[task_name]
        
        if task_name in REGRESSION_TASKS:
            # Regression task: use MSE, MAE, R² instead of AUC
            task_stats[task_name] = {
                'task_type': 'regression',
                'label_mean': float(labels.mean()),
                'label_std': float(labels.std()),
                'pred_mean': float(probs.mean()),
                'pred_std': float(probs.std()),
                'mse': float(mean_squared_error(labels, probs)),
                'mae': float(mean_absolute_error(labels, probs)),
                'r2': float(r2_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
            }
        else:
            # Binary task: use AUC-ROC
            task_stats[task_name] = {
                'task_type': 'binary',
                'positive_rate': float(labels.mean()),
                'pred_mean': float(probs.mean()),
                'pred_std': float(probs.std()),
                'auc_roc': float(roc_auc_score(labels, probs)),
            }
    
    # Compute pairwise task correlations (labels and predictions)
    task_correlations = {'label': {}, 'prediction': {}}
    task_list = model.task_names
    for i, task1 in enumerate(task_list):
        for task2 in task_list[i+1:]:
            if task1 in test_labels and task2 in test_labels:
                corr = float(np.corrcoef(test_labels[task1], test_labels[task2])[0, 1])
                task_correlations['label'][f'{task1}_{task2}'] = corr
            if task1 in all_probs and task2 in all_probs:
                corr = float(np.corrcoef(all_probs[task1], all_probs[task2])[0, 1])
                task_correlations['prediction'][f'{task1}_{task2}'] = corr
    
    # Compute inverse positive rate weights (handle edge cases)
    # For regression tasks, use 1.0 as default weight
    inverse_pos_weights = {}
    for task in model.task_names:
        if task in REGRESSION_TASKS:
            # Regression task: no positive rate, use default weight
            inverse_pos_weights[task] = 1.0
        else:
            pos_rate = task_stats[task].get('positive_rate', 0)
            if pos_rate > 0:
                inverse_pos_weights[task] = 1.0 / pos_rate
            else:
                inverse_pos_weights[task] = 1.0  # fallback
    
    value_function_config = {
        'task_names': model.task_names,
        'task_stats': task_stats,
        'gate_weights_summary': gate_weights_summary,
        'task_correlations': task_correlations,
        'suggested_weights': {
            # Equal weighting across all tasks
            'equal': {task: 1.0 for task in model.task_names},
            # Weight inversely proportional to positive rate
            'inverse_positive_rate': inverse_pos_weights,
            # Business-oriented: subtract dislike penalty
            'engagement_focused': {
                'engagement': 1.0,
                'completion': 2.0,
                'play_ratio': 0.5,  # Regression: predicted listen percentage
                'like': 5.0,
                'dislike': -10.0,  # Negative! Penalize predicted dislikes
            },
            # Long-term satisfaction: prioritize likes, heavily penalize dislikes
            'satisfaction_focused': {
                'engagement': 0.5,
                'completion': 1.0,
                'play_ratio': 1.0,  # Regression: predicted listen percentage
                'like': 10.0,
                'dislike': -20.0,
            },
            # Expected listen time focused (uses regression prediction)
            'listen_time_focused': {
                'engagement': 0.5,
                'completion': 0.5,
                'play_ratio': 3.0,  # High weight on predicted percentage listened
                'like': 2.0,
                'dislike': -5.0,
            },
        },
        'usage_example': (
            "# Load predictions and compute value:\n"
            "df = pd.read_parquet('predictions.parquet')\n"
            "# Example: Value function with regression task (play_ratio)\n"
            "weights = {'engagement': 1.0, 'completion': 2.0, 'play_ratio': 1.0, 'like': 5.0, 'dislike': -10.0}\n"
            "df['value'] = (\n"
            "    weights['engagement'] * df['prob_engagement'] +\n"
            "    weights['completion'] * df['prob_completion'] +\n"
            "    weights['play_ratio'] * df['prob_play_ratio'] +\n"  # Regression: 0-1 predicted percentage
            "    weights['like'] * df['prob_like'] +\n"
            "    weights['dislike'] * df['prob_dislike']  # Negative weight!\n"
            ")\n"
            "# Alternative: Expected listen time\n"
            "df['expected_listen_time'] = df['prob_engagement'] * df['prob_play_ratio'] * track_length_seconds\n"
            "# Rank by value for final recommendations\n"
            "df_ranked = df.sort_values('value', ascending=False)"
        ),
    }
    
    with open(output_path / 'value_function_config.json', 'w') as f:
        json.dump(value_function_config, f, indent=2)
    logger.info(f"  Saved value_function_config.json")
    
    logger.info("  Artifact export complete!")
    
    return {
        'predictions_df': predictions_df,
        'calibration_data': calibration_data,
        'value_function_config': value_function_config,
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    task_losses = {name: 0.0 for name in model.task_names}
    n_batches = 0
    
    for sparse_batch, dense_batch, labels_batch in train_loader:
        sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
        if dense_batch is not None:
            dense_batch = dense_batch.to(device)
        labels_batch = {k: v.to(device) for k, v in labels_batch.items()}
        
        optimizer.zero_grad()
        outputs = model(sparse_batch, dense_batch)
        loss, batch_task_losses = criterion(outputs, labels_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for name, task_loss in batch_task_losses.items():
            task_losses[name] += task_loss.item()
        n_batches += 1
    
    return total_loss / n_batches, {name: loss / n_batches for name, loss in task_losses.items()}


def evaluate(model, data_loader, criterion, device):
    model.eval()
    
    all_preds = {name: [] for name in model.task_names}
    all_labels = {name: [] for name in model.task_names}
    total_loss = 0.0
    task_losses = {name: 0.0 for name in model.task_names}
    n_batches = 0
    
    with torch.no_grad():
        for sparse_batch, dense_batch, labels_batch in data_loader:
            sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
            if dense_batch is not None:
                dense_batch = dense_batch.to(device)
            labels_batch = {k: v.to(device) for k, v in labels_batch.items()}
            
            outputs = model(sparse_batch, dense_batch)
            loss, batch_task_losses = criterion(outputs, labels_batch)
            
            for name in model.task_names:
                probs = torch.sigmoid(outputs[name])
                all_preds[name].append(probs.cpu().numpy())
                all_labels[name].append(labels_batch[name].cpu().numpy())
            
            total_loss += loss.item()
            for name, task_loss in batch_task_losses.items():
                task_losses[name] += task_loss.item()
            n_batches += 1
    
    results = {'overall': {'loss': total_loss / n_batches}}
    
    for name in model.task_names:
        preds = np.concatenate(all_preds[name]).flatten()
        labels = np.concatenate(all_labels[name]).flatten()
        
        # Check if this is a regression task
        if name in REGRESSION_TASKS:
            # Regression metrics: MSE, MAE, R²
            results[name] = {
                'loss': task_losses[name] / n_batches,
                'mse': float(mean_squared_error(labels, preds)),
                'mae': float(mean_absolute_error(labels, preds)),
                'r2': float(r2_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0,
                'mean_pred': float(preds.mean()),
                'mean_label': float(labels.mean()),
            }
        elif len(np.unique(labels)) == 1:
            # Binary task with only one class in labels
            results[name] = {
                'loss': task_losses[name] / n_batches, 
                'auc_roc': 0.5, 
                'auc_pr': labels.mean(), 
                'f1_max': 0.0,
                'f1_threshold': 0.5,
                'positive_rate': float(labels.mean())
            }
        else:
            # Binary task: Compute F1-max: optimal threshold F1 (better for imbalanced data)
            precision, recall, thresholds = precision_recall_curve(labels, preds)
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_f1_idx = np.argmax(f1_scores)
            f1_max = f1_scores[best_f1_idx]
            best_threshold = thresholds[best_f1_idx]
            
            results[name] = {
                'loss': task_losses[name] / n_batches,
                'auc_roc': roc_auc_score(labels, preds),
                'auc_pr': average_precision_score(labels, preds),
                'f1_max': float(f1_max),  # F1 at optimal threshold
                'f1_threshold': float(best_threshold),  # The optimal threshold
                'positive_rate': float(labels.mean()),
            }
    
    return results


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MMoE on Yambda dataset (Multi-Task)')
    
    parser.add_argument('--data_dir', type=str, default=os.environ.get('YAMBDA_DATA_DIR', ''))
    parser.add_argument('--train_days', type=int, default=30)
    parser.add_argument('--test_days', type=int, default=1)
    
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--num_experts', type=int, default=4)  # 4 experts performed best in experiments
    parser.add_argument('--expert_dims', type=str, default='256,128')
    parser.add_argument('--tower_dims', type=str, default='64,32')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Task weights for multi-task loss (5 tasks: 4 binary + 1 regression)
    parser.add_argument('--engagement_weight', type=float, default=1.0)
    parser.add_argument('--completion_weight', type=float, default=1.0)
    parser.add_argument('--play_ratio_weight', type=float, default=1.0)  # Regression task weight
    parser.add_argument('--like_weight', type=float, default=2.0)  # Higher weight for rare positive task
    parser.add_argument('--dislike_weight', type=float, default=2.0)  # Higher weight for rare positive task
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss for imbalanced tasks (like, dislike)')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=3)
    
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    expert_dims = [int(x) for x in args.expert_dims.split(',')]
    tower_dims = [int(x) for x in args.tower_dims.split(',')]
    
    if not args.data_dir:
        logger.error("Data directory not specified. Set YAMBDA_DATA_DIR or use --data_dir")
        sys.exit(1)
    
    train_days = None if args.train_days == 0 else args.train_days
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_days_str = 'full' if train_days is None else f'{train_days}d'
    focal_str = '_focal' if args.use_focal_loss else ''
    exp_name = f'mmoe_{train_days_str}_{args.num_experts}exp{focal_str}_{timestamp}'
    output_path = Path(args.output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MMoE Multi-Task Training on Yambda Dataset")
    logger.info("=" * 70)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Binary tasks: engagement (w={args.engagement_weight}), completion (w={args.completion_weight}), "
                f"like (w={args.like_weight}), dislike (w={args.dislike_weight})")
    logger.info(f"Regression task: play_ratio (w={args.play_ratio_weight})")
    logger.info(f"Focal loss: {'enabled for like/dislike' if args.use_focal_loss else 'disabled'}")
    
    start_time = time.time()
    
    # Load Data
    logger.info("\n[1/5] Loading multi-task data...")
    train_lf, test_lf = load_multitask_data(data_dir=args.data_dir, train_days=train_days, test_days=args.test_days)
    
    # Feature Engineering
    logger.info("\n[2/5] Running feature pipeline...")
    pipeline = PolarsPipeline()
    pipeline.fit(train_lf)
    
    train_polars_df, _ = pipeline.transform(train_lf)
    test_polars_df, _ = pipeline.transform(test_lf)
    
    train_df = train_polars_df.to_pandas()
    test_df = test_polars_df.to_pandas()
    
    logger.info(f"Train samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # Prepare Features
    logger.info("\n[3/5] Preparing features for MMoE...")
    feature_processor = MMoEFeatureProcessor(SPARSE_FEATURES, DENSE_FEATURES, TASK_NAMES)
    
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    for task_name in TASK_NAMES:
        if task_name in train_labels:
            if task_name in REGRESSION_TASKS:
                logger.info(f"Task '{task_name}' (regression) - Train mean: {train_labels[task_name].mean():.3f}, std: {train_labels[task_name].std():.3f}")
            else:
                logger.info(f"Task '{task_name}' (binary) - Train positive rate: {train_labels[task_name].mean():.2%}")
    
    train_dataset = MMoEDataset(train_sparse, train_dense, train_labels)
    test_dataset = MMoEDataset(test_sparse, test_dense, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Train Model
    logger.info("\n[4/5] Training MMoE...")
    
    config = MMoEConfig(
        sparse_features=feature_processor.vocab_sizes,
        dense_features=DENSE_FEATURES,
        num_tasks=len(TASK_NAMES),
        task_names=TASK_NAMES,
        embed_dim=args.embed_dim,
        num_experts=args.num_experts,
        expert_dims=expert_dims,
        tower_dims=tower_dims,
        dropout=args.dropout,
    )
    
    model = MMoE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Configure multi-task loss with 5 tasks (4 binary + 1 regression)
    # Use focal loss for imbalanced binary tasks (like, dislike)
    # Use MSE loss for regression task (play_ratio)
    criterion = MultiTaskLoss(
        task_names=TASK_NAMES,
        task_weights={
            'engagement': args.engagement_weight,
            'completion': args.completion_weight,
            'play_ratio': args.play_ratio_weight,  # Regression task
            'like': args.like_weight,
            'dislike': args.dislike_weight,
        },
        regression_tasks=REGRESSION_TASKS,  # ['play_ratio'] - uses MSE loss
        use_focal_loss={
            'engagement': False,  # High positive rate, no focal loss needed
            'completion': False,  # High positive rate, no focal loss needed
            'play_ratio': False,  # Regression task, not applicable
            'like': args.use_focal_loss,  # Low positive rate (~0.8%)
            'dislike': args.use_focal_loss,  # Very low positive rate (~0.2%)
        },
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'test': []}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_task_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        completion_auc = test_metrics['completion']['auc_roc']
        like_auc_pr = test_metrics['like']['auc_pr']  # Use AUC-PR for imbalanced task
        
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Completion AUC: {completion_auc:.4f} | "
            f"Like AUC-PR: {like_auc_pr:.4f} | "  # AUC-PR better for imbalanced
            f"Time: {time.time() - epoch_start:.1f}s"
        )
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        scheduler.step(completion_auc)
        
        if completion_auc > best_auc:
            best_auc = completion_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'metrics': test_metrics}, output_path / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - start_time
    
    # Final Evaluation
    logger.info("\n[5/5] Final Evaluation...")
    checkpoint = torch.load(output_path / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_metrics = evaluate(model, train_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    
    for task_name in TASK_NAMES:
        metrics = test_metrics[task_name]
        logger.info(f"\n{task_name.upper()} Task:")
        
        if task_name in REGRESSION_TASKS:
            # Regression task metrics
            logger.info(f"  MSE:        {metrics['mse']:.4f}")
            logger.info(f"  MAE:        {metrics['mae']:.4f}")
            logger.info(f"  R²:         {metrics['r2']:.4f}")
            logger.info(f"  Mean pred:  {metrics['mean_pred']:.3f}")
            logger.info(f"  Mean label: {metrics['mean_label']:.3f}")
        else:
            # Binary task metrics
            logger.info(f"  Test AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"  Test AUC-PR:  {metrics['auc_pr']:.4f}  (better for imbalanced data)")
            logger.info(f"  F1-max:       {metrics['f1_max']:.4f}  (at threshold {metrics['f1_threshold']:.3f})")
            logger.info(f"  Positive rate: {metrics['positive_rate']:.2%}")
    
    results = {
        'experiment': exp_name,
        'train_days': train_days,
        'best_epoch': best_epoch,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'n_params': n_params,
        'train_time_seconds': train_time,
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # =========================================================================
    # Export artifacts for Chapter 8: Value Functions
    # =========================================================================
    
    # 1. Export inference artifacts (model + feature processor + pipeline)
    #    This allows Chapter 8 to load the model and run inference on new data
    export_inference_artifacts(
        feature_processor=feature_processor,
        pipeline=pipeline,
        output_path=output_path,
    )
    
    # 2. Export pre-computed predictions for quick analysis
    export_predictions_for_value_functions(
        model=model,
        test_loader=test_loader,
        test_df=test_df,
        test_labels=test_labels,
        device=device,
        output_path=output_path,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("OUTPUTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Directory: {output_path}")
    logger.info("\nFor Chapter 8 - Option A: Run inference from scratch")
    logger.info("  inference/")
    logger.info("    - feature_processor.pkl     : Encoders + scaler for feature transform")
    logger.info("    - polars_pipeline/          : Polars pipeline state")
    logger.info("    - inference_example.py      : Example script for running inference")
    logger.info("  best_model.pt                 : Model checkpoint")
    logger.info("\nFor Chapter 8 - Option B: Use pre-computed predictions")
    logger.info("  predictions.parquet           : Test set predictions")
    logger.info("  calibration.json              : Calibration data")
    logger.info("  value_function_config.json    : Task stats & suggested weights")
    logger.info("  gate_weights_sample.parquet   : MMoE gate weights")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

