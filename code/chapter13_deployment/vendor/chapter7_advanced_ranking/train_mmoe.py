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
from utils.polars_pipeline import (
    load_yambda_polars, PolarsPipeline, PolarsHistoricalFeatures,
    split_validation_window, split_training_history,
)

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
# Yambda timestamps are seconds (rounded down to multiples of 5), so
# 24 hours = 24 * 60 * 60 = 86,400 units
LIKE_WINDOW_HOURS = 24
LIKE_WINDOW_TS = LIKE_WINDOW_HOURS * 60 * 60  # 86,400


# ============================================================================
# Multi-Task Data Loading
# ============================================================================

def _add_time_windowed_labels(
    listens_lf: pl.LazyFrame,
    events_lf: pl.LazyFrame,
    label_name: str,
    window_ts: int,
    label_end_ts: Optional[int] = None,
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
        label_end_ts: End of the listens' split (exclusive). Events at or after
            it are ignored, so a training or validation label never depends
            on events in a later split (validation, gap or test). A listen
            close to the end of its split therefore has a shorter window.
        
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
    # (and event_ts < label_end_ts when given)
    # Then aggregate to get a single flag per listen
    in_window = (
        col('event_ts').is_not_null() &
        (col('event_ts') >= col('timestamp')) &
        (col('event_ts') <= col('timestamp') + window_ts)
    )
    if label_end_ts is not None:
        in_window = in_window & (col('event_ts') < label_end_ts)
    result = joined.with_columns([
        # Check if event is within window (and event exists, i.e., not null)
        pl.when(in_window).then(1).otherwise(0).alias('_event_in_window')
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


def _add_multitask_labels(
    lf: pl.LazyFrame,
    data_path: Path,
    label_end_ts: Optional[int] = None,
) -> pl.LazyFrame:
    """
    Add the engagement, play_ratio, like and dislike labels to one split.

    label_end_ts is the end of the split (exclusive): like and dislike events
    at or after it are ignored (see _add_time_windowed_labels).
    """
    # Add engagement label (played_ratio_pct > 0)
    # Note: In listens.parquet, all rows have some listening, so engagement is mostly 1
    # This becomes more meaningful when comparing against impression data
    lf = lf.with_columns([
        (col('played_ratio_pct') > 0).cast(pl.Int8).alias('label_engagement'),
        # Regression target: play_ratio as fraction [0, 1]
        (col('played_ratio_pct') / 100.0).clip(0, 1).alias('label_play_ratio')
    ])
    
    # Add like/dislike labels with TIME-WINDOWED JOIN (prevents temporal leakage)
    for file_name, label_name in [('likes.parquet', 'label_liked'),
                                  ('dislikes.parquet', 'label_disliked')]:
        events_path = data_path / file_name
        if events_path.exists():
            events_lf = pl.scan_parquet(events_path)
            lf = _add_time_windowed_labels(lf, events_lf, label_name, LIKE_WINDOW_TS, label_end_ts)
        else:
            lf = lf.with_columns([pl.lit(0).cast(pl.Int8).alias(label_name)])
    return lf


def _log_feedback_files(data_path: Path) -> None:
    for file_name in ['likes.parquet', 'dislikes.parquet']:
        events_path = data_path / file_name
        if events_path.exists():
            logger.info(f"Loading {file_name.split('.')[0]} from {events_path}")
            logger.info(f"  Using {LIKE_WINDOW_HOURS}-hour window, cut at the end of each split")
        else:
            logger.warning(f"{file_name} not found, creating a dummy all-zero label column")


def _log_label_stats(train_lf: pl.LazyFrame) -> None:
    logger.info("Computing label statistics...")
    train_df_sample = train_lf.head(100000).collect()
    logger.info("Estimated label statistics (from 100K sample):")
    logger.info(f"  engagement (binary): {train_df_sample['label_engagement'].mean():.2%}")
    logger.info(f"  play_ratio (regression): mean={train_df_sample['label_play_ratio'].mean():.3f}, std={train_df_sample['label_play_ratio'].std():.3f}")
    # Note: 'label' (completion) is added by PolarsPipeline later
    logger.info(f"  like (binary):       {train_df_sample['label_liked'].mean():.2%}")
    logger.info(f"  dislike (binary):    {train_df_sample['label_disliked'].mean():.2%}")


def _train_end_timestamp(data_path: Path, test_days: int, gap_seconds: int) -> int:
    """End of the train window (exclusive), computed as in load_yambda_polars."""
    max_ts = (
        pl.scan_parquet(data_path / 'listens.parquet')
        .select(col('timestamp').max()).collect().item()
    )
    return int(max_ts) - test_days * 24 * 60 * 60 - gap_seconds


def load_multitask_data(data_dir: str, train_days: Optional[int] = None, test_days: int = 1, gap_seconds: int = 1800):
    """
    Load Yambda data with multi-task labels for 4 tasks:
    
    1. engagement:  played_ratio_pct > 0   (user started listening)
    2. completion:  played_ratio_pct >= 50 (user finished listening)
    3. like:        user liked the item within 24 hours of this listen
    4. dislike:     user disliked the item within 24 hours of this listen
    
    IMPORTANT: Like/dislike labels use a time-windowed join to prevent temporal leakage.
    A listen is only labeled as "liked" if the like action occurred AFTER the listen
    and within the configured time window (default: 24 hours), and before the end
    of the listen's split: train-window labels never use events in the gap or
    test window. Use load_multitask_splits() to also hold out a validation
    window with its own cut.
    
    Returns LazyFrames with all label columns added.
    """
    data_path = Path(data_dir)
    logger.info(f"Loading listens from {data_path / 'listens.parquet'}")
    
    train_lf, test_lf = load_yambda_polars(
        data_dir=data_dir, train_days=train_days, test_days=test_days, gap_seconds=gap_seconds, lazy=True
    )
    
    _log_feedback_files(data_path)
    train_end = _train_end_timestamp(data_path, test_days, gap_seconds)
    train_lf = _add_multitask_labels(train_lf, data_path, label_end_ts=train_end)
    test_lf = _add_multitask_labels(test_lf, data_path)
    
    _log_label_stats(train_lf)
    return train_lf, test_lf


def load_multitask_splits(
    data_dir: str,
    train_days: Optional[int] = None,
    test_days: int = 1,
    gap_seconds: int = 1800,
    val_split_ratio: float = 0.1,
):
    """
    Load the fit, validation and test splits with multi-task labels.

    Same labels as load_multitask_data, but the validation window (the last
    val_split_ratio of the train window) is held out first, and each split's
    like and dislike labels are cut at that split's end: fit rows at the start
    of the validation window, validation rows at the end of the train window,
    test rows at the end of the data.

    Returns:
        Tuple of (fit_lf, val_lf, test_lf) LazyFrames with all label columns.
    """
    data_path = Path(data_dir)
    logger.info(f"Loading listens from {data_path / 'listens.parquet'}")
    
    train_lf, test_lf = load_yambda_polars(
        data_dir=data_dir, train_days=train_days, test_days=test_days, gap_seconds=gap_seconds, lazy=True
    )
    fit_lf, val_lf = split_validation_window(train_lf, val_split_ratio)
    val_start = int(val_lf.select(col('timestamp').min()).collect().item())
    train_end = _train_end_timestamp(data_path, test_days, gap_seconds)
    
    _log_feedback_files(data_path)
    fit_lf = _add_multitask_labels(fit_lf, data_path, label_end_ts=val_start)
    val_lf = _add_multitask_labels(val_lf, data_path, label_end_ts=train_end)
    test_lf = _add_multitask_labels(test_lf, data_path)
    
    _log_label_stats(fit_lf)
    return fit_lf, val_lf, test_lf


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


def _collect_mmoe_outputs(
    model: MMoE,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run the model over a loader; return per-task logits, probabilities and gate weights."""
    model.eval()
    
    all_logits = {name: [] for name in model.task_names}
    all_probs = {name: [] for name in model.task_names}
    all_gate_weights = {name: [] for name in model.task_names}
    
    with torch.no_grad():
        for sparse_batch, dense_batch, _ in loader:
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
    
    return all_logits, all_probs, all_gate_weights


def _predictions_frame(
    df: pd.DataFrame,
    labels: Dict[str, np.ndarray],
    probs: Dict[str, np.ndarray],
    logits: Dict[str, np.ndarray],
    task_names,
) -> pd.DataFrame:
    """Build the uid / item_id / timestamp + label_, prob_, logit_ columns frame."""
    predictions_data = {
        'uid': df['uid'].values,
        'item_id': df['item_id'].values,
        'timestamp': df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)),
    }
    for task_name in task_names:
        if task_name in labels:
            predictions_data[f'label_{task_name}'] = labels[task_name]
        if task_name in probs:
            predictions_data[f'prob_{task_name}'] = probs[task_name]
        if task_name in logits:
            predictions_data[f'logit_{task_name}'] = logits[task_name]
    return pd.DataFrame(predictions_data)


def export_predictions_for_value_functions(
    model: MMoE,
    val_loader: DataLoader,
    val_df: pd.DataFrame,
    val_labels: Dict[str, np.ndarray],
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
    
    Everything derived from labels (calibration curves, task statistics,
    task correlations, suggested weights) and the gate-weight summaries
    comes from the VALIDATION split, so Chapter 8 can tune on it:
    - val_predictions.parquet: validation predictions and labels
      (calibration analysis, value function weight tuning)
    - calibration.json, value_function_config.json,
      gate_weights_sample.parquet
    
    The TEST split is exported for final reporting only:
    - predictions.parquet: test predictions and labels. Do not tune
      weights, thresholds or calibration on this file.
    """
    logger.info("\nExporting pre-computed predictions for analysis...")
    
    val_logits, val_probs, val_gate_weights = _collect_mmoe_outputs(model, val_loader, device)
    test_logits, test_probs, _ = _collect_mmoe_outputs(model, test_loader, device)
    
    # 1. Export predictions DataFrames (all 5 task outputs)
    val_predictions_df = _predictions_frame(val_df, val_labels, val_probs, val_logits, model.task_names)
    val_predictions_df.to_parquet(output_path / 'val_predictions.parquet', index=False)
    logger.info(f"  Saved val_predictions.parquet ({len(val_predictions_df):,} samples, "
                f"validation split: calibration and weight tuning)")
    
    predictions_df = _predictions_frame(test_df, test_labels, test_probs, test_logits, model.task_names)
    predictions_df.to_parquet(output_path / 'predictions.parquet', index=False)
    logger.info(f"  Saved predictions.parquet ({len(predictions_df):,} samples, "
                f"test split: final reporting only)")
    logger.info(f"    Columns: {list(predictions_df.columns)}")
    
    # 2. Export gate weights (aggregated statistics, validation split)
    gate_weights_summary = {}
    for task_name in model.task_names:
        gate_w = val_gate_weights[task_name]
        gate_weights_summary[task_name] = {
            'mean_weights': gate_w.mean(axis=0).tolist(),
            'std_weights': gate_w.std(axis=0).tolist(),
            'expert_names': [f'expert_{i}' for i in range(gate_w.shape[1])],
        }
    
    # Save detailed gate weights for a sample (first 10000 validation rows)
    sample_size = min(10000, len(val_df))
    gate_weights_df = pd.DataFrame({
        'sample_idx': np.arange(sample_size),
    })
    for task_name in model.task_names:
        gate_w = val_gate_weights[task_name][:sample_size]
        for exp_idx in range(gate_w.shape[1]):
            gate_weights_df[f'{task_name}_expert_{exp_idx}'] = gate_w[:, exp_idx]
    
    gate_weights_df.to_parquet(output_path / 'gate_weights_sample.parquet', index=False)
    logger.info(f"  Saved gate_weights_sample.parquet ({sample_size:,} validation samples)")
    
    # 3. Export calibration data (validation split; binary tasks only,
    # calibration curves are undefined for regression targets like play_ratio)
    calibration_data = {}
    for task_name in model.task_names:
        if task_name in REGRESSION_TASKS:
            calibration_data[task_name] = {
                'note': 'regression task - calibration curve not applicable'
            }
            continue
        probs = val_probs[task_name]
        labels = val_labels[task_name]
        
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
                'split': 'validation',
            }
        except Exception as e:
            logger.warning(f"  Could not compute calibration for {task_name}: {e}")
            calibration_data[task_name] = {'error': str(e)}
    
    with open(output_path / 'calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=2)
    logger.info(f"  Saved calibration.json (validation split)")
    
    # 4. Export value function configuration (validation split)
    task_stats = {}
    for task_name in model.task_names:
        probs = val_probs[task_name]
        labels = val_labels[task_name]
        
        if task_name in REGRESSION_TASKS:
            # Regression task: use MSE, MAE, R2 instead of AUC
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
            # Binary task: use AUC-ROC (undefined when only one class is present)
            task_stats[task_name] = {
                'task_type': 'binary',
                'positive_rate': float(labels.mean()),
                'pred_mean': float(probs.mean()),
                'pred_std': float(probs.std()),
                'auc_roc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else None,
            }
    
    # Compute pairwise task correlations (labels and predictions)
    task_correlations = {'label': {}, 'prediction': {}}
    task_list = model.task_names
    for i, task1 in enumerate(task_list):
        for task2 in task_list[i+1:]:
            if task1 in val_labels and task2 in val_labels:
                corr = float(np.corrcoef(val_labels[task1], val_labels[task2])[0, 1])
                task_correlations['label'][f'{task1}_{task2}'] = corr
            if task1 in val_probs and task2 in val_probs:
                corr = float(np.corrcoef(val_probs[task1], val_probs[task2])[0, 1])
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
        'statistics_split': 'validation',
        'prediction_files': {
            'val_predictions.parquet': 'validation split: calibration and weight tuning',
            'predictions.parquet': 'test split: final reporting only',
        },
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
            "# Tune weights on validation predictions; keep predictions.parquet\n"
            "# (test split) for the final report only.\n"
            "df = pd.read_parquet('val_predictions.parquet')\n"
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
    logger.info(f"  Saved value_function_config.json (validation split)")
    
    logger.info("  Artifact export complete!")
    
    return {
        'predictions_df': predictions_df,
        'val_predictions_df': val_predictions_df,
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


def evaluate(model, data_loader, criterion, device, thresholds: Optional[Dict[str, float]] = None):
    """Evaluate all tasks.

    f1_max / f1_threshold are tuned on the data being evaluated, so they are
    optimistic diagnostics. Pass `thresholds` (chosen on the validation set)
    to also get f1_at_threshold, the F1 to report on the test set.
    """
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
            precision, recall, pr_thresholds = precision_recall_curve(labels, preds)
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_f1_idx = np.argmax(f1_scores)
            f1_max = f1_scores[best_f1_idx]
            best_threshold = pr_thresholds[best_f1_idx]
            
            results[name] = {
                'loss': task_losses[name] / n_batches,
                'auc_roc': roc_auc_score(labels, preds),
                'auc_pr': average_precision_score(labels, preds),
                'f1_max': float(f1_max),  # F1 at optimal threshold
                'f1_threshold': float(best_threshold),  # The optimal threshold
                'positive_rate': float(labels.mean()),
            }
        
        if thresholds is not None and name in thresholds and name not in REGRESSION_TASKS:
            results[name]['f1_at_threshold'] = float(
                f1_score(labels, (preds >= thresholds[name]).astype(int), zero_division=0)
            )
            results[name]['applied_threshold'] = float(thresholds[name])
    
    return results


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MMoE on Yambda dataset (Multi-Task)')
    
    parser.add_argument('--data_dir', type=str, default=os.environ.get('YAMBDA_DATA_DIR', ''))
    parser.add_argument('--train_days', type=int, default=30)
    parser.add_argument('--test_days', type=int, default=1)
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                        help='Fraction of the train window, taken from its end, held out '
                             'as the validation set for model selection')
    parser.add_argument('--history_days', type=float, default=0,
                        help='Days of listens before the fit window used for the training '
                             "rows' user and item aggregates (0: same length as the fit window)")
    
    parser.add_argument('--embed_dim', type=int, default=16)
    # Default of 4 came from earlier runs compared on the test set, not on
    # validation; re-tune on the validation window before relying on it.
    parser.add_argument('--num_experts', type=int, default=4)
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
    # Validation window: the last val_split_ratio of the train window by time.
    # It is held out before the pipeline is fitted, so its historical features
    # come only from earlier data, exactly as for the test window. Like and
    # dislike labels are cut at the end of each split.
    train_lf, val_lf, test_lf = load_multitask_splits(
        data_dir=args.data_dir, train_days=train_days, test_days=args.test_days,
        val_split_ratio=args.val_split_ratio,
    )
    
    # Feature Engineering
    logger.info("\n[2/5] Running feature pipeline...")
    pipeline = PolarsPipeline()
    pipeline.fit(train_lf)
    
    # Training rows get user and item aggregates from the window just before
    # the fit window, so no row's own label is in its features. Validation
    # and test rows use the fit window, which ends before their windows.
    train_lf, history_df = split_training_history(args.data_dir, train_lf, args.history_days)
    train_history = PolarsHistoricalFeatures().fit(history_df)
    
    train_polars_df, _ = pipeline.transform(train_lf, history=train_history)
    val_polars_df, _ = pipeline.transform(val_lf)
    test_polars_df, _ = pipeline.transform(test_lf)
    
    train_df = train_polars_df.to_pandas()
    val_df = val_polars_df.to_pandas()
    test_df = test_polars_df.to_pandas()
    
    logger.info(f"Train samples: {len(train_df):,}")
    logger.info(f"Validation samples: {len(val_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # Prepare Features
    logger.info("\n[3/5] Preparing features for MMoE...")
    feature_processor = MMoEFeatureProcessor(SPARSE_FEATURES, DENSE_FEATURES, TASK_NAMES)
    
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    val_sparse, val_dense, val_labels = feature_processor.transform(val_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    for task_name in TASK_NAMES:
        if task_name in train_labels:
            if task_name in REGRESSION_TASKS:
                logger.info(f"Task '{task_name}' (regression) - Train mean: {train_labels[task_name].mean():.3f}, std: {train_labels[task_name].std():.3f}")
            else:
                logger.info(f"Task '{task_name}' (binary) - Train positive rate: {train_labels[task_name].mean():.2%}")
    
    train_dataset = MMoEDataset(train_sparse, train_dense, train_labels)
    val_dataset = MMoEDataset(val_sparse, val_dense, val_labels)
    test_dataset = MMoEDataset(test_sparse, test_dense, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
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
    history = {'train': [], 'val': []}
    
    # Validation metrics drive LR scheduling, early stopping and checkpoint
    # selection; the test set is only touched after training.
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_task_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        completion_auc = val_metrics['completion']['auc_roc']
        like_auc_pr = val_metrics['like']['auc_pr']  # Use AUC-PR for imbalanced task
        
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Val Completion AUC: {completion_auc:.4f} | "
            f"Val Like AUC-PR: {like_auc_pr:.4f} | "  # AUC-PR better for imbalanced
            f"Time: {time.time() - epoch_start:.1f}s"
        )
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        scheduler.step(completion_auc)
        
        if completion_auc > best_auc:
            best_auc = completion_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'metrics': val_metrics}, output_path / 'best_model.pt')
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
    val_metrics = evaluate(model, val_loader, criterion, device)
    # Classification thresholds are chosen on validation and applied to test
    val_thresholds = {
        name: val_metrics[name]['f1_threshold']
        for name in TASK_NAMES if name not in REGRESSION_TASKS
    }
    test_metrics = evaluate(model, test_loader, criterion, device, thresholds=val_thresholds)
    
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
            if 'f1_at_threshold' in metrics:
                logger.info(f"  Test F1:      {metrics['f1_at_threshold']:.4f}  (at val-tuned threshold {metrics['applied_threshold']:.3f})")
            logger.info(f"  F1-max:       {metrics['f1_max']:.4f}  (oracle: threshold {metrics['f1_threshold']:.3f} tuned on test, diagnostic only)")
            logger.info(f"  Positive rate: {metrics['positive_rate']:.2%}")
    
    results = {
        'experiment': exp_name,
        'train_days': train_days,
        'val_split_ratio': args.val_split_ratio,
        'history_days': args.history_days,
        'best_epoch': best_epoch,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
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
    
    # 2. Export pre-computed predictions for quick analysis. Calibration,
    #    task statistics and suggested weights come from the validation
    #    split; test predictions are exported for final reporting only.
    export_predictions_for_value_functions(
        model=model,
        val_loader=val_loader,
        val_df=val_df,
        val_labels=val_labels,
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
    logger.info("  val_predictions.parquet       : Validation predictions (calibration, weight tuning)")
    logger.info("  predictions.parquet           : Test predictions (final reporting only)")
    logger.info("  calibration.json              : Calibration data (validation split)")
    logger.info("  value_function_config.json    : Task stats & suggested weights (validation split)")
    logger.info("  gate_weights_sample.parquet   : MMoE gate weights (validation split)")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

