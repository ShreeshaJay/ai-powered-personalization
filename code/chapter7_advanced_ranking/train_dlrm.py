"""
Chapter 7: DLRM Training Script for Yambda Dataset

This script trains a DLRM (Deep Learning Recommendation Model) using the same 
data pipeline as DeepFM (Chapter 6), enabling direct comparison.

DLRM is Facebook's architecture for recommendation systems, featuring:
- Bottom MLP: Transforms dense features to embedding space
- Embedding tables: Direct lookup for sparse features
- Feature interactions: Pairwise dot products between all embeddings
- Top MLP: Final prediction from combined features

Usage:
    # Quick run (30 days)
    python train_dlrm.py --train_days 30
    
    # Full dataset
    python train_dlrm.py --train_days 0
    
    # With custom hyperparameters
    python train_dlrm.py --train_days 30 --embed_dim 32 --top_mlp_dims 512,256,128

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.dlrm import DLRM, DLRMConfig, DLRMDataset, collate_fn
from utils.polars_pipeline import load_yambda_polars, PolarsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Configuration (Same as DeepFM for fair comparison)
# ============================================================================

SPARSE_FEATURES = ['uid', 'item_id', 'hour_of_day', 'day_of_week']

DENSE_FEATURES = [
    'is_organic',
    'track_length_seconds',
    'user_total_listens',
    'user_avg_completion',
    'user_std_completion',
    'user_median_completion',
    'user_unique_items',
    'user_organic_ratio',
    'user_active_span',
    'user_listen_rate',
    'item_total_plays',
    'item_avg_completion',
    'item_std_completion',
    'item_unique_listeners',
    'item_organic_ratio',
    'item_repeat_ratio',
    'has_listened_before',
    'previous_listen_count',
]

LABEL_COL = 'label'


# ============================================================================
# Data Preparation
# ============================================================================

class DLRMFeatureProcessor:
    """Prepares features for DLRM model."""
    
    def __init__(self, sparse_features: list, dense_features: list):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_encoders: Dict[str, LabelEncoder] = {}
        self.dense_scaler: Optional[StandardScaler] = None
        self.vocab_sizes: Dict[str, int] = {}
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DLRMFeatureProcessor':
        logger.info("Fitting feature processor...")
        
        for col in self.sparse_features:
            if col in df.columns:
                le = LabelEncoder()
                values = df[col].fillna(-1).astype(str)
                le.fit(values)
                self.sparse_encoders[col] = le
                self.vocab_sizes[col] = len(le.classes_) + 1
                logger.info(f"  {col}: vocab_size={self.vocab_sizes[col]:,}")
        
        if self.dense_features:
            dense_cols = [c for c in self.dense_features if c in df.columns]
            dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
            self.dense_scaler = StandardScaler()
            self.dense_scaler.fit(dense_data)
            logger.info(f"  Dense features: {len(dense_cols)} columns scaled")
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        sparse_data = {}
        for col in self.sparse_features:
            if col in df.columns and col in self.sparse_encoders:
                values = df[col].fillna(-1).astype(str)
                le = self.sparse_encoders[col]
                encoded = np.zeros(len(values), dtype=np.int64)
                known_mask = values.isin(le.classes_)
                encoded[known_mask] = le.transform(values[known_mask]) + 1
                sparse_data[col] = encoded
        
        dense_cols = [c for c in self.dense_features if c in df.columns]
        dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
        if self.dense_scaler is not None:
            dense_data = self.dense_scaler.transform(dense_data)
        
        labels = df[LABEL_COL].values.astype(np.float32)
        return sparse_data, dense_data, labels
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model: DLRM, train_loader: DataLoader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for sparse_batch, dense_batch, labels in train_loader:
        sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
        if dense_batch is not None:
            dense_batch = dense_batch.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(sparse_batch, dense_batch)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model: DLRM, data_loader: DataLoader, criterion, device) -> Dict[str, float]:
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for sparse_batch, dense_batch, labels in data_loader:
            sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
            if dense_batch is not None:
                dense_batch = dense_batch.to(device)
            labels = labels.to(device)
            
            logits = model(sparse_batch, dense_batch)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
    
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    return {
        'loss': total_loss / n_batches,
        'auc_roc': roc_auc_score(labels, preds),
        'auc_pr': average_precision_score(labels, preds),
        'f1': f1_score(labels, (preds >= 0.5).astype(int)),
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DLRM on Yambda dataset')
    
    parser.add_argument('--data_dir', type=str, default=os.environ.get('YAMBDA_DATA_DIR', ''))
    parser.add_argument('--train_days', type=int, default=30)
    parser.add_argument('--test_days', type=int, default=1)
    
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--bottom_mlp_dims', type=str, default='64,32,16')
    parser.add_argument('--top_mlp_dims', type=str, default='256,128,64')
    parser.add_argument('--interaction', type=str, default='dot', choices=['dot', 'cat'])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=3)
    
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    bottom_mlp_dims = [int(x) for x in args.bottom_mlp_dims.split(',')]
    top_mlp_dims = [int(x) for x in args.top_mlp_dims.split(',')]
    
    if bottom_mlp_dims[-1] != args.embed_dim:
        logger.warning(f"Adjusting bottom_mlp_dims[-1] from {bottom_mlp_dims[-1]} to {args.embed_dim}")
        bottom_mlp_dims[-1] = args.embed_dim
    
    if not args.data_dir:
        logger.error("Data directory not specified. Set YAMBDA_DATA_DIR or use --data_dir")
        sys.exit(1)
    
    train_days = None if args.train_days == 0 else args.train_days
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_days_str = 'full' if train_days is None else f'{train_days}d'
    exp_name = f'dlrm_{train_days_str}_{args.interaction}_{timestamp}'
    output_path = Path(args.output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DLRM Training on Yambda Dataset")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Train window: {'FULL' if train_days is None else f'{train_days} days'}")
    logger.info(f"Interaction type: {args.interaction}")
    
    start_time = time.time()
    
    # Load Data
    logger.info("\n[1/5] Loading data with Polars...")
    train_lf, test_lf = load_yambda_polars(
        data_dir=args.data_dir,
        train_days=train_days,
        test_days=args.test_days,
    )
    
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
    logger.info("\n[3/5] Preparing features for DLRM...")
    feature_processor = DLRMFeatureProcessor(SPARSE_FEATURES, DENSE_FEATURES)
    
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    logger.info(f"Train positive rate: {train_labels.mean():.1%}")
    
    train_dataset = DLRMDataset(train_sparse, train_dense, train_labels)
    test_dataset = DLRMDataset(test_sparse, test_dense, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Train Model
    logger.info("\n[4/5] Training DLRM...")
    
    config = DLRMConfig(
        sparse_features=feature_processor.vocab_sizes,
        dense_features=DENSE_FEATURES,
        embed_dim=args.embed_dim,
        bottom_mlp_dims=bottom_mlp_dims,
        top_mlp_dims=top_mlp_dims,
        dropout=args.dropout,
        interaction_type=args.interaction,
    )
    
    model = DLRM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'test': []}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train AUC: {train_metrics['auc_roc']:.4f} | "
            f"Test AUC: {test_metrics['auc_roc']:.4f} | "
            f"Time: {time.time() - epoch_start:.1f}s"
        )
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        scheduler.step(test_metrics['auc_roc'])
        
        if test_metrics['auc_roc'] > best_auc:
            best_auc = test_metrics['auc_roc']
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
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    
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
    
    logger.info(f"\nOutputs saved to: {output_path}")


if __name__ == '__main__':
    main()

