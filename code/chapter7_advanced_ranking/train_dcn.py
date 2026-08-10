"""
Chapter 7: DCN-V2 Training Script for Yambda Dataset

This script trains a DCN-V2 (Deep & Cross Network V2) model using the same 
data pipeline as DeepFM (Chapter 6), enabling direct comparison.

DCN-V2 is Google's state-of-the-art architecture for CTR prediction, featuring:
- Cross Network with low-rank matrix factorization (more expressive than DCN-V1)
- Parallel or stacked structure with deep network
- Efficient computation for large-scale deployment

Usage:
    # Quick run (30 days)
    python train_dcn.py --train_days 30
    
    # Full dataset
    python train_dcn.py --train_days 0
    
    # With custom hyperparameters
    python train_dcn.py --train_days 30 --embed_dim 32 --cross_layers 4

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

from models.dcn import DCNV2, DCNV2Config, DCNV2Dataset, collate_fn
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

# Sparse features: High cardinality categoricals that benefit from embeddings
SPARSE_FEATURES = ['uid', 'item_id', 'hour_of_day', 'day_of_week']

# Dense features: Numerical and low-cardinality categoricals
DENSE_FEATURES = [
    # Binary
    'is_organic',
    # Numerical
    'track_length_seconds',
    # User historical
    'user_total_listens',
    'user_avg_completion',
    'user_std_completion',
    'user_median_completion',
    'user_unique_items',
    'user_organic_ratio',
    'user_active_span',
    'user_listen_rate',
    # Item historical
    'item_total_plays',
    'item_avg_completion',
    'item_std_completion',
    'item_unique_listeners',
    'item_organic_ratio',
    'item_repeat_ratio',
    # User-item
    'has_listened_before',
    'previous_listen_count',
]

# Label column
LABEL_COL = 'label'


# ============================================================================
# Data Preparation
# ============================================================================

class DCNV2FeatureProcessor:
    """Prepares features for DCN-V2 model."""
    
    def __init__(
        self,
        sparse_features: list,
        dense_features: list,
    ):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.sparse_encoders: Dict[str, LabelEncoder] = {}
        self.dense_scaler: Optional[StandardScaler] = None
        self.vocab_sizes: Dict[str, int] = {}
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DCNV2FeatureProcessor':
        """Fit encoders on training data."""
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
    
    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Transform data for DCN-V2."""
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
    
    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: DCNV2,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
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


def evaluate(
    model: DCNV2,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on dataset."""
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
    
    auc_roc = roc_auc_score(labels, preds)
    auc_pr = average_precision_score(labels, preds)
    
    pred_binary = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, pred_binary)
    
    return {
        'loss': total_loss / n_batches,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1': f1,
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DCN-V2 on Yambda dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('YAMBDA_DATA_DIR', ''),
                        help='Path to Yambda flat data directory')
    parser.add_argument('--train_days', type=int, default=30,
                        help='Training window in days (0 for full dataset)')
    parser.add_argument('--test_days', type=int, default=1,
                        help='Test window in days')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension for sparse features')
    parser.add_argument('--cross_layers', type=int, default=3,
                        help='Number of cross network layers')
    parser.add_argument('--cross_rank', type=int, default=32,
                        help='Low-rank dimension for cross network')
    parser.add_argument('--mlp_dims', type=str, default='256,128,64',
                        help='Comma-separated MLP hidden dimensions')
    parser.add_argument('--structure', type=str, default='parallel',
                        choices=['parallel', 'stacked'],
                        help='DCN-V2 structure: parallel or stacked')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and results')
    
    args = parser.parse_args()
    
    mlp_dims = [int(x) for x in args.mlp_dims.split(',')]
    
    if not args.data_dir:
        logger.error("Data directory not specified. Set YAMBDA_DATA_DIR or use --data_dir")
        sys.exit(1)
    
    train_days = None if args.train_days == 0 else args.train_days
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_days_str = 'full' if train_days is None else f'{train_days}d'
    exp_name = f'dcnv2_{train_days_str}_{args.structure}_{timestamp}'
    output_path = Path(args.output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DCN-V2 Training on Yambda Dataset")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Train window: {'FULL' if train_days is None else f'{train_days} days'}")
    logger.info(f"Structure: {args.structure}")
    logger.info(f"Output directory: {output_path}")
    
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
    
    train_polars_df, feature_cols = pipeline.transform(train_lf)
    test_polars_df, _ = pipeline.transform(test_lf)
    
    train_df = train_polars_df.to_pandas()
    test_df = test_polars_df.to_pandas()
    
    logger.info(f"Train samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # Prepare Features
    logger.info("\n[3/5] Preparing features for DCN-V2...")
    feature_processor = DCNV2FeatureProcessor(
        sparse_features=SPARSE_FEATURES,
        dense_features=DENSE_FEATURES,
    )
    
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    logger.info(f"Sparse features: {list(feature_processor.vocab_sizes.keys())}")
    logger.info(f"Dense features: {len(DENSE_FEATURES)}")
    logger.info(f"Train positive rate: {train_labels.mean():.1%}")
    logger.info(f"Test positive rate: {test_labels.mean():.1%}")
    
    train_dataset = DCNV2Dataset(train_sparse, train_dense, train_labels)
    test_dataset = DCNV2Dataset(test_sparse, test_dense, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Train Model
    logger.info("\n[4/5] Training DCN-V2...")
    
    config = DCNV2Config(
        sparse_features=feature_processor.vocab_sizes,
        dense_features=DENSE_FEATURES,
        embed_dim=args.embed_dim,
        cross_layers=args.cross_layers,
        cross_rank=args.cross_rank,
        mlp_dims=mlp_dims,
        dropout=args.dropout,
        structure=args.structure,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    model = DCNV2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Embedding dim: {args.embed_dim}")
    logger.info(f"Cross layers: {args.cross_layers}, rank: {args.cross_rank}")
    logger.info(f"MLP dims: {mlp_dims}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'test': []}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train AUC: {train_metrics['auc_roc']:.4f} | "
            f"Test AUC: {test_metrics['auc_roc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        scheduler.step(test_metrics['auc_roc'])
        
        if test_metrics['auc_roc'] > best_auc:
            best_auc = test_metrics['auc_roc']
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': test_metrics,
            }, output_path / 'best_model.pt')
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
    
    logger.info("\nTraining Metrics:")
    logger.info(f"  AUC-ROC: {train_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:  {train_metrics['auc_pr']:.4f}")
    logger.info(f"  F1:      {train_metrics['f1']:.4f}")
    
    logger.info("\nTest Metrics:")
    logger.info(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:  {test_metrics['auc_pr']:.4f}")
    logger.info(f"  F1:      {test_metrics['f1']:.4f}")
    
    results = {
        'experiment': exp_name,
        'train_days': train_days,
        'config': {
            'embed_dim': args.embed_dim,
            'cross_layers': args.cross_layers,
            'cross_rank': args.cross_rank,
            'mlp_dims': mlp_dims,
            'structure': args.structure,
            'dropout': args.dropout,
            'lr': args.lr,
            'batch_size': args.batch_size,
        },
        'best_epoch': best_epoch,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'n_params': n_params,
        'train_time_seconds': train_time,
        'history': history,
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")
    logger.info(f"Total time: {train_time:.1f}s")
    logger.info(f"Outputs: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

