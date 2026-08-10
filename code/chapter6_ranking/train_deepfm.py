"""
DeepFM Training Script for Yambda Dataset

This script trains a DeepFM model using the same data pipeline as XGBoost,
enabling direct comparison between gradient boosting and deep learning approaches.

Usage:
    # Quick run (30 days)
    python train_deepfm.py --train_days 30
    
    # Full dataset
    python train_deepfm.py --train_days 0
    
    # With custom hyperparameters
    python train_deepfm.py --train_days 30 --embed_dim 32 --batch_size 4096

Author: Chapter 6 - Ranking
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

from models.deepfm import DeepFM, DeepFMConfig, DeepFMDataset, collate_fn
from utils.polars_pipeline import load_yambda_polars, PolarsPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Configuration
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

class DeepFMFeatureProcessor:
    """Prepares features for DeepFM model.
    
    Handles:
    - Label encoding for sparse (categorical) features
    - StandardScaler for dense (numerical) features
    - Vocabulary size tracking for embeddings
    """
    
    def __init__(
        self,
        sparse_features: list,
        dense_features: list,
    ):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        
        # Encoders
        self.sparse_encoders: Dict[str, LabelEncoder] = {}
        self.dense_scaler: Optional[StandardScaler] = None
        
        # Vocabulary sizes (for embedding layers)
        self.vocab_sizes: Dict[str, int] = {}
        
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DeepFMFeatureProcessor':
        """Fit encoders on training data."""
        logger.info("Fitting feature processor...")
        
        # Fit label encoders for sparse features
        for col in self.sparse_features:
            if col in df.columns:
                le = LabelEncoder()
                # Handle potential NaN/unknown values
                values = df[col].fillna(-1).astype(str)
                le.fit(values)
                self.sparse_encoders[col] = le
                # +1 for unknown/padding token at index 0
                self.vocab_sizes[col] = len(le.classes_) + 1
                logger.info(f"  {col}: vocab_size={self.vocab_sizes[col]:,}")
        
        # Fit scaler for dense features
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
        """Transform data for DeepFM.
        
        Returns:
            sparse_data: Dict[feature_name, encoded_indices]
            dense_data: Scaled dense features array
            labels: Binary labels
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Transform sparse features
        sparse_data = {}
        for col in self.sparse_features:
            if col in df.columns and col in self.sparse_encoders:
                values = df[col].fillna(-1).astype(str)
                le = self.sparse_encoders[col]
                
                # Handle unseen values: map to index 0 (padding/unknown)
                encoded = np.zeros(len(values), dtype=np.int64)
                known_mask = values.isin(le.classes_)
                encoded[known_mask] = le.transform(values[known_mask]) + 1  # +1 to reserve 0
                
                sparse_data[col] = encoded
        
        # Transform dense features
        dense_cols = [c for c in self.dense_features if c in df.columns]
        dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
        if self.dense_scaler is not None:
            dense_data = self.dense_scaler.transform(dense_data)
        
        # Extract labels
        labels = df[LABEL_COL].values.astype(np.float32)
        
        return sparse_data, dense_data, labels
    
    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str) -> None:
        """Save feature processor to disk for later inference."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'sparse_features': self.sparse_features,
            'dense_features': self.dense_features,
            'sparse_encoders': self.sparse_encoders,
            'dense_scaler': self.dense_scaler,
            'vocab_sizes': self.vocab_sizes,
            '_fitted': self._fitted,
        }
        
        with open(path / 'feature_processor.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Feature processor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DeepFMFeatureProcessor':
        """Load feature processor from disk."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        
        with open(path / 'feature_processor.pkl', 'rb') as f:
            state = pickle.load(f)
        
        processor = cls(
            sparse_features=state['sparse_features'],
            dense_features=state['dense_features'],
        )
        processor.sparse_encoders = state['sparse_encoders']
        processor.dense_scaler = state['dense_scaler']
        processor.vocab_sizes = state['vocab_sizes']
        processor._fitted = state['_fitted']
        
        return processor


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: DeepFM,
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
        # Move to device
        sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
        if dense_batch is not None:
            dense_batch = dense_batch.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sparse_batch, dense_batch)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(
    model: DeepFM,
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
            # Move to device
            sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
            if dense_batch is not None:
                dense_batch = dense_batch.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(sparse_batch, dense_batch)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
    
    # Concatenate
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    # Compute metrics
    auc_roc = roc_auc_score(labels, preds)
    auc_pr = average_precision_score(labels, preds)
    
    # F1 at threshold 0.5
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
    parser = argparse.ArgumentParser(description='Train DeepFM on Yambda dataset')
    
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
    parser.add_argument('--mlp_dims', type=str, default='256,128,64',
                        help='Comma-separated MLP hidden dimensions')
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
    
    # Parse MLP dimensions
    mlp_dims = [int(x) for x in args.mlp_dims.split(',')]
    
    # Validate data directory
    if not args.data_dir:
        logger.error("Data directory not specified. Set YAMBDA_DATA_DIR or use --data_dir")
        sys.exit(1)
    
    # Handle train_days=0 as full dataset
    train_days = None if args.train_days == 0 else args.train_days
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_days_str = 'full' if train_days is None else f'{train_days}d'
    exp_name = f'deepfm_{train_days_str}_{timestamp}'
    output_path = Path(args.output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DeepFM Training on Yambda Dataset")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Train window: {'FULL' if train_days is None else f'{train_days} days'}")
    logger.info(f"Output directory: {output_path}")
    
    start_time = time.time()
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("\n[1/5] Loading data with Polars...")
    
    train_lf, test_lf = load_yambda_polars(
        data_dir=args.data_dir,
        train_days=train_days,
        test_days=args.test_days,
    )
    
    # =========================================================================
    # Step 2: Feature Engineering (using existing pipeline)
    # =========================================================================
    logger.info("\n[2/5] Running feature pipeline...")
    
    pipeline = PolarsPipeline()
    pipeline.fit(train_lf)
    
    # Transform - returns (DataFrame, feature_columns) tuple
    train_polars_df, feature_cols = pipeline.transform(train_lf)
    test_polars_df, _ = pipeline.transform(test_lf)
    
    # Convert to pandas
    train_df = train_polars_df.to_pandas()
    test_df = test_polars_df.to_pandas()
    
    logger.info(f"Train samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # =========================================================================
    # Diagnostic: Check user/item overlap (to assess potential memorization)
    # =========================================================================
    train_users = set(train_df['uid'].unique())
    test_users = set(test_df['uid'].unique())
    train_items = set(train_df['item_id'].unique())
    test_items = set(test_df['item_id'].unique())
    
    user_overlap = len(train_users & test_users) / len(test_users) * 100
    item_overlap = len(train_items & test_items) / len(test_items) * 100
    
    logger.info(f"\nOverlap Analysis (potential memorization check):")
    logger.info(f"  Users in test seen in train: {user_overlap:.1f}%")
    logger.info(f"  Items in test seen in train: {item_overlap:.1f}%")
    if user_overlap > 95 and item_overlap > 95:
        logger.warning("  HIGH OVERLAP: Model may benefit from memorization of user/item embeddings")
        logger.warning("  This is NOT data leakage, but explains why embeddings >> frequency encoding")
    
    # =========================================================================
    # Step 3: Prepare Features for DeepFM
    # =========================================================================
    logger.info("\n[3/5] Preparing features for DeepFM...")
    
    feature_processor = DeepFMFeatureProcessor(
        sparse_features=SPARSE_FEATURES,
        dense_features=DENSE_FEATURES,
    )
    
    # Fit on training data
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    logger.info(f"Sparse features: {list(feature_processor.vocab_sizes.keys())}")
    logger.info(f"Dense features: {len(DENSE_FEATURES)}")
    logger.info(f"Train positive rate: {train_labels.mean():.1%}")
    logger.info(f"Test positive rate: {test_labels.mean():.1%}")
    
    # Create datasets
    train_dataset = DeepFMDataset(train_sparse, train_dense, train_labels)
    test_dataset = DeepFMDataset(test_sparse, test_dense, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type == 'cuda' else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # =========================================================================
    # Step 4: Create and Train Model
    # =========================================================================
    logger.info("\n[4/5] Training DeepFM...")
    
    # Create model config
    config = DeepFMConfig(
        sparse_features=feature_processor.vocab_sizes,
        dense_features=DENSE_FEATURES,
        embed_dim=args.embed_dim,
        mlp_dims=mlp_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create model
    model = DeepFM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Embedding dim: {args.embed_dim}")
    logger.info(f"MLP dims: {mlp_dims}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'test': []}
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        train_metrics = evaluate(model, train_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train AUC: {train_metrics['auc_roc']:.4f} | "
            f"Test AUC: {test_metrics['auc_roc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Store history
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        # Learning rate scheduling
        scheduler.step(test_metrics['auc_roc'])
        
        # Early stopping check
        if test_metrics['auc_roc'] > best_auc:
            best_auc = test_metrics['auc_roc']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
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
    
    # =========================================================================
    # Step 5: Final Evaluation
    # =========================================================================
    logger.info("\n[5/5] Final Evaluation...")
    
    # Load best model
    checkpoint = torch.load(output_path / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final metrics
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
    
    # Save results
    results = {
        'experiment': exp_name,
        'train_days': train_days,
        'config': {
            'embed_dim': args.embed_dim,
            'mlp_dims': mlp_dims,
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
    
    # =========================================================================
    # Save all artifacts for later chapters
    # =========================================================================
    logger.info("\nSaving artifacts for inference...")
    
    # Save Polars pipeline (encodings, historical feature lookup tables)
    pipeline.save(str(output_path / 'polars_pipeline'))
    logger.info(f"  Polars pipeline saved to {output_path / 'polars_pipeline'}")
    
    # Save DeepFM feature processor (sparse encoders, dense scaler)
    feature_processor.save(str(output_path / 'feature_processor'))
    logger.info(f"  Feature processor saved to {output_path / 'feature_processor'}")
    
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
    
    # Comparison hint
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH XGBOOST")
    logger.info("=" * 60)
    logger.info("\nTo compare with XGBoost baseline, run:")
    logger.info(f"  python train_xgboost_polars.py --train_days {args.train_days}")
    logger.info("\nExpected comparison:")
    logger.info("  - DeepFM: Better with more data, learns embeddings")
    logger.info("  - XGBoost: Faster training, better with engineered features")


if __name__ == '__main__':
    main()

