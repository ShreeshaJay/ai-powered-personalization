"""
Chapter 7: DCN-V2 with Pre-trained Item Embeddings

This script uses pre-trained item embeddings from embeddings.parquet instead of
learning item embeddings from scratch. This helps reduce overfitting on high-cardinality
item_id features.

Key differences from train_dcn.py:
1. item_id is removed from sparse features (no learned embedding)
2. Pre-trained 128-dim item embeddings are added to dense features
3. Items without embeddings use a zero vector (cold-start handling)

Usage:
    python train_dcn_pretrained.py --train_days 30
    
    # With fine-tuning of embeddings (slower but potentially better)
    python train_dcn_pretrained.py --train_days 30 --finetune_embeddings

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
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))

from models.dcn import DCNV2, DCNV2Config, DCNV2Dataset, collate_fn
from utils.polars_pipeline import load_yambda_polars, PolarsPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Configuration
# ============================================================================

# Sparse features: REMOVED item_id - using pre-trained embeddings instead
SPARSE_FEATURES = ['uid', 'hour_of_day', 'day_of_week']

# Per-feature embedding dimensions (based on vocabulary size heuristics)
# Rule of thumb: embed_dim ≈ min(50, vocab_size ** 0.25)
# - uid: ~8K vocab → 4th root = 9.4 → use 16 (allows more user representation)
# - hour_of_day: 24 vocab → 4th root = 2.2 → use 4
# - day_of_week: 7 vocab → 4th root = 1.6 → use 2
SPARSE_EMBED_DIMS = {
    'uid': 16,           # Main user representation
    'hour_of_day': 4,    # Small vocab, small embedding
    'day_of_week': 2,    # Very small vocab, minimal embedding
}

# Dense features: Numerical and low-cardinality categoricals
DENSE_FEATURES = [
    'is_organic',
    'track_length_seconds',
    # User historical
    'user_total_listens', 'user_avg_completion', 'user_std_completion',
    'user_median_completion', 'user_unique_items', 'user_organic_ratio',
    'user_active_span', 'user_listen_rate',
    # Item historical
    'item_total_plays', 'item_avg_completion', 'item_std_completion',
    'item_unique_listeners', 'item_organic_ratio', 'item_repeat_ratio',
    # User-item
    'has_listened_before', 'previous_listen_count',
]

LABEL_COL = 'label'
ITEM_EMBED_DIM = 128  # Pre-trained embedding dimension


# ============================================================================
# Pre-trained Embedding Loader
# ============================================================================

class PretrainedItemEmbeddings:
    """Loads and manages pre-trained item embeddings."""
    
    def __init__(self, embeddings_path: str, use_normalized: bool = True):
        self.embeddings_path = Path(embeddings_path)
        self.use_normalized = use_normalized
        self.item_to_idx: Dict[int, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.embed_dim: int = ITEM_EMBED_DIM
        
    def load(self, item_ids: Optional[set] = None) -> 'PretrainedItemEmbeddings':
        """
        Load embeddings from parquet file.
        
        Args:
            item_ids: If provided, only load embeddings for these items (memory efficient)
        """
        logger.info(f"Loading pre-trained embeddings from {self.embeddings_path}...")
        
        embed_col = 'normalized_embed' if self.use_normalized else 'embed'
        
        # Read in batches to avoid memory issues
        pf = pq.ParquetFile(self.embeddings_path)
        
        embeddings_list = []
        idx = 0
        
        for batch in pf.iter_batches(batch_size=100000, columns=['item_id', embed_col]):
            batch_ids = batch.column('item_id').to_pylist()
            batch_embeds = batch.column(embed_col).to_pylist()
            
            for item_id, embed in zip(batch_ids, batch_embeds):
                # Filter to relevant items if specified
                if item_ids is not None and item_id not in item_ids:
                    continue
                    
                self.item_to_idx[item_id] = idx
                embeddings_list.append(embed)
                idx += 1
                
            if idx % 500000 == 0:
                logger.info(f"  Loaded {idx:,} embeddings...")
        
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        self.embed_dim = self.embeddings.shape[1]
        
        logger.info(f"  Loaded {len(self.item_to_idx):,} item embeddings (dim={self.embed_dim})")
        
        return self
    
    def get_embeddings(self, item_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for item IDs.
        Returns zero vector for unknown items (cold-start handling).
        """
        result = np.zeros((len(item_ids), self.embed_dim), dtype=np.float32)
        
        for i, item_id in enumerate(item_ids):
            if item_id in self.item_to_idx:
                idx = self.item_to_idx[item_id]
                result[i] = self.embeddings[idx]
        
        return result
    
    def coverage(self, item_ids: np.ndarray) -> float:
        """Calculate what fraction of items have embeddings."""
        found = sum(1 for item_id in item_ids if item_id in self.item_to_idx)
        return found / len(item_ids) if len(item_ids) > 0 else 0.0


# ============================================================================
# Feature Processor with Pre-trained Embeddings
# ============================================================================

class DCNV2PretrainedFeatureProcessor:
    """Prepares features for DCN-V2 with pre-trained item embeddings."""
    
    def __init__(
        self,
        sparse_features: list,
        dense_features: list,
        item_embeddings: PretrainedItemEmbeddings,
    ):
        self.sparse_features = sparse_features  # Excludes item_id
        self.dense_features = dense_features
        self.item_embeddings = item_embeddings
        self.sparse_encoders: Dict[str, LabelEncoder] = {}
        self.dense_scaler: Optional[StandardScaler] = None
        self.vocab_sizes: Dict[str, int] = {}
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DCNV2PretrainedFeatureProcessor':
        """Fit encoders on training data."""
        logger.info("Fitting feature processor (with pre-trained item embeddings)...")
        
        # Fit sparse encoders (excluding item_id)
        for col in self.sparse_features:
            if col in df.columns:
                le = LabelEncoder()
                values = df[col].fillna(-1).astype(str)
                le.fit(values)
                self.sparse_encoders[col] = le
                self.vocab_sizes[col] = len(le.classes_) + 1
                logger.info(f"  {col}: vocab_size={self.vocab_sizes[col]:,}")
        
        # Fit dense scaler (original dense features only, item embeddings are pre-normalized)
        if self.dense_features:
            dense_cols = [c for c in self.dense_features if c in df.columns]
            dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
            self.dense_scaler = StandardScaler()
            self.dense_scaler.fit(dense_data)
            logger.info(f"  Dense features: {len(dense_cols)} columns scaled")
        
        # Log item embedding coverage
        if 'item_id' in df.columns:
            coverage = self.item_embeddings.coverage(df['item_id'].values)
            logger.info(f"  Item embedding coverage: {coverage:.1%}")
        
        self._fitted = True
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Transform data for DCN-V2 with pre-trained embeddings."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Sparse features (excluding item_id)
        sparse_data = {}
        for col in self.sparse_features:
            if col in df.columns and col in self.sparse_encoders:
                values = df[col].fillna(-1).astype(str)
                le = self.sparse_encoders[col]
                encoded = np.zeros(len(values), dtype=np.int64)
                known_mask = values.isin(le.classes_)
                encoded[known_mask] = le.transform(values[known_mask]) + 1
                sparse_data[col] = encoded
        
        # Dense features (original)
        dense_cols = [c for c in self.dense_features if c in df.columns]
        dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
        if self.dense_scaler is not None:
            dense_data = self.dense_scaler.transform(dense_data)
        
        # Add pre-trained item embeddings to dense features
        if 'item_id' in df.columns:
            item_embeds = self.item_embeddings.get_embeddings(df['item_id'].values)
            dense_data = np.concatenate([dense_data, item_embeds], axis=1)
        
        labels = df[LABEL_COL].values.astype(np.float32)
        
        return sparse_data, dense_data, labels
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)
    
    @property
    def total_dense_dim(self) -> int:
        """Total dense dimension including item embeddings."""
        return len(self.dense_features) + self.item_embeddings.embed_dim


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, data_loader, criterion, device):
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
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            n_batches += 1
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    metrics = {
        'loss': total_loss / n_batches,
        'auc_roc': roc_auc_score(labels, preds),
        'auc_pr': average_precision_score(labels, preds),
        'f1': f1_score(labels, (preds > 0.5).astype(int)),
    }
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DCN-V2 with pre-trained item embeddings')
    
    # Data
    parser.add_argument('--data_dir', type=str, default=os.environ.get('YAMBDA_DATA_DIR', ''))
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Path to embeddings.parquet (default: auto-detect)')
    parser.add_argument('--train_days', type=int, default=30)
    parser.add_argument('--test_days', type=int, default=1)
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--cross_layers', type=int, default=3)
    parser.add_argument('--cross_rank', type=int, default=32)
    parser.add_argument('--mlp_dims', type=str, default='256,128,64')
    parser.add_argument('--structure', type=str, default='parallel', choices=['parallel', 'stacked'])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=3)
    
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    mlp_dims = [int(x) for x in args.mlp_dims.split(',')]
    
    if not args.data_dir:
        logger.error("Data directory not specified. Set YAMBDA_DATA_DIR or use --data_dir")
        sys.exit(1)
    
    # Auto-detect embeddings path
    if args.embeddings_path is None:
        data_path = Path(args.data_dir)
        # Try common locations
        candidates = [
            data_path.parent / 'embeddings.parquet',  # Dataset/Yandex/embeddings.parquet
            data_path / 'embeddings.parquet',
        ]
        for candidate in candidates:
            if candidate.exists():
                args.embeddings_path = str(candidate)
                break
        if args.embeddings_path is None:
            logger.error("Could not find embeddings.parquet. Specify with --embeddings_path")
            sys.exit(1)
    
    train_days = None if args.train_days == 0 else args.train_days
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_days_str = 'full' if train_days is None else f'{train_days}d'
    exp_name = f'dcnv2_pretrained_{train_days_str}_{args.structure}_{timestamp}'
    output_path = Path(args.output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("DCN-V2 with Pre-trained Item Embeddings")
    logger.info("=" * 70)
    logger.info(f"Using pre-trained 128-dim item embeddings")
    logger.info(f"Sparse features (learned): {SPARSE_FEATURES}")
    logger.info(f"Dense features: {len(DENSE_FEATURES)} + 128 (item embeddings)")
    
    # ========== Step 1: Load Pre-trained Embeddings ==========
    logger.info("\n[1/5] Loading pre-trained item embeddings...")
    start_time = time.time()
    
    item_embeddings = PretrainedItemEmbeddings(args.embeddings_path, use_normalized=True)
    
    # First pass: get unique item IDs from data to load only relevant embeddings
    logger.info("  Scanning data for unique item IDs...")
    train_lf, test_lf = load_yambda_polars(
        data_dir=args.data_dir,
        train_days=train_days,
        test_days=args.test_days,
        lazy=True
    )
    
    # Get unique items (sampling for efficiency)
    unique_items = set(
        train_lf.select('item_id').unique().collect()['item_id'].to_list()
    )
    logger.info(f"  Found {len(unique_items):,} unique items in training data")
    
    # Load only relevant embeddings
    item_embeddings.load(item_ids=unique_items)
    
    # ========== Step 2: Load and Process Data ==========
    logger.info("\n[2/5] Loading data...")
    
    # PolarsPipeline uses categorical_columns and numerical_columns
    # It auto-generates derived and historical features
    pipeline = PolarsPipeline(
        categorical_columns=['uid', 'item_id', 'is_organic'],
        numerical_columns=['track_length_seconds'],
        add_derived_features=True,
        add_historical_features=True,
    )
    
    # fit_transform and transform return (DataFrame, feature_columns) tuple
    # They already collect by default, so just unpack and convert to pandas
    train_pl_df, feature_cols = pipeline.fit_transform(train_lf)
    train_df = train_pl_df.to_pandas()
    
    test_pl_df, _ = pipeline.transform(test_lf)
    test_df = test_pl_df.to_pandas()
    
    logger.info(f"Train samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # ========== Step 3: Prepare Features ==========
    logger.info("\n[3/5] Preparing features with pre-trained embeddings...")
    
    feature_processor = DCNV2PretrainedFeatureProcessor(
        sparse_features=SPARSE_FEATURES,  # Excludes item_id
        dense_features=DENSE_FEATURES,
        item_embeddings=item_embeddings,
    )
    
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    logger.info(f"Sparse features: {list(train_sparse.keys())}")
    logger.info(f"Dense features: {train_dense.shape[1]} (including 128-dim item embeddings)")
    logger.info(f"Train positive rate: {train_labels.mean():.1%}")
    logger.info(f"Test positive rate: {test_labels.mean():.1%}")
    
    # Create datasets
    train_dataset = DCNV2Dataset(train_sparse, train_dense, train_labels)
    test_dataset = DCNV2Dataset(test_sparse, test_dense, test_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # ========== Step 4: Build Model ==========
    logger.info("\n[4/5] Training DCN-V2...")
    
    config = DCNV2Config(
        sparse_features=feature_processor.vocab_sizes,
        dense_dim=feature_processor.total_dense_dim,
        embed_dim=args.embed_dim,  # Default (not used if all features have custom dims)
        embed_dims=SPARSE_EMBED_DIMS,  # Per-feature embedding dimensions
        cross_layers=args.cross_layers,
        cross_rank=args.cross_rank,
        mlp_dims=mlp_dims,
        structure=args.structure,
        dropout=args.dropout,
    )
    
    # Log embedding dimensions
    logger.info("Embedding dimensions per feature:")
    for feat, vocab in feature_processor.vocab_sizes.items():
        dim = config.get_embed_dim(feat)
        logger.info(f"  {feat}: vocab={vocab:,} → embed_dim={dim}")
    
    model = DCNV2(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"  (Reduced from ~3M to ~{n_params:,} by using pre-trained item embeddings)")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # ========== Training Loop ==========
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train AUC: {train_metrics['auc_roc']:.4f} | "
            f"Test AUC: {test_metrics['auc_roc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        scheduler.step(test_metrics['auc_roc'])
        
        if test_metrics['auc_roc'] > best_auc:
            best_auc = test_metrics['auc_roc']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'test_metrics': test_metrics,
            }, output_path / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # ========== Step 5: Final Evaluation ==========
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
    
    # Save results
    results = {
        'experiment': exp_name,
        'config': {
            'train_days': train_days,
            'embed_dim': args.embed_dim,
            'cross_layers': args.cross_layers,
            'cross_rank': args.cross_rank,
            'mlp_dims': mlp_dims,
            'structure': args.structure,
            'pretrained_item_embeddings': True,
            'item_embed_dim': ITEM_EMBED_DIM,
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Outputs: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

