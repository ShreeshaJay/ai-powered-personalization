"""
Chapter 7: MMoE Single-Task Training Script with Pre-trained Item Embeddings

This script trains MMoE (Mixture-of-Experts) for SINGLE TASK learning to provide
a clean architectural comparison with DCN-V2. Both models are trained on the
same task (completion) with the same pre-trained item embeddings.

Comparison Design:
    - DCN-V2: Cross Network + Deep Network
    - MMoE (single-task): Shared Experts + Gate + Tower
    
This isolates the architectural difference without confounding from multi-task learning.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    MMoE Single-Task                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Sparse Features              Dense Features                    │
    │  (uid, hour, day)             (numerical + 128-dim item embed)  │
    │       │                            │                            │
    │       ▼                            │                            │
    │  ┌─────────┐                       │                            │
    │  │Embedding│                       │                            │
    │  │ Tables  │                       │                            │
    │  └────┬────┘                       │                            │
    │       │                            │                            │
    │       └────────────┬───────────────┘                            │
    │                    ▼                                            │
    │            ┌──────────────┐                                     │
    │            │   Concat     │                                     │
    │            └──────┬───────┘                                     │
    │                   │                                             │
    │   ┌───────────────┼───────────────┬───────────────┐             │
    │   ▼               ▼               ▼               ▼             │
    │ ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐            │
    │ │Expert│        │Expert│        │Expert│        │Expert│            │
    │ │  1  │        │  2  │        │  3  │        │  4  │            │
    │ └──┬──┘        └──┬──┘        └──┬──┘        └──┬──┘            │
    │    └──────────────┴──────────────┴──────────────┘               │
    │                         │                                       │
    │                         ▼                                       │
    │                  ┌─────────────┐                                │
    │                  │   Gate      │  ← Learns expert weighting     │
    │                  │ (Softmax)   │                                │
    │                  └──────┬──────┘                                │
    │                         │                                       │
    │                         ▼                                       │
    │                  ┌─────────────┐                                │
    │                  │   Tower     │  ← Task-specific MLP           │
    │                  │    MLP      │                                │
    │                  └──────┬──────┘                                │
    │                         │                                       │
    │                         ▼                                       │
    │                  ┌─────────────┐                                │
    │                  │ P(complete) │                                │
    │                  └─────────────┘                                │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    python train_mmoe_single.py --train_days 30 --data_dir /path/to/yambda/flat
    
Author: Chapter 7 - Advanced Ranking Models
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import polars as pl
from polars import col
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))

from utils.polars_pipeline import load_yambda_polars, PolarsPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Configuration (matches DCN-V2 pretrained for fair comparison)
# ============================================================================

# Sparse features: item_id removed (using pre-trained embeddings instead)
SPARSE_FEATURES = ['uid', 'hour_of_day', 'day_of_week']

# Per-feature embedding dimensions (same as DCN-V2 pretrained)
SPARSE_EMBED_DIMS = {
    'uid': 16,          # ~8K users, primary personalization signal
    'hour_of_day': 4,   # 24 values, captures time-of-day patterns
    'day_of_week': 2,   # 7 values, weekday/weekend distinction
}

DENSE_FEATURES = [
    'is_organic', 'track_length_seconds',
    'user_total_listens', 'user_avg_completion', 'user_std_completion',
    'user_median_completion', 'user_unique_items', 'user_organic_ratio',
    'user_active_span', 'user_listen_rate',
    'item_total_plays', 'item_avg_completion', 'item_std_completion',
    'item_unique_listeners', 'item_organic_ratio', 'item_repeat_ratio',
    'has_listened_before', 'previous_listen_count',
]

LABEL_COL = 'label'  # Completion: played_ratio_pct >= 50
ITEM_EMBED_DIM = 128  # Pre-trained item embedding dimension


# ============================================================================
# Pre-trained Item Embeddings (same as DCN-V2 pretrained)
# ============================================================================

class PretrainedItemEmbeddings:
    """Memory-efficient loader for pre-trained item embeddings."""
    
    def __init__(self, embeddings_path: str, embed_dim: int = 128):
        self.embeddings_path = Path(embeddings_path)
        self.embed_dim = embed_dim
        self.embeddings: Optional[np.ndarray] = None
        self.item_to_idx: Dict[int, int] = {}
        
    def load(self, item_ids: Optional[Set[int]] = None, batch_size: int = 100_000):
        """Load embeddings, optionally filtering to specific item IDs."""
        logger.info(f"Loading pre-trained embeddings from {self.embeddings_path}...")
        
        pf = pq.ParquetFile(self.embeddings_path)
        
        all_embeddings = []
        all_item_ids = []
        
        for batch in pf.iter_batches(batch_size=batch_size, columns=['item_id', 'normalized_embed']):
            batch_df = batch.to_pandas()
            
            if item_ids is not None:
                mask = batch_df['item_id'].isin(item_ids)
                batch_df = batch_df[mask]
            
            if len(batch_df) > 0:
                batch_item_ids = batch_df['item_id'].values
                batch_embeds = np.stack(batch_df['normalized_embed'].values).astype(np.float32)
                
                all_item_ids.extend(batch_item_ids)
                all_embeddings.append(batch_embeds)
        
        if all_embeddings:
            self.embeddings = np.vstack(all_embeddings)
            self.item_to_idx = {int(item_id): idx for idx, item_id in enumerate(all_item_ids)}
        else:
            self.embeddings = np.zeros((0, self.embed_dim), dtype=np.float32)
            self.item_to_idx = {}
            
        logger.info(f"  Loaded {len(self.item_to_idx):,} item embeddings (dim={self.embed_dim})")
        
    def get_embeddings(self, item_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for array of item IDs. Unknown items get zero vectors."""
        result = np.zeros((len(item_ids), self.embed_dim), dtype=np.float32)
        for i, item_id in enumerate(item_ids):
            if item_id in self.item_to_idx:
                idx = self.item_to_idx[item_id]
                result[i] = self.embeddings[idx]
        return result


# ============================================================================
# Feature Processor with Pre-trained Embeddings
# ============================================================================

class MMoESingleFeatureProcessor:
    """Feature processor for single-task MMoE with pre-trained item embeddings."""
    
    def __init__(
        self,
        sparse_features: list,
        dense_features: list,
        item_embeddings: PretrainedItemEmbeddings,
    ):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.item_embeddings = item_embeddings
        
        self.sparse_encoders: Dict[str, LabelEncoder] = {}
        self.dense_scaler: Optional[StandardScaler] = None
        self.vocab_sizes: Dict[str, int] = {}
        self._fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit encoders and scaler on training data."""
        logger.info("Fitting feature processor (with pre-trained item embeddings)...")
        
        # Fit sparse encoders (excluding item_id)
        for col_name in self.sparse_features:
            if col_name in df.columns:
                le = LabelEncoder()
                le.fit(df[col_name].fillna(-1).astype(str))
                self.sparse_encoders[col_name] = le
                self.vocab_sizes[col_name] = len(le.classes_) + 1  # +1 for padding
                logger.info(f"  {col_name}: vocab_size={self.vocab_sizes[col_name]:,}")
        
        # Fit dense scaler
        dense_cols = [c for c in self.dense_features if c in df.columns]
        if dense_cols:
            dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
            self.dense_scaler = StandardScaler()
            self.dense_scaler.fit(dense_data)
            logger.info(f"  Dense features: {len(dense_cols)} columns scaled")
        
        # Report item embedding coverage
        if 'item_id' in df.columns:
            unique_items = set(df['item_id'].unique())
            covered = sum(1 for item in unique_items if item in self.item_embeddings.item_to_idx)
            coverage = covered / len(unique_items) * 100
            logger.info(f"  Item embedding coverage: {coverage:.1f}%")
        
        self._fitted = True
        return self
        
    def transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Transform features for model input."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Transform sparse features
        sparse_data = {}
        for col_name in self.sparse_features:
            if col_name in df.columns and col_name in self.sparse_encoders:
                values = df[col_name].fillna(-1).astype(str)
                le = self.sparse_encoders[col_name]
                encoded = np.zeros(len(values), dtype=np.int64)
                known_mask = values.isin(le.classes_)
                encoded[known_mask] = le.transform(values[known_mask]) + 1
                sparse_data[col_name] = encoded
        
        # Transform dense features
        dense_cols = [c for c in self.dense_features if c in df.columns]
        dense_data = df[dense_cols].fillna(0).values.astype(np.float32)
        if self.dense_scaler is not None:
            dense_data = self.dense_scaler.transform(dense_data)
        
        # Add pre-trained item embeddings to dense features
        if 'item_id' in df.columns:
            item_embeds = self.item_embeddings.get_embeddings(df['item_id'].values)
            dense_data = np.concatenate([dense_data, item_embeds], axis=1)
        
        # Get labels
        labels = df[LABEL_COL].values.astype(np.float32)
        
        return sparse_data, dense_data, labels
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)
    
    @property
    def total_dense_dim(self) -> int:
        return len(self.dense_features) + self.item_embeddings.embed_dim


# ============================================================================
# Dataset
# ============================================================================

class MMoESingleDataset(Dataset):
    """PyTorch Dataset for single-task MMoE."""
    
    def __init__(
        self,
        sparse_data: Dict[str, np.ndarray],
        dense_data: np.ndarray,
        labels: np.ndarray,
    ):
        self.sparse_data = sparse_data
        self.dense_data = torch.FloatTensor(dense_data)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sparse = {k: v[idx] for k, v in self.sparse_data.items()}
        return sparse, self.dense_data[idx], self.labels[idx]


def collate_fn(batch):
    """Collate function for DataLoader."""
    sparse_list, dense_list, label_list = zip(*batch)
    
    sparse_batch = {}
    for key in sparse_list[0].keys():
        sparse_batch[key] = torch.LongTensor([s[key] for s in sparse_list])
    
    dense_batch = torch.stack(dense_list)
    labels_batch = torch.stack(label_list)
    
    return sparse_batch, dense_batch, labels_batch


# ============================================================================
# MMoE Single-Task Model
# ============================================================================

class Expert(nn.Module):
    """Expert network - a standard MLP."""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
    def forward(self, x):
        return self.network(x)


class Tower(nn.Module):
    """Task tower - MLP ending with single output."""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


class MMoESingleTask(nn.Module):
    """MMoE model for single task (completion prediction).
    
    This is a simplified MMoE with one gate and one tower, essentially
    becoming a "Mixture of Experts" model for single-task learning.
    """
    
    def __init__(
        self,
        sparse_features: Dict[str, int],  # {name: vocab_size}
        dense_dim: int,
        embed_dim: int = 16,
        embed_dims: Optional[Dict[str, int]] = None,  # Per-feature dims
        num_experts: int = 4,
        expert_dims: list = [256, 128],
        tower_dims: list = [64, 32],
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        
        self.sparse_features = sparse_features
        self.embed_dim = embed_dim
        self.embed_dims = embed_dims or {}
        
        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_sparse_dim = 0
        for feat_name, vocab_size in sparse_features.items():
            feat_embed_dim = self.embed_dims.get(feat_name, embed_dim)
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=feat_embed_dim,
                padding_idx=0,
            )
            total_sparse_dim += feat_embed_dim
        
        # Input dimension
        self.input_dim = total_sparse_dim + dense_dim
        
        # Experts
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(self.input_dim, expert_dims, dropout, use_bn)
            for _ in range(num_experts)
        ])
        self.expert_output_dim = expert_dims[-1] if expert_dims else self.input_dim
        
        # Gate (single task)
        self.gate = nn.Linear(self.input_dim, num_experts)
        
        # Tower (single task)
        self.tower = Tower(self.expert_output_dim, tower_dims, dropout, use_bn)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, sparse_inputs: Dict[str, torch.Tensor], dense_inputs: torch.Tensor):
        # Embed sparse features
        sparse_embeds = []
        for feat_name in self.sparse_features:
            if feat_name in sparse_inputs:
                embed = self.embeddings[feat_name](sparse_inputs[feat_name])
                sparse_embeds.append(embed)
        
        # Concatenate all inputs
        if sparse_embeds:
            sparse_concat = torch.cat(sparse_embeds, dim=1)
            x = torch.cat([sparse_concat, dense_inputs], dim=1)
        else:
            x = dense_inputs
        
        # Expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Shape: (batch_size, num_experts, expert_dim)
        
        # Gate weights
        gate_weights = torch.softmax(self.gate(x), dim=1)  # (batch_size, num_experts)
        
        # Weighted combination of expert outputs
        gate_weights = gate_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        mixed = torch.sum(expert_outputs * gate_weights, dim=1)  # (batch_size, expert_dim)
        
        # Tower
        logits = self.tower(mixed)
        
        return logits, gate_weights.squeeze(-1)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for sparse_batch, dense_batch, labels_batch in dataloader:
        sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
        dense_batch = dense_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(sparse_batch, dense_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels_batch)
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_gate_weights = []
    
    with torch.no_grad():
        for sparse_batch, dense_batch, labels_batch in dataloader:
            sparse_batch = {k: v.to(device) for k, v in sparse_batch.items()}
            dense_batch = dense_batch.to(device)
            
            logits, gate_weights = model(sparse_batch, dense_batch)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
            all_gate_weights.append(gate_weights.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_gate_weights = np.vstack(all_gate_weights)
    
    metrics = {
        'auc_roc': roc_auc_score(all_labels, all_preds),
        'auc_pr': average_precision_score(all_labels, all_preds),
        'f1': f1_score(all_labels, (all_preds > 0.5).astype(int)),
    }
    
    return metrics, all_preds, all_labels, all_gate_weights


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MMoE Single-Task with Pre-trained Embeddings')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('YAMBDA_DATA_DIR', ''))
    parser.add_argument('--train_days', type=int, default=30)
    parser.add_argument('--test_days', type=int, default=1)
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--expert_dims', type=str, default='256,128')
    parser.add_argument('--tower_dims', type=str, default='64,32')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    
    args = parser.parse_args()
    
    # Parse dimensions
    expert_dims = [int(x) for x in args.expert_dims.split(',')]
    tower_dims = [int(x) for x in args.tower_dims.split(',')]
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"mmoe_single_{args.train_days}d_{timestamp}"
    output_path = Path('outputs') / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MMoE Single-Task with Pre-trained Item Embeddings")
    logger.info("=" * 70)
    logger.info("Comparison with DCN-V2: Same features, different architecture")
    logger.info(f"  - DCN-V2: Cross Network + Deep Network")
    logger.info(f"  - MMoE:   {args.num_experts} Experts + Gate + Tower")
    logger.info(f"Using pre-trained {ITEM_EMBED_DIM}-dim item embeddings")
    logger.info(f"Sparse features (learned): {SPARSE_FEATURES}")
    logger.info(f"Dense features: {len(DENSE_FEATURES)} + {ITEM_EMBED_DIM} (item embeddings)")
    
    # ========== Step 1: Load Pre-trained Embeddings ==========
    logger.info("\n[1/5] Loading pre-trained item embeddings...")
    
    embeddings_path = Path(args.data_dir).parent / 'embeddings.parquet'
    item_embeddings = PretrainedItemEmbeddings(str(embeddings_path), embed_dim=ITEM_EMBED_DIM)
    
    # First pass to get unique item IDs
    logger.info("  Scanning data for unique item IDs...")
    train_lf, test_lf = load_yambda_polars(
        args.data_dir,
        train_days=args.train_days if args.train_days > 0 else None,
        test_days=args.test_days,
    )
    
    unique_items = set(train_lf.select('item_id').unique().collect()['item_id'].to_list())
    unique_items.update(test_lf.select('item_id').unique().collect()['item_id'].to_list())
    logger.info(f"  Found {len(unique_items):,} unique items in data")
    
    item_embeddings.load(item_ids=unique_items)
    
    # ========== Step 2: Load and Process Data ==========
    logger.info("\n[2/5] Loading data...")
    
    pipeline = PolarsPipeline(
        categorical_columns=['uid', 'item_id', 'is_organic'],
        numerical_columns=['track_length_seconds'],
        add_derived_features=True,
        add_historical_features=True,
    )
    
    train_pl_df, feature_cols = pipeline.fit_transform(train_lf)
    train_df = train_pl_df.to_pandas()
    
    test_pl_df, _ = pipeline.transform(test_lf)
    test_df = test_pl_df.to_pandas()
    
    logger.info(f"Train samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    
    # ========== Step 3: Prepare Features ==========
    logger.info("\n[3/5] Preparing features with pre-trained embeddings...")
    
    feature_processor = MMoESingleFeatureProcessor(
        sparse_features=SPARSE_FEATURES,
        dense_features=DENSE_FEATURES,
        item_embeddings=item_embeddings,
    )
    
    train_sparse, train_dense, train_labels = feature_processor.fit_transform(train_df)
    test_sparse, test_dense, test_labels = feature_processor.transform(test_df)
    
    logger.info(f"Sparse features: {SPARSE_FEATURES}")
    logger.info(f"Dense features: {feature_processor.total_dense_dim} (including {ITEM_EMBED_DIM}-dim item embeddings)")
    logger.info(f"Train positive rate: {train_labels.mean()*100:.1f}%")
    logger.info(f"Test positive rate: {test_labels.mean()*100:.1f}%")
    
    # Create datasets
    train_dataset = MMoESingleDataset(train_sparse, train_dense, train_labels)
    test_dataset = MMoESingleDataset(test_sparse, test_dense, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn, pin_memory=True)
    
    # ========== Step 4: Train Model ==========
    logger.info("\n[4/5] Training MMoE Single-Task...")
    
    logger.info("Embedding dimensions per feature:")
    for feat in SPARSE_FEATURES:
        dim = SPARSE_EMBED_DIMS.get(feat, args.embed_dim)
        logger.info(f"  {feat}: vocab={feature_processor.vocab_sizes[feat]:,} → embed_dim={dim}")
    
    model = MMoESingleTask(
        sparse_features=feature_processor.vocab_sizes,
        dense_dim=feature_processor.total_dense_dim,
        embed_dim=args.embed_dim,
        embed_dims=SPARSE_EMBED_DIMS,
        num_experts=args.num_experts,
        expert_dims=expert_dims,
        tower_dims=tower_dims,
        dropout=args.dropout,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"  Experts: {args.num_experts}, dims: {expert_dims}")
    logger.info(f"  Tower dims: {tower_dims}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics, _, _, _ = evaluate(model, test_loader, device)
        test_auc = test_metrics['auc_roc']
        
        scheduler.step(test_auc)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1:2d}/{args.epochs} | "
                   f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | "
                   f"Time: {epoch_time:.1f}s")
        
        if test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_auc': test_auc,
                'config': {
                    'sparse_features': feature_processor.vocab_sizes,
                    'dense_dim': feature_processor.total_dense_dim,
                    'embed_dim': args.embed_dim,
                    'embed_dims': SPARSE_EMBED_DIMS,
                    'num_experts': args.num_experts,
                    'expert_dims': expert_dims,
                    'tower_dims': tower_dims,
                }
            }, output_path / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # ========== Step 5: Final Evaluation ==========
    logger.info("\n[5/5] Final Evaluation...")
    
    checkpoint = torch.load(output_path / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_metrics, _, _, train_gate_weights = evaluate(model, train_loader, device)
    test_metrics, test_preds, test_labels_arr, test_gate_weights = evaluate(model, test_loader, device)
    
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
    
    # Expert utilization analysis
    logger.info("\nExpert Utilization (Gate Weights):")
    mean_gate = test_gate_weights.mean(axis=0)
    for i, weight in enumerate(mean_gate):
        logger.info(f"  Expert {i+1}: {weight:.3f} ({weight*100:.1f}%)")
    
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Best Epoch: {checkpoint['epoch'] + 1}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Outputs: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

