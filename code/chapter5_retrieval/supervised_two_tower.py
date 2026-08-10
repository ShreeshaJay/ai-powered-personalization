"""
Chapter 5: Supervised Two-Tower Model for Personalized Retrieval

This script implements a learned two-tower architecture where:
- User Tower: A neural network that encodes user history sequences
- Item Tower: Projects item embeddings into a shared space

Unlike the unsupervised approach (mean pooling / last item), this model:
- Learns how to weight and combine history items
- Is trained end-to-end to optimize for next-item prediction
- Can capture sequential patterns in user behavior

Architecture:
    User History Sequence -> [User Tower (Attention/GRU)] -> User Embedding
    Item Embedding ------> [Item Tower (Projection)] -----> Item Embedding
    
    Similarity = dot_product(user_emb, item_emb)

Training:
    - Positive pairs: (user_history, next_item)
    - Negative sampling: Random items from batch (in-batch negatives)
    - Loss: Contrastive loss (softmax cross-entropy over similarities)

Author: Shreesha Jagadeesh
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import faiss

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
DATA_PATH = Path("path/to/Dataset/SIGIR-ecom-data-challenge/train")
CATALOG_FILE = "sku_to_content.csv"
BROWSING_FILE = "browsing_train.csv"

# Model configuration
EMBEDDING_DIM = 50  # Description embedding dimension
HIDDEN_DIM = 64     # Hidden dimension for user tower
USER_EMB_DIM = 50   # Output user embedding dimension (same as item for dot product)
MAX_SEQ_LEN = 20    # Maximum sequence length for user history

# User tower type: 'attention' (simple) or 'transformer' (full self-attention)
USER_TOWER_TYPE = 'attention'  # Options: 'attention' or 'transformer'

# Transformer-specific config (only used if USER_TOWER_TYPE == 'transformer')
NUM_TRANSFORMER_HEADS = 2
NUM_TRANSFORMER_LAYERS = 2

# Training configuration
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 1e-5
NUM_NEGATIVES = 10  # Additional hard negatives per sample

# Evaluation
MIN_SESSION_LENGTH = 3
GROUND_TRUTH_ACTIONS = ['detail', 'add', 'purchase']
TOP_K_RETRIEVAL = [5, 10, 20, 50]

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING (Reused from personalized_retrieval.py)
# =============================================================================

def parse_embedding_batch(embedding_strs: List[str]) -> np.ndarray:
    """Parse a batch of embedding strings efficiently."""
    embeddings = []
    for emb_str in embedding_strs:
        if pd.isna(emb_str) or emb_str == '':
            embeddings.append(None)
        else:
            try:
                emb = np.array(json.loads(emb_str), dtype=np.float32)
                embeddings.append(emb)
            except:
                embeddings.append(None)
    return embeddings


def load_product_catalog(data_path: Path) -> pl.DataFrame:
    """Load product catalog with embeddings."""
    print("\nLoading product catalog...")
    catalog_path = data_path / CATALOG_FILE
    
    catalog_df = pl.read_csv(catalog_path)
    print(f"  Total products: {len(catalog_df):,}")
    
    # Parse description embeddings
    print("  Parsing description embeddings...")
    desc_emb_strs = catalog_df['description_vector'].to_list()
    desc_embeddings = parse_embedding_batch(desc_emb_strs)
    
    # Filter to products with valid embeddings
    valid_mask = [emb is not None for emb in desc_embeddings]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    
    catalog_df = catalog_df[valid_indices]
    valid_embeddings = [desc_embeddings[i] for i in valid_indices]
    
    # Stack embeddings
    embedding_matrix = np.vstack(valid_embeddings)
    catalog_df = catalog_df.with_columns(pl.Series('embedding', list(embedding_matrix)))
    
    print(f"  Products with valid embeddings: {len(catalog_df):,}")
    print(f"  Embedding dimension: {embedding_matrix.shape[1]}")
    
    return catalog_df


def load_browsing_sessions_lazy(data_path: Path, filename: str) -> pl.LazyFrame:
    """Load browsing data as lazy frame for memory efficiency."""
    print(f"\nLoading browsing sessions (lazy)...")
    browsing_path = data_path / filename
    
    lf = pl.scan_csv(browsing_path)
    return lf


def build_user_sessions(
    browsing_lf: pl.LazyFrame,
    min_session_length: int = 3,
    ground_truth_actions: List[str] = None
) -> List[Dict]:
    """
    Build user sessions for training and evaluation.
    
    For each session with N items:
    - history_items: Items 1 to N-1
    - ground_truth_item: Item N (the next item to predict)
    - max_timestamp: Latest timestamp in session (for time-based splitting)
    """
    if ground_truth_actions is None:
        ground_truth_actions = ['detail', 'add', 'purchase']
    
    print("\nBuilding user sessions...")
    print(f"  Ground truth actions: {ground_truth_actions}")
    print(f"  Minimum session length: {min_session_length}")
    
    # Query to get ordered items per session WITH max timestamp for sorting
    sessions_query = (
        browsing_lf
        .filter(
            (pl.col('product_sku_hash').is_not_null()) &
            (pl.col('product_action').is_in(ground_truth_actions))
        )
        .sort(['session_id_hash', 'server_timestamp_epoch_ms'])
        .group_by('session_id_hash')
        .agg([
            pl.col('product_sku_hash').alias('items'),
            pl.col('server_timestamp_epoch_ms').max().alias('max_timestamp')
        ])
    )
    
    sessions_df = sessions_query.collect(engine="streaming")
    
    # Process into training format
    user_sessions = []
    for row in tqdm(sessions_df.iter_rows(named=True), total=len(sessions_df), desc="  Processing sessions"):
        items = row['items']
        # Deduplicate consecutive items (user may view same item multiple times)
        unique_items = []
        for item in items:
            if not unique_items or item != unique_items[-1]:
                unique_items.append(item)
        
        if len(unique_items) >= min_session_length:
            user_sessions.append({
                'session_id': row['session_id_hash'],
                'history_items': unique_items[:-1],  # N-1 items
                'ground_truth_item': unique_items[-1],  # Nth item
                'max_timestamp': row['max_timestamp']  # For time-based split
            })
    
    print(f"  Valid sessions: {len(user_sessions):,}")
    
    # Statistics
    history_lens = [len(s['history_items']) for s in user_sessions]
    print(f"  History length: min={min(history_lens)}, median={np.median(history_lens):.0f}, max={max(history_lens)}")
    
    return user_sessions


def compute_popularity_scores(
    browsing_lf: pl.LazyFrame,
    actions: List[str] = None
) -> Dict[str, int]:
    """Compute popularity scores for all items."""
    if actions is None:
        actions = ['detail', 'add', 'purchase']
    
    print("\nComputing popularity scores...")
    
    popularity_query = (
        browsing_lf
        .filter(
            (pl.col('product_sku_hash').is_not_null()) &
            (pl.col('product_action').is_in(actions))
        )
        .group_by('product_sku_hash')
        .agg(pl.len().alias('engagement_count'))
    )
    
    popularity_df = popularity_query.collect(engine="streaming")
    
    popularity_scores = {
        row['product_sku_hash']: row['engagement_count']
        for row in popularity_df.iter_rows(named=True)
    }
    
    print(f"  Items with engagement: {len(popularity_scores):,}")
    
    return popularity_scores


# =============================================================================
# PYTORCH DATASET
# =============================================================================

@dataclass
class ItemEmbeddingLookup:
    """Fast lookup for item embeddings."""
    sku_to_idx: Dict[str, int]
    idx_to_sku: Dict[int, str]
    embeddings: np.ndarray  # Shape: (num_items, embedding_dim)
    
    @classmethod
    def from_catalog(cls, catalog_df: pl.DataFrame) -> 'ItemEmbeddingLookup':
        """Build lookup from catalog dataframe."""
        skus = catalog_df['product_sku_hash'].to_list()
        embeddings = np.vstack(catalog_df['embedding'].to_list())
        
        sku_to_idx = {sku: idx for idx, sku in enumerate(skus)}
        idx_to_sku = {idx: sku for idx, sku in enumerate(skus)}
        
        return cls(sku_to_idx, idx_to_sku, embeddings)
    
    def get_embedding(self, sku: str) -> Optional[np.ndarray]:
        """Get embedding for a SKU."""
        idx = self.sku_to_idx.get(sku)
        if idx is not None:
            return self.embeddings[idx]
        return None
    
    def get_embeddings(self, skus: List[str]) -> Tuple[np.ndarray, List[bool]]:
        """Get embeddings for multiple SKUs. Returns embeddings and validity mask."""
        embeddings = []
        valid_mask = []
        for sku in skus:
            emb = self.get_embedding(sku)
            if emb is not None:
                embeddings.append(emb)
                valid_mask.append(True)
            else:
                embeddings.append(np.zeros(self.embeddings.shape[1], dtype=np.float32))
                valid_mask.append(False)
        return np.array(embeddings), valid_mask


class TwoTowerDataset(Dataset):
    """
    Dataset for training the two-tower model.
    
    Each sample contains:
    - User history: Sequence of item embeddings (padded/truncated to max_seq_len)
    - History mask: Valid positions in the sequence
    - Positive item: The next item embedding
    - Positive item index: For in-batch negative sampling
    """
    
    def __init__(
        self,
        user_sessions: List[Dict],
        item_lookup: ItemEmbeddingLookup,
        max_seq_len: int = 20
    ):
        self.user_sessions = user_sessions
        self.item_lookup = item_lookup
        self.max_seq_len = max_seq_len
        self.embedding_dim = item_lookup.embeddings.shape[1]
        
        # Pre-filter sessions to those with valid items
        self.valid_sessions = self._filter_valid_sessions()
        print(f"  Dataset samples: {len(self.valid_sessions):,}")
    
    def _filter_valid_sessions(self) -> List[Dict]:
        """Filter to sessions where ground truth item exists in catalog."""
        valid = []
        for session in self.user_sessions:
            gt_item = session['ground_truth_item']
            if gt_item in self.item_lookup.sku_to_idx:
                # Also need at least one valid history item
                history_valid = any(
                    sku in self.item_lookup.sku_to_idx 
                    for sku in session['history_items']
                )
                if history_valid:
                    valid.append(session)
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_sessions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        session = self.valid_sessions[idx]
        
        # Get history embeddings
        history_items = session['history_items'][-self.max_seq_len:]  # Take last max_seq_len items
        history_embs, valid_mask = self.item_lookup.get_embeddings(history_items)
        
        # Pad if needed
        seq_len = len(history_items)
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            history_embs = np.vstack([
                np.zeros((pad_len, self.embedding_dim), dtype=np.float32),
                history_embs
            ])
            valid_mask = [False] * pad_len + valid_mask
        
        # Get positive item
        gt_sku = session['ground_truth_item']
        gt_idx = self.item_lookup.sku_to_idx[gt_sku]
        gt_emb = self.item_lookup.embeddings[gt_idx]
        
        return {
            'history_embs': torch.tensor(history_embs, dtype=torch.float32),
            'history_mask': torch.tensor(valid_mask, dtype=torch.bool),
            'positive_emb': torch.tensor(gt_emb, dtype=torch.float32),
            'positive_idx': torch.tensor(gt_idx, dtype=torch.long),
            'session_idx': idx
        }


# =============================================================================
# TWO-TOWER MODEL
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence aggregation.
    
    Learns which items in the history are most relevant for predicting
    the next item. This is more expressive than mean pooling.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            mask: (batch, seq_len) - True for valid positions
            
        Returns:
            (batch, embedding_dim) - Attention-weighted sum
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        # Mask invalid positions
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        
        # Handle all-masked sequences
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Weighted sum
        output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, embedding_dim)
        
        return output


class UserTower(nn.Module):
    """
    User Tower: Encodes user history sequence into a user embedding.
    
    Architecture:
        Input: Sequence of item embeddings (batch, seq_len, embedding_dim)
        -> Attention pooling to aggregate sequence
        -> MLP projection to user embedding space
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Attention pooling
        self.attention_pool = AttentionPooling(embedding_dim, hidden_dim)
        
        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_embs: (batch, seq_len, embedding_dim)
            history_mask: (batch, seq_len)
            
        Returns:
            user_emb: (batch, output_dim)
        """
        # Aggregate sequence
        pooled = self.attention_pool(history_embs, history_mask)
        
        # Project to user space
        user_emb = self.projection(pooled)
        user_emb = self.layer_norm(user_emb)
        
        return user_emb


class TransformerUserTower(nn.Module):
    """
    Transformer-based User Tower: Uses self-attention to encode user history.
    
    This is the full Transformer-style attention where:
    - Each item attends to all other items in the sequence
    - A learnable [CLS] token aggregates the sequence representation
    - More expressive than simple attention pooling but more expensive
    
    Architecture:
        [CLS] + [item1, item2, ..., itemN] 
        -> Transformer Encoder (self-attention layers)
        -> Output [CLS] token as user embedding
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Positional encoding (learnable)
        # +1 for CLS token, max_seq_len for items
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN + 1, embedding_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_embs: (batch, seq_len, embedding_dim)
            history_mask: (batch, seq_len) - True for valid positions
            
        Returns:
            user_emb: (batch, output_dim)
        """
        batch_size, seq_len, _ = history_embs.shape
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embedding_dim)
        
        # Concatenate: [CLS, item1, item2, ..., itemN]
        x = torch.cat([cls_tokens, history_embs], dim=1)  # (batch, seq_len+1, embedding_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len+1, :]
        
        # Create attention mask for transformer
        # Transformer uses True for positions to MASK (opposite of our convention)
        # Add False for CLS token (never masked)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=history_mask.device)
        attn_mask = torch.cat([cls_mask, ~history_mask], dim=1)  # (batch, seq_len+1)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Extract CLS token output
        cls_output = x[:, 0, :]  # (batch, embedding_dim)
        
        # Project to output dimension
        user_emb = self.projection(cls_output)
        user_emb = self.layer_norm(user_emb)
        
        return user_emb


class ItemTower(nn.Module):
    """
    Item Tower: Projects item embeddings into shared space.
    
    Simple projection layer to transform item embeddings
    into the same space as user embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_embs: (batch, embedding_dim) or (num_items, embedding_dim)
            
        Returns:
            projected: (batch, output_dim)
        """
        projected = self.projection(item_embs)
        projected = self.layer_norm(projected)
        return projected


class TwoTowerModel(nn.Module):
    """
    Complete Two-Tower Model for personalized retrieval.
    
    Combines User Tower and Item Tower with:
    - Dot product similarity
    - Temperature-scaled softmax for training
    
    Supports two user tower architectures:
    - 'attention': Simple attention pooling (faster, fewer params)
    - 'transformer': Full transformer encoder with [CLS] token (more expressive)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float = 0.1,
        dropout: float = 0.1,
        user_tower_type: str = 'attention',  # 'attention' or 'transformer'
        num_transformer_heads: int = 2,
        num_transformer_layers: int = 2
    ):
        super().__init__()
        
        self.user_tower_type = user_tower_type
        
        if user_tower_type == 'attention':
            self.user_tower = UserTower(embedding_dim, hidden_dim, output_dim, dropout)
        elif user_tower_type == 'transformer':
            self.user_tower = TransformerUserTower(
                embedding_dim, hidden_dim, output_dim,
                num_heads=num_transformer_heads,
                num_layers=num_transformer_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown user_tower_type: {user_tower_type}")
        
        self.item_tower = ItemTower(embedding_dim, hidden_dim, output_dim, dropout)
        self.temperature = temperature
    
    def encode_user(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """Encode user from history."""
        return self.user_tower(history_embs, history_mask)
    
    def encode_items(self, item_embs: torch.Tensor) -> torch.Tensor:
        """Encode items."""
        return self.item_tower(item_embs)
    
    def forward(
        self,
        history_embs: torch.Tensor,
        history_mask: torch.Tensor,
        positive_embs: torch.Tensor,
        negative_embs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            history_embs: (batch, seq_len, embedding_dim)
            history_mask: (batch, seq_len)
            positive_embs: (batch, embedding_dim)
            negative_embs: (batch, num_neg, embedding_dim) - optional
            
        Returns:
            Dict with user_emb, pos_item_emb, logits
        """
        batch_size = history_embs.shape[0]
        
        # Encode user
        user_emb = self.encode_user(history_embs, history_mask)  # (batch, output_dim)
        
        # Encode positive items
        pos_item_emb = self.encode_items(positive_embs)  # (batch, output_dim)
        
        # Compute similarities with in-batch negatives
        # All positive items serve as negatives for other users
        all_item_embs = pos_item_emb  # (batch, output_dim)
        
        # Logits: (batch, batch) - each row is similarity of user to all items
        logits = torch.matmul(user_emb, all_item_embs.T) / self.temperature
        
        # If explicit negatives provided, add them
        if negative_embs is not None:
            neg_item_emb = self.encode_items(negative_embs.view(-1, negative_embs.shape[-1]))
            neg_item_emb = neg_item_emb.view(batch_size, -1, neg_item_emb.shape[-1])
            
            # (batch, num_neg)
            neg_logits = torch.bmm(user_emb.unsqueeze(1), neg_item_emb.transpose(1, 2)).squeeze(1)
            neg_logits = neg_logits / self.temperature
            
            # Concatenate: (batch, batch + num_neg)
            logits = torch.cat([logits, neg_logits], dim=-1)
        
        return {
            'user_emb': user_emb,
            'pos_item_emb': pos_item_emb,
            'logits': logits
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        # Move to device
        history_embs = batch['history_embs'].to(device)
        history_mask = batch['history_mask'].to(device)
        positive_embs = batch['positive_emb'].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(history_embs, history_mask, positive_embs)
        
        # In-batch contrastive loss
        # Labels: each sample's positive is at its own index
        batch_size = history_embs.shape[0]
        labels = torch.arange(batch_size, device=device)
        
        loss = F.cross_entropy(outputs['logits'], labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': total_loss / num_batches
    }


def train_model(
    model: TwoTowerModel,
    train_dataloader: DataLoader,
    val_sessions: List[Dict],
    item_lookup: ItemEmbeddingLookup,
    num_epochs: int,
    learning_rate: float,
    warmup_epochs: int,
    device: torch.device
) -> TwoTowerModel:
    """Full training loop."""
    print("\n" + "=" * 60)
    print("TRAINING TWO-TOWER MODEL")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {train_dataloader.batch_size}")
    print(f"  Training samples: {len(train_dataloader.dataset):,}")
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Warmup + cosine decay scheduler
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_hr = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch
        )
        
        # Validate every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            val_hr = quick_evaluate(model, val_sessions, item_lookup, device, k=10)
            print(f"  Epoch {epoch+1}: Loss={train_metrics['loss']:.4f}, Val HR@10={val_hr:.4f}")
            
            if val_hr > best_hr:
                best_hr = val_hr
                best_model_state = model.state_dict().copy()
        else:
            print(f"  Epoch {epoch+1}: Loss={train_metrics['loss']:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (HR@10={best_hr:.4f})")
    
    return model


def quick_evaluate(
    model: TwoTowerModel,
    sessions: List[Dict],
    item_lookup: ItemEmbeddingLookup,
    device: torch.device,
    k: int = 10,
    max_sessions: int = 1000
) -> float:
    """Quick evaluation during training (Hit Rate@K)."""
    model.eval()
    
    # Build item index with projected embeddings
    with torch.no_grad():
        all_item_embs = torch.tensor(item_lookup.embeddings, dtype=torch.float32).to(device)
        projected_items = model.encode_items(all_item_embs).cpu().numpy()
    
    # Build FAISS index
    dim = projected_items.shape[1]
    index = faiss.IndexFlatIP(dim)
    projected_items = np.ascontiguousarray(projected_items, dtype=np.float32)
    faiss.normalize_L2(projected_items)
    index.add(projected_items)
    
    # Evaluate
    hits = 0
    evaluated = 0
    
    for session in sessions[:max_sessions]:
        if session['ground_truth_item'] not in item_lookup.sku_to_idx:
            continue
        
        # Get user embedding
        history_items = session['history_items'][-MAX_SEQ_LEN:]
        history_embs, valid_mask = item_lookup.get_embeddings(history_items)
        
        # Pad
        if len(history_items) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(history_items)
            history_embs = np.vstack([
                np.zeros((pad_len, item_lookup.embeddings.shape[1]), dtype=np.float32),
                history_embs
            ])
            valid_mask = [False] * pad_len + valid_mask
        
        with torch.no_grad():
            h_emb = torch.tensor(history_embs, dtype=torch.float32).unsqueeze(0).to(device)
            h_mask = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(device)
            user_emb = model.encode_user(h_emb, h_mask).cpu().numpy()
        
        # Ensure contiguous for FAISS
        user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        
        # Search
        faiss.normalize_L2(user_emb)
        _, indices = index.search(user_emb, k + len(session['history_items']))
        
        # Get results excluding history
        history_set = set(session['history_items'])
        results = []
        for idx in indices[0]:
            sku = item_lookup.idx_to_sku[idx]
            if sku not in history_set:
                results.append(sku)
                if len(results) >= k:
                    break
        
        # Check hit
        if session['ground_truth_item'] in results:
            hits += 1
        evaluated += 1
    
    return hits / evaluated if evaluated > 0 else 0.0


# =============================================================================
# INFERENCE & RETRIEVAL
# =============================================================================

class SupervisedRetriever:
    """Retriever using the trained two-tower model."""
    
    def __init__(
        self,
        model: TwoTowerModel,
        item_lookup: ItemEmbeddingLookup,
        device: torch.device
    ):
        self.model = model
        self.item_lookup = item_lookup
        self.device = device
        
        # Build index with projected item embeddings
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index with model-projected item embeddings."""
        print("\nBuilding retrieval index with projected embeddings...")
        
        self.model.eval()
        with torch.no_grad():
            all_item_embs = torch.tensor(
                self.item_lookup.embeddings, dtype=torch.float32
            ).to(self.device)
            
            # Project in batches for memory efficiency
            batch_size = 10000
            projected_list = []
            for i in range(0, len(all_item_embs), batch_size):
                batch = all_item_embs[i:i+batch_size]
                projected = self.model.encode_items(batch).cpu().numpy()
                projected_list.append(projected)
            
            self.projected_items = np.vstack(projected_list)
        
        # Build FAISS index
        dim = self.projected_items.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        # Ensure contiguous and normalize for cosine similarity
        self.projected_items = np.ascontiguousarray(self.projected_items, dtype=np.float32)
        faiss.normalize_L2(self.projected_items)
        self.index.add(self.projected_items)
        
        print(f"  Index built with {self.index.ntotal:,} items")
    
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Tuple[str, float]]:
        """Retrieve top-K items for a user given their history."""
        # Get history embeddings
        history_items_trimmed = history_items[-MAX_SEQ_LEN:]
        history_embs, valid_mask = self.item_lookup.get_embeddings(history_items_trimmed)
        
        # Pad
        embedding_dim = self.item_lookup.embeddings.shape[1]
        if len(history_items_trimmed) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(history_items_trimmed)
            history_embs = np.vstack([
                np.zeros((pad_len, embedding_dim), dtype=np.float32),
                history_embs
            ])
            valid_mask = [False] * pad_len + valid_mask
        
        # Encode user
        self.model.eval()
        with torch.no_grad():
            h_emb = torch.tensor(history_embs, dtype=torch.float32).unsqueeze(0).to(self.device)
            h_mask = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            user_emb = self.model.encode_user(h_emb, h_mask).cpu().numpy()
        
        # Ensure contiguous and correct dtype for FAISS
        user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        
        # Normalize
        faiss.normalize_L2(user_emb)
        
        # Search
        k_search = top_k + len(history_items) if exclude_history else top_k
        distances, indices = self.index.search(user_emb, k_search)
        
        # Filter and format results
        history_set = set(history_items) if exclude_history else set()
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            sku = self.item_lookup.idx_to_sku[idx]
            if sku not in history_set:
                results.append((sku, float(dist)))
                if len(results) >= top_k:
                    break
        
        return results


# =============================================================================
# BASELINES (Same as personalized_retrieval.py)
# =============================================================================

class RandomBaseline:
    """Random sampling from catalog."""
    
    def __init__(self, all_skus: List[str]):
        self.all_skus = all_skus
        self.all_skus_set = set(all_skus)
        
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Tuple[str, float]]:
        if exclude_history:
            candidates = list(self.all_skus_set - set(history_items))
        else:
            candidates = self.all_skus
        
        k = min(top_k, len(candidates))
        sampled = np.random.choice(candidates, size=k, replace=False)
        return [(sku, 0.0) for sku in sampled]


class PopularityBaseline:
    """Recommend most popular items."""
    
    def __init__(self, popularity_scores: Dict[str, int], all_skus: List[str]):
        self.popularity_scores = popularity_scores
        self.sorted_by_popularity = sorted(
            all_skus,
            key=lambda x: popularity_scores.get(x, 0),
            reverse=True
        )
        
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Tuple[str, float]]:
        history_set = set(history_items) if exclude_history else set()
        
        results = []
        for sku in self.sorted_by_popularity:
            if sku not in history_set:
                score = self.popularity_scores.get(sku, 0)
                results.append((sku, float(score)))
                if len(results) >= top_k:
                    break
        
        return results


class UnsupervisedRetriever:
    """
    Unsupervised retriever (mean pooling / last item).
    Baseline from personalized_retrieval.py for comparison.
    """
    
    def __init__(self, item_lookup: ItemEmbeddingLookup, aggregation: str = 'mean'):
        self.item_lookup = item_lookup
        self.aggregation = aggregation
        
        # Build FAISS index
        self.embeddings = np.ascontiguousarray(item_lookup.embeddings, dtype=np.float32)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        normalized = np.ascontiguousarray(self.embeddings.copy(), dtype=np.float32)
        faiss.normalize_L2(normalized)
        self.index.add(normalized)
    
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Tuple[str, float]]:
        # Get history embeddings
        history_embs, valid_mask = self.item_lookup.get_embeddings(history_items)
        valid_embs = history_embs[valid_mask]
        
        if len(valid_embs) == 0:
            return []
        
        # Aggregate
        if self.aggregation == 'mean':
            user_emb = valid_embs.mean(axis=0, keepdims=True)
        else:  # last
            user_emb = valid_embs[-1:, :]
        
        # Ensure contiguous and normalize
        user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        faiss.normalize_L2(user_emb)
        
        # Search
        k_search = top_k + len(history_items) if exclude_history else top_k
        distances, indices = self.index.search(user_emb, k_search)
        
        # Filter
        history_set = set(history_items) if exclude_history else set()
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            sku = self.item_lookup.idx_to_sku[idx]
            if sku not in history_set:
                results.append((sku, float(dist)))
                if len(results) >= top_k:
                    break
        
        return results


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_retriever(
    retriever,
    user_sessions: List[Dict],
    top_k_values: List[int] = [5, 10, 20, 50],
    max_sessions: int = None,
    model_name: str = "Model"
) -> pd.DataFrame:
    """Evaluate a retriever on user sessions."""
    if max_sessions:
        user_sessions = user_sessions[:max_sessions]
    
    print(f"\n{'-' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'-' * 60}")
    
    max_k = max(top_k_values)
    
    # Generate predictions
    all_predictions = {}
    for session in tqdm(user_sessions, desc="  Generating predictions"):
        retrieved = retriever.retrieve_for_user(
            session['history_items'],
            top_k=max_k,
            exclude_history=True
        )
        all_predictions[session['session_id']] = [sku for sku, _ in retrieved]
    
    # Compute metrics
    results = []
    for k in top_k_values:
        hits = 0
        mrr_sum = 0.0
        evaluated = 0
        
        for session in user_sessions:
            preds = all_predictions.get(session['session_id'], [])[:k]
            gt = session['ground_truth_item']
            
            if gt in preds:
                hits += 1
                rank = preds.index(gt) + 1
                mrr_sum += 1.0 / rank
            
            evaluated += 1
        
        hit_rate = hits / evaluated if evaluated > 0 else 0.0
        mrr = mrr_sum / evaluated if evaluated > 0 else 0.0
        
        results.append({
            'K': k,
            'Hit Rate@K': hit_rate,
            'MRR@K': mrr
        })
    
    # Print
    df = pd.DataFrame(results)
    print(f"\n{' K':>3}  {'Hit Rate@K':>12}  {'MRR@K':>8}")
    for _, row in df.iterrows():
        print(f"{row['K']:>3}  {row['Hit Rate@K']*100:>11.2f}%  {row['MRR@K']:>8.4f}")
    
    return df


def run_full_comparison(
    supervised_retriever: SupervisedRetriever,
    random_baseline: RandomBaseline,
    popularity_baseline: PopularityBaseline,
    unsupervised_mean: UnsupervisedRetriever,
    unsupervised_last: UnsupervisedRetriever,
    user_sessions: List[Dict],
    top_k_values: List[int] = [5, 10, 20, 50],
    max_sessions: int = 5000
) -> pd.DataFrame:
    """Compare all models."""
    print("\n" + "=" * 70)
    print("FULL MODEL COMPARISON")
    print("=" * 70)
    
    all_results = []
    
    # 1. Random
    df = evaluate_retriever(
        random_baseline, user_sessions, top_k_values, max_sessions, "Random"
    )
    df['Model'] = 'Random'
    all_results.append(df)
    
    # 2. Popularity
    df = evaluate_retriever(
        popularity_baseline, user_sessions, top_k_values, max_sessions, "Popularity"
    )
    df['Model'] = 'Popularity'
    all_results.append(df)
    
    # 3. Unsupervised (mean)
    df = evaluate_retriever(
        unsupervised_mean, user_sessions, top_k_values, max_sessions, "Unsupervised (mean)"
    )
    df['Model'] = 'Unsupervised (mean)'
    all_results.append(df)
    
    # 4. Unsupervised (last)
    df = evaluate_retriever(
        unsupervised_last, user_sessions, top_k_values, max_sessions, "Unsupervised (last)"
    )
    df['Model'] = 'Unsupervised (last)'
    all_results.append(df)
    
    # 5. Supervised Two-Tower
    df = evaluate_retriever(
        supervised_retriever, user_sessions, top_k_values, max_sessions, "Supervised Two-Tower"
    )
    df['Model'] = 'Supervised Two-Tower'
    all_results.append(df)
    
    # Combine
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY (K=10)")
    print("=" * 70)
    
    summary = combined_df[combined_df['K'] == 10][['Model', 'Hit Rate@K', 'MRR@K']].copy()
    
    # Lift calculation
    random_hr = summary[summary['Model'] == 'Random']['Hit Rate@K'].values[0]
    summary['Lift vs Random'] = summary['Hit Rate@K'].apply(
        lambda x: f"+{(x/random_hr - 1)*100:.0f}%" if random_hr > 0 and x > random_hr else "0%"
    )
    
    # Lift vs unsupervised best
    unsup_best = summary[summary['Model'] == 'Unsupervised (last)']['Hit Rate@K'].values[0]
    summary['Lift vs Unsup'] = summary['Hit Rate@K'].apply(
        lambda x: f"+{(x/unsup_best - 1)*100:.1f}%" if unsup_best > 0 else "N/A"
    )
    
    # Format
    summary['Hit Rate@K'] = summary['Hit Rate@K'].apply(lambda x: f"{x*100:.2f}%")
    summary['MRR@K'] = summary['MRR@K'].apply(lambda x: f"{x:.4f}")
    
    print(summary.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print("- Random: Lower bound sanity check")
    print("- Popularity: Non-personalized baseline")
    print("- Unsupervised (mean/last): Simple aggregation of history embeddings")
    print("- Supervised Two-Tower: Learned sequence encoding with contrastive loss")
    print("- The 'Lift vs Unsup' column shows improvement over best unsupervised method")
    
    return combined_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    print("=" * 70)
    print("Chapter 5: Supervised Two-Tower Model")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"\nUser Tower Type: {USER_TOWER_TYPE.upper()}")
    print("\nModel Architecture:")
    if USER_TOWER_TYPE == 'attention':
        print("  User Tower: Attention-based sequence aggregation + MLP")
    else:
        print(f"  User Tower: Transformer Encoder ({NUM_TRANSFORMER_LAYERS} layers, {NUM_TRANSFORMER_HEADS} heads)")
        print("              with learnable [CLS] token for aggregation")
    print("  Item Tower: MLP projection")
    print("  Training: In-batch contrastive loss")
    
    # 1. Load data
    catalog_df = load_product_catalog(DATA_PATH)
    browsing_lf = load_browsing_sessions_lazy(DATA_PATH, BROWSING_FILE)
    
    # 2. Build item lookup
    item_lookup = ItemEmbeddingLookup.from_catalog(catalog_df)
    
    # 3. Build user sessions
    user_sessions = build_user_sessions(
        browsing_lf,
        min_session_length=MIN_SESSION_LENGTH,
        ground_truth_actions=GROUND_TRUTH_ACTIONS
    )
    
    # 4. TIME-BASED split into train/val/test (chronological order)
    # Sort sessions by timestamp - older sessions for training, newer for testing
    # This is the correct approach for temporal data to avoid data leakage
    print("\nSorting sessions chronologically for time-based split...")
    user_sessions.sort(key=lambda x: x['max_timestamp'])
    
    n_train = int(len(user_sessions) * 0.7)
    n_val = int(len(user_sessions) * 0.15)
    
    train_sessions = user_sessions[:n_train]      # Oldest 70%
    val_sessions = user_sessions[n_train:n_train+n_val]  # Next 15%
    test_sessions = user_sessions[n_train+n_val:]  # Most recent 15%
    
    # Show time ranges for verification
    def ts_to_date(ts):
        from datetime import datetime
        return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
    
    print(f"\nTime-based data split:")
    print(f"  Train: {len(train_sessions):,} sessions")
    print(f"    Time range: {ts_to_date(train_sessions[0]['max_timestamp'])} to {ts_to_date(train_sessions[-1]['max_timestamp'])}")
    print(f"  Val: {len(val_sessions):,} sessions")
    print(f"    Time range: {ts_to_date(val_sessions[0]['max_timestamp'])} to {ts_to_date(val_sessions[-1]['max_timestamp'])}")
    print(f"  Test: {len(test_sessions):,} sessions")
    print(f"    Time range: {ts_to_date(test_sessions[0]['max_timestamp'])} to {ts_to_date(test_sessions[-1]['max_timestamp'])}")
    
    # 5. Create dataset and dataloader
    train_dataset = TwoTowerDataset(train_sessions, item_lookup, MAX_SEQ_LEN)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        drop_last=True
    )
    
    # 6. Initialize model
    print(f"\nInitializing model with user tower type: {USER_TOWER_TYPE}")
    model = TwoTowerModel(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=USER_EMB_DIM,
        temperature=0.1,
        dropout=0.1,
        user_tower_type=USER_TOWER_TYPE,
        num_transformer_heads=NUM_TRANSFORMER_HEADS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. Train
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_sessions=val_sessions,
        item_lookup=item_lookup,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_epochs=WARMUP_EPOCHS,
        device=DEVICE
    )
    
    # 8. Build baselines
    print("\n" + "=" * 60)
    print("BUILDING BASELINES")
    print("=" * 60)
    
    all_skus = list(item_lookup.sku_to_idx.keys())
    
    random_baseline = RandomBaseline(all_skus)
    
    browsing_lf_pop = load_browsing_sessions_lazy(DATA_PATH, BROWSING_FILE)
    popularity_scores = compute_popularity_scores(browsing_lf_pop, GROUND_TRUTH_ACTIONS)
    popularity_baseline = PopularityBaseline(popularity_scores, all_skus)
    
    unsupervised_mean = UnsupervisedRetriever(item_lookup, aggregation='mean')
    unsupervised_last = UnsupervisedRetriever(item_lookup, aggregation='last')
    
    # 9. Build supervised retriever
    supervised_retriever = SupervisedRetriever(model, item_lookup, DEVICE)
    
    # 10. Run comparison on TEST set
    comparison_df = run_full_comparison(
        supervised_retriever=supervised_retriever,
        random_baseline=random_baseline,
        popularity_baseline=popularity_baseline,
        unsupervised_mean=unsupervised_mean,
        unsupervised_last=unsupervised_last,
        user_sessions=test_sessions,
        top_k_values=TOP_K_RETRIEVAL,
        max_sessions=5000
    )
    
    # 11. Save results and model
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / 'supervised_comparison.csv', index=False)
    
    torch.save(model.state_dict(), output_dir / 'two_tower_model.pt')
    
    print(f"\nResults saved to: {output_dir / 'supervised_comparison.csv'}")
    print(f"Model saved to: {output_dir / 'two_tower_model.pt'}")
    
    # 12. Show example
    print("\n" + "=" * 60)
    print("EXAMPLE: Comparing Recommendations")
    print("=" * 60)
    
    example_session = test_sessions[0]
    print(f"\nSession: {example_session['session_id'][:40]}...")
    print(f"History length: {len(example_session['history_items'])} items")
    print(f"Ground Truth: {example_session['ground_truth_item'][:40]}...")
    
    print("\nTop 5 Recommendations:")
    print("-" * 50)
    
    models = [
        ("Random", random_baseline),
        ("Popularity", popularity_baseline),
        ("Unsup (last)", unsupervised_last),
        ("Supervised", supervised_retriever)
    ]
    
    for name, retriever in models:
        recs = retriever.retrieve_for_user(example_session['history_items'], top_k=5)
        hit = any(sku == example_session['ground_truth_item'] for sku, _ in recs)
        hit_marker = " [HIT!]" if hit else ""
        print(f"\n{name}:{hit_marker}")
        for i, (sku, score) in enumerate(recs[:3], 1):
            marker = " <--" if sku == example_session['ground_truth_item'] else ""
            print(f"  {i}. {sku[:45]}...{marker}")
    
    return comparison_df, model, test_sessions


if __name__ == '__main__':
    comparison_df, model, test_sessions = main()

