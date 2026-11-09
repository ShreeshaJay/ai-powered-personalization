"""
Chapter 7: DLRM - Deep Learning Recommendation Model

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                          DLRM                                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Sparse Features                    Dense Features              │
    │  (uid, item_id,                    (numerical,                 │
    │   hour, day_of_week)                historical, is_organic)     │
    │       │                                  │                      │
    │       ▼                                  ▼                      │
    │  ┌─────────────┐                  ┌──────────────┐              │
    │  │  Embedding  │                  │  Bottom MLP  │              │
    │  │   Tables    │                  │              │              │
    │  │             │                  │  FC → ReLU   │              │
    │  │  e₁ e₂ e₃..│                  │  FC → ReLU   │              │
    │  └──────┬──────┘                  └──────┬───────┘              │
    │         │                                │                      │
    │         │         ┌───────────────┐      │                      │
    │         │         │               │      │                      │
    │         └─────────►   Feature     ◄──────┘                      │
    │                   │ Interactions  │                             │
    │                   │               │                             │
    │                   │  (Dot Product │                             │
    │                   │   Pairwise)   │                             │
    │                   └───────┬───────┘                             │
    │                           │                                     │
    │                           ▼                                     │
    │                   ┌───────────────┐                             │
    │                   │   Top MLP     │                             │
    │                   │               │                             │
    │                   │  FC → ReLU    │                             │
    │                   │  FC → ReLU    │                             │
    │                   │  FC → Sigmoid │                             │
    │                   └───────────────┘                             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Key Design Principles:
1. Dense features → Bottom MLP → Embedding-sized vector
2. All embeddings (sparse + processed dense) → Pairwise dot products
3. Concatenate: original embeddings + interaction results
4. Top MLP for final prediction

Reference:
    Naumov et al. "Deep Learning Recommendation Model for 
    Personalization and Recommendation Systems" (Facebook, 2019)

Usage:
    from models.dlrm import DLRM, DLRMConfig
    
    config = DLRMConfig(
        sparse_features={'uid': 10000, 'item_id': 900000},
        dense_features=['track_length_seconds', 'user_avg_completion', ...],
        embed_dim=16,
        bottom_mlp_dims=[64, 32, 16],
        top_mlp_dims=[256, 128, 64],
    )
    model = DLRM(config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DLRMConfig:
    """Configuration for DLRM model.
    
    Attributes:
        sparse_features: Dict mapping feature name to vocabulary size
        dense_features: List of dense feature names
        embed_dim: Embedding dimension for sparse features (default: 16)
        bottom_mlp_dims: Hidden dimensions for bottom MLP (default: [64, 32, 16])
        top_mlp_dims: Hidden dimensions for top MLP (default: [256, 128, 64])
        dropout: Dropout rate (default: 0.1)
        interaction_type: 'dot' or 'cat' for feature interaction (default: 'dot')
    """
    # Feature configuration
    sparse_features: Dict[str, int] = field(default_factory=dict)
    dense_features: List[str] = field(default_factory=list)
    
    # Model hyperparameters
    embed_dim: int = 16
    bottom_mlp_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    top_mlp_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    interaction_type: str = 'dot'  # 'dot' or 'cat'
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.sparse_features and not self.dense_features:
            raise ValueError("Must have at least one sparse or dense feature")
        if self.interaction_type not in ['dot', 'cat']:
            raise ValueError("interaction_type must be 'dot' or 'cat'")
        # Bottom MLP output must match embed_dim for dot product interactions
        if self.bottom_mlp_dims and self.bottom_mlp_dims[-1] != self.embed_dim:
            # Auto-adjust to match embed_dim
            self.bottom_mlp_dims = list(self.bottom_mlp_dims[:-1]) + [self.embed_dim]


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_bn: bool = True,
        activation: str = 'relu',
        output_activation: bool = False,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
            if output_activation:
                layers.append(nn.ReLU())
            prev_dim = output_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DotInteraction(nn.Module):
    """Pairwise dot product interaction between embeddings.
    
    Given n embeddings of dimension d, computes n*(n-1)/2 dot products
    (upper triangular, excluding diagonal).
    """
    
    def __init__(self, include_self_interaction: bool = False):
        super().__init__()
        self.include_self = include_self_interaction
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise dot products.
        
        Args:
            embeddings: (batch_size, num_embeddings, embed_dim)
            
        Returns:
            interactions: (batch_size, num_interactions)
                where num_interactions = n*(n+1)/2 if include_self else n*(n-1)/2
        """
        batch_size, num_embeds, embed_dim = embeddings.shape
        
        # Compute all pairwise dot products: (batch, n, n)
        # Using batch matrix multiplication: (batch, n, d) @ (batch, d, n)
        dot_products = torch.bmm(embeddings, embeddings.transpose(1, 2))
        
        # Extract upper triangular (including or excluding diagonal)
        if self.include_self:
            # Include diagonal: triu with offset 0
            indices = torch.triu_indices(num_embeds, num_embeds, offset=0)
        else:
            # Exclude diagonal: triu with offset 1
            indices = torch.triu_indices(num_embeds, num_embeds, offset=1)
        
        # Extract interactions
        interactions = dot_products[:, indices[0], indices[1]]
        
        return interactions


class DLRM(nn.Module):
    """Deep Learning Recommendation Model (Facebook).
    
    Architecture:
    1. Bottom MLP: Transform dense features to embedding space
    2. Embedding tables: Lookup for sparse features
    3. Feature interaction: Pairwise dot products
    4. Top MLP: Final prediction from concatenated features
    """
    
    def __init__(self, config: DLRMConfig):
        super().__init__()
        self.config = config
        
        # ===== Embedding layers for sparse features =====
        self.embeddings = nn.ModuleDict()
        
        for feat_name, vocab_size in config.sparse_features.items():
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config.embed_dim,
                padding_idx=0,
            )
        
        # ===== Bottom MLP for dense features =====
        num_dense = len(config.dense_features)
        if num_dense > 0:
            self.bottom_mlp = MLP(
                input_dim=num_dense,
                hidden_dims=config.bottom_mlp_dims[:-1] if len(config.bottom_mlp_dims) > 1 else [],
                output_dim=config.embed_dim,  # Output matches embedding dimension
                dropout=config.dropout,
                use_bn=True,
                output_activation=True,  # ReLU on output
            )
        else:
            self.bottom_mlp = None
        
        # ===== Feature Interaction =====
        # Number of embeddings = sparse features + (1 if dense features)
        num_sparse = len(config.sparse_features)
        num_embeddings = num_sparse + (1 if num_dense > 0 else 0)
        
        if config.interaction_type == 'dot':
            self.interaction = DotInteraction(include_self_interaction=False)
            # Number of interactions: n*(n-1)/2
            num_interactions = (num_embeddings * (num_embeddings - 1)) // 2
            # Top MLP input: all embeddings flattened + interactions
            top_input_dim = num_embeddings * config.embed_dim + num_interactions
        else:
            # Concatenation: just flatten all embeddings
            self.interaction = None
            top_input_dim = num_embeddings * config.embed_dim
        
        # ===== Top MLP =====
        self.top_mlp = MLP(
            input_dim=top_input_dim,
            hidden_dims=config.top_mlp_dims,
            output_dim=1,
            dropout=config.dropout,
            use_bn=True,
            output_activation=False,  # No activation before loss
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                if 'bias' in name:
                    nn.init.zeros_(param)
                continue
            if 'embedding' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and 'bn' not in name:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self,
        sparse_inputs: Dict[str, torch.Tensor],
        dense_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sparse_inputs: Dict mapping feature name to indices (batch_size,)
            dense_inputs: Dense features tensor (batch_size, num_dense) or None
            
        Returns:
            logits: (batch_size, 1) - raw scores before sigmoid
        """
        batch_size = next(iter(sparse_inputs.values())).size(0)
        
        # ===== Process sparse features =====
        sparse_embeds = []
        for feat_name, indices in sparse_inputs.items():
            embed = self.embeddings[feat_name](indices)  # (batch, embed_dim)
            sparse_embeds.append(embed)
        
        # ===== Process dense features =====
        if dense_inputs is not None and self.bottom_mlp is not None:
            dense_embed = self.bottom_mlp(dense_inputs)  # (batch, embed_dim)
            all_embeds = sparse_embeds + [dense_embed]
        else:
            all_embeds = sparse_embeds
        
        # Stack embeddings: (batch, num_embeddings, embed_dim)
        stacked_embeds = torch.stack(all_embeds, dim=1)
        
        # ===== Feature Interaction =====
        # Flatten embeddings
        flat_embeds = stacked_embeds.view(batch_size, -1)  # (batch, num_embeds * embed_dim)
        
        if self.interaction is not None:
            # Dot product interactions
            interactions = self.interaction(stacked_embeds)  # (batch, num_interactions)
            # Concatenate flat embeddings with interactions
            combined = torch.cat([flat_embeds, interactions], dim=1)
        else:
            combined = flat_embeds
        
        # ===== Top MLP =====
        logits = self.top_mlp(combined)
        
        return logits
    
    def predict_proba(
        self,
        sparse_inputs: Dict[str, torch.Tensor],
        dense_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(sparse_inputs, dense_inputs)
        return torch.sigmoid(logits)


class DLRMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for DLRM training."""
    
    def __init__(
        self,
        sparse_data: Dict[str, np.ndarray],
        dense_data: Optional[np.ndarray],
        labels: np.ndarray,
    ):
        """
        Args:
            sparse_data: Dict mapping feature name to encoded indices
            dense_data: 2D array of dense features (n_samples, n_dense)
            labels: Binary labels
        """
        self.sparse_data = {k: torch.LongTensor(v) for k, v in sparse_data.items()}
        self.dense_data = torch.FloatTensor(dense_data) if dense_data is not None else None
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        
        self.n_samples = len(labels)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        sparse = {k: v[idx] for k, v in self.sparse_data.items()}
        dense = self.dense_data[idx] if self.dense_data is not None else None
        label = self.labels[idx]
        return sparse, dense, label


def collate_fn(batch):
    """Custom collate function for DLRM batches."""
    sparse_list, dense_list, label_list = zip(*batch)
    
    # Stack sparse features
    sparse_batch = {}
    for key in sparse_list[0].keys():
        sparse_batch[key] = torch.stack([s[key] for s in sparse_list])
    
    # Stack dense features
    if dense_list[0] is not None:
        dense_batch = torch.stack(dense_list)
    else:
        dense_batch = None
    
    # Stack labels
    labels = torch.stack(label_list)
    
    return sparse_batch, dense_batch, labels


# ===== Default configuration for Yambda dataset =====
def get_yambda_dlrm_config(
    uid_vocab_size: int = 10000,
    item_vocab_size: int = 900000,
) -> DLRMConfig:
    """Get default DLRM config for Yambda dataset."""
    return DLRMConfig(
        sparse_features={
            'uid': uid_vocab_size,
            'item_id': item_vocab_size,
            'hour_of_day': 24,
            'day_of_week': 7,
        },
        dense_features=[
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
        ],
        embed_dim=16,
        bottom_mlp_dims=[64, 32, 16],  # Last dim must match embed_dim
        top_mlp_dims=[256, 128, 64],
        dropout=0.1,
        interaction_type='dot',
        learning_rate=1e-3,
        weight_decay=1e-5,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing DLRM model...")
    
    config = DLRMConfig(
        sparse_features={'uid': 1000, 'item_id': 5000},
        dense_features=['feat1', 'feat2', 'feat3'],
        embed_dim=16,
        bottom_mlp_dims=[32, 16],
        top_mlp_dims=[64, 32],
        interaction_type='dot',
    )
    
    model = DLRM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 32
    sparse_inputs = {
        'uid': torch.randint(0, 1000, (batch_size,)),
        'item_id': torch.randint(0, 5000, (batch_size,)),
    }
    dense_inputs = torch.randn(batch_size, 3)
    
    logits = model(sparse_inputs, dense_inputs)
    probs = model.predict_proba(sparse_inputs, dense_inputs)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    
    print("✓ DLRM test passed!")

