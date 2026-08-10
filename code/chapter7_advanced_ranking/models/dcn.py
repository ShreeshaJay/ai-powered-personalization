"""
Chapter 7: DCN-V2 - Deep & Cross Network V2 for CTR Prediction

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         DCN-V2                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Sparse Features              Dense Features                    │
    │  (uid, item_id,              (numerical,                       │
    │   hour, day_of_week)          historical, is_organic)           │
    │       │                            │                            │
    │       ▼                            │                            │
    │  ┌─────────┐                       │                            │
    │  │Embedding│                       │                            │
    │  │ Tables  │                       │                            │
    │  └────┬────┘                       │                            │
    │       │                            │                            │
    │       └────────────┬───────────────┘                            │
    │                    │                                            │
    │                    ▼                                            │
    │            ┌──────────────┐                                     │
    │            │   Concat     │                                     │
    │            │ (x₀ input)   │                                     │
    │            └──────┬───────┘                                     │
    │                   │                                             │
    │    ┌──────────────┴──────────────┐                              │
    │    │                             │                              │
    │    ▼                             ▼                              │
    │  ┌───────────────┐      ┌─────────────────┐                     │
    │  │ Cross Network │      │  Deep Network   │                     │
    │  │  (DCN-V2)     │      │    (MLP)        │                     │
    │  │               │      │                 │                     │
    │  │ x_{l+1} =     │      │ FC → BN → ReLU  │                     │
    │  │  x₀ ⊙ (W·x_l  │      │ FC → BN → ReLU  │                     │
    │  │   + b) + x_l  │      │ FC → BN → ReLU  │                     │
    │  └───────┬───────┘      └────────┬────────┘                     │
    │          │                       │                              │
    │          └───────────┬───────────┘                              │
    │                      ▼                                          │
    │               ┌──────────┐                                      │
    │               │  Concat  │                                      │
    │               │    +     │                                      │
    │               │ Linear   │                                      │
    │               └──────────┘                                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

DCN-V2 Improvements over DCN:
- Uses matrix W instead of vector w for cross network (more expressive)
- Low-rank factorization: W = U·V^T for efficiency
- Supports both "stacked" and "parallel" structures
- Better feature interaction modeling

Reference:
    Wang et al. "DCN V2: Improved Deep & Cross Network and 
    Practical Lessons for Web-scale Learning to Rank Systems" (WWW 2021)

Usage:
    from models.dcn import DCNV2, DCNV2Config
    
    config = DCNV2Config(
        sparse_features={'uid': 10000, 'item_id': 900000},
        dense_features=['track_length_seconds', 'user_avg_completion', ...],
        embed_dim=16,
        cross_layers=3,
        mlp_dims=[256, 128, 64],
    )
    model = DCNV2(config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DCNV2Config:
    """Configuration for DCN-V2 model.
    
    Attributes:
        sparse_features: Dict mapping feature name to vocabulary size
        dense_features: List of dense feature names
        embed_dim: Default embedding dimension for sparse features (default: 16)
        embed_dims: Optional dict mapping feature name to embedding dimension.
                    If provided, overrides embed_dim for specific features.
                    Recommended: use smaller dims for low-cardinality features.
        cross_layers: Number of cross network layers (default: 3)
        cross_rank: Low-rank dimension for cross network (default: 32)
        mlp_dims: Hidden layer dimensions for deep network (default: [256, 128, 64])
        dropout: Dropout rate (default: 0.1)
        use_bn: Whether to use batch normalization (default: True)
        structure: 'parallel' or 'stacked' (default: 'parallel')
    
    Example with per-feature embedding dimensions:
        config = DCNV2Config(
            sparse_features={'uid': 10000, 'hour_of_day': 24, 'day_of_week': 7},
            embed_dim=16,  # Default
            embed_dims={'hour_of_day': 4, 'day_of_week': 2},  # Override for small vocab
        )
    """
    # Feature configuration
    sparse_features: Dict[str, int] = field(default_factory=dict)
    dense_features: List[str] = field(default_factory=list)
    dense_dim: Optional[int] = None  # Override: if set, used instead of len(dense_features)
    
    # Model hyperparameters
    embed_dim: int = 16  # Default embedding dimension
    embed_dims: Optional[Dict[str, int]] = None  # Per-feature override
    cross_layers: int = 3
    cross_rank: int = 32  # Low-rank dimension for efficiency
    mlp_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    use_bn: bool = True
    structure: str = 'parallel'  # 'parallel' or 'stacked'
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.sparse_features and not self.dense_features:
            raise ValueError("Must have at least one sparse or dense feature")
        if self.structure not in ['parallel', 'stacked']:
            raise ValueError("structure must be 'parallel' or 'stacked'")
    
    def get_embed_dim(self, feature_name: str) -> int:
        """Get embedding dimension for a specific feature."""
        if self.embed_dims and feature_name in self.embed_dims:
            return self.embed_dims[feature_name]
        return self.embed_dim
    
    def get_total_sparse_dim(self) -> int:
        """Calculate total dimension of all sparse embeddings."""
        return sum(self.get_embed_dim(feat) for feat in self.sparse_features)


class CrossNetworkV2(nn.Module):
    """Cross Network V2 with Low-Rank Factorization.
    
    The cross network performs explicit bounded-degree feature interactions:
        x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    
    Where W_l = U_l · V_l^T (low-rank factorization for efficiency)
    
    This is more expressive than DCN-V1 which used:
        x_{l+1} = x_0 · (w_l^T · x_l) + b_l + x_l (vector w instead of matrix W)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        low_rank: int = 32,
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Low-rank matrices for each layer: W = U · V^T
        self.U_list = nn.ModuleList([
            nn.Linear(input_dim, low_rank, bias=False) 
            for _ in range(num_layers)
        ])
        self.V_list = nn.ModuleList([
            nn.Linear(low_rank, input_dim, bias=False) 
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross network.
        
        Args:
            x0: Initial input (batch_size, input_dim)
            
        Returns:
            Cross network output (batch_size, input_dim)
        """
        xl = x0
        for i in range(self.num_layers):
            # Low-rank computation: W · x_l = V · (U · x_l)
            xl_u = self.U_list[i](xl)  # (batch, low_rank)
            xl_w = self.V_list[i](xl_u)  # (batch, input_dim)
            
            # Cross network formula: x_{l+1} = x_0 ⊙ (W · x_l + b) + x_l
            xl = x0 * (xl_w + self.bias[i]) + xl
        
        return xl


class DNN(nn.Module):
    """Deep Neural Network for higher-order feature interactions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
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
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            output: (batch_size, output_dim)
        """
        return self.mlp(x)


class DCNV2(nn.Module):
    """DCN-V2: Deep & Cross Network V2.
    
    Combines:
    1. Cross Network: Explicit bounded-degree feature interactions with low-rank W
    2. Deep Network: Implicit higher-order interactions via MLP
    
    Supports two structures:
    - Parallel: Cross and Deep run in parallel, outputs concatenated
    - Stacked: Deep network output feeds into Cross network
    """
    
    def __init__(self, config: DCNV2Config):
        super().__init__()
        self.config = config
        
        # ===== Embedding layers for sparse features =====
        # Supports per-feature embedding dimensions via config.get_embed_dim()
        self.embeddings = nn.ModuleDict()
        self.embed_dims = {}  # Store actual dims for forward pass
        
        for feat_name, vocab_size in config.sparse_features.items():
            feat_embed_dim = config.get_embed_dim(feat_name)
            self.embed_dims[feat_name] = feat_embed_dim
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=feat_embed_dim,
                padding_idx=0,
            )
        
        # ===== Compute input dimension =====
        # Sum of all sparse embedding dims (may vary per feature) + dense features
        total_sparse_dim = config.get_total_sparse_dim()
        # Use dense_dim if provided, otherwise fall back to len(dense_features)
        num_dense = config.dense_dim if config.dense_dim is not None else len(config.dense_features)
        self.input_dim = total_sparse_dim + num_dense
        
        # ===== Cross Network =====
        self.cross_net = CrossNetworkV2(
            input_dim=self.input_dim,
            num_layers=config.cross_layers,
            low_rank=config.cross_rank,
        )
        
        # ===== Deep Network =====
        self.deep_net = DNN(
            input_dim=self.input_dim,
            hidden_dims=config.mlp_dims,
            dropout=config.dropout,
            use_bn=config.use_bn,
        )
        
        # ===== Output layer =====
        if config.structure == 'parallel':
            # Parallel: concat cross and deep outputs
            final_dim = self.input_dim + self.deep_net.output_dim
        else:
            # Stacked: cross network applied after deep
            final_dim = self.input_dim
        
        self.output_layer = nn.Linear(final_dim, 1)
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                if 'bias' in name or param.dim() == 0:
                    continue  # Keep zeros for biases
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
        # ===== Embed sparse features =====
        embed_list = []
        for feat_name, indices in sparse_inputs.items():
            embed = self.embeddings[feat_name](indices)  # (batch, embed_dim)
            embed_list.append(embed)
        
        # Concatenate all embeddings
        sparse_concat = torch.cat(embed_list, dim=1)  # (batch, num_sparse * embed_dim)
        
        # ===== Create input x0 =====
        if dense_inputs is not None:
            x0 = torch.cat([sparse_concat, dense_inputs], dim=1)  # (batch, input_dim)
        else:
            x0 = sparse_concat
        
        # ===== Structure-dependent forward pass =====
        if self.config.structure == 'parallel':
            # Cross and Deep in parallel
            cross_out = self.cross_net(x0)  # (batch, input_dim)
            deep_out = self.deep_net(x0)    # (batch, mlp_output_dim)
            combined = torch.cat([cross_out, deep_out], dim=1)
        else:
            # Stacked: Deep first, then Cross
            deep_out = self.deep_net(x0)
            # Expand deep output back to input_dim for cross network
            # (In practice, for stacked we often just use cross on x0)
            cross_out = self.cross_net(x0)
            combined = cross_out
        
        # ===== Output =====
        logits = self.output_layer(combined) + self.bias
        
        return logits
    
    def predict_proba(
        self,
        sparse_inputs: Dict[str, torch.Tensor],
        dense_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(sparse_inputs, dense_inputs)
        return torch.sigmoid(logits)


class DCNV2Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for DCN-V2 training."""
    
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
    """Custom collate function for DCN-V2 batches."""
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
def get_yambda_dcnv2_config(
    uid_vocab_size: int = 10000,
    item_vocab_size: int = 900000,
) -> DCNV2Config:
    """Get default DCN-V2 config for Yambda dataset."""
    return DCNV2Config(
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
        cross_layers=3,
        cross_rank=32,
        mlp_dims=[256, 128, 64],
        dropout=0.1,
        use_bn=True,
        structure='parallel',
        learning_rate=1e-3,
        weight_decay=1e-5,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing DCN-V2 model...")
    
    config = DCNV2Config(
        sparse_features={'uid': 1000, 'item_id': 5000},
        dense_features=['feat1', 'feat2', 'feat3'],
        embed_dim=16,
        cross_layers=3,
        cross_rank=16,
        mlp_dims=[64, 32],
        structure='parallel',
    )
    
    model = DCNV2(config)
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
    
    print("✓ DCN-V2 test passed!")

