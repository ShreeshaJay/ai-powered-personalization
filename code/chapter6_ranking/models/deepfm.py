"""
DeepFM: Deep Factorization Machine for CTR Prediction

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                     DeepFM                               │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Sparse Features              Dense Features            │
    │  (uid, item_id,              (numerical,               │
    │   hour, day_of_week)          historical, is_organic)   │
    │       │                            │                    │
    │       ▼                            │                    │
    │  ┌─────────┐                       │                    │
    │  │Embedding│                       │                    │
    │  │ Tables  │                       │                    │
    │  └────┬────┘                       │                    │
    │       │                            │                    │
    │       ├────────────────────────────┤                    │
    │       │                            │                    │
    │       ▼                            ▼                    │
    │  ┌─────────┐              ┌─────────────────┐           │
    │  │   FM    │              │ Concat + DNN    │           │
    │  │ (2nd    │              │ (Higher order)  │           │
    │  │ order)  │              │                 │           │
    │  └────┬────┘              └────────┬────────┘           │
    │       │                            │                    │
    │       └──────────┬─────────────────┘                    │
    │                  ▼                                      │
    │           ┌──────────┐                                  │
    │           │   Sum    │                                  │
    │           │    +     │                                  │
    │           │ Sigmoid  │                                  │
    │           └──────────┘                                  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

Reference:
    Guo et al. "DeepFM: A Factorization-Machine based Neural Network 
    for CTR Prediction" (IJCAI 2017)

Usage:
    from models.deepfm import DeepFM, DeepFMConfig
    
    config = DeepFMConfig(
        sparse_features={'uid': 10000, 'item_id': 900000},
        dense_features=['track_length_seconds', 'user_avg_completion', ...],
        embed_dim=16,
        mlp_dims=[256, 128, 64],
    )
    model = DeepFM(config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DeepFMConfig:
    """Configuration for DeepFM model.
    
    Attributes:
        sparse_features: Dict mapping feature name to vocabulary size
        dense_features: List of dense feature names
        embed_dim: Embedding dimension for sparse features (default: 16)
        mlp_dims: Hidden layer dimensions for DNN (default: [256, 128, 64])
        dropout: Dropout rate (default: 0.1)
        use_bn: Whether to use batch normalization (default: True)
    """
    # Feature configuration
    sparse_features: Dict[str, int] = field(default_factory=dict)
    dense_features: List[str] = field(default_factory=list)
    
    # Model hyperparameters
    embed_dim: int = 16
    mlp_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    use_bn: bool = True
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.sparse_features and not self.dense_features:
            raise ValueError("Must have at least one sparse or dense feature")


class FMLayer(nn.Module):
    """Factorization Machine layer for 2nd-order feature interactions.
    
    Computes: sum_{i<j} <v_i, v_j> * x_i * x_j
    
    Using the efficient formula:
        0.5 * (||sum(v_i * x_i)||^2 - sum(||v_i * x_i||^2))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, num_fields, embed_dim) - stacked embeddings
            
        Returns:
            fm_output: (batch_size, 1) - FM interaction score
        """
        # Sum of squared embeddings
        sum_of_square = torch.sum(embeddings ** 2, dim=1)  # (batch, embed_dim)
        
        # Square of summed embeddings
        square_of_sum = torch.sum(embeddings, dim=1) ** 2  # (batch, embed_dim)
        
        # FM formula: 0.5 * sum(square_of_sum - sum_of_square)
        fm_output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        return fm_output  # (batch, 1)


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


class DeepFM(nn.Module):
    """DeepFM: Factorization Machine + Deep Neural Network.
    
    Combines:
    1. Linear term (1st order)
    2. FM term (2nd order interactions via embeddings)
    3. DNN term (higher order interactions)
    """
    
    def __init__(self, config: DeepFMConfig):
        super().__init__()
        self.config = config
        
        # ===== Embedding layers for sparse features =====
        self.embeddings = nn.ModuleDict()
        self.linear_embeddings = nn.ModuleDict()  # For 1st order term
        
        for feat_name, vocab_size in config.sparse_features.items():
            # Main embedding for FM and DNN
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config.embed_dim,
                padding_idx=0,  # Index 0 reserved for unknown/padding
            )
            # Linear embedding for 1st order term (output dim = 1)
            self.linear_embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=1,
                padding_idx=0,
            )
        
        # ===== Dense feature processing =====
        num_dense = len(config.dense_features)
        if num_dense > 0:
            # Linear weights for dense features (1st order)
            self.dense_linear = nn.Linear(num_dense, 1, bias=False)
            # Project dense to embedding space for FM interaction
            self.dense_embedding = nn.Linear(num_dense, config.embed_dim, bias=False)
        else:
            self.dense_linear = None
            self.dense_embedding = None
        
        # ===== FM Layer =====
        self.fm = FMLayer()
        
        # ===== DNN =====
        # DNN input: flattened sparse embeddings + dense features
        num_sparse = len(config.sparse_features)
        dnn_input_dim = num_sparse * config.embed_dim + num_dense
        
        self.dnn = DNN(
            input_dim=dnn_input_dim,
            hidden_dims=config.mlp_dims,
            dropout=config.dropout,
            use_bn=config.use_bn,
        )
        
        # ===== Output layer =====
        # Combines: linear (1) + FM (1) + DNN output
        self.output_layer = nn.Linear(self.dnn.output_dim, 1)
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                # Skip 1D parameters (biases, scalars)
                if 'bias' in name or param.dim() == 0:
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
        
        # ===== 1st Order (Linear) Term =====
        linear_output = self.bias.expand(batch_size, 1)
        
        # Sparse linear terms
        for feat_name, indices in sparse_inputs.items():
            linear_embed = self.linear_embeddings[feat_name](indices)  # (batch, 1)
            linear_output = linear_output + linear_embed
        
        # Dense linear terms
        if dense_inputs is not None and self.dense_linear is not None:
            linear_output = linear_output + self.dense_linear(dense_inputs)
        
        # ===== 2nd Order (FM) Term =====
        # Collect embeddings for FM
        embed_list = []
        for feat_name, indices in sparse_inputs.items():
            embed = self.embeddings[feat_name](indices)  # (batch, embed_dim)
            embed_list.append(embed)
        
        # Add dense features projected to embedding space
        if dense_inputs is not None and self.dense_embedding is not None:
            dense_embed = self.dense_embedding(dense_inputs)  # (batch, embed_dim)
            embed_list.append(dense_embed)
        
        # Stack embeddings: (batch, num_fields, embed_dim)
        stacked_embeds = torch.stack(embed_list, dim=1)
        
        # FM interaction
        fm_output = self.fm(stacked_embeds)  # (batch, 1)
        
        # ===== DNN Term =====
        # Flatten sparse embeddings
        sparse_concat = torch.cat(
            [self.embeddings[feat_name](indices) for feat_name, indices in sparse_inputs.items()],
            dim=1
        )  # (batch, num_sparse * embed_dim)
        
        # Concatenate with dense features
        if dense_inputs is not None:
            dnn_input = torch.cat([sparse_concat, dense_inputs], dim=1)
        else:
            dnn_input = sparse_concat
        
        dnn_hidden = self.dnn(dnn_input)  # (batch, dnn_output_dim)
        dnn_output = self.output_layer(dnn_hidden)  # (batch, 1)
        
        # ===== Combine all terms =====
        logits = linear_output + fm_output + dnn_output
        
        return logits
    
    def predict_proba(
        self,
        sparse_inputs: Dict[str, torch.Tensor],
        dense_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(sparse_inputs, dense_inputs)
        return torch.sigmoid(logits)


class DeepFMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for DeepFM training.
    
    Handles conversion of pandas/numpy data to torch tensors with proper
    feature encoding for sparse and dense features.
    """
    
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
    """Custom collate function for DeepFM batches."""
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
def get_yambda_deepfm_config(
    uid_vocab_size: int = 10000,
    item_vocab_size: int = 900000,
) -> DeepFMConfig:
    """Get default DeepFM config for Yambda dataset.
    
    Sparse features (learned embeddings):
        - uid: User ID
        - item_id: Track ID
        - hour_of_day: Hour (0-23) - cyclical patterns
        - day_of_week: Day (0-6) - weekly patterns
    
    Dense features:
        - is_organic (binary)
        - track_length_seconds
        - All historical user/item features
    """
    return DeepFMConfig(
        sparse_features={
            'uid': uid_vocab_size,
            'item_id': item_vocab_size,
            'hour_of_day': 24,
            'day_of_week': 7,
        },
        dense_features=[
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
        ],
        embed_dim=16,
        mlp_dims=[256, 128, 64],
        dropout=0.1,
        use_bn=True,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing DeepFM model...")
    
    config = DeepFMConfig(
        sparse_features={'uid': 1000, 'item_id': 5000},
        dense_features=['feat1', 'feat2', 'feat3'],
        embed_dim=16,
        mlp_dims=[64, 32],
    )
    
    model = DeepFM(config)
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
    print("✓ DeepFM test passed!")

