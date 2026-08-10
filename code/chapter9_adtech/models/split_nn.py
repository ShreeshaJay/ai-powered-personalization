"""
Split Neural Network for Vertical Federated Learning
=====================================================

Implements a Split Neural Network for privacy-preserving CVR prediction,
following the VanillaVFL architecture from the FedAds paper.

Reference:
    Wei et al. "FedAds: A Benchmark for Privacy-Preserving CVR Estimation
    with Vertical Federated Learning" (SIGIR 2023)

Scenario:
    Two parties jointly train a CVR model without sharing raw data:
    
    PUBLISHER (NON-LABEL PARTY, f_N):
        - Features: l_i_fea_*, l_u_fea_*, l_c_fea (17 features in public dataset)
        - Tower: embed + DNN[128, 32]
        
    ADVERTISER (LABEL PARTY, f_L):
        - Features: f_u_fea_*, f_uc_fea_*, f_c (5 features in public dataset)
        - Bottom tower: embed + DNN[256, 128]
        - Top layer (aggregation): single layer (160 -> 1) with sigmoid

    Note on feature count discrepancy:
        Paper's internal system: 7 non-label + 16 label features
        Public dataset release:  17 non-label + 5 label features
        The ratio is essentially flipped. Absolute AUC numbers will differ
        from Table 3, but relative ordering (VFL > Local) should hold.

Training Settings (Paper Sec 5.1.4):
    - Embed dim: 8 per feature
    - Batch size: 256
    - Epochs: 1 (standard for large-scale ad systems)
    - Loss: standard cross-entropy
    - Train/test split: time-based (last week = test)

Usage:
    config = SplitNNConfig()
    model = SplitNN(config)
    outputs = model(local_ids, fed_ids)
    loss = model.compute_loss(outputs, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration (matches FedAds paper Sec 5.1.4)
# =============================================================================

@dataclass
class SplitNNConfig:
    """Configuration for Split Neural Network.
    
    Defaults match the FedAds paper (Wei et al., SIGIR 2023).
    
    Paper Table 3 expected results:
        Local (label party only): AUC = 0.609
        VanillaVFL:               AUC = 0.620
        ORALE (centralized):      AUC = 0.658
    """
    # Vocabulary sizes for feature hashing
    local_vocab_size: int = 50_000
    fed_vocab_size: int = 50_000
    
    # Feature counts (from public FedAds CSV)
    n_local_features: int = 17
    n_fed_features: int = 5
    
    # Embedding: 8-dim per feature (paper Sec 5.1.4)
    embed_dim: int = 8
    
    # f_N (non-label/publisher): 2-layer DNN with output sizes [128, 32]
    local_layers: List[int] = field(default_factory=lambda: [128])
    local_hidden_dim: int = 32
    
    # f_L bottom (label/advertiser): 2-layer DNN with output sizes [256, 128]
    fed_layers: List[int] = field(default_factory=lambda: [256])
    fed_hidden_dim: int = 128
    
    # f_L top (aggregation): single layer with sigmoid, no hidden layers
    agg_layers: List[int] = field(default_factory=list)
    
    # Paper does not mention dropout or weight decay
    dropout: float = 0.0
    use_bn: bool = True
    
    # Standard cross-entropy (paper does not use focal loss)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    use_focal_loss: bool = False
    focal_alpha: float = 0.25   # Only used if use_focal_loss=True
    focal_gamma: float = 2.0    # Only used if use_focal_loss=True


# Training defaults from paper
PAPER_TRAINING_DEFAULTS = {
    'batch_size': 256,       # Paper Sec 5.1.4
    'epochs': 1,             # "number of epochs is usually set to one"
    'grad_clip': None,       # Paper does not mention gradient clipping
    'val_ratio': 0.115,      # ~1.3M test / 11.3M total
}


# =============================================================================
# Tower Modules
# =============================================================================

def _build_tower(input_dim, hidden_dims, output_dim, config):
    """Build a tower: sequence of Linear + BN + ReLU (+ Dropout) layers."""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if config.use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _init_tower_weights(module):
    """Xavier init for Linear layers in a tower."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _embed_and_pool(embedding, ids):
    """Embed feature IDs and mean-pool over features, masking padding (id=0)."""
    embeddings = embedding(ids)
    mask = (ids != 0).float().unsqueeze(-1)
    masked = embeddings * mask
    return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _compute_loss(cvr, labels, config):
    """Shared loss: BCE or focal loss."""
    if config.use_focal_loss:
        pred = torch.clamp(cvr, min=1e-7, max=1 - 1e-7)
        ce = -labels * torch.log(pred) - (1 - labels) * torch.log(1 - pred)
        p_t = torch.where(labels == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** config.focal_gamma
        alpha_t = torch.where(labels == 1, config.focal_alpha, 1 - config.focal_alpha)
        return (alpha_t * focal_weight * ce).mean()
    else:
        return F.binary_cross_entropy(cvr, labels, reduction='mean')


class LocalTower(nn.Module):
    """Publisher-side tower (f_N in paper). Only hidden rep is shared."""
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.local_vocab_size + 1, config.embed_dim, padding_idx=0)
        self.network = _build_tower(
            config.embed_dim, config.local_layers, config.local_hidden_dim, config)
        nn.init.xavier_uniform_(self.embedding.weight[1:])
        _init_tower_weights(self.network)
    
    def forward(self, local_ids: torch.Tensor) -> torch.Tensor:
        return self.network(_embed_and_pool(self.embedding, local_ids))


class FederatedTower(nn.Module):
    """Advertiser-side bottom tower (f_L bottom in paper). Only hidden rep is shared."""
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.fed_vocab_size + 1, config.embed_dim, padding_idx=0)
        self.network = _build_tower(
            config.embed_dim, config.fed_layers, config.fed_hidden_dim, config)
        nn.init.xavier_uniform_(self.embedding.weight[1:])
        _init_tower_weights(self.network)
    
    def forward(self, fed_ids: torch.Tensor) -> torch.Tensor:
        return self.network(_embed_and_pool(self.embedding, fed_ids))


class Aggregator(nn.Module):
    """Aggregation / f_L top in paper. Combines h_local and h_fed -> P(cvr).
    
    With default config (agg_layers=[]), this is a single linear layer:
    concat(h_local=32, h_fed=128) = 160 -> 1, matching the paper.
    """
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        input_dim = config.local_hidden_dim + config.fed_hidden_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.agg_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        _init_tower_weights(self.network)
    
    def forward(self, h_local: torch.Tensor, h_fed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_local, h_fed], dim=1)
        return torch.sigmoid(self.network(combined).squeeze(-1))


# =============================================================================
# Complete Models
# =============================================================================

class SplitNN(nn.Module):
    """VanillaVFL: Split Neural Network for federated CVR prediction."""
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        self.config = config
        self.local_tower = LocalTower(config)
        self.fed_tower = FederatedTower(config)
        self.aggregator = Aggregator(config)
    
    def forward(self, local_ids, fed_ids) -> Dict[str, torch.Tensor]:
        h_local = self.local_tower(local_ids)
        h_fed = self.fed_tower(fed_ids)
        cvr = self.aggregator(h_local, h_fed)
        return {'cvr': cvr, 'h_local': h_local, 'h_fed': h_fed}
    
    def forward_local_only(self, local_ids: torch.Tensor) -> torch.Tensor:
        """Forward with only local features (zero federated hidden)."""
        h_local = self.local_tower(local_ids)
        h_fed = torch.zeros(local_ids.size(0), self.config.fed_hidden_dim,
                            device=local_ids.device)
        return self.aggregator(h_local, h_fed)
    
    def compute_loss(self, outputs, labels) -> Dict[str, torch.Tensor]:
        return {'loss': _compute_loss(outputs['cvr'], labels, self.config)}
    
    def get_parameter_count(self) -> Dict[str, int]:
        lp = sum(p.numel() for p in self.local_tower.parameters())
        fp = sum(p.numel() for p in self.fed_tower.parameters())
        ap = sum(p.numel() for p in self.aggregator.parameters())
        return {'local_tower': lp, 'fed_tower': fp, 'aggregator': ap, 'total': lp+fp+ap}


class CentralizedBaseline(nn.Module):
    """ORALE upper bound: sees ALL features as if both parties shared raw data."""
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        self.config = config
        total_vocab = config.local_vocab_size + config.fed_vocab_size
        self.embedding = nn.Embedding(total_vocab + 1, config.embed_dim, padding_idx=0)
        
        # Use local_layers for the DNN (paper doesn't specify centralized arch)
        self.network = _build_tower(
            config.embed_dim, config.local_layers, 1, config)
        
        nn.init.xavier_uniform_(self.embedding.weight[1:])
        _init_tower_weights(self.network)
    
    def forward(self, local_ids, fed_ids) -> Dict[str, torch.Tensor]:
        fed_offset = fed_ids + self.config.local_vocab_size
        fed_offset = torch.where(fed_ids == 0, torch.zeros_like(fed_ids), fed_offset)
        all_ids = torch.cat([local_ids, fed_offset], dim=1)
        pooled = _embed_and_pool(self.embedding, all_ids)
        logit = self.network(pooled).squeeze(-1)
        return {'cvr': torch.sigmoid(logit)}
    
    def compute_loss(self, outputs, labels) -> Dict[str, torch.Tensor]:
        return {'loss': _compute_loss(outputs['cvr'], labels, self.config)}


class LabelPartyOnlyModel(nn.Module):
    """Paper's 'Local' baseline: uses only advertiser (label party) features.
    
    Note: In the paper's internal system, the label party has 16 features.
    In the public dataset, it has only 5 features, so this baseline will
    underperform relative to paper's reported AUC=0.609.
    """
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.fed_vocab_size + 1, config.embed_dim, padding_idx=0)
        self.network = _build_tower(
            config.embed_dim, config.fed_layers, 1, config)
        nn.init.xavier_uniform_(self.embedding.weight[1:])
        _init_tower_weights(self.network)
    
    def forward(self, local_ids, fed_ids) -> Dict[str, torch.Tensor]:
        """Uses only fed_ids; local_ids accepted for API compatibility."""
        pooled = _embed_and_pool(self.embedding, fed_ids)
        logit = self.network(pooled).squeeze(-1)
        return {'cvr': torch.sigmoid(logit)}
    
    def compute_loss(self, outputs, labels) -> Dict[str, torch.Tensor]:
        return {'loss': _compute_loss(outputs['cvr'], labels, self.config)}


class NonLabelPartyOnlyModel(nn.Module):
    """Publisher-only baseline: uses only non-label party (publisher) features.
    
    Not in the paper but useful to understand the value of each party's data.
    """
    
    def __init__(self, config: SplitNNConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.local_vocab_size + 1, config.embed_dim, padding_idx=0)
        self.network = _build_tower(
            config.embed_dim, config.local_layers, 1, config)
        nn.init.xavier_uniform_(self.embedding.weight[1:])
        _init_tower_weights(self.network)
    
    def forward(self, local_ids) -> Dict[str, torch.Tensor]:
        pooled = _embed_and_pool(self.embedding, local_ids)
        logit = self.network(pooled).squeeze(-1)
        return {'cvr': torch.sigmoid(logit)}
    
    def compute_loss(self, outputs, labels) -> Dict[str, torch.Tensor]:
        return {'loss': _compute_loss(outputs['cvr'], labels, self.config)}


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Split Neural Network (paper config)...")
    
    config = SplitNNConfig()
    print(f"Config: local_layers={config.local_layers}, local_hidden={config.local_hidden_dim}")
    print(f"        fed_layers={config.fed_layers}, fed_hidden={config.fed_hidden_dim}")
    print(f"        agg_layers={config.agg_layers}")
    print(f"        dropout={config.dropout}, use_focal_loss={config.use_focal_loss}")
    
    model = SplitNN(config)
    print(f"\nParameter counts: {model.get_parameter_count()}")
    
    batch_size = 32
    local_ids = torch.randint(0, 50000, (batch_size, 17))
    fed_ids = torch.randint(0, 50000, (batch_size, 5))
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    outputs = model(local_ids, fed_ids)
    losses = model.compute_loss(outputs, labels)
    print(f"VanillaVFL loss: {losses['loss'].item():.4f}")
    print(f"CVR range: [{outputs['cvr'].min():.4f}, {outputs['cvr'].max():.4f}]")
    
    # Test all baselines
    central = CentralizedBaseline(config)
    print(f"\nCentralized output: {central(local_ids, fed_ids)['cvr'].shape}")
    
    label_only = LabelPartyOnlyModel(config)
    print(f"Label-party-only output: {label_only(local_ids, fed_ids)['cvr'].shape}")
    
    nonlabel_only = NonLabelPartyOnlyModel(config)
    print(f"NonLabel-party-only output: {nonlabel_only(local_ids)['cvr'].shape}")
    
    print("\nAll tests passed!")
