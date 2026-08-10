"""
ESMM: Entire Space Multi-Task Model for Click and Conversion Prediction
========================================================================

This module implements ESMM, a seminal model for addressing selection bias
in Conversion Rate (CVR) prediction.

The Problem:
    Traditional CVR models train on clicked samples only:
        P(conversion | click, features)
    
    But at inference, we apply this to ALL users (including non-clickers).
    The feature distribution of clicked users ≠ all users, causing:
        - Selection bias (SSB - Sample Selection Bias)
        - Poor generalization to the full user space
        - Data sparsity (conversions are rare even among clickers)

The ESMM Solution:
    Model the entire user journey as a multi-task problem:
    
    1. CTR Tower:   P(click | impression)
    2. CVR Tower:   P(conversion | click)  [No direct supervision!]
    3. CTCVR:       P(click) × P(conversion | click) = P(click AND conversion)
    
    The ESSM formulation assumes user experience flow is such that conversions can only happen if there is a click.
     (ie. there is no 1-click conversions of the sort Amazon has)
    CTCVR = P(click AND conversion | impression)
      = P(click | impression) × P(conversion | click)
      = CTR × CVR

    Key insight: CTCVR can be supervised on ALL impressions (not just clicks)
    because we observe conversions on the entire space.

CRITICAL ASSUMPTION - Sequential Funnel:
    ESMM assumes conversions can ONLY happen AFTER a click:
    
        Impression → Click → Conversion (sequential, no shortcuts)
    
    This is VALID for:
        - Traditional display advertising
        - Search ads with landing pages
        - Any funnel where purchase requires clicking through
    
    This is NOT VALID for:
        - View-through conversions (user sees ad, converts later without click)
        - Amazon 1-click purchases (conversion without ad click)
        - Multi-touch attribution scenarios
    
    For non-sequential funnels, consider:
        - Direct P(conversion | impression) modeling
        - Parallel path model: P(conv) = P(click)×P(conv|click) + P(no_click)×P(conv|no_click)
        - Separate click-through and view-through conversion models

Architecture:
    Basic idea is to have separate towers for CTR and CVR and then multiply them to get CTCVR.
    
    ┌────────────────────────────────────────────────────────────────────┐
    │                        ESMM Architecture                           │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │  Sparse Features (feature_ids)                                     │
    │        │                                                           │
    │        ▼                                                           │
    │  ┌─────────────────────────────────────────────┐                   │
    │  │        Shared Embedding Layer               │                   │
    │  │    (vocab_size × embed_dim)                 │                   │
    │  └──────────────────┬──────────────────────────┘                   │
    │                     │                                              │
    │            Feature Pooling (sum/mean)                              │
    │                     │                                              │
    │         ┌───────────┴───────────┐                                  │
    │         │                       │                                  │
    │         ▼                       ▼                                  │
    │  ┌─────────────┐         ┌─────────────┐                           │
    │  │  CTR Tower  │         │  CVR Tower  │                           │
    │  │    (MLP)    │         │    (MLP)    │                           │
    │  │             │         │             │                           │
    │  │  256→128→64 │         │  256→128→64 │                           │
    │  └──────┬──────┘         └──────┬──────┘                           │
    │         │                       │                                  │
    │         ▼                       ▼                                  │
    │      P(click)              P(conv|click)                           │
    │         │                       │                                  │
    │         └───────────┬───────────┘                                  │
    │                     │                                              │
    │                     ▼                                              │
    │              CTCVR = CTR × CVR                                     │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

Training:
    Loss = L_ctr + L_ctcvr
    
    L_ctr:   BCE(P(click), click_label)        on ALL samples
    L_ctcvr: BCE(P(click) × P(conv|click), conversion_label)  on ALL samples
    
    Note: CVR tower learns ONLY through the CTCVR gradient path.
    No direct CVR loss (which would require clicked samples only).

Reference:
    Ma et al. "Entire Space Multi-Task Model: An Effective Approach for 
    Estimating Post-Click Conversion Rate" (SIGIR 2018)
    
Usage:
    config = ESMMConfig(vocab_size=100000, embed_dim=16, tower_dims=[256, 128, 64])
    model = ESMM(config)
    
    # Forward pass
    outputs = model(feature_ids, feature_values)
    # outputs['ctr']: P(click)
    # outputs['cvr']: P(conversion | click)
    # outputs['ctcvr']: P(click AND conversion)
    
    # Compute loss
    loss = model.compute_loss(outputs, click_labels, conversion_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ESMMConfig:
    """Configuration for ESMM model.
    
    Attributes:
        vocab_size: Size of feature ID vocabulary (hash buckets)
        num_features: Number of feature slots per sample
        embed_dim: Embedding dimension for sparse features
        tower_dims: Hidden layer dimensions for CTR/CVR towers
        dropout: Dropout rate
        use_bn: Whether to use batch normalization
        ctr_loss_weight: Weight for CTR loss
        ctcvr_loss_weight: Weight for CTCVR loss
        use_auxiliary_cvr_loss: Whether to add auxiliary CVR loss on clicked samples
        auxiliary_cvr_weight: Weight for auxiliary CVR loss
        use_focal_loss: Whether to use focal loss for class imbalance
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
    """
    # Feature configuration
    vocab_size: int = 100_000
    num_features: int = 50
    
    # Embedding
    embed_dim: int = 16
    
    # Tower architecture
    tower_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    use_bn: bool = True
    
    # Loss configuration
    ctr_loss_weight: float = 1.0
    ctcvr_loss_weight: float = 1.0
    
    # EXPERIMENTAL: Auxiliary CVR loss
    # NOT in original ESMM paper - this is our own extension based on gradient analysis.
    # Hypothesis: May help when CTR is very low and CVR tower gets weak gradients.
    # Disabled by default to match original paper.
    use_auxiliary_cvr_loss: bool = False
    auxiliary_cvr_weight: float = 0.1
    
    # Focal loss for class imbalance
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class Tower(nn.Module):
    """MLP Tower for CTR or CVR prediction.
    
    Architecture: input → [Linear → BN → ReLU → Dropout] × N → Linear → Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
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
        
        # Output layer (logit, no activation)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.network(x).squeeze(-1)


class ESMM(nn.Module):
    """Entire Space Multi-Task Model for CTR and CVR prediction.
    
    Key insight: By modeling CTCVR = CTR × CVR and training on the entire
    impression space, we avoid selection bias in CVR estimation.
    """
    
    def __init__(self, config: ESMMConfig):
        super().__init__()
        self.config = config
        
        # Shared embedding layer
        # +1 for padding index 0
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size + 1,
            embedding_dim=config.embed_dim,
            padding_idx=0,
        )
        
        # Feature value projection (optional, for weighted embeddings)
        self.value_proj = nn.Linear(1, config.embed_dim, bias=False)
        
        # Pooled embedding dimension
        pooled_dim = config.embed_dim
        
        # CTR Tower
        self.ctr_tower = Tower(
            input_dim=pooled_dim,
            hidden_dims=config.tower_dims,
            dropout=config.dropout,
            use_bn=config.use_bn,
        )
        
        # CVR Tower
        self.cvr_tower = Tower(
            input_dim=pooled_dim,
            hidden_dims=config.tower_dims,
            dropout=config.dropout,
            use_bn=config.use_bn,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.embedding.weight[1:])  # Skip padding
        
        # Initialize towers
        for module in [self.ctr_tower, self.cvr_tower]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing CTR, CVR, and CTCVR.
        
        Args:
            feature_ids: (batch_size, num_features) tensor of feature IDs
            feature_values: (batch_size, num_features) tensor of feature values
            
        Returns:
            Dictionary with keys:
                'ctr': P(click | impression)
                'cvr': P(conversion | click)
                'ctcvr': P(click AND conversion)
                'ctr_logit': Raw CTR logit (for loss computation)
                'cvr_logit': Raw CVR logit
        """
        batch_size = feature_ids.size(0)
        
        # Embed features: (batch, num_features) → (batch, num_features, embed_dim)
        embeddings = self.embedding(feature_ids)
        
        # Weight embeddings by feature values if provided
        if feature_values is not None:
            # (batch, num_features, 1)
            values = feature_values.unsqueeze(-1)
            # Scale embeddings by values
            embeddings = embeddings * values
        
        # Pool embeddings: mean over features
        # (batch, num_features, embed_dim) → (batch, embed_dim)
        # Mask padding (feature_id = 0)
        mask = (feature_ids != 0).float().unsqueeze(-1)  # (batch, num_features, 1)
        masked_embeddings = embeddings * mask
        
        # Sum and normalize
        summed = masked_embeddings.sum(dim=1)  # (batch, embed_dim)
        counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = summed / counts  # (batch, embed_dim)
        
        # CTR Tower
        ctr_logit = self.ctr_tower(pooled)
        ctr = torch.sigmoid(ctr_logit)
        
        # CVR Tower
        cvr_logit = self.cvr_tower(pooled)
        cvr = torch.sigmoid(cvr_logit)
        
        # CTCVR = CTR × CVR
        # Clamp to avoid numerical issues
        ctr_clamped = torch.clamp(ctr, min=1e-7, max=1 - 1e-7)
        cvr_clamped = torch.clamp(cvr, min=1e-7, max=1 - 1e-7)
        ctcvr = ctr_clamped * cvr_clamped
        
        return {
            'ctr': ctr,
            'cvr': cvr,
            'ctcvr': ctcvr,
            'ctr_logit': ctr_logit,
            'cvr_logit': cvr_logit,
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        click_labels: torch.Tensor,
        conversion_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute ESMM training loss.
        
        Loss = w_ctr * L_ctr + w_ctcvr * L_ctcvr [+ w_cvr * L_cvr]
        
        Args:
            outputs: Model outputs from forward()
            click_labels: (batch_size,) binary click labels
            conversion_labels: (batch_size,) binary conversion labels
            
        Returns:
            Dictionary with loss components and total loss
        """
        ctr = outputs['ctr']
        ctcvr = outputs['ctcvr']
        cvr = outputs['cvr']
        
        if self.config.use_focal_loss:
            # Focal loss for CTR
            loss_ctr = self._focal_loss(
                ctr, click_labels,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
            )
            
            # Focal loss for CTCVR
            loss_ctcvr = self._focal_loss(
                ctcvr, conversion_labels,
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
            )
        else:
            # Standard BCE loss
            loss_ctr = F.binary_cross_entropy(
                ctr, click_labels, reduction='mean'
            )
            loss_ctcvr = F.binary_cross_entropy(
                ctcvr, conversion_labels, reduction='mean'
            )
        
        # Total loss
        total_loss = (
            self.config.ctr_loss_weight * loss_ctr +
            self.config.ctcvr_loss_weight * loss_ctcvr
        )
        
        losses = {
            'loss': total_loss,
            'loss_ctr': loss_ctr,
            'loss_ctcvr': loss_ctcvr,
        }
        
        # EXPERIMENTAL: Optional auxiliary CVR loss on clicked samples
        # NOT in original ESMM paper - our extension to address weak CVR gradients
        if self.config.use_auxiliary_cvr_loss:
            clicked_mask = click_labels == 1
            if clicked_mask.sum() > 0:
                cvr_clicked = cvr[clicked_mask]
                conv_clicked = conversion_labels[clicked_mask]
                
                if self.config.use_focal_loss:
                    loss_cvr = self._focal_loss(
                        cvr_clicked, conv_clicked,
                        alpha=self.config.focal_alpha,
                        gamma=self.config.focal_gamma,
                    )
                else:
                    loss_cvr = F.binary_cross_entropy(
                        cvr_clicked, conv_clicked, reduction='mean'
                    )
                
                losses['loss_cvr'] = loss_cvr
                losses['loss'] = total_loss + self.config.auxiliary_cvr_weight * loss_cvr
        
        return losses
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Compute focal loss for handling class imbalance.
        
        Focal loss down-weights easy examples, focusing learning on hard ones.
        
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        
        where p_t = p if y=1 else 1-p
        
        Args:
            pred: Predicted probabilities (after sigmoid)
            target: Binary targets
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            
        Returns:
            Mean focal loss
        """
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Compute cross entropy
        ce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Compute p_t
        p_t = torch.where(target == 1, pred, 1 - pred)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * ce
        
        return focal_loss.mean()
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        ctr_params = sum(p.numel() for p in self.ctr_tower.parameters())
        cvr_params = sum(p.numel() for p in self.cvr_tower.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'embedding': embedding_params,
            'ctr_tower': ctr_params,
            'cvr_tower': cvr_params,
            'total': total_params,
        }


class CVRBaseline(nn.Module):
    """Naive CVR baseline that trains on clicked samples only.
    
    This model demonstrates the selection bias problem:
    - Trains on: P(conversion | click, features)
    - Applied to: ALL impressions (including non-clickers)
    
    Comparison with ESMM shows the benefit of entire-space training.
    """
    
    def __init__(self, config: ESMMConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size + 1,
            embedding_dim=config.embed_dim,
            padding_idx=0,
        )
        
        # Single tower for CVR
        pooled_dim = config.embed_dim
        self.tower = Tower(
            input_dim=pooled_dim,
            hidden_dims=config.tower_dims,
            dropout=config.dropout,
            use_bn=config.use_bn,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight[1:])
        for m in self.tower.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing CVR.
        
        Args:
            feature_ids: (batch_size, num_features) tensor
            feature_values: Optional (batch_size, num_features) tensor
            
        Returns:
            Dictionary with 'cvr' probability
        """
        # Embed and pool (same as ESMM)
        embeddings = self.embedding(feature_ids)
        
        if feature_values is not None:
            values = feature_values.unsqueeze(-1)
            embeddings = embeddings * values
        
        mask = (feature_ids != 0).float().unsqueeze(-1)
        masked_embeddings = embeddings * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        
        # CVR prediction
        cvr_logit = self.tower(pooled)
        cvr = torch.sigmoid(cvr_logit)
        
        return {
            'cvr': cvr,
            'cvr_logit': cvr_logit,
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        conversion_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute CVR loss (BCE or focal).
        
        Note: This should be called on CLICKED samples only during training.
        """
        cvr = outputs['cvr']
        
        if self.config.use_focal_loss:
            loss = self._focal_loss(cvr, conversion_labels)
        else:
            loss = F.binary_cross_entropy(cvr, conversion_labels, reduction='mean')
        
        return {'loss': loss, 'loss_cvr': loss}
    
    def _focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Same focal loss as ESMM."""
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        ce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** gamma
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        return (alpha_t * focal_weight * ce).mean()


if __name__ == "__main__":
    # Quick test of ESMM model
    print("Testing ESMM model...")
    
    config = ESMMConfig(
        vocab_size=1000,
        num_features=20,
        embed_dim=8,
        tower_dims=[64, 32],
    )
    
    model = ESMM(config)
    print(f"\nModel architecture:\n{model}")
    print(f"\nParameter counts: {model.get_parameter_count()}")
    
    # Test forward pass
    batch_size = 32
    feature_ids = torch.randint(0, 1000, (batch_size, 20))
    feature_values = torch.rand(batch_size, 20)
    click_labels = torch.randint(0, 2, (batch_size,)).float()
    conversion_labels = (click_labels * torch.randint(0, 2, (batch_size,))).float()
    
    outputs = model(feature_ids, feature_values)
    print(f"\nForward pass outputs:")
    for k, v in outputs.items():
        print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
    
    losses = model.compute_loss(outputs, click_labels, conversion_labels)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("\nESMM model test passed!")

