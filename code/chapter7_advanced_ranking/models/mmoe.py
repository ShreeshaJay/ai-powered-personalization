"""
Chapter 7: MMoE - Multi-gate Mixture-of-Experts for Multi-Task Learning

This implementation supports 4 tasks for comprehensive user engagement modeling:
    1. engagement:  P(played_ratio_pct > 0)   - User started listening
    2. completion:  P(played_ratio_pct >= 50) - User finished listening  
    3. like:        P(user liked item)        - Positive explicit feedback
    4. dislike:     P(user disliked item)     - Negative explicit feedback

Value Function:
    value = w1*P(engagement) + w2*P(completion) + w3*P(like) - w4*P(dislike)

Architecture:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                              MMoE (4 Tasks)                              │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Sparse Features              Dense Features                             │
    │  (uid, item_id,              (numerical,                                │
    │   hour, day_of_week)          historical, is_organic)                    │
    │       │                            │                                     │
    │       ▼                            │                                     │
    │  ┌─────────┐                       │                                     │
    │  │Embedding│                       │                                     │
    │  │ Tables  │                       │                                     │
    │  └────┬────┘                       │                                     │
    │       │                            │                                     │
    │       └────────────┬───────────────┘                                     │
    │                    │                                                     │
    │                    ▼                                                     │
    │            ┌──────────────┐                                              │
    │            │   Concat     │                                              │
    │            │   (Input)    │                                              │
    │            └──────┬───────┘                                              │
    │                   │                                                      │
    │   ┌───────────────┼───────────────┬───────────────┐                      │
    │   │               │               │               │                      │
    │   ▼               ▼               ▼               ▼                      │
    │ ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐                     │
    │ │Expert│        │Expert│        │Expert│   ... │Expert│                     │
    │ │  1  │        │  2  │        │  3  │        │  n  │                     │
    │ └──┬──┘        └──┬──┘        └──┬──┘        └──┬──┘                     │
    │    │              │              │              │                        │
    │    └──────────────┴──────────────┴──────────────┘                        │
    │                         │                                                │
    │  ┌─────────────────┼────────────────┬────────────────┬────────────────┐   │
    │  │                 │                │                │                │   │
    │  ▼                 ▼                ▼                ▼                ▼   │
    │ ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐    │
    │ │ Gate1 │      │ Gate2 │      │ Gate3 │      │ Gate4 │      │ Gate5 │    │
    │ │Engage │      │Compl. │      │PlayRat│      │ Like  │      │Dislike│    │
    │ └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘    │
    │     │              │              │              │              │        │
    │     ▼              ▼              ▼              ▼              ▼        │
    │ ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐    │
    │ │Tower1 │      │Tower2 │      │Tower3 │      │Tower4 │      │Tower5 │    │
    │ │  MLP  │      │  MLP  │      │  MLP  │      │  MLP  │      │  MLP  │    │
    │ └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘    │
    │     │              │              │              │              │        │
    │     ▼              ▼              ▼              ▼              ▼        │
    │ ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐    │
    │ │P(eng) │      │P(comp)│      │E[ratio│      │P(like)│      │P(disl)│    │
    │ │Binary │      │Binary │      │Regress│      │Binary │      │Binary │    │
    │ └───────┘      └───────┘      └───────┘      └───────┘      └───────┘    │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

Key Components:
1. Shared Experts: Learn general representations useful for all tasks
2. Task-specific Gates: Learn to weight expert outputs per task  
3. Task-specific Towers: Final prediction networks per task
4. Mixed task types: 4 binary classification + 1 regression (play_ratio)

Reference:
    Ma et al. "Modeling Task Relationships in Multi-task Learning with 
    Multi-gate Mixture-of-Experts" (KDD 2018)

Usage:
    from models.mmoe import MMoE, MMoEConfig
    
    config = MMoEConfig(
        sparse_features={'uid': 10000, 'item_id': 900000},
        dense_features=['track_length_seconds', 'user_avg_completion', ...],
        num_tasks=5,
        task_names=['engagement', 'completion', 'play_ratio', 'like', 'dislike'],
        num_experts=6,  # 6 experts for 5 tasks
        expert_dims=[256, 128],
        tower_dims=[64, 32],
    )
    model = MMoE(config)
    
    # Note: play_ratio is a regression task (predicts [0,1] ratio)
    # Use MSE loss for play_ratio, BCE for others
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MMoEConfig:
    """Configuration for MMoE model.
    
    Attributes:
        sparse_features: Dict mapping feature name to vocabulary size
        dense_features: List of dense feature names
        num_tasks: Number of tasks (default: 4)
        task_names: Names for each task (default: ['engagement', 'completion', 'like', 'dislike'])
        embed_dim: Embedding dimension for sparse features (default: 16)
        num_experts: Number of shared experts (default: 4)
        expert_dims: Hidden dimensions for expert networks (default: [256, 128])
        tower_dims: Hidden dimensions for task towers (default: [64, 32])
        dropout: Dropout rate (default: 0.1)
        use_bn: Whether to use batch normalization (default: True)
    """
    # Feature configuration
    sparse_features: Dict[str, int] = field(default_factory=dict)
    dense_features: List[str] = field(default_factory=list)
    
    # Multi-task configuration
    # 5 tasks: 4 binary (engagement, completion, like, dislike) + 1 regression (play_ratio)
    num_tasks: int = 5
    task_names: List[str] = field(default_factory=lambda: ['engagement', 'completion', 'play_ratio', 'like', 'dislike'])
    
    # Model hyperparameters
    embed_dim: int = 16
    num_experts: int = 4
    expert_dims: List[int] = field(default_factory=lambda: [256, 128])
    tower_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    use_bn: bool = True
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.sparse_features and not self.dense_features:
            raise ValueError("Must have at least one sparse or dense feature")
        if len(self.task_names) != self.num_tasks:
            raise ValueError(f"task_names length ({len(self.task_names)}) must match num_tasks ({self.num_tasks})")


class Expert(nn.Module):
    """Expert network - a standard MLP."""
    
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
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Gate(nn.Module):
    """Task-specific gating network.
    
    Outputs a softmax distribution over experts for weighted combination.
    """
    
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            gate_weights: Softmax weights over experts (batch_size, num_experts)
        """
        return F.softmax(self.gate(x), dim=-1)


class Tower(nn.Module):
    """Task-specific tower network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts model.
    
    Learns shared representations via multiple expert networks,
    then uses task-specific gates to combine experts and
    task-specific towers for final predictions.
    """
    
    def __init__(self, config: MMoEConfig):
        super().__init__()
        self.config = config
        self.num_tasks = config.num_tasks
        self.task_names = config.task_names
        
        # ===== Embedding layers for sparse features =====
        self.embeddings = nn.ModuleDict()
        
        for feat_name, vocab_size in config.sparse_features.items():
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config.embed_dim,
                padding_idx=0,
            )
        
        # ===== Compute input dimension =====
        num_sparse = len(config.sparse_features)
        num_dense = len(config.dense_features)
        self.input_dim = num_sparse * config.embed_dim + num_dense
        
        # ===== Shared Expert Networks =====
        self.experts = nn.ModuleList([
            Expert(
                input_dim=self.input_dim,
                hidden_dims=config.expert_dims,
                dropout=config.dropout,
                use_bn=config.use_bn,
            )
            for _ in range(config.num_experts)
        ])
        
        # Get expert output dimension
        expert_output_dim = self.experts[0].output_dim
        
        # ===== Task-specific Gates =====
        self.gates = nn.ModuleList([
            Gate(self.input_dim, config.num_experts)
            for _ in range(config.num_tasks)
        ])
        
        # ===== Task-specific Towers =====
        self.towers = nn.ModuleList([
            Tower(
                input_dim=expert_output_dim,
                hidden_dims=config.tower_dims,
                output_dim=1,
                dropout=config.dropout,
                use_bn=config.use_bn,
            )
            for _ in range(config.num_tasks)
        ])
        
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sparse_inputs: Dict mapping feature name to indices (batch_size,)
            dense_inputs: Dense features tensor (batch_size, num_dense) or None
            
        Returns:
            outputs: Dict mapping task name to logits (batch_size, 1)
        """
        # ===== Embed sparse features =====
        embed_list = []
        for feat_name, indices in sparse_inputs.items():
            embed = self.embeddings[feat_name](indices)  # (batch, embed_dim)
            embed_list.append(embed)
        
        # Concatenate embeddings
        sparse_concat = torch.cat(embed_list, dim=1)  # (batch, num_sparse * embed_dim)
        
        # ===== Create input =====
        if dense_inputs is not None:
            x = torch.cat([sparse_concat, dense_inputs], dim=1)  # (batch, input_dim)
        else:
            x = sparse_concat
        
        # ===== Expert outputs =====
        # expert_outputs: List of (batch, expert_output_dim)
        expert_outputs = [expert(x) for expert in self.experts]
        # Stack: (batch, num_experts, expert_output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # ===== Task-specific outputs =====
        outputs = {}
        
        for task_idx, task_name in enumerate(self.task_names):
            # Gate weights: (batch, num_experts)
            gate_weights = self.gates[task_idx](x)
            
            # Weighted combination of experts
            # gate_weights: (batch, num_experts) -> (batch, num_experts, 1)
            # expert_outputs: (batch, num_experts, expert_output_dim)
            # Result: (batch, expert_output_dim)
            gated_output = torch.sum(
                expert_outputs * gate_weights.unsqueeze(-1),
                dim=1
            )
            
            # Tower output
            logits = self.towers[task_idx](gated_output)
            outputs[task_name] = logits
        
        return outputs
    
    def predict_proba(
        self,
        sparse_inputs: Dict[str, torch.Tensor],
        dense_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get probability predictions for all tasks."""
        outputs = self.forward(sparse_inputs, dense_inputs)
        return {name: torch.sigmoid(logits) for name, logits in outputs.items()}


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t = p if y=1, else 1-p
    - alpha_t = alpha if y=1, else 1-alpha
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions (batch_size, 1)
            targets: Binary labels (batch_size, 1)
            
        Returns:
            Focal loss
        """
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Final loss
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning.
    
    Supports:
    - Weighted combination of task losses
    - Focal loss for imbalanced binary tasks
    - MSE loss for regression tasks
    - Uncertainty weighting (Kendall et al.)
    """
    
    def __init__(
        self,
        task_names: List[str],
        task_weights: Optional[Dict[str, float]] = None,
        use_focal_loss: Optional[Dict[str, bool]] = None,
        regression_tasks: Optional[List[str]] = None,  # Tasks using MSE loss
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.task_names = task_names
        self.regression_tasks = regression_tasks or []
        
        # Default equal weights
        if task_weights is None:
            task_weights = {name: 1.0 for name in task_names}
        self.task_weights = task_weights
        
        # Default: no focal loss
        if use_focal_loss is None:
            use_focal_loss = {name: False for name in task_names}
        
        # Create loss functions per task
        self.loss_fns = nn.ModuleDict()
        
        for name in task_names:
            if name in self.regression_tasks:
                # Regression task: use MSE loss
                self.loss_fns[name] = nn.MSELoss()
            elif use_focal_loss.get(name, False):
                # Imbalanced binary task: use focal loss
                self.loss_fns[name] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            else:
                # Standard binary task: use BCE
                self.loss_fns[name] = nn.BCEWithLogitsLoss()
        
        # Uncertainty weighting (learnable)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # log(sigma^2) for each task (learnable)
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in task_names
            })
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-task loss.
        
        Args:
            predictions: Dict mapping task name to logits
            targets: Dict mapping task name to labels
            
        Returns:
            total_loss: Combined loss
            task_losses: Dict of individual task losses
        """
        task_losses = {}
        
        for name in self.task_names:
            if name not in predictions or name not in targets:
                continue
            
            pred = predictions[name]
            target = targets[name]
            
            # For regression tasks, apply sigmoid to convert logits to [0, 1]
            if name in self.regression_tasks:
                pred = torch.sigmoid(pred)
            
            loss = self.loss_fns[name](pred, target)
            task_losses[name] = loss
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L = sum(L_i / (2 * sigma_i^2) + log(sigma_i))
            total_loss = 0
            for name, loss in task_losses.items():
                precision = torch.exp(-self.log_vars[name])
                total_loss += precision * loss + self.log_vars[name]
        else:
            # Simple weighted sum
            total_loss = sum(
                self.task_weights.get(name, 1.0) * loss 
                for name, loss in task_losses.items()
            )
        
        return total_loss, task_losses


class MMoEDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for MMoE multi-task training.
    
    Handles multiple labels for multi-task learning.
    """
    
    def __init__(
        self,
        sparse_data: Dict[str, np.ndarray],
        dense_data: Optional[np.ndarray],
        labels: Dict[str, np.ndarray],
    ):
        """
        Args:
            sparse_data: Dict mapping feature name to encoded indices
            dense_data: 2D array of dense features (n_samples, n_dense)
            labels: Dict mapping task name to binary labels
        """
        self.sparse_data = {k: torch.LongTensor(v) for k, v in sparse_data.items()}
        self.dense_data = torch.FloatTensor(dense_data) if dense_data is not None else None
        self.labels = {k: torch.FloatTensor(v).unsqueeze(1) for k, v in labels.items()}
        
        # Get sample count from first label array
        self.n_samples = len(next(iter(labels.values())))
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor]
    ]:
        sparse = {k: v[idx] for k, v in self.sparse_data.items()}
        dense = self.dense_data[idx] if self.dense_data is not None else None
        labels = {k: v[idx] for k, v in self.labels.items()}
        return sparse, dense, labels


def collate_fn(batch):
    """Custom collate function for MMoE batches."""
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
    
    # Stack labels for each task
    labels_batch = {}
    for key in label_list[0].keys():
        labels_batch[key] = torch.stack([l[key] for l in label_list])
    
    return sparse_batch, dense_batch, labels_batch


# ===== Default configuration for Yambda dataset =====
def get_yambda_mmoe_config(
    uid_vocab_size: int = 10000,
    item_vocab_size: int = 900000,
) -> MMoEConfig:
    """Get default MMoE config for Yambda dataset.
    
    Tasks (5 total: 4 binary + 1 regression):
        - engagement: P(user started listening) - binary
        - completion: P(user listened >= 50%) - binary
        - play_ratio: E[played_ratio_pct / 100] - regression [0, 1]
        - like: P(user liked) - binary, imbalanced
        - dislike: P(user disliked) - binary, imbalanced
    """
    return MMoEConfig(
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
        # 5 tasks: 4 binary + 1 regression for comprehensive engagement modeling
        num_tasks=5,
        task_names=['engagement', 'completion', 'play_ratio', 'like', 'dislike'],
        embed_dim=16,
        num_experts=6,  # 6 experts for 5 tasks
        expert_dims=[256, 128],
        tower_dims=[64, 32],
        dropout=0.1,
        use_bn=True,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )


if __name__ == "__main__":
    # Quick test with 4 tasks
    print("Testing MMoE model with 4 tasks...")
    
    config = MMoEConfig(
        sparse_features={'uid': 1000, 'item_id': 5000},
        dense_features=['feat1', 'feat2', 'feat3'],
        num_tasks=4,
        task_names=['engagement', 'completion', 'like', 'dislike'],
        embed_dim=16,
        num_experts=6,  # More experts for 4 tasks
        expert_dims=[64, 32],
        tower_dims=[32, 16],
    )
    
    model = MMoE(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Tasks: {config.task_names}")
    print(f"Experts: {config.num_experts}")
    
    # Test forward pass
    batch_size = 32
    sparse_inputs = {
        'uid': torch.randint(0, 1000, (batch_size,)),
        'item_id': torch.randint(0, 5000, (batch_size,)),
    }
    dense_inputs = torch.randn(batch_size, 3)
    
    outputs = model(sparse_inputs, dense_inputs)
    probs = model.predict_proba(sparse_inputs, dense_inputs)
    
    print(f"\nOutputs (logits):")
    for task_name, logits in outputs.items():
        print(f"  {task_name}: shape={logits.shape}")
    
    print(f"\nProbabilities:")
    for task_name, prob in probs.items():
        print(f"  {task_name}: mean={prob.mean():.4f}")
    
    # Test value function computation
    print(f"\nExample value function:")
    weights = {'engagement': 1.0, 'completion': 2.0, 'like': 5.0, 'dislike': -10.0}
    value = sum(weights[t] * probs[t].mean().item() for t in config.task_names)
    print(f"  weights: {weights}")
    print(f"  value: {value:.4f}")
    
    print("✓ MMoE test passed!")

