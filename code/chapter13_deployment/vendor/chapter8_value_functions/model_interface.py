"""
Model Interface for Chapter 7 Multi-Task Model

This module defines the expected interface for the multi-task model
trained in Chapter 7. The OrderingPipeline expects models to conform
to this interface.

When implementing Chapter 7, ensure your model:
1. Inherits from MultiTaskModelBase or implements the same interface
2. Returns a dictionary with the expected keys
3. Saves artifacts in the expected format
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskModelConfig:
    """Configuration for multi-task ranking model.
    
    Attributes:
        sparse_features: List of sparse (categorical) feature names
        dense_features: List of dense (numerical) feature names  
        sparse_cardinalities: Dict of feature_name -> cardinality
        embed_dim: Embedding dimension for sparse features
        hidden_dims: Hidden layer dimensions for shared network
        task_hidden_dims: Hidden dimensions for task-specific towers
        dropout: Dropout rate
    """
    sparse_features: list
    dense_features: list
    sparse_cardinalities: dict
    embed_dim: int = 16
    hidden_dims: list = None
    task_hidden_dims: list = None
    dropout: float = 0.1
    
    def __post_init__(self):
        self.hidden_dims = self.hidden_dims or [128, 64]
        self.task_hidden_dims = self.task_hidden_dims or [32]
    
    def to_dict(self) -> dict:
        return {
            'sparse_features': self.sparse_features,
            'dense_features': self.dense_features,
            'sparse_cardinalities': self.sparse_cardinalities,
            'embed_dim': self.embed_dim,
            'hidden_dims': self.hidden_dims,
            'task_hidden_dims': self.task_hidden_dims,
            'dropout': self.dropout,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MultiTaskModelConfig':
        return cls(**d)


class MultiTaskModelBase(ABC, nn.Module):
    """
    Abstract base class for multi-task ranking models.
    
    Chapter 7 implementations should inherit from this class
    to ensure compatibility with the Chapter 8 ordering pipeline.
    """
    
    @abstractmethod
    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-task predictions.
        
        Args:
            sparse_features: Tensor of shape (batch, num_sparse_features)
                Contains encoded categorical feature indices
            dense_features: Tensor of shape (batch, num_dense_features)
                Contains numerical feature values
        
        Returns:
            Dictionary with keys:
                'p_listen': Tensor (batch,) - P(listen) probability
                'p_like': Tensor (batch,) - P(like|listen) probability
                'e_engagement': Tensor (batch,) - E[play_ratio] in [0, 1]
                'p_dislike': Tensor (batch,) - P(dislike|listen) probability (optional)
        
        All outputs should be in range [0, 1] after sigmoid activation.
        """
        pass
    
    def save(self, output_dir: str, config: MultiTaskModelConfig):
        """
        Save model and config to directory.
        
        Args:
            output_dir: Directory to save artifacts
            config: Model configuration
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = output_path / "mtl_model.pt"
        torch.save(self.state_dict(), model_path)
        logger.info(f"Saved model weights to {model_path}")
        
        # Save config
        config_path = output_path / "mtl_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    @classmethod
    def load(cls, model_dir: str, device: str = 'cpu') -> 'MultiTaskModelBase':
        """
        Load model from directory.
        
        Args:
            model_dir: Directory containing model artifacts
            device: Device to load model to
        
        Returns:
            Loaded model instance
        """
        model_path = Path(model_dir)
        
        # Load config
        config_path = model_path / "mtl_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = MultiTaskModelConfig.from_dict(config_dict)
        
        # Create model instance
        model = cls(config)
        
        # Load weights
        weights_path = model_path / "mtl_model.pt"
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {model_dir}")
        return model


# ============================================================================
# Example Implementation (Reference for Chapter 7)
# ============================================================================

class SharedBottomMultiTask(MultiTaskModelBase):
    """
    Shared-Bottom Multi-Task Learning Model.
    
    Architecture:
        [Sparse Embeddings + Dense Features]
                      |
                [Shared Network]
                      |
            +---------+---------+---------+
            |         |         |         |
        [Listen]  [Like]  [Engagement] [Dislike]
          Head      Head      Head       Head
    
    This is a reference implementation for Chapter 7.
    """
    
    def __init__(self, config: MultiTaskModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers for sparse features
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for feat_name in config.sparse_features:
            cardinality = config.sparse_cardinalities[feat_name]
            self.embeddings[feat_name] = nn.Embedding(
                cardinality + 1,  # +1 for unknown
                config.embed_dim,
            )
            total_embed_dim += config.embed_dim
        
        # Calculate input dimension
        input_dim = total_embed_dim + len(config.dense_features)
        
        # Shared network
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim
        self.shared_network = nn.Sequential(*shared_layers)
        
        shared_output_dim = config.hidden_dims[-1]
        
        # Task-specific towers
        def make_tower(output_dim: int = 1):
            layers = []
            prev = shared_output_dim
            for hidden in config.task_hidden_dims:
                layers.extend([
                    nn.Linear(prev, hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ])
                prev = hidden
            layers.append(nn.Linear(prev, output_dim))
            return nn.Sequential(*layers)
        
        self.listen_tower = make_tower(1)
        self.like_tower = make_tower(1)
        self.engagement_tower = make_tower(1)
        self.dislike_tower = make_tower(1)
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Embed sparse features
        embedded = []
        for i, feat_name in enumerate(self.config.sparse_features):
            feat_idx = sparse_features[:, i]
            embedded.append(self.embeddings[feat_name](feat_idx))
        
        # Concatenate embeddings and dense features
        x = torch.cat(embedded + [dense_features], dim=1)
        
        # Shared representation
        shared_out = self.shared_network(x)
        
        # Task-specific outputs
        p_listen = torch.sigmoid(self.listen_tower(shared_out)).squeeze(-1)
        p_like = torch.sigmoid(self.like_tower(shared_out)).squeeze(-1)
        e_engagement = torch.sigmoid(self.engagement_tower(shared_out)).squeeze(-1)
        p_dislike = torch.sigmoid(self.dislike_tower(shared_out)).squeeze(-1)
        
        return {
            'p_listen': p_listen,
            'p_like': p_like,
            'e_engagement': e_engagement,
            'p_dislike': p_dislike,
        }


def load_multitask_model(
    model_path: str,
    config: dict,
    model_class: type = SharedBottomMultiTask,
    device: str = 'cpu',
) -> nn.Module:
    """
    Load a multi-task model from saved weights.
    
    This function is called by the OrderingPipeline to load
    Chapter 7 model artifacts.
    
    Args:
        model_path: Path to model weights (.pt file)
        config: Model configuration dictionary
        model_class: Model class to instantiate
        device: Device to load model to
    
    Returns:
        Loaded model instance
    """
    # Create config object
    model_config = MultiTaskModelConfig.from_dict(config)
    
    # Instantiate model
    model = model_class(model_config)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model


# ============================================================================
# Training Utilities (for Chapter 7)
# ============================================================================

def compute_multitask_loss(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    task_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Compute weighted multi-task loss.
    
    Args:
        predictions: Model output dictionary
        labels: Ground truth labels dictionary with same keys
        task_weights: Optional weights for each task loss
    
    Returns:
        Combined loss tensor
    """
    task_weights = task_weights or {
        'p_listen': 1.0,
        'p_like': 1.0,
        'e_engagement': 1.0,
        'p_dislike': 0.5,
    }
    
    total_loss = 0.0
    
    # Binary cross-entropy for classification tasks
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    
    for task_name, weight in task_weights.items():
        if task_name not in predictions or task_name not in labels:
            continue
        
        pred = predictions[task_name]
        label = labels[task_name]
        
        if task_name == 'e_engagement':
            # Regression task
            loss = mse(pred, label)
        else:
            # Classification task
            loss = bce(pred, label.float())
        
        total_loss += weight * loss
    
    return total_loss

