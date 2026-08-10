"""
Chapter 7: Advanced Ranking Models - Configuration
==================================================
Centralized configuration for paths, hyperparameters, and settings.

This chapter builds on Chapter 6's foundation with more sophisticated
neural ranking architectures:
- DCN-V2: Deep & Cross Network V2 (Google)
- DLRM: Deep Learning Recommendation Model (Facebook)
- MMoE: Multi-gate Mixture-of-Experts (multi-task learning)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


# =============================================================================
# Path Configuration
# =============================================================================

# Base paths - UPDATE THESE FOR YOUR ENVIRONMENT
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Data paths (shared with Chapter 6)
DATA_DIR = PROJECT_ROOT / "Dataset" / "Yandex" / "flat"
EMBEDDINGS_PATH = PROJECT_ROOT / "Dataset" / "Yandex" / "embeddings.parquet"
LIKES_PATH = DATA_DIR / "likes.parquet"
DISLIKES_PATH = DATA_DIR / "dislikes.parquet"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


# =============================================================================
# Global Temporal Split Configuration (same as Chapter 6)
# =============================================================================

@dataclass
class GTSConfig:
    """Global Temporal Split configuration."""
    train_days: int = 300
    gap_minutes: int = 30
    test_days: int = 1
    timestamp_bin_seconds: int = 5
    
    @property
    def train_duration_ts(self) -> int:
        seconds_per_day = 86400
        return (self.train_days * seconds_per_day) // self.timestamp_bin_seconds
    
    @property
    def gap_duration_ts(self) -> int:
        gap_seconds = self.gap_minutes * 60
        return gap_seconds // self.timestamp_bin_seconds
    
    @property
    def test_duration_ts(self) -> int:
        seconds_per_day = 86400
        return (self.test_days * seconds_per_day) // self.timestamp_bin_seconds


# =============================================================================
# Feature Configuration (shared across all models)
# =============================================================================

# Sparse features: High cardinality categoricals → Embeddings
SPARSE_FEATURES = ['uid', 'item_id', 'hour_of_day', 'day_of_week']

# Dense features: Numerical and low-cardinality categoricals
DENSE_FEATURES = [
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
]


# =============================================================================
# Model-Specific Configurations
# =============================================================================

@dataclass
class DCNV2HyperParams:
    """DCN-V2 hyperparameters."""
    embed_dim: int = 16
    cross_layers: int = 3
    cross_rank: int = 32
    mlp_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    structure: str = 'parallel'  # 'parallel' or 'stacked'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class DLRMHyperParams:
    """DLRM hyperparameters."""
    embed_dim: int = 16
    bottom_mlp_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    top_mlp_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    interaction_type: str = 'dot'  # 'dot' or 'cat'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class MMoEHyperParams:
    """MMoE hyperparameters."""
    embed_dim: int = 16
    num_experts: int = 4
    expert_dims: List[int] = field(default_factory=lambda: [256, 128])
    tower_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    # Multi-task settings
    task_names: List[str] = field(default_factory=lambda: ['completion', 'like'])
    completion_weight: float = 1.0
    like_weight: float = 2.0
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 4096
    patience: int = 3  # Early stopping patience
    val_split_ratio: float = 0.1
    listen_completion_threshold: int = 50
    verbose: bool = True


# =============================================================================
# Default Configurations
# =============================================================================

DEFAULT_GTS_CONFIG = GTSConfig()
DEFAULT_DCNV2_CONFIG = DCNV2HyperParams()
DEFAULT_DLRM_CONFIG = DLRMHyperParams()
DEFAULT_MMOE_CONFIG = MMoEHyperParams()
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        'paths': {
            'data_dir': str(DATA_DIR),
            'output_dir': str(OUTPUT_DIR),
        },
        'gts': {
            'train_days': DEFAULT_GTS_CONFIG.train_days,
            'gap_minutes': DEFAULT_GTS_CONFIG.gap_minutes,
            'test_days': DEFAULT_GTS_CONFIG.test_days,
        },
        'features': {
            'sparse': SPARSE_FEATURES,
            'dense': DENSE_FEATURES,
        },
    }


def ensure_directories():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

