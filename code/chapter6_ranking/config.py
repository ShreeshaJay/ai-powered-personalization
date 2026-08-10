"""
Chapter 6: Ranking - Configuration
==================================
Centralized configuration for paths, hyperparameters, and settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# =============================================================================
# Path Configuration
# =============================================================================

# Base paths - UPDATE THESE FOR YOUR ENVIRONMENT
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "Dataset" / "Yandex" / "flat"
EMBEDDINGS_PATH = PROJECT_ROOT / "Dataset" / "Yandex" / "embeddings.parquet"
ALBUM_MAPPING_PATH = PROJECT_ROOT / "Dataset" / "Yandex" / "album_item_mapping.parquet"
ARTIST_MAPPING_PATH = PROJECT_ROOT / "Dataset" / "Yandex" / "artist_item_mapping.parquet"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


# =============================================================================
# Global Temporal Split Configuration
# =============================================================================

@dataclass
class GTSConfig:
    """
    Global Temporal Split configuration.
    
    Default values match the Yambda paper's evaluation protocol:
    - Training: First 300 days
    - Gap: 30 minutes (to prevent information leakage)
    - Test: Next 1 day
    
    Note: Timestamps in Yambda are delta values binned into 5-second intervals.
    """
    train_days: int = 300
    gap_minutes: int = 30
    test_days: int = 1
    timestamp_bin_seconds: int = 5  # Each timestamp unit = 5 seconds
    
    @property
    def train_duration_ts(self) -> int:
        """Training duration in timestamp units."""
        seconds_per_day = 86400
        return (self.train_days * seconds_per_day) // self.timestamp_bin_seconds
    
    @property
    def gap_duration_ts(self) -> int:
        """Gap duration in timestamp units."""
        gap_seconds = self.gap_minutes * 60
        return gap_seconds // self.timestamp_bin_seconds
    
    @property
    def test_duration_ts(self) -> int:
        """Test duration in timestamp units."""
        seconds_per_day = 86400
        return (self.test_days * seconds_per_day) // self.timestamp_bin_seconds


# =============================================================================
# Model Hyperparameters
# =============================================================================

@dataclass
class XGBoostConfig:
    """XGBoost model hyperparameters."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 1.0  # Adjust for class imbalance (n_neg / n_pos)
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 10
    
    # Feature configuration
    categorical_columns: tuple = ('uid', 'item_id', 'is_organic')
    # Note: raw 'timestamp' excluded - it's a monotonic index, not a meaningful feature
    # Time patterns are captured via derived features (hour_of_day, day_of_week)
    numerical_columns: tuple = ('track_length_seconds',)
    high_cardinality_threshold: int = 10000  # Use frequency encoding above this
    
    def to_xgb_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dict."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False
        }


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data sampling (set to None for full data)
    sample_frac: Optional[float] = None
    
    # Validation split (from end of training data, temporal)
    val_split_ratio: float = 0.1
    
    # Label configuration
    listen_completion_threshold: int = 50  # played_ratio_pct >= this = positive
    
    # Logging
    verbose: bool = True
    log_every_n_rounds: int = 10


# =============================================================================
# Default Configurations
# =============================================================================

# Create default instances
DEFAULT_GTS_CONFIG = GTSConfig()
DEFAULT_XGBOOST_CONFIG = XGBoostConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary (for logging/saving)."""
    return {
        'paths': {
            'data_dir': str(DATA_DIR),
            'output_dir': str(OUTPUT_DIR),
            'models_dir': str(MODELS_DIR),
        },
        'gts': {
            'train_days': DEFAULT_GTS_CONFIG.train_days,
            'gap_minutes': DEFAULT_GTS_CONFIG.gap_minutes,
            'test_days': DEFAULT_GTS_CONFIG.test_days,
        },
        'xgboost': {
            'n_estimators': DEFAULT_XGBOOST_CONFIG.n_estimators,
            'max_depth': DEFAULT_XGBOOST_CONFIG.max_depth,
            'learning_rate': DEFAULT_XGBOOST_CONFIG.learning_rate,
        },
        'training': {
            'sample_frac': DEFAULT_TRAINING_CONFIG.sample_frac,
            'val_split_ratio': DEFAULT_TRAINING_CONFIG.val_split_ratio,
            'listen_completion_threshold': DEFAULT_TRAINING_CONFIG.listen_completion_threshold,
        }
    }


def ensure_directories():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

