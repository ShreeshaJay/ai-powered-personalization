"""
Chapter 9: Adtech Modeling - Configuration
==========================================
Centralized configuration for paths, hyperparameters, and settings.

This chapter covers adtech-specific modeling techniques:
- ESMM: Entire Space Multi-Task Model for CVR prediction
- Delayed Feedback: Handling attribution windows
- Calibration: Ensuring accurate probabilities for bidding
- Split Neural Networks: Privacy-preserving federated learning

Datasets:
- Ali-CCP: Alibaba Click and Conversion Prediction (primary)
- FedAds: Alibaba Federated Advertising Dataset (secondary)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


# =============================================================================
# Path Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Ali-CCP Dataset
ALI_CCP_DIR = PROJECT_ROOT / "Dataset" / "Ali-CCP Entire State Model"
ALI_CCP_SKELETON = ALI_CCP_DIR / "sample_skeleton_train.csv"
ALI_CCP_FEATURES = ALI_CCP_DIR / "common_features_train.csv"

# FedAds Dataset
FEDADS_DIR = PROJECT_ROOT / "Dataset" / "Alibaba Federated Ads"
FEDADS_ALIGNED = FEDADS_DIR / "sample_train_aligned.csv"
FEDADS_UNALIGNED = FEDADS_DIR / "sample_train_unaligned.csv"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Processed data cache
CACHE_DIR = BASE_DIR / "cache"


# =============================================================================
# Ali-CCP Feature Configuration
# =============================================================================

# Ali-CCP uses a unique sparse feature format where feature_id and value
# are concatenated. We'll hash feature_ids to a fixed vocabulary.
ALI_CCP_VOCAB_SIZE = 100_000  # Hash feature IDs to this size
ALI_CCP_MAX_FEATURES_PER_SAMPLE = 50  # Max features to extract per sample


# =============================================================================
# FedAds Feature Configuration
# =============================================================================

# Local party features (publisher side)
FEDADS_LOCAL_FEATURES = [
    'l_i_fea_1', 'l_i_fea_2', 'l_i_fea_3', 'l_i_fea_4', 'l_i_fea_5',
    'l_i_fea_6', 'l_i_fea_7', 'l_i_fea_8', 'l_i_fea_9', 'l_i_fea_10',
    'l_u_fea_1', 'l_u_fea_2', 'l_u_fea_3', 'l_u_fea_4', 'l_u_fea_5', 'l_u_fea_6',
    'l_c_fea',
]

# Federated party features (advertiser side)
FEDADS_FEDERATED_FEATURES = [
    'f_u_fea_1', 'f_u_fea_2',
    'f_uc_fea_1', 'f_uc_fea_2',
    'f_c',
]

# All FedAds features
FEDADS_ALL_FEATURES = FEDADS_LOCAL_FEATURES + FEDADS_FEDERATED_FEATURES


# =============================================================================
# Model Configurations
# =============================================================================

@dataclass
class ESMMConfig:
    """ESMM (Entire Space Multi-Task Model) configuration.
    
    Architecture:
        - Shared embedding layer for all sparse features
        - CTR Tower: Predicts P(click | impression)
        - CVR Tower: Predicts P(conversion | click)
        - CTCVR = CTR × CVR (trained end-to-end)
    
    Reference:
        Ma et al. "Entire Space Multi-Task Model" (SIGIR 2018)
    """
    # Feature configuration
    vocab_size: int = 100_000  # Hash bucket size for sparse features
    num_features: int = 50  # Number of feature slots per sample
    
    # Embedding
    embed_dim: int = 16
    
    # Tower architecture (shared by CTR and CVR towers)
    tower_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    use_bn: bool = True
    
    # Loss weights
    ctr_loss_weight: float = 1.0
    ctcvr_loss_weight: float = 1.0
    
    # EXPERIMENTAL: Auxiliary CVR loss (on clicked samples only)
    # NOT in original ESMM paper - our own extension based on gradient analysis.
    # Hypothesis: May help when CTR is very low and CVR tower gets weak gradients.
    # Disabled by default to match original paper.
    use_auxiliary_cvr_loss: bool = False
    auxiliary_cvr_weight: float = 0.1
    
    # Focal loss for handling class imbalance
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class SplitNNConfig:
    """Split Neural Network configuration for federated learning.
    
    Defaults match the FedAds paper (Wei et al., SIGIR 2023, Sec 5.1.4).
    
    Architecture:
        f_N (non-label/publisher): embed + DNN[128, 32]
        f_L bottom (label/advertiser): embed + DNN[256, 128]
        f_L top (aggregation): single layer (32+128=160 -> 1)
    
    Privacy:
        Only hidden representations are exchanged, not raw features.
    """
    # Vocabulary sizes for hashed features
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
    
    # f_L top (aggregation): single layer, no hidden aggregation layers
    agg_layers: List[int] = field(default_factory=list)
    
    # Paper does not mention dropout or weight decay
    dropout: float = 0.0
    use_bn: bool = True
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # Standard cross-entropy (paper does not use focal loss)
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class CVRBaselineConfig:
    """Naive CVR baseline configuration (for comparison with ESMM).
    
    This model trains on CLICKED samples only, demonstrating selection bias.
    """
    vocab_size: int = 100_000
    num_features: int = 50
    embed_dim: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    use_bn: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass 
class TrainingConfig:
    """General training configuration."""
    epochs: int = 10
    batch_size: int = 4096
    patience: int = 3  # Early stopping patience
    val_split_ratio: float = 0.1
    
    # Data sampling for development
    dev_sample_size: int = 2_000_000  # 2M samples for quick experiments
    use_full_data: bool = False  # Set True for final training
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    verbose: bool = True


@dataclass
class CalibrationConfig:
    """Calibration configuration for bid optimization."""
    # Number of bins for ECE calculation
    n_bins: int = 10
    
    # Calibration method: 'platt', 'isotonic', 'temperature'
    method: str = 'isotonic'
    
    # Validation split for calibration fitting
    calibration_val_ratio: float = 0.2


@dataclass
class DelayedFeedbackConfig:
    """Delayed feedback modeling configuration."""
    # Attribution window in hours
    attribution_window_hours: int = 24
    
    # Method: 'cutoff', 'importance_weighting', 'dfm'
    method: str = 'importance_weighting'
    
    # For importance weighting
    min_weight: float = 0.1  # Minimum weight for recent samples


# =============================================================================
# Default Configurations
# =============================================================================

DEFAULT_ESMM_CONFIG = ESMMConfig()
DEFAULT_SPLIT_NN_CONFIG = SplitNNConfig()
DEFAULT_CVR_BASELINE_CONFIG = CVRBaselineConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_CALIBRATION_CONFIG = CalibrationConfig()
DEFAULT_DELAYED_FEEDBACK_CONFIG = DelayedFeedbackConfig()


# =============================================================================
# Utility Functions
# =============================================================================

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        'paths': {
            'ali_ccp_dir': str(ALI_CCP_DIR),
            'fedads_dir': str(FEDADS_DIR),
            'output_dir': str(OUTPUT_DIR),
        },
        'esmm': {
            'vocab_size': DEFAULT_ESMM_CONFIG.vocab_size,
            'embed_dim': DEFAULT_ESMM_CONFIG.embed_dim,
            'tower_dims': DEFAULT_ESMM_CONFIG.tower_dims,
        },
        'training': {
            'epochs': DEFAULT_TRAINING_CONFIG.epochs,
            'batch_size': DEFAULT_TRAINING_CONFIG.batch_size,
            'dev_sample_size': DEFAULT_TRAINING_CONFIG.dev_sample_size,
        },
    }


def ensure_directories():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR, CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def validate_data_paths():
    """Check if required data files exist."""
    missing = []
    
    if not ALI_CCP_SKELETON.exists():
        missing.append(f"Ali-CCP skeleton: {ALI_CCP_SKELETON}")
    if not ALI_CCP_FEATURES.exists():
        missing.append(f"Ali-CCP features: {ALI_CCP_FEATURES}")
    if not FEDADS_ALIGNED.exists():
        missing.append(f"FedAds aligned: {FEDADS_ALIGNED}")
        
    if missing:
        print("WARNING: Missing data files:")
        for m in missing:
            print(f"  - {m}")
        return False
    return True


if __name__ == "__main__":
    # Print configuration summary
    print("Chapter 9: Adtech Modeling Configuration")
    print("=" * 50)
    
    print("\nData Paths:")
    print(f"  Ali-CCP Dir: {ALI_CCP_DIR}")
    print(f"  FedAds Dir: {FEDADS_DIR}")
    
    print("\nESMM Config:")
    print(f"  Vocab Size: {DEFAULT_ESMM_CONFIG.vocab_size:,}")
    print(f"  Embed Dim: {DEFAULT_ESMM_CONFIG.embed_dim}")
    print(f"  Tower Dims: {DEFAULT_ESMM_CONFIG.tower_dims}")
    
    print("\nValidating paths...")
    validate_data_paths()

