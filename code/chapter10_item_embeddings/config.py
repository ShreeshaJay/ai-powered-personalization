"""
Configuration for Chapter 10: Item Embeddings

Centralizes all paths, model settings, and hyperparameters for reproducibility.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================================
# Base Paths
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
CHAPTER_ROOT = Path(__file__).parent

# Dataset-specific paths
MIND_DATA_PATH = DATASET_ROOT / "Microsoft News" / "MINDsmall_train"
IJCAI_DATA_PATH = DATASET_ROOT / "Alibaba IJCAI-18 Sponsored Search CVR"
AMAZON_KDD_DATA_PATH = DATASET_ROOT / "Amazon KDD 2023"
YANDEX_DATA_PATH = DATASET_ROOT / "Yandex"
SIGIR_DATA_PATH = DATASET_ROOT / "SIGIR-ecom-data-challenge"
MUSIC4ALL_DATA_PATH = DATASET_ROOT / "Music4all Onion"

# Output paths
CACHE_DIR = CHAPTER_ROOT / "cache"
OUTPUTS_DIR = CHAPTER_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Ensure directories exist
for dir_path in [CACHE_DIR, OUTPUTS_DIR, EMBEDDINGS_DIR, MODELS_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Section 10.1: Pre-trained Text Encoder Baselines
# ============================================================================

@dataclass
class TextEncoderConfig:
    """Configuration for text-based item embedding generation."""
    
    # Model settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # SBERT baseline
    max_seq_length: int = 128
    batch_size: int = 64
    device: str = "cuda"  # or "cpu"
    
    # Alternative models for comparison
    alternative_models: List[str] = field(default_factory=lambda: [
        "sentence-transformers/all-mpnet-base-v2",  # Higher quality SBERT
        "rexbert-base",  # E-commerce domain-specialized (thebajajra/RexBERT-base)
    ])
    
    # Text preprocessing
    lowercase: bool = True
    remove_special_chars: bool = False
    max_text_length: int = 512  # Before tokenization
    
    # Output settings
    normalize_embeddings: bool = True  # L2 normalization
    embedding_dim: int = 384  # MiniLM-L6 output dim
    
    # Caching
    use_cache: bool = True
    cache_dir: Path = CACHE_DIR / "text_encoders"


@dataclass
class MINDConfig:
    """Configuration for MIND News Dataset."""
    
    # Data paths
    data_dir: Path = MIND_DATA_PATH
    news_file: str = "news.tsv"
    behaviors_file: str = "behaviors.tsv"
    entity_embedding_file: str = "entity_embedding.vec"
    relation_embedding_file: str = "relation_embedding.vec"
    
    # Text field selection for encoding
    use_title: bool = True
    use_abstract: bool = True
    use_category: bool = True
    use_subcategory: bool = True
    
    # Text combination template
    text_template: str = "{title} [SEP] {abstract} [SEP] {category} {subcategory}"
    
    # Sampling (for faster experimentation)
    sample_size: Optional[int] = None  # None = use all data
    random_seed: int = 42
    
    # Output
    output_file: str = "mind_item_embeddings.npz"


@dataclass
class IJCAIConfig:
    """Configuration for Alibaba IJCAI-18 CVR Dataset."""
    
    # Data paths
    data_dir: Path = IJCAI_DATA_PATH
    train_file: str = "round1_ijcai_18_train_20180301.txt"
    test_a_file: str = "round1_ijcai_18_test_a_20180301.txt"
    test_b_file: str = "round1_ijcai_18_test_b_20180418.txt"
    
    # Text serialization strategy for structured data
    serialization_template: str = (
        "Category: {categories} | "
        "Properties: {properties} | "
        "Brand: {brand} | "
        "Price: {price_level} | "
        "Sales: {sales_level} | "
        "City: {city}"
    )
    
    # Feature selection
    use_item_features: bool = True
    use_shop_features: bool = False  # Shop features for items
    
    # Query-side text (for bi-encoder evaluation)
    query_field: str = "predict_category_property"  # From context table
    query_template: str = "User interested in: {categories_properties}"
    
    # Sampling
    sample_size: Optional[int] = None  # None = use all
    min_interactions: int = 1  # Filter items with < N interactions
    
    # Output
    output_file: str = "ijcai_item_embeddings.npz"


@dataclass
class AmazonKDDConfig:
    """Configuration for Amazon KDD Cup 2023 Dataset (English/UK locale)."""

    # Data paths
    data_dir: Path = AMAZON_KDD_DATA_PATH
    products_file: str = "products_train.csv"
    sessions_file: str = "sessions_train.csv"

    # Locale filter
    locale: str = "UK"  # English products only

    # Text fields for encoding
    use_title: bool = True
    use_brand: bool = True
    use_description: bool = True
    use_color: bool = False      # 24% null — optional enrichment
    use_material: bool = False   # 40% null — optional enrichment

    # Text combination template
    text_template: str = "{title} [SEP] {brand} [SEP] {description}"

    # Session filtering
    min_session_length: int = 3  # ignore very short sessions

    # Output
    output_file: str = "amazon_kdd_item_embeddings.npz"


@dataclass
class BiEncoderEvalConfig:
    """Configuration for bi-encoder evaluation (query vs. item)."""
    
    # Evaluation metrics
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    
    # Relevance definition
    relevance_threshold: str = "conversion"  # "conversion" or "click"
    
    # Retrieval settings
    use_faiss: bool = True  # Use FAISS for fast nearest neighbor search
    faiss_index_type: str = "IndexFlatIP"  # Inner product (for normalized vectors)
    
    # Temporal evaluation split
    # IMPORTANT: Always split chronologically to avoid data leakage.
    # Train on earlier timestamps, evaluate on later timestamps.
    temporal_split: bool = True  # Enforce chronological splitting
    train_ratio: float = 0.7    # Earliest 70% of timestamps for training
    val_ratio: float = 0.1      # Next 10% for validation
    test_ratio: float = 0.2     # Latest 20% for testing
    
    # Output
    results_file: str = "biencoder_evaluation_results.json"


# ============================================================================
# Section 10.2: Item2Vec (Collaborative Embeddings from Sequences)
# ============================================================================

@dataclass
class Item2VecConfig:
    """Configuration for Item2Vec (Word2Vec on item sequences)."""

    # Word2Vec hyperparameters
    embedding_dim: int = 128
    window_size: int = 5        # context window (items within this distance are "co-occurring")
    min_count: Optional[int] = None   # if set, overrides percentile-based cutoff
    min_count_percentile: float = 5.0  # drop items below this percentile of frequency distribution
    sg: int = 1                 # 1 = Skip-gram, 0 = CBOW
    negative: int = 10          # number of negative samples
    epochs: int = 10
    workers: int = 4
    seed: int = 42

    # Output
    output_subdir: str = "item2vec"


@dataclass
class YandexConfig:
    """Configuration for Yandex Yambda music dataset."""

    data_dir: Path = YANDEX_DATA_PATH
    listens_file: str = "listens.parquet"
    embeddings_file: str = "embeddings.parquet"

    # Filtering
    min_played_ratio: int = 50   # only count listens where >= 50% of track was played
    min_listens_per_user: int = 10
    max_listens_per_user: int = 5000  # cap very active users for memory

    # Sequence construction strategy for Item2Vec:
    #   "full_history" — one sequence per user (entire chronological history).
    #       The Word2Vec window_size naturally limits co-occurrence distance.
    #       Simpler, avoids arbitrary gap threshold; follows original Item2Vec paper.
    #   "session" — split each user's history into sessions by time gaps.
    #       Prevents cross-session co-occurrence (respects intent shifts).
    sequence_mode: str = "full_history"
    session_gap_seconds: int = 1800  # only used when sequence_mode="session"

    # Output
    output_file: str = "yandex_item2vec_embeddings.npz"


# ============================================================================
# Section 10.4: Multi-Modal Fusion (Music4all)
# ============================================================================

@dataclass
class Music4allConfig:
    """Configuration for Music4all-Onion dataset."""

    data_dir: Path = MUSIC4ALL_DATA_PATH

    # Audio modality — i-vectors (100-dim, from 256 GMM components)
    audio_file: str = "id_ivec256.tsv.bz2"
    audio_dim: int = 100

    # Lyrics modality — averaged Word2Vec (300-dim)
    lyrics_file: str = "id_lyrics_word2vec.tsv.bz2"
    lyrics_dim: int = 300

    # Genre modality — TF-IDF over genre labels (685-dim, sparse)
    genre_file: str = "id_genres_tf-idf.tsv.bz2"
    genre_dim: int = 685

    # User-track interactions (for co-listen evaluation)
    interactions_file: str = "userid_trackid_count.tsv.bz2"

    # Interaction filtering
    min_listen_count: int = 2       # min plays for a user-track pair to count
    min_tracks_per_user: int = 10   # users with fewer tracks are excluded
    max_tracks_per_user: int = 500  # cap heavy users for balanced evaluation

    # Output
    output_file: str = "music4all_embeddings.npz"


@dataclass
class MultiModalFusionConfig:
    """Configuration for multi-modal fusion training."""

    # Shared projection dimension (CLIP-style)
    shared_dim: int = 128

    # Projection head architecture
    hidden_dim: int = 256       # MLP hidden layer (0 = linear projection only)
    dropout: float = 0.1

    # CLIP training hyperparameters
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07  # InfoNCE temperature (learnable if init > 0)
    learnable_temperature: bool = True

    # PCA baseline
    pca_dim: int = 128

    # Evaluation
    eval_num_users: int = 10_000   # sample users for co-listen eval
    max_queries: int = 5_000
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])

    # Training
    random_seed: int = 42
    device: str = "cuda"

    # Output
    output_subdir: str = "multimodal_fusion"


# ============================================================================
# Appendix: Superlinked Multi-Modal Integration (MIND)
# (Config kept here; script lives in appendix_superlinked/)
# ============================================================================

@dataclass
class SuperlinkedConfig:
    """Configuration for Superlinked multi-modal integration demo."""

    # Text encoding model (same as 10.1 for fair comparison)
    text_model: str = "all-MiniLM-L6-v2"

    # Weight presets for evaluation: name -> {space: weight}
    weight_presets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "text_only": {"text": 1.0, "category": 0.0, "subcategory": 0.0},
        "balanced": {"text": 1.0, "category": 0.5, "subcategory": 0.3},
        "category_heavy": {"text": 0.3, "category": 1.0, "subcategory": 0.5},
        "subcategory_heavy": {"text": 0.5, "category": 0.3, "subcategory": 1.0},
    })

    # Evaluation
    max_queries: int = 2_000
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    coclick_min_support: int = 2
    random_seed: int = 42

    # Qualitative examples: sample articles per category for NN inspection
    qualitative_samples: int = 3

    # Output
    output_subdir: str = "superlinked"


# ============================================================================
# Section 10.3: Contrastive Fine-Tuning
# ============================================================================

@dataclass
class ContrastiveFineTuneConfig:
    """Configuration for contrastive fine-tuning of text encoders."""

    # Base model to fine-tune
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 128

    # Loss function: MultipleNegativesRankingLoss (in-batch negatives)
    loss_type: str = "mnr"  # "mnr" or "triplet"

    # Hard negatives (optional, improves quality)
    use_hard_negatives: bool = True
    hard_neg_per_positive: int = 1

    # Data splitting — train on most pairs, hold out some for validation
    train_fraction: float = 0.9
    random_seed: int = 42

    # Amazon KDD specific
    amazon_min_session_length: int = 3
    amazon_max_train_pairs: int = 500_000

    # MIND specific
    mind_min_coclick_support: int = 2
    mind_max_train_pairs: int = 200_000

    # Evaluation during training
    eval_steps: int = 500
    save_best_model: bool = True

    # Output paths (relative to MODELS_DIR)
    output_subdir: str = "contrastive_finetuned"


# ============================================================================
# Global Defaults
# ============================================================================

# Default configurations for Section 10.1
DEFAULT_TEXT_ENCODER = TextEncoderConfig()
DEFAULT_MIND_CONFIG = MINDConfig()
DEFAULT_IJCAI_CONFIG = IJCAIConfig()
DEFAULT_AMAZON_KDD_CONFIG = AmazonKDDConfig()
DEFAULT_BIENCODER_EVAL = BiEncoderEvalConfig()

# Default configurations for Section 10.2
DEFAULT_ITEM2VEC_CONFIG = Item2VecConfig()
DEFAULT_YANDEX_CONFIG = YandexConfig()

# Default configuration for Section 10.3
DEFAULT_CONTRASTIVE_CONFIG = ContrastiveFineTuneConfig()

# Default configurations for Section 10.4
DEFAULT_MUSIC4ALL_CONFIG = Music4allConfig()
DEFAULT_MULTIMODAL_CONFIG = MultiModalFusionConfig()

# Default configuration for Appendix (Superlinked)
DEFAULT_SUPERLINKED_CONFIG = SuperlinkedConfig()


# ============================================================================
# Utility Functions
# ============================================================================

def get_config(dataset: str = "mind", section: str = "10.1") -> Any:
    """
    Get configuration for a specific dataset and chapter section.
    
    Args:
        dataset: Dataset name ("mind", "ijcai", "amazon", "amazon_kdd")
        section: Chapter section ("10.1", "10.2", "10.3", etc.)
    
    Returns:
        Configuration dataclass instance
    """
    if section == "10.1":
        if dataset.lower() == "mind":
            return DEFAULT_MIND_CONFIG
        elif dataset.lower() == "ijcai":
            return DEFAULT_IJCAI_CONFIG
        elif dataset.lower() in ("amazon", "amazon_kdd"):
            return DEFAULT_AMAZON_KDD_CONFIG
        else:
            raise ValueError(f"Unknown dataset for section 10.1: {dataset}")
    elif section == "10.2":
        if dataset.lower() in ("amazon", "amazon_kdd"):
            return DEFAULT_ITEM2VEC_CONFIG, DEFAULT_AMAZON_KDD_CONFIG
        elif dataset.lower() in ("yandex", "yambda"):
            return DEFAULT_ITEM2VEC_CONFIG, DEFAULT_YANDEX_CONFIG
        else:
            raise ValueError(f"Unknown dataset for section 10.2: {dataset}")
    elif section == "10.3":
        return DEFAULT_CONTRASTIVE_CONFIG
    elif section == "10.4":
        return DEFAULT_MUSIC4ALL_CONFIG, DEFAULT_MULTIMODAL_CONFIG
    elif section in ("10.5", "superlinked", "appendix_superlinked"):
        return DEFAULT_SUPERLINKED_CONFIG, DEFAULT_MIND_CONFIG
    else:
        raise ValueError(f"Section {section} not yet implemented")


def print_config(config: Any) -> None:
    """Pretty print a configuration dataclass."""
    print(f"\n{'='*80}")
    print(f"{config.__class__.__name__}")
    print(f"{'='*80}")
    for field_name, field_value in config.__dict__.items():
        if isinstance(field_value, Path):
            print(f"  {field_name:.<40} {field_value}")
        else:
            print(f"  {field_name:.<40} {field_value}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test configuration loading
    print("Chapter 10 Configuration Test")
    print("="*80)
    
    print("\n[Section 10.1] Text Encoder Configuration:")
    print_config(DEFAULT_TEXT_ENCODER)
    
    print("\n[MIND Dataset] Configuration:")
    print_config(DEFAULT_MIND_CONFIG)
    
    print("\n[IJCAI Dataset] Configuration:")
    print_config(DEFAULT_IJCAI_CONFIG)
    
    print("\n[Bi-Encoder Evaluation] Configuration:")
    print_config(DEFAULT_BIENCODER_EVAL)
    
    print("\n[Amazon KDD Dataset] Configuration:")
    print_config(DEFAULT_AMAZON_KDD_CONFIG)

    print("\n[Section 10.2] Item2Vec Configuration:")
    print_config(DEFAULT_ITEM2VEC_CONFIG)

    print("\n[Yandex Dataset] Configuration:")
    print_config(DEFAULT_YANDEX_CONFIG)

    print("\n[Section 10.3] Contrastive Fine-Tuning Configuration:")
    print_config(DEFAULT_CONTRASTIVE_CONFIG)

    # Check if paths exist
    print("\n[Path Validation]")
    print(f"MIND data exists: {MIND_DATA_PATH.exists()}")
    print(f"IJCAI data exists: {IJCAI_DATA_PATH.exists()}")
    print(f"Amazon KDD data exists: {AMAZON_KDD_DATA_PATH.exists()}")
    print(f"Yandex data exists: {YANDEX_DATA_PATH.exists()}")
    print(f"Music4all data exists: {MUSIC4ALL_DATA_PATH.exists()}")

    print("\n[Section 10.4] Music4all Configuration:")
    print_config(DEFAULT_MUSIC4ALL_CONFIG)

    print("\n[Section 10.4] Multi-Modal Fusion Configuration:")
    print_config(DEFAULT_MULTIMODAL_CONFIG)

    print("\n[Appendix] Superlinked Configuration:")
    print_config(DEFAULT_SUPERLINKED_CONFIG)
