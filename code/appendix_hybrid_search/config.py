"""
Configuration for Appendix: Hybrid Search with Vector Databases.

Follows the dataclass-based config pattern from Chapters 10 and 11.
All paths, model settings, and hyperparameters are centralized here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
CHAPTER_ROOT = Path(__file__).parent

# Dataset path
ESCI_DATA_PATH = DATASET_ROOT / "Amazon ESCI" / "shopping_queries_dataset"

# Output paths
OUTPUTS_DIR = CHAPTER_ROOT / "outputs"
CACHE_DIR = CHAPTER_ROOT / "cache"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
SPLADE_CACHE_DIR = CACHE_DIR / "splade"
BM25_CACHE_DIR = CACHE_DIR / "bm25"

# Auto-create directories
for dir_path in [OUTPUTS_DIR, CACHE_DIR, METRICS_DIR, PLOTS_DIR,
                 EMBEDDINGS_DIR, SPLADE_CACHE_DIR, BM25_CACHE_DIR,
                 SPLADE_CACHE_DIR / "prithivi",
                 SPLADE_CACHE_DIR / "qdrant_esci",
                 SPLADE_CACHE_DIR / "naver"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ESCI Label Configuration
# =============================================================================

# Graded relevance gains for NDCG computation
ESCI_GAINS: Dict[str, int] = {"E": 3, "S": 2, "C": 1, "I": 0}

# Qdrant-compatible gains (for curated pool evaluation, enables comparison
# with published SPLADE results from qdrant.tech/articles/sparse-embeddings)
ESCI_GAINS_QDRANT: Dict[str, float] = {"E": 1.0, "S": 0.7, "C": 0.5, "I": 0.0}

# Ordered labels for consistent plotting
ESCI_LABELS_ORDERED: List[str] = ["E", "S", "C", "I"]

# Colors for ESCI labels in plots
ESCI_COLORS: Dict[str, str] = {
    "E": "#2ecc71",   # Green  - Exact
    "S": "#3498db",   # Blue   - Substitute
    "C": "#e67e22",   # Orange - Complement
    "I": "#e74c3c",   # Red    - Irrelevant
}

# Human-readable label names
ESCI_LABEL_NAMES: Dict[str, str] = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
}


# =============================================================================
# Dataclass Configurations
# =============================================================================

@dataclass
class ESCIDataConfig:
    """Configuration for Amazon ESCI dataset loading."""
    data_dir: Path = ESCI_DATA_PATH
    examples_file: str = "shopping_queries_dataset_examples.parquet"
    products_file: str = "shopping_queries_dataset_products.parquet"
    locale: str = "us"
    split: str = "train"

    # Text serialization template for product representation
    # Available fields: {title}, {bullet_point}, {brand}, {color}, {description}
    text_template: str = "{title}. {bullet_point}. Brand: {brand}"

    # Sampling controls
    max_queries: int = 5000     # 0 = all ~97K queries
    max_products: int = 0       # 0 = all ~1.2M products
    random_seed: int = 42


@dataclass
class BM25Config:
    """Configuration for BM25 search using bm25s."""
    method: str = "lucene"       # BM25 variant: "lucene", "robertson", "atire", "bm25l", "bm25+"
    k1: float = 1.5              # Term frequency saturation parameter
    b: float = 0.75              # Document length normalization
    cache_dir: Path = BM25_CACHE_DIR


@dataclass
class SPLADEConfig:
    """Configuration for SPLADE learned sparse search.

    Supports multiple SPLADE variants via use_sparse_encoder flag:
      - AutoModelForMaskedLM: prithivida/Splade_PP_en_v2 (Apache 2.0)
      - SparseEncoder (sentence-transformers v5): Qdrant ESCI-tuned, Naver
    """
    model_name: str = "prithivida/Splade_PP_en_v2"
    use_sparse_encoder: bool = False  # True = SparseEncoder, False = AutoModelForMaskedLM
    label: str = "SPLADE (Prithivi)"  # Display name for plots/tables
    batch_size: int = 32
    max_seq_length: int = 256
    vocab_size: int = 30522      # BERT vocabulary size
    cache_dir: Path = SPLADE_CACHE_DIR
    product_vectors_file: str = "splade_product_vectors.npz"
    product_ids_file: str = "splade_product_ids.npy"
    top_k_expansion_tokens: int = 20    # For visualization


# Three SPLADE variants for comparison
SPLADE_MODELS: Dict[str, SPLADEConfig] = {
    "prithivi": SPLADEConfig(
        model_name="prithivida/Splade_PP_en_v2",
        use_sparse_encoder=False,
        label="SPLADE (Prithivi)",
        cache_dir=SPLADE_CACHE_DIR / "prithivi",
    ),
    "qdrant_esci": SPLADEConfig(
        model_name="thierrydamiba/splade-ecommerce-esci",
        use_sparse_encoder=True,
        label="SPLADE (Qdrant ESCI)",
        cache_dir=SPLADE_CACHE_DIR / "qdrant_esci",
    ),
    "naver": SPLADEConfig(
        model_name="naver/splade-cocondenser-ensembledistil",
        use_sparse_encoder=True,
        label="SPLADE (Naver)",
        cache_dir=SPLADE_CACHE_DIR / "naver",
    ),
}


@dataclass
class DenseConfig:
    """Configuration for dense semantic search using SBERT + FAISS."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    max_seq_length: int = 128
    embedding_dim: int = 384
    normalize_embeddings: bool = True
    faiss_index_type: str = "IndexFlatIP"
    cache_dir: Path = EMBEDDINGS_DIR
    cache_file: str = "dense_product_embeddings.npz"


@dataclass
class HybridConfig:
    """Configuration for hybrid fusion methods."""
    # Reciprocal Rank Fusion parameters
    rrf_k: int = 60              # RRF smoothing constant (standard default)

    # Weighted score fusion weights
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "bm25": 0.3,
        "splade": 0.4,
        "dense": 0.3,
    })

    # Top-K for full-catalog retrieval per individual method (before fusion)
    retrieval_top_k: int = 100


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    k_values: List[int] = field(default_factory=lambda: [10, 20, 50])
    esci_gains: Dict[str, int] = field(default_factory=lambda: dict(ESCI_GAINS))

    # Separability analysis: relevant vs irrelevant classification
    separability_positive_labels: List[str] = field(
        default_factory=lambda: ["E", "S", "C"]
    )
    separability_negative_labels: List[str] = field(
        default_factory=lambda: ["I"]
    )

    # Boxplot settings
    max_points_per_label: int = 50000   # Subsample for plotting speed

    # Output files
    results_file: str = "hybrid_search_results.json"
    comparison_csv: str = "method_comparison.csv"

    # Curated pool evaluation (Qdrant-style)
    curated_pool_gains: Dict[str, float] = field(
        default_factory=lambda: dict(ESCI_GAINS_QDRANT)
    )
    curated_pool_split: str = "test"      # Use test split for curated pool
    curated_pool_max_queries: int = 2000  # Qdrant uses 2000 test queries

    # BM25 re-ranking
    bm25_rerank_cache_top_k: int = 1000  # Retrieve BM25 top-1000 for score caching


# =============================================================================
# Default Instances
# =============================================================================

DEFAULT_DATA_CONFIG = ESCIDataConfig()
DEFAULT_BM25_CONFIG = BM25Config()
DEFAULT_SPLADE_CONFIG = SPLADE_MODELS["prithivi"]  # Default to Prithivi (Apache 2.0)
DEFAULT_DENSE_CONFIG = DenseConfig()
DEFAULT_HYBRID_CONFIG = HybridConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()

# For backwards compatibility
DEFAULT_SPLADE_CONFIGS = SPLADE_MODELS


# =============================================================================
# Utility Functions
# =============================================================================

def print_config(config: Any, title: str = "") -> None:
    """Pretty-print a dataclass configuration."""
    if title:
        logger.info(f"\n{'='*60}")
        logger.info(f"  {title}")
        logger.info(f"{'='*60}")

    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        if isinstance(value, Path):
            value = str(value)
        logger.info(f"  {field_name}: {value}")


def validate_paths() -> Dict[str, bool]:
    """Check that required dataset files exist."""
    checks = {
        "ESCI examples": (DEFAULT_DATA_CONFIG.data_dir / DEFAULT_DATA_CONFIG.examples_file).exists(),
        "ESCI products": (DEFAULT_DATA_CONFIG.data_dir / DEFAULT_DATA_CONFIG.products_file).exists(),
    }

    logger.info("\nPath validation:")
    for name, exists in checks.items():
        status = "OK" if exists else "MISSING"
        logger.info(f"  {name}: {status}")

    return checks
