"""
Configuration for Chapter 11: User/Customer Embeddings

Centralizes all paths, model settings, and hyperparameters for reproducibility.
Follows the same pattern as Chapter 10's config.py.

Section 11.1: Aggregation Baselines (No Training)
- Simple Mean, Last-K Mean, Exponential Decay, TF-IDF Weighted, TF-IDF + Recency
- Datasets: Amazon KDD (session embeddings), MIND (user embeddings)

Section 11.2: Sequence Models (SASRec / GRU4Rec)
- Frozen SBERT embeddings from Chapter 10 as input
- SASRec: Causal Transformer encoder with learnable positional embeddings
- GRU4Rec: Stacked GRU with packed sequence handling
- MNR loss (in-batch negatives) for next-item prediction
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================================
# Base Paths
# ============================================================================

# Project root (three levels up: config.py -> chapter11_user_embeddings/ -> RecSys book code/ -> RecSys ML Book Writing/)
# Matches Chapter 10's convention: PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
CHAPTER_ROOT = Path(__file__).parent

# Chapter 10 paths (for reusing cached item embeddings)
# CHAPTER10_ROOT is a sibling directory under "RecSys book code/"
CHAPTER10_ROOT = CHAPTER_ROOT.parent / "chapter10_item_embeddings"
CHAPTER10_EMBEDDINGS_DIR = CHAPTER10_ROOT / "outputs" / "embeddings"
CHAPTER10_MODELS_DIR = CHAPTER10_ROOT / "outputs" / "models"

# Dataset-specific paths
MIND_DATA_PATH = DATASET_ROOT / "Microsoft News" / "MINDsmall_train"
MIND_DEV_DATA_PATH = DATASET_ROOT / "Microsoft News" / "MINDsmall_dev"
AMAZON_KDD_DATA_PATH = DATASET_ROOT / "Amazon KDD 2023"

# Output paths
OUTPUTS_DIR = CHAPTER_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
METRICS_DIR = OUTPUTS_DIR / "metrics"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
MODELS_DIR = OUTPUTS_DIR / "models"

# Ensure directories exist
for dir_path in [OUTPUTS_DIR, EMBEDDINGS_DIR, METRICS_DIR, ANALYSIS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Chapter 10 Cached Embedding Paths
# ============================================================================

# Item embedding files from Chapter 10 (DO NOT re-encode)
AMAZON_FINETUNED_SBERT_EMBEDDINGS = (
    CHAPTER10_EMBEDDINGS_DIR
    / "amazon_kdd_embeddings_finetuned_sentence-transformers_all-MiniLM-L6-v2.npz"
)
AMAZON_ZEROSHOT_SBERT_EMBEDDINGS = (
    CHAPTER10_EMBEDDINGS_DIR
    / "amazon_kdd_embeddings_sentence-transformers_all-MiniLM-L6-v2.npz"
)
AMAZON_ITEM2VEC_EMBEDDINGS = (
    CHAPTER10_MODELS_DIR
    / "item2vec" / "amazon_item2vec" / "amazon_item2vec_embeddings.npz"
)
MIND_ZEROSHOT_SBERT_EMBEDDINGS = (
    CHAPTER10_EMBEDDINGS_DIR
    / "mind_embeddings_sentence-transformers_all-MiniLM-L6-v2.npz"
)


# ============================================================================
# Section 11.1: Aggregation Baselines
# ============================================================================

@dataclass
class AmazonSessionConfig:
    """Configuration for Amazon KDD session-based user embeddings.

    Note on terminology: Amazon KDD has no user IDs -- each row is an anonymous
    browsing session.  The "user embedding" here is really a *session embedding*
    representing the shopper's current intent.  This is actually the dominant
    industrial pattern for anonymous / logged-out traffic.
    """

    # Data paths
    data_dir: Path = AMAZON_KDD_DATA_PATH
    products_file: str = "products_train.csv"
    sessions_file: str = "sessions_train.csv"

    # Locale filter (English only, matching Chapter 10)
    locale: str = "UK"

    # Session filtering
    min_session_length: int = 3  # Need at least 3 prev_items: 2+ for aggregation, 1 held out

    # Item embedding source (primary: best Chapter 10 result)
    item_embedding_file: Path = AMAZON_FINETUNED_SBERT_EMBEDDINGS
    item_embedding_dim: int = 384

    # Secondary comparison (Item2Vec, for "embedding quality matters" lesson)
    item2vec_embedding_file: Path = AMAZON_ITEM2VEC_EMBEDDINGS
    item2vec_embedding_dim: int = 128


@dataclass
class MINDUserConfig:
    """Configuration for MIND user-based embeddings.

    MIND has real user IDs and timestamped impression logs with click histories.
    This enables true user embeddings aggregated over reading histories,
    evaluated by predicting future clicks.
    """

    # Data paths
    data_dir: Path = MIND_DATA_PATH
    dev_data_dir: Path = MIND_DEV_DATA_PATH
    news_file: str = "news.tsv"
    behaviors_file: str = "behaviors.tsv"

    # User filtering
    min_history_length: int = 5  # Users with fewer clicks are excluded

    # Item embedding source
    item_embedding_file: Path = MIND_ZEROSHOT_SBERT_EMBEDDINGS
    item_embedding_dim: int = 384


@dataclass
class AggregationConfig:
    """Configuration for aggregation-based user embedding methods.

    All hyperparameters are either derived from the data (empirical distributions)
    or explicitly flagged as pedagogical simplifications.
    """

    # --- Aggregation method selection ---
    methods: List[str] = field(default_factory=lambda: [
        "simple_mean",
        "last_k_mean",
        "exponential_decay",
        "tfidf_weighted",
        "tfidf_recency",
    ])

    # --- Last-K Mean ---
    # K values to sweep; the "best" K is reported alongside the empirical median.
    # For Amazon: K is bounded by short session lengths (typically 2-8).
    # For MIND: K can range widely (5-100+).
    # These defaults are starting points; data_analysis.py refines them.
    last_k_values_amazon: List[int] = field(default_factory=lambda: [2, 3, 5])
    last_k_values_mind: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # --- Exponential Decay ---
    # For Amazon (no timestamps): position-based decay.
    #   weight_i = exp(-lambda * (N - i))  where i=0 is oldest, N-1 is most recent.
    #   lambda is set so that the oldest item in a median-length session gets weight ~0.5.
    # For MIND (has timestamps): time-based decay.
    #   weight_i = exp(-lambda * (t_latest - t_i) / 3600)  in hours.
    #   lambda is set so that a click 24 hours old gets weight ~0.5.
    decay_half_life_positions: Optional[float] = None  # Computed from data in data_analysis.py
    decay_half_life_hours: float = 24.0  # MIND: 24-hour half-life (pedagogical default)

    # --- TF-IDF Weighting ---
    # IDF = log(N_sessions / df_item) for Amazon, log(N_users / df_item) for MIND.
    # Computed only on training data to prevent test leakage.
    # TF is binary (1 if item appears in history, 0 otherwise) since most items
    # appear once per session/history.
    idf_smooth: float = 1.0  # Additive smoothing to prevent log(0) for unseen items

    # --- Output ---
    normalize_user_embeddings: bool = True  # L2-normalize after aggregation (for FAISS IP)


@dataclass
class EvaluationConfig:
    """Configuration for user embedding evaluation."""

    # Retrieval metrics
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])

    # FAISS settings
    faiss_index_type: str = "IndexFlatIP"  # Inner product on L2-normalized vectors

    # Evaluation scale
    max_eval_sessions_amazon: int = 20_000  # Match Chapter 10 for comparability
    max_eval_users_mind: int = 10_000

    # Random seed for reproducible sampling
    random_seed: int = 42

    # Output files
    amazon_results_file: str = "aggregation_amazon_results.json"
    mind_results_file: str = "aggregation_mind_results.json"
    comparison_file: str = "aggregation_comparison.json"


# ============================================================================
# Section 11.2: Sequence Models (SASRec / GRU4Rec)
# ============================================================================

@dataclass
class SequenceModelConfig:
    """Configuration for Section 11.2: Sequence Models.

    Both SASRec and GRU4Rec consume frozen SBERT embeddings (384-dim) from
    Chapter 10.  A learnable projection maps 384 → hidden_dim for the
    sequence model, and an output projection maps hidden_dim → 384 so the
    user embedding lives in the same space as the item embeddings (for
    direct FAISS retrieval without rebuilding the index).

    Training uses MNR (Multiple Negatives Ranking) loss with in-batch
    negatives — the same loss family from Chapter 10's contrastive
    fine-tuning — applied at every valid sequence position.

    All hyperparameters are justified in Chapter11_Design_Notes.md.
    """

    # --- Model architecture ---
    model_type: str = "sasrec"      # "sasrec" or "gru4rec"
    input_dim: int = 384            # Frozen SBERT embedding dimension
    hidden_dim: int = 128           # 3x compression; matches Ch10 Item2Vec dim
    num_layers: int = 2             # Original SASRec default; sufficient for short sequences
    num_heads: int = 2              # SASRec only; head_dim = hidden_dim / num_heads = 64
    ffn_dim: int = 512              # SASRec only; 4 * hidden_dim (standard ratio)
    dropout: float = 0.2            # Original SASRec default

    # --- Sequence handling ---
    # Max sequence lengths are data-driven from session/history length distributions:
    #   Amazon: P95 = 12 → max_seq_len = 20 (covers P95+ with margin)
    #   MIND:   P90 = 42 → max_seq_len = 50 (covers P90; longer would strain memory)
    # Truncation keeps the MOST RECENT items (right-aligned), since recency is
    # the strongest signal (confirmed by Section 11.1 results).
    max_seq_len_amazon: int = 20
    max_seq_len_mind: int = 50

    # --- Training ---
    learning_rate: float = 1e-3     # Standard for training-from-scratch; original SASRec
    batch_size: int = 256           # Matches Ch10; provides 65K in-batch negatives
    epochs_amazon: int = 30         # Larger dataset, short sequences; early stopping
    epochs_mind: int = 20           # Smaller dataset, longer histories; early stopping
    warmup_ratio: float = 0.1      # 10% of total steps; matches Ch10 pattern
    weight_decay: float = 0.01     # Standard AdamW regularization
    temperature: float = 0.05      # Contrastive loss sharpness (= 1/20; same as Ch10's *20)
    patience: int = 5              # Early stopping: epochs without val improvement
    gradient_clip: float = 1.0     # Matches Ch10; prevents gradient explosion

    # --- Data split ---
    train_fraction: float = 0.9    # 90% train, 10% validation (of non-eval users/sessions)
    random_seed: int = 42

    # --- Multi-position training ---
    # Train on next-item prediction at ALL positions, not just the last.
    # Critical for short Amazon sessions (median 4 → only 3-4 training signals).
    multi_position_loss: bool = True

    # --- Evaluation ---
    eval_during_training: bool = True
    eval_every_n_epochs: int = 2

    # --- Output ---
    output_subdir: str = "sequence_models"


# ============================================================================
# Section 11.3: LightGCN (Graph Neural Network)
# ============================================================================

@dataclass
class LightGCNConfig:
    """Configuration for Section 11.3: LightGCN.

    LightGCN (He et al., SIGIR 2020) learns user and item embeddings via
    graph neural network propagation on the user-item bipartite interaction
    graph.  It removes ALL feature transformations and nonlinearities from
    standard GCN, relying solely on neighborhood aggregation.

    MIND-only: Amazon KDD has no persistent user IDs (anonymous sessions),
    so a bipartite user-item graph cannot be constructed.

    Training uses BPR (Bayesian Personalized Ranking) loss with per-edge
    negative sampling, avoiding the mode collapse problem observed with
    MNR loss on MIND's dense topic clusters (Section 11.2 failure analysis).
    """

    # --- Model architecture ---
    hidden_dim: int = 64            # LightGCN convention; smaller since propagation is parameter-free
    num_layers: int = 3             # 3-hop neighborhood (original LightGCN default)
    dropout: float = 0.0            # No dropout (standard for LightGCN)

    # --- Training ---
    loss_type: str = "bpr"          # "bpr" or "mnr" (mnr for pedagogical comparison)
    learning_rate: float = 1e-3     # Standard for LightGCN
    l2_reg_weight: float = 1e-4     # L2 regularization on initial (pre-GCN) embeddings
    batch_size: int = 1024          # Edges per batch (not users)
    epochs: int = 100               # With early stopping
    patience: int = 10              # LightGCN converges slowly; needs more patience
    train_fraction: float = 0.9     # Edge split for train/val
    random_seed: int = 42

    # --- MNR comparison ---
    mnr_temperature: float = 0.05   # Same as Section 11.2 for fair comparison

    # --- Evaluation ---
    eval_during_training: bool = True
    eval_every_n_epochs: int = 5

    # --- Output ---
    output_subdir: str = "lightgcn"


# ============================================================================
# Global Defaults
# ============================================================================

DEFAULT_AMAZON_SESSION_CONFIG = AmazonSessionConfig()
DEFAULT_MIND_USER_CONFIG = MINDUserConfig()
DEFAULT_AGGREGATION_CONFIG = AggregationConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_SEQUENCE_MODEL_CONFIG = SequenceModelConfig()
DEFAULT_LIGHTGCN_CONFIG = LightGCNConfig()


# ============================================================================
# Utility Functions
# ============================================================================

def get_config(dataset: str = "amazon", section: str = "11.1") -> Any:
    """
    Get configuration for a specific dataset and chapter section.

    Args:
        dataset: Dataset name ("amazon", "amazon_kdd", "mind")
        section: Chapter section ("11.1", "11.2", "11.3")

    Returns:
        Configuration dataclass instance(s)
    """
    if section == "11.1":
        if dataset.lower() in ("amazon", "amazon_kdd"):
            return DEFAULT_AMAZON_SESSION_CONFIG, DEFAULT_AGGREGATION_CONFIG
        elif dataset.lower() == "mind":
            return DEFAULT_MIND_USER_CONFIG, DEFAULT_AGGREGATION_CONFIG
        else:
            raise ValueError(f"Unknown dataset for section 11.1: {dataset}")
    elif section == "11.2":
        if dataset.lower() in ("amazon", "amazon_kdd"):
            return DEFAULT_AMAZON_SESSION_CONFIG, DEFAULT_SEQUENCE_MODEL_CONFIG
        elif dataset.lower() == "mind":
            return DEFAULT_MIND_USER_CONFIG, DEFAULT_SEQUENCE_MODEL_CONFIG
        else:
            raise ValueError(f"Unknown dataset for section 11.2: {dataset}")
    elif section == "11.3":
        if dataset.lower() == "mind":
            return DEFAULT_MIND_USER_CONFIG, DEFAULT_LIGHTGCN_CONFIG
        else:
            raise ValueError(
                f"Section 11.3 (LightGCN) is MIND-only. "
                f"Amazon KDD has no user IDs for bipartite graph construction."
            )
    else:
        raise ValueError(f"Section {section} not yet implemented")


def print_config(config: Any) -> None:
    """Pretty-print a configuration dataclass."""
    print(f"\n{'=' * 80}")
    print(f"{config.__class__.__name__}")
    print(f"{'=' * 80}")
    for field_name, field_value in config.__dict__.items():
        if isinstance(field_value, Path):
            # Show relative path for readability
            try:
                rel = field_value.relative_to(PROJECT_ROOT)
                print(f"  {field_name:.<50} {rel}")
            except ValueError:
                print(f"  {field_name:.<50} {field_value}")
        else:
            print(f"  {field_name:.<50} {field_value}")
    print(f"{'=' * 80}\n")


def validate_paths() -> Dict[str, bool]:
    """Check that all required data files and embedding caches exist."""
    checks = {
        "Amazon KDD data dir": AMAZON_KDD_DATA_PATH.exists(),
        "MIND data dir": MIND_DATA_PATH.exists(),
        "Amazon fine-tuned SBERT embeddings": AMAZON_FINETUNED_SBERT_EMBEDDINGS.exists(),
        "Amazon zero-shot SBERT embeddings": AMAZON_ZEROSHOT_SBERT_EMBEDDINGS.exists(),
        "Amazon Item2Vec embeddings": AMAZON_ITEM2VEC_EMBEDDINGS.exists(),
        "MIND zero-shot SBERT embeddings": MIND_ZEROSHOT_SBERT_EMBEDDINGS.exists(),
    }

    print("\n[Path Validation]")
    all_ok = True
    for name, exists in checks.items():
        status = "OK" if exists else "MISSING"
        print(f"  {name:.<55} {status}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  WARNING: Some paths are missing. Run Chapter 10 first to generate embeddings.")

    return checks


if __name__ == "__main__":
    print("Chapter 11 Configuration Test")
    print("=" * 80)

    print("\n[Amazon KDD Session Config]:")
    print_config(DEFAULT_AMAZON_SESSION_CONFIG)

    print("\n[MIND User Config]:")
    print_config(DEFAULT_MIND_USER_CONFIG)

    print("\n[Aggregation Config]:")
    print_config(DEFAULT_AGGREGATION_CONFIG)

    print("\n[Evaluation Config]:")
    print_config(DEFAULT_EVALUATION_CONFIG)

    print("\n[Sequence Model Config]:")
    print_config(DEFAULT_SEQUENCE_MODEL_CONFIG)

    print("\n[LightGCN Config]:")
    print_config(DEFAULT_LIGHTGCN_CONFIG)

    validate_paths()
