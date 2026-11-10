"""
Configuration for Chapter 12: Generative AI for Recommender Systems.

Every script reads and writes inside one *work directory* so that a tiny CPU run,
a full GPU run and the ranker-sized run never overwrite each other:

    <work_dir>/data/sessions.parquet       prepared sessions with a stable session_id and split
    <work_dir>/data/items.parquet          catalog text for the items in scope
    <work_dir>/embeddings/                 item embeddings, Semantic IDs, LLM feature embeddings
    <work_dir>/slates/                     candidate slates shared by every ranker
    <work_dir>/llm/                        raw LLM generations (JSONL, resumable)
    <work_dir>/models/                     checkpoints and adapters
    <work_dir>/metrics/                    JSON results

Paths can be redirected without editing code:
    RECSYS_DATASET_ROOT  folder that contains "Amazon KDD 2023/" (default: <repo>/Dataset)
    CH12_WORK_DIR        default work directory (default: ./outputs/default)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

CHAPTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CHAPTER_ROOT.parent.parent

DATASET_ROOT = Path(os.environ.get("RECSYS_DATASET_ROOT", PROJECT_ROOT / "Dataset"))
AMAZON_KDD_DATA_PATH = DATASET_ROOT / "Amazon KDD 2023"

DEFAULT_WORK_DIR = Path(os.environ.get("CH12_WORK_DIR", CHAPTER_ROOT / "outputs" / "default"))

# Open-weights models only. Licences checked on the Hugging Face model cards (2026-11-08):
# Qwen2.5-0.5B-Instruct, -1.5B-Instruct and -7B-Instruct are Apache 2.0.
# Qwen2.5-3B-Instruct is under the Qwen Research licence and is deliberately not a default.
DEFAULT_TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
CPU_LLM = "Qwen/Qwen2.5-0.5B-Instruct"

SEED = 42


@dataclass(frozen=True)
class WorkPaths:
    """All file locations derived from one work directory."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def sessions(self) -> Path:
        return self.data_dir / "sessions.parquet"

    @property
    def items(self) -> Path:
        return self.data_dir / "items.parquet"

    @property
    def prepare_meta(self) -> Path:
        return self.data_dir / "prepare_meta.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def item_embeddings(self) -> Path:
        return self.embeddings_dir / "item_embeddings.npz"

    @property
    def semantic_ids(self) -> Path:
        return self.embeddings_dir / "item_semantic_ids.npz"

    @property
    def llm_item_embeddings(self) -> Path:
        return self.embeddings_dir / "llm_item_knowledge_embeddings.npz"

    @property
    def llm_intent_embeddings(self) -> Path:
        return self.embeddings_dir / "llm_session_intent_embeddings.npz"

    @property
    def slates_dir(self) -> Path:
        return self.root / "slates"

    def slate_file(self, split: str, mode: str, size: int) -> Path:
        return self.slates_dir / f"{split}_{mode}_n{size}.jsonl"

    @property
    def llm_dir(self) -> Path:
        return self.root / "llm"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    def ensure(self) -> "WorkPaths":
        for path in [self.data_dir, self.embeddings_dir, self.slates_dir,
                     self.llm_dir, self.models_dir, self.metrics_dir]:
            path.mkdir(parents=True, exist_ok=True)
        return self


def work_paths(work_dir: Path | str | None = None) -> WorkPaths:
    return WorkPaths(Path(work_dir) if work_dir is not None else DEFAULT_WORK_DIR)


# ============================================================================
# Data preparation
# ============================================================================

@dataclass
class PrepareConfig:
    locale: str = "UK"
    min_prev_items: int = 3           # sessions need at least this many history items
    max_sessions: int | None = None   # None keeps every eligible session
    split_fractions: tuple = (0.8, 0.1, 0.1)   # train, val, test
    catalog: str = "all"              # "all" UK products, or "referenced" (items in the kept sessions)
    extra_random_items: int = 0       # with catalog="referenced", add this many unreferenced items as distractors
    desc_max_chars: int = 400
    seed: int = SEED


# ============================================================================
# Section 12.2: Semantic IDs and generative retrieval
# ============================================================================

@dataclass
class RQVAEConfig:
    """Residual-quantised autoencoder that turns item embeddings into Semantic IDs (Rajput et al., 2023)."""

    encoder_hidden_dim: int = 256
    latent_dim: int = 64
    num_levels: int = 3
    codebook_size: int = 256
    learning_rate: float = 1e-3
    batch_size: int = 1024
    epochs: int = 30
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    kmeans_init_samples: int = 20_000
    kmeans_iters: int = 20
    dead_code_fraction: float = 0.1   # reset an entry whose EMA usage is below this fraction of the mean entry's
    weight_decay: float = 1e-4
    val_fraction: float = 0.05
    seed: int = SEED


@dataclass
class TIGERConfig:
    """Encoder-decoder transformer that generates the next item's Semantic ID."""

    hidden_dim: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_history_items: int = 15
    learning_rate: float = 5e-4
    batch_size: int = 256
    epochs: int = 20
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    patience: int = 3
    label_smoothing: float = 0.1
    # Per-level loss weights for the semantic levels; the collision-suffix level always gets 1.0.
    level_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    beam_width: int = 20
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    max_eval_sessions: int = 5_000
    seed: int = SEED


# ============================================================================
# Section 12.5: LLMs as rankers (evaluated on shared candidate slates)
# ============================================================================

@dataclass
class SlateConfig:
    size: int = 20
    modes: List[str] = field(default_factory=lambda: ["random", "popular", "retrieved", "injected"])
    max_sessions: int = 1_000
    seed: int = SEED


@dataclass
class ZeroShotLLMConfig:
    model_name: str = DEFAULT_LLM
    max_new_tokens: int = 80
    max_history_items: int = 10
    batch_size: int = 8
    max_sessions: int = 1_000
    slate_mode: str = "retrieved"
    slate_size: int = 20
    seed: int = SEED


@dataclass
class TALLRecConfig:
    """TALLRec-style LoRA tuning for a Yes/No engagement prompt (Bao et al., 2023)."""

    model_name: str = DEFAULT_LLM
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_train_sessions: int = 5_000
    num_negatives_per_positive: int = 3
    max_history_items: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    eval_batch_size: int = 20
    max_eval_sessions: int = 1_000
    slate_mode: str = "retrieved"
    slate_size: int = 20
    seed: int = SEED


# ============================================================================
# Section 12.4: LLMs as offline feature generators
# ============================================================================

@dataclass
class LLMFeatureConfig:
    """Offline item-knowledge and session-intent generation (KAR-style; Xi et al., 2024)."""

    model_name: str = DEFAULT_LLM
    item_max_new_tokens: int = 120
    intent_max_new_tokens: int = 48
    batch_size: int = 32
    max_items: int | None = None      # None covers every item in the work directory
    max_sessions: int | None = None   # None covers every session in the work directory
    seed: int = SEED


@dataclass
class AugmentedRankerConfig:
    """MLP re-ranker over frozen embedding tables; feature sets are compared in an ablation."""

    hidden_dims: List[int] = field(default_factory=lambda: [256, 64])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    patience: int = 3
    weight_decay: float = 1e-4
    max_history_items: int = 10
    num_hard_negatives: int = 2
    num_random_negatives: int = 2
    hard_negative_pool: int = 50
    slate_size: int = 50
    cold_item_threshold: int = 5      # targets seen fewer times than this in *train* sessions count as cold
    min_llm_coverage: float = 0.99
    seed: int = SEED


FEATURE_SETS = ["content", "content_history", "content_history_llm_item", "full"]
