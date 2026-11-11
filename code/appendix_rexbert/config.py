"""Protocol constants and run profiles for the RexBERT appendix.

Reported numbers come from the gpu_full profile. The cpu_small profile is a
teaching run: same code paths, smaller data, and its metrics are not the
appendix's reported results.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent

LOCALE = "UK"
TEXT_TEMPLATE = "{title} [SEP] {brand} [SEP] {description}"

# Hash of the filtered session row index. Buckets 0-7 train, 8 val, 9 test.
SPLIT_SEED = 42
TRAIN_FRACTION_BUCKETS = 8
PAIR_CAP_SEED = 42
SAMPLE_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05

CHAPTER10_QUERY_COUNT = 5000
CHAPTER10_QUERY_SEED = 42
CHAPTER10_MIN_PREV_ITEMS = 2

MAX_SEQ_LENGTH = 128
GPU_BATCH_SIZE = 128
CPU_BATCH_SIZE = 64
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MNR_SCALE = 20.0
LEARNING_RATES = (2e-5, 5e-5, 1e-4)
GPU_EPOCHS = 3
TRAIN_SEEDS = (0, 1, 2)
GPU_TRAIN_PAIR_CAP = 500_000
GPU_SWEEP_PAIRS = 100_000
VAL_SESSIONS = 2_000
VAL_CORPUS = 50_000
GPU_TEST_QUERIES = 20_000
UNIT_NORM_TOLERANCE = 1e-3
DECISION_REQUIRED_SEEDS = 3
RETRIEVAL_TOP_K = 100

# Hub commits recorded 2026-11-11. Loading passes revision= and trust_remote_code=False.
MODEL_REVISIONS = {
    "rexbert_micro": "25587117376236b769710249fdcb66d6041edcee",
    "rexbert_base": "55bd1c52eb0714beefcb61cae5e304352197f44c",
    "ettin_17m": "987607455c61e7a5bbc85f7758e0512ea6d0ae4c",
    "minilm": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "mpnet": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hub_id: str
    revision: str
    kind: str  # "mlm" or "sbert"
    expected_dim: int


MODELS: dict[str, ModelSpec] = {
    "rexbert_micro": ModelSpec(
        "rexbert_micro",
        "thebajajra/RexBERT-micro",
        MODEL_REVISIONS["rexbert_micro"],
        "mlm",
        256,
    ),
    "rexbert_base": ModelSpec(
        "rexbert_base",
        "thebajajra/RexBERT-base",
        MODEL_REVISIONS["rexbert_base"],
        "mlm",
        768,
    ),
    "ettin_17m": ModelSpec(
        "ettin_17m",
        "jhu-clsp/ettin-encoder-17m",
        MODEL_REVISIONS["ettin_17m"],
        "mlm",
        256,
    ),
    "minilm": ModelSpec(
        "minilm",
        "sentence-transformers/all-MiniLM-L6-v2",
        MODEL_REVISIONS["minilm"],
        "sbert",
        384,
    ),
    "mpnet": ModelSpec(
        "mpnet",
        "sentence-transformers/all-mpnet-base-v2",
        MODEL_REVISIONS["mpnet"],
        "sbert",
        768,
    ),
}


@dataclass(frozen=True)
class Profile:
    name: str
    zero_shot_models: tuple[str, ...]
    finetune_models: tuple[str, ...]
    n_catalog: int | None
    n_test_queries: int | None
    n_train_pairs: int
    n_val_sessions: int
    n_val_corpus: int
    epochs: int
    seeds: tuple[int, ...]
    learning_rates: tuple[float, ...]
    run_lr_sweep: bool
    sweep_pairs: int
    batch_size: int
    encode_batch_size: int
    max_seq_length: int
    bootstrap_resamples: int
    chapter10_queries: int
    request_bf16: bool
    n_cosine_pairs: int
    reported: bool


PROFILES: dict[str, Profile] = {
    "cpu_small": Profile(
        name="cpu_small",
        zero_shot_models=("minilm", "rexbert_micro", "ettin_17m"),
        finetune_models=("rexbert_micro", "ettin_17m"),
        n_catalog=50_000,
        n_test_queries=2_000,
        n_train_pairs=50_000,
        n_val_sessions=VAL_SESSIONS,
        n_val_corpus=VAL_CORPUS,
        epochs=1,
        seeds=(0,),
        learning_rates=(5e-5,),
        run_lr_sweep=False,
        sweep_pairs=0,
        # 64 rather than 128: the teaching machine has 16 GB of RAM.
        # gpu_full keeps the protocol batch of 128 in-batch negatives.
        batch_size=CPU_BATCH_SIZE,
        encode_batch_size=32,
        max_seq_length=MAX_SEQ_LENGTH,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        chapter10_queries=CHAPTER10_QUERY_COUNT,
        request_bf16=False,
        n_cosine_pairs=10_000,
        reported=False,
    ),
    "gpu_full": Profile(
        name="gpu_full",
        zero_shot_models=(
            "minilm",
            "mpnet",
            "rexbert_base",
            "rexbert_micro",
            "ettin_17m",
        ),
        finetune_models=("rexbert_micro", "ettin_17m", "minilm"),
        n_catalog=None,
        n_test_queries=GPU_TEST_QUERIES,
        n_train_pairs=GPU_TRAIN_PAIR_CAP,
        n_val_sessions=VAL_SESSIONS,
        n_val_corpus=VAL_CORPUS,
        epochs=GPU_EPOCHS,
        seeds=TRAIN_SEEDS,
        learning_rates=LEARNING_RATES,
        run_lr_sweep=True,
        sweep_pairs=GPU_SWEEP_PAIRS,
        batch_size=GPU_BATCH_SIZE,
        encode_batch_size=128,
        max_seq_length=MAX_SEQ_LENGTH,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        chapter10_queries=CHAPTER10_QUERY_COUNT,
        request_bf16=True,
        n_cosine_pairs=10_000,
        reported=True,
    ),
}

RUN_IDS = ("R0", "R1", "R2", "R3", "R4")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when it is already imported.

    Importing PyTorch here would inflate the data-load measurement, so this
    function seeds PyTorch only after another module has imported it.
    The session split uses SHA-256, not this RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" not in sys.modules:
        return
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_bucket(row_index: int, seed: int = SPLIT_SEED) -> int:
    """Return a digit 0-9 from SHA-256 of '{seed}:{row_index}'."""
    import hashlib

    digest = hashlib.sha256(f"{seed}:{row_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def bucket_to_split(bucket: int) -> int:
    """Map a digit to 0 train, 1 val, 2 test."""
    if bucket < TRAIN_FRACTION_BUCKETS:
        return 0
    if bucket == TRAIN_FRACTION_BUCKETS:
        return 1
    return 2


SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
