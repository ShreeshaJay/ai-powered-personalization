"""
Section 10.2: Item2Vec — Collaborative Embeddings from Sequences

Learns item embeddings purely from behavioral co-occurrence using Word2Vec
(Skip-gram with negative sampling). No item metadata is used.

Key idea: treat each user session as a "sentence" and each item as a "word",
then train Word2Vec so items that appear in similar contexts (sessions) get
similar embeddings.

Datasets:
    - Amazon KDD (primary): Shopping sessions → product embeddings
    - Yandex Yambda (secondary): Listening sessions → track embeddings

Reference: Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for
           Collaborative Filtering", MLSP 2016.

Usage:
    python item2vec.py --dataset amazon   # train on Amazon KDD sessions
    python item2vec.py --dataset yandex   # train on Yandex music sessions
    python item2vec.py --dataset both     # train on both
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
from collections import Counter
from gensim.models import Word2Vec

from config import (
    AMAZON_KDD_DATA_PATH,
    YANDEX_DATA_PATH,
    MODELS_DIR,
    DEFAULT_ITEM2VEC_CONFIG,
    DEFAULT_YANDEX_CONFIG,
    Item2VecConfig,
)
from data.amazon_kdd_dataset import AmazonKDDDataset
from data.yandex_dataset import YandexDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_amazon_sessions() -> List[List[str]]:
    """Load Amazon KDD shopping sessions as item sequences."""
    dataset = AmazonKDDDataset(data_dir=AMAZON_KDD_DATA_PATH, locale="UK")
    dataset.load_products()
    product_ids = set(dataset.products_df["id"])
    dataset.load_sessions(product_ids=product_ids)
    return dataset.get_sessions_as_sequences()


def load_yandex_sessions() -> List[List[str]]:
    """Load Yandex listening sequences (full history or session-split)."""
    cfg = DEFAULT_YANDEX_CONFIG
    dataset = YandexDataset(
        data_dir=cfg.data_dir,
        min_played_ratio=cfg.min_played_ratio,
        min_listens_per_user=cfg.min_listens_per_user,
        max_listens_per_user=cfg.max_listens_per_user,
        session_gap_seconds=cfg.session_gap_seconds,
    )
    return dataset.build_sequences(mode=cfg.sequence_mode)


def compute_min_count(
    sessions: List[List[str]],
    percentile: float = 5.0,
) -> int:
    """
    Derive min_count from the empirical item frequency distribution.

    Counts how often each item appears across all sessions, then returns
    the value at the given percentile. Items below this threshold carry
    too little co-occurrence signal for meaningful embeddings.

    Args:
        sessions: List of item-ID sequences.
        percentile: Percentile of the frequency distribution to use as
                    the cutoff (e.g., 5.0 means drop the bottom 5%).

    Returns:
        The min_count value (at least 1).
    """
    freq = Counter(item for session in sessions for item in session)
    counts = np.array(list(freq.values()))

    cutoff = int(np.percentile(counts, percentile))
    cutoff = max(cutoff, 1)

    total_items = len(freq)
    retained = int((counts >= cutoff).sum())

    logger.info(f"Item frequency distribution:")
    logger.info(f"  Total unique items: {total_items:,}")
    logger.info(f"  Frequency percentiles: "
                f"p5={np.percentile(counts, 5):.0f}, "
                f"p25={np.percentile(counts, 25):.0f}, "
                f"p50={np.percentile(counts, 50):.0f}, "
                f"p75={np.percentile(counts, 75):.0f}, "
                f"p95={np.percentile(counts, 95):.0f}")
    logger.info(f"  Chosen percentile: p{percentile} -> min_count={cutoff}")
    logger.info(f"  Retained items: {retained:,} / {total_items:,} "
                f"({retained / total_items * 100:.1f}%)")

    return cutoff


def resolve_min_count(
    config: Item2VecConfig,
    sessions: List[List[str]],
) -> int:
    """
    Determine the min_count to use for training.

    If config.min_count is set explicitly, use that (allows manual override).
    Otherwise, compute from the data using config.min_count_percentile.
    """
    if config.min_count is not None:
        logger.info(f"Using explicit min_count={config.min_count}")
        return config.min_count

    return compute_min_count(sessions, percentile=config.min_count_percentile)


def train_item2vec(
    sessions: List[List[str]],
    config: Item2VecConfig,
    model_name: str = "item2vec",
) -> Word2Vec:
    """
    Train Word2Vec (Skip-gram) on item sessions.

    Each session is a "sentence" of item IDs. The skip-gram objective
    learns to predict context items given a target item, producing
    embeddings where co-occurring items are nearby.

    Returns:
        Trained gensim Word2Vec model
    """
    min_count = resolve_min_count(config, sessions)

    logger.info(f"Training Item2Vec ({model_name})")
    logger.info(f"  Sessions: {len(sessions):,}")
    logger.info(f"  Embedding dim: {config.embedding_dim}")
    logger.info(f"  Window: {config.window_size}")
    logger.info(f"  Min count: {min_count} "
                f"(from {'explicit override' if config.min_count is not None else f'p{config.min_count_percentile} percentile'})")
    logger.info(f"  Negative samples: {config.negative}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Algorithm: {'Skip-gram' if config.sg == 1 else 'CBOW'}")

    start = time.time()

    model = Word2Vec(
        sentences=sessions,
        vector_size=config.embedding_dim,
        window=config.window_size,
        min_count=min_count,
        sg=config.sg,
        negative=config.negative,
        epochs=config.epochs,
        workers=config.workers,
        seed=config.seed,
    )

    elapsed = time.time() - start
    vocab_size = len(model.wv)

    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"  Vocabulary size: {vocab_size:,} items")

    # Save model and embeddings
    output_dir = MODELS_DIR / config.output_subdir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "word2vec.model"
    model.save(str(model_path))
    logger.info(f"  Saved model: {model_path}")

    # Save as .npz for interoperability with our FAISS evaluation pipeline
    item_ids = list(model.wv.index_to_key)
    embeddings = np.array([model.wv[item_id] for item_id in item_ids], dtype=np.float32)
    npz_path = output_dir / f"{model_name}_embeddings.npz"
    np.savez_compressed(str(npz_path), ids=np.array(item_ids), embeddings=embeddings)
    logger.info(f"  Saved embeddings: {npz_path} (shape: {embeddings.shape})")

    # Save training metadata
    meta = {
        "model_name": model_name,
        "vocab_size": vocab_size,
        "embedding_dim": config.embedding_dim,
        "window_size": config.window_size,
        "min_count": min_count,
        "min_count_source": ("explicit" if config.min_count is not None
                             else f"p{config.min_count_percentile}_percentile"),
        "negative_samples": config.negative,
        "epochs": config.epochs,
        "sg": config.sg,
        "num_sessions": len(sessions),
        "training_seconds": round(elapsed, 1),
    }
    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model


def print_model_summary(model: Word2Vec, model_name: str, sample_items: int = 5):
    """Print summary and sample nearest neighbors."""
    print(f"\n{'=' * 80}")
    print(f"Item2Vec Model Summary: {model_name}")
    print(f"{'=' * 80}")
    print(f"  Vocabulary:     {len(model.wv):,} items")
    print(f"  Embedding dim:  {model.wv.vector_size}")

    sample_ids = list(model.wv.index_to_key[:sample_items])
    for item_id in sample_ids:
        neighbors = model.wv.most_similar(item_id, topn=3)
        nn_str = ", ".join([f"{nid} ({sim:.3f})" for nid, sim in neighbors])
        print(f"\n  [{item_id}] nearest neighbors:")
        print(f"    {nn_str}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Train Item2Vec embeddings")
    parser.add_argument(
        "--dataset",
        choices=["amazon", "yandex", "both"],
        default="amazon",
        help="Which dataset(s) to train on",
    )
    parser.add_argument("--dim", type=int, default=None, help="Embedding dimension override")
    parser.add_argument("--window", type=int, default=None, help="Window size override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--min-count", type=int, default=None,
                        help="Explicit min_count override (bypasses percentile)")
    parser.add_argument("--min-count-percentile", type=float, default=None,
                        help="Percentile of item frequency distribution for min_count "
                             f"(default: {DEFAULT_ITEM2VEC_CONFIG.min_count_percentile})")
    args = parser.parse_args()

    config = Item2VecConfig(
        embedding_dim=args.dim or DEFAULT_ITEM2VEC_CONFIG.embedding_dim,
        window_size=args.window or DEFAULT_ITEM2VEC_CONFIG.window_size,
        min_count=args.min_count,  # None means "use percentile"
        min_count_percentile=(args.min_count_percentile
                              if args.min_count_percentile is not None
                              else DEFAULT_ITEM2VEC_CONFIG.min_count_percentile),
        epochs=args.epochs or DEFAULT_ITEM2VEC_CONFIG.epochs,
        sg=DEFAULT_ITEM2VEC_CONFIG.sg,
        negative=DEFAULT_ITEM2VEC_CONFIG.negative,
        workers=DEFAULT_ITEM2VEC_CONFIG.workers,
        seed=DEFAULT_ITEM2VEC_CONFIG.seed,
    )

    if args.dataset in ("amazon", "both"):
        logger.info("=" * 80)
        logger.info("Training Item2Vec on Amazon KDD sessions")
        logger.info("=" * 80)

        sessions = load_amazon_sessions()
        model = train_item2vec(sessions, config, model_name="amazon_item2vec")
        print_model_summary(model, "Amazon KDD Item2Vec")

    if args.dataset in ("yandex", "both"):
        logger.info("=" * 80)
        logger.info("Training Item2Vec on Yandex Yambda sessions")
        logger.info("=" * 80)

        sessions = load_yandex_sessions()
        model = train_item2vec(sessions, config, model_name="yandex_item2vec")
        print_model_summary(model, "Yandex Yambda Item2Vec")

    logger.info("Done!")


if __name__ == "__main__":
    main()
