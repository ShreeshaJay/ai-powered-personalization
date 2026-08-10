"""
Evaluate Item2Vec (Section 10.2) — Comparison on Amazon KDD

Evaluates collaborative Item2Vec embeddings and compares against:
  - Zero-shot text encoder (Section 10.1)
  - [Optional] Contrastive fine-tuned encoder (Section 10.3, if available)

After running Sections 10.1 and 10.2, this produces a 2-way comparison
(zero-shot vs. Item2Vec). After completing Section 10.3 (fine-tuning),
re-run to get the full 3-way comparison.

Key insight: Item2Vec learns purely from behavioral co-occurrence (no metadata).
This reveals what behavioral signals capture vs. content signals, and how they
complement each other (a theme Chapter 11 will explore).

Two evaluation modes:
  1. Full catalog: Each model evaluated on all items it covers
     (text models cover 100%, Item2Vec covers frequent items only)
  2. Common subset: All models restricted to the Item2Vec vocabulary
     for a head-to-head apples-to-apples comparison

Usage:
    python evaluate_item2vec.py                          # 2- or 3-way comparison
    python evaluate_item2vec.py --item2vec_only           # Item2Vec alone
    python evaluate_item2vec.py --max_queries 20000       # more queries
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_AMAZON_KDD_CONFIG,
    DEFAULT_ITEM2VEC_CONFIG,
    DEFAULT_CONTRASTIVE_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
    EMBEDDINGS_DIR,
)
from data.amazon_kdd_dataset import AmazonKDDDataset
from utils import evaluate_ranking_batch
from utils.faiss_index import build_faiss_index
from utils.metrics import print_ranking_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Embedding loaders
# ============================================================================

def load_item2vec_embeddings(
    model_name: str = "amazon_item2vec",
) -> Tuple[np.ndarray, List[str]]:
    """Load Item2Vec embeddings from the .npz file saved by item2vec.py."""
    npz_path = (MODELS_DIR / DEFAULT_ITEM2VEC_CONFIG.output_subdir
                / model_name / f"{model_name}_embeddings.npz")

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Item2Vec embeddings not found: {npz_path}\n"
            f"Run training first:  python item2vec.py --dataset amazon"
        )

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    item_ids = list(data["ids"])

    logger.info(f"Loaded Item2Vec embeddings: {embeddings.shape} from {npz_path}")
    return embeddings, item_ids


def load_text_embeddings(
    model_tag: str,
) -> Tuple[np.ndarray, List[str]]:
    """Load cached text embeddings (.npz) from Section 10.1 or 10.2."""
    npz_path = EMBEDDINGS_DIR / f"{model_tag}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Text embeddings not found: {npz_path}\n"
            f"Run evaluation first:  python evaluate_amazon.py --compare_models"
        )

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    item_ids = list(data["item_ids"])

    logger.info(f"Loaded text embeddings: {embeddings.shape} from {npz_path}")
    return embeddings, item_ids


# ============================================================================
# Next-item retrieval evaluation (reusable)
# ============================================================================

def next_item_retrieval(
    embeddings: np.ndarray,
    item_ids: List[str],
    next_item_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
    restrict_to: Optional[Set[str]] = None,
    label: str = "",
) -> Dict[str, Any]:
    """
    Evaluate next-item retrieval on a given set of embeddings.

    Args:
        restrict_to: If provided, only use items in this set
                     (for apples-to-apples comparison across models).
    """
    id_to_idx = {pid: i for i, pid in enumerate(item_ids)}
    item_set = set(item_ids)

    if restrict_to is not None:
        item_set = item_set & restrict_to

    query_ids = [
        pid for pid in item_ids
        if pid in item_set
        and pid in next_item_map
        and len(next_item_map[pid]) > 0
    ]

    if len(query_ids) > max_queries:
        rng = np.random.RandomState(42)
        query_ids = list(rng.choice(query_ids, size=max_queries, replace=False))

    if len(query_ids) == 0:
        logger.warning(f"[{label}] No valid queries — skipping")
        return {"num_queries": 0, "coverage": len(item_set)}

    max_k = max(k_values)
    faiss_index = build_faiss_index(embeddings, index_type="IndexFlatIP")

    query_indices = np.array([id_to_idx[pid] for pid in query_ids])
    query_embs = embeddings[query_indices].astype(np.float32)
    distances, nn_indices = faiss_index.search(query_embs, max_k + 1)

    scores_list = []
    relevance_list = []

    for i, q_id in enumerate(query_ids):
        q_idx = query_indices[i]
        targets = next_item_map[q_id]
        target_idx_set = {id_to_idx[p] for p in targets if p in id_to_idx}

        row_idx = nn_indices[i]
        row_dist = distances[i]
        mask = row_idx != q_idx
        row_idx = row_idx[mask][:max_k]
        row_dist = row_dist[mask][:max_k]

        relevance = np.array([1 if idx in target_idx_set else 0 for idx in row_idx])
        scores_list.append(row_dist)
        relevance_list.append(relevance)

    metrics = evaluate_ranking_batch(scores_list, relevance_list, k_values=k_values)
    metrics["num_queries"] = len(query_ids)
    metrics["coverage"] = len(item_set)
    metrics["avg_next_items"] = np.mean([len(next_item_map[pid]) for pid in query_ids])

    return metrics


# ============================================================================
# 3-way comparison
# ============================================================================

def run_comparison(
    next_item_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
    include_zeroshot: bool = True,
    include_finetuned: bool = True,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run 3-way comparison: zero-shot vs fine-tuned vs Item2Vec.

    Returns:
        (full_catalog_df, common_subset_df) — two DataFrames,
        one for each evaluation mode.
    """
    safe_base = base_model.replace("/", "_").replace(" ", "_")

    # --- Load all embedding sets ---
    models = {}

    # Item2Vec (always included)
    i2v_emb, i2v_ids = load_item2vec_embeddings("amazon_item2vec")
    # L2-normalize for cosine similarity via IndexFlatIP
    norms = np.linalg.norm(i2v_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    i2v_emb = i2v_emb / norms
    models["Item2Vec (behavioral)"] = (i2v_emb, i2v_ids)

    if include_zeroshot:
        zs_tag = f"amazon_kdd_embeddings_{safe_base}"
        zs_emb, zs_ids = load_text_embeddings(zs_tag)
        models[f"Zero-shot ({base_model.split('/')[-1]})"] = (zs_emb, zs_ids)

    if include_finetuned:
        ft_tag = f"amazon_kdd_embeddings_finetuned_{safe_base}"
        try:
            ft_emb, ft_ids = load_text_embeddings(ft_tag)
            models[f"Fine-tuned ({base_model.split('/')[-1]})"] = (ft_emb, ft_ids)
        except FileNotFoundError:
            logger.warning("Fine-tuned embeddings not found — skipping. "
                           "Run: python finetune_contrastive.py --dataset amazon")

    # --- Evaluation 1: Full catalog (each model uses all items it covers) ---
    logger.info("=" * 80)
    logger.info("EVALUATION 1: Full Catalog (each model uses its own item set)")
    logger.info("=" * 80)

    full_rows = []
    for label, (emb, ids) in models.items():
        metrics = next_item_retrieval(
            emb, ids, next_item_map,
            k_values=k_values, max_queries=max_queries,
            label=label,
        )
        print_ranking_metrics(metrics, f"Full Catalog — {label}")
        row = {"model": label, "dim": emb.shape[1], **metrics}
        full_rows.append(row)

    full_df = pd.DataFrame(full_rows)

    # --- Evaluation 2: Common subset (items in ALL models' vocabularies) ---
    common_items = set(i2v_ids)
    for label, (emb, ids) in models.items():
        common_items &= set(ids)

    logger.info("=" * 80)
    logger.info(f"EVALUATION 2: Common Subset ({len(common_items):,} items in ALL models)")
    logger.info("=" * 80)

    common_rows = []
    for label, (emb, ids) in models.items():
        metrics = next_item_retrieval(
            emb, ids, next_item_map,
            k_values=k_values, max_queries=max_queries,
            restrict_to=common_items,
            label=label,
        )
        print_ranking_metrics(metrics, f"Common Subset — {label}")
        row = {"model": label, "dim": emb.shape[1], **metrics}
        common_rows.append(row)

    common_df = pd.DataFrame(common_rows)

    return full_df, common_df


def print_comparison_table(df: pd.DataFrame, title: str):
    """Pretty-print a comparison table."""
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")

    display_cols = ["model", "dim", "coverage", "num_queries"]
    for col in df.columns:
        if col.startswith("precision@") or col.startswith("recall@") or col in ("mrr", "ndcg@10"):
            display_cols.append(col)
    if "avg_next_items" in df.columns:
        display_cols.append("avg_next_items")

    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'=' * 110}\n")


# ============================================================================
# Yandex evaluation (standalone)
# ============================================================================

def evaluate_yandex_item2vec(
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
) -> Dict[str, Any]:
    """
    Evaluate Yandex Item2Vec with next-track retrieval.

    Optionally compare against pre-computed audio CNN embeddings
    if available.
    """
    from data.yandex_dataset import YandexDataset
    from config import DEFAULT_YANDEX_CONFIG

    cfg = DEFAULT_YANDEX_CONFIG

    # Load Item2Vec embeddings
    npz_path = (MODELS_DIR / DEFAULT_ITEM2VEC_CONFIG.output_subdir
                / "yandex_item2vec" / "yandex_item2vec_embeddings.npz")

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Yandex Item2Vec not found: {npz_path}\n"
            f"Run: python item2vec.py --dataset yandex"
        )

    data = np.load(npz_path, allow_pickle=True)
    i2v_emb = data["embeddings"]
    i2v_ids = list(data["ids"])

    # Normalize
    norms = np.linalg.norm(i2v_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    i2v_emb = i2v_emb / norms

    # Build ground truth (uses same sequence mode as training)
    dataset = YandexDataset(
        data_dir=cfg.data_dir,
        min_played_ratio=cfg.min_played_ratio,
        min_listens_per_user=cfg.min_listens_per_user,
        max_listens_per_user=cfg.max_listens_per_user,
        session_gap_seconds=cfg.session_gap_seconds,
    )
    dataset.build_sequences(mode=cfg.sequence_mode)
    next_item_map = dataset.get_next_item_pairs()

    metrics = next_item_retrieval(
        i2v_emb, i2v_ids, next_item_map,
        k_values=k_values, max_queries=max_queries,
        label="Yandex Item2Vec",
    )
    print_ranking_metrics(metrics, "Yandex Item2Vec — Next-Track Retrieval")

    # If audio CNN embeddings exist, compare
    audio_emb = dataset.load_audio_embeddings()
    if audio_emb:
        common_ids = [tid for tid in i2v_ids if tid in audio_emb]
        if len(common_ids) > 100:
            logger.info(f"Audio CNN embeddings available for {len(common_ids):,} tracks — comparing")
            audio_arr = np.array([audio_emb[tid] for tid in common_ids], dtype=np.float32)
            audio_norms = np.linalg.norm(audio_arr, axis=1, keepdims=True)
            audio_norms[audio_norms == 0] = 1.0
            audio_arr = audio_arr / audio_norms

            audio_metrics = next_item_retrieval(
                audio_arr, common_ids, next_item_map,
                k_values=k_values, max_queries=max_queries,
                label="Yandex Audio CNN",
            )
            print_ranking_metrics(audio_metrics, "Yandex Audio CNN — Next-Track Retrieval")

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section 10.2 — Evaluate Item2Vec and compare with text embeddings"
    )
    parser.add_argument("--dataset", choices=["amazon", "yandex", "both"],
                        default="amazon")
    parser.add_argument("--item2vec_only", action="store_true",
                        help="Skip text model comparison, evaluate Item2Vec alone")
    parser.add_argument("--max_queries", type=int, default=20000)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--base_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset in ("amazon", "both"):
        logger.info("=" * 80)
        logger.info("Amazon KDD — Item2Vec Evaluation (Section 10.2)")
        logger.info("=" * 80)

        # Load dataset and ground truth
        config = DEFAULT_AMAZON_KDD_CONFIG
        dataset = AmazonKDDDataset(data_dir=config.data_dir, locale=config.locale)
        dataset.load_products()
        product_ids = set(dataset.products_df["id"])
        dataset.load_sessions(product_ids=product_ids)
        next_item_map = dataset.get_next_item_pairs(
            min_prev_items=config.min_session_length - 1
        )

        if args.item2vec_only:
            i2v_emb, i2v_ids = load_item2vec_embeddings("amazon_item2vec")
            norms = np.linalg.norm(i2v_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            i2v_emb = i2v_emb / norms

            metrics = next_item_retrieval(
                i2v_emb, i2v_ids, next_item_map,
                k_values=args.k_values, max_queries=args.max_queries,
                label="Item2Vec",
            )
            print_ranking_metrics(metrics, "Amazon KDD — Item2Vec Next-Item Retrieval")
        else:
            full_df, common_df = run_comparison(
                next_item_map=next_item_map,
                k_values=args.k_values,
                max_queries=args.max_queries,
                base_model=args.base_model,
            )

            print_comparison_table(full_df, "FULL CATALOG COMPARISON (Amazon KDD)")
            print_comparison_table(common_df, "COMMON SUBSET COMPARISON (Amazon KDD)")

            # Save results
            full_csv = METRICS_DIR / f"item2vec_comparison_full_{ts}.csv"
            common_csv = METRICS_DIR / f"item2vec_comparison_common_{ts}.csv"
            full_df.to_csv(full_csv, index=False)
            common_df.to_csv(common_csv, index=False)

            # Combined JSON
            results = {
                "full_catalog": full_df.to_dict(orient="records"),
                "common_subset": common_df.to_dict(orient="records"),
                "timestamp": ts,
            }
            json_path = METRICS_DIR / f"item2vec_comparison_{ts}.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Saved: {full_csv}")
            logger.info(f"Saved: {common_csv}")
            logger.info(f"Saved: {json_path}")

    if args.dataset in ("yandex", "both"):
        logger.info("=" * 80)
        logger.info("Yandex Yambda — Item2Vec Evaluation (Section 10.2)")
        logger.info("=" * 80)

        evaluate_yandex_item2vec(
            k_values=args.k_values,
            max_queries=args.max_queries,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
