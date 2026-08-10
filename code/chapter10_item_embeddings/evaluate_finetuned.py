"""
Evaluate Fine-Tuned vs Zero-Shot Item Embeddings

Section 10.3: Side-by-side comparison of contrastive-finetuned encoders
against the zero-shot baselines from Section 10.1.

Runs the same evaluation suite on both the original pre-trained model and
the fine-tuned checkpoint, producing a direct comparison table.

For Amazon KDD:  next-item retrieval (same as evaluate_amazon.py)
For MIND:        category retrieval + co-click retrieval (same as evaluate_mind.py)

Usage:
    # Compare on Amazon KDD
    python evaluate_finetuned.py --dataset amazon

    # Compare on MIND
    python evaluate_finetuned.py --dataset mind

    # Point to a specific fine-tuned checkpoint
    python evaluate_finetuned.py --dataset amazon --finetuned_path outputs/models/...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_CONTRASTIVE_CONFIG,
    DEFAULT_AMAZON_KDD_CONFIG,
    DEFAULT_MIND_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
    EMBEDDINGS_DIR,
)
from data.amazon_kdd_dataset import AmazonKDDDataset
from data.mind_dataset import MINDDataset
from models import create_encoder, TextEncoder
from utils import evaluate_ranking_batch
from utils.faiss_index import build_faiss_index
from utils.metrics import print_ranking_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Encoding helpers
# ============================================================================

def _safe_model_tag(model_path_or_name: str) -> str:
    """Create a filesystem-safe tag from a model name or path."""
    return str(model_path_or_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def encode_items(
    model_path_or_name: str,
    item_texts: Dict[str, str],
    cache_tag: str,
    batch_size: int = 64,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode all items with a model (zero-shot or fine-tuned).

    Caches to EMBEDDINGS_DIR / f"{cache_tag}.npz".
    """
    emb_file = EMBEDDINGS_DIR / f"{cache_tag}.npz"
    expected = len(item_texts)

    if not force_recompute and emb_file.exists():
        data = np.load(emb_file, allow_pickle=True)
        if data["embeddings"].shape[0] == expected:
            logger.info(f"Loaded cached embeddings: {emb_file} ({expected} items)")
            return data["embeddings"], list(data["item_ids"])
        logger.warning(f"Stale cache ({data['embeddings'].shape[0]} != {expected}). Re-encoding.")

    logger.info(f"Encoding {expected} items with {model_path_or_name}")

    model_path = Path(model_path_or_name)
    if model_path.exists():
        encoder = TextEncoder(
            model_name=str(model_path),
            max_seq_length=128,
            normalize_embeddings=True,
        )
    else:
        encoder = create_encoder(
            model_name=model_path_or_name,
            max_seq_length=128,
            normalize_embeddings=True,
        )

    embedding_dict = encoder.encode_dict(item_texts, batch_size=batch_size, show_progress=True)

    item_ids = list(embedding_dict.keys())
    embeddings = np.stack([embedding_dict[pid] for pid in item_ids])

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        emb_file,
        embeddings=embeddings,
        item_ids=np.array(item_ids),
    )
    logger.info(f"Saved embeddings to {emb_file}  shape={embeddings.shape}")

    return embeddings, item_ids


# ============================================================================
# Amazon KDD evaluation (mirrors evaluate_amazon.py)
# ============================================================================

def evaluate_amazon(
    embeddings: np.ndarray,
    item_ids: List[str],
    next_item_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
) -> Dict[str, float]:
    """Next-item retrieval evaluation using FAISS."""
    logger.info(f"Amazon next-item retrieval (max_queries={max_queries})")

    id_to_idx = {pid: i for i, pid in enumerate(item_ids)}
    query_ids = [pid for pid in item_ids
                 if pid in next_item_map and len(next_item_map[pid]) > 0]

    if len(query_ids) > max_queries:
        rng = np.random.RandomState(42)
        query_ids = list(rng.choice(query_ids, size=max_queries, replace=False))

    if len(query_ids) == 0:
        logger.warning("No query items found")
        return {"num_queries": 0}

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
    metrics["avg_next_items"] = np.mean([len(next_item_map[pid]) for pid in query_ids])

    return metrics


# ============================================================================
# MIND evaluation (mirrors evaluate_mind.py)
# ============================================================================

def evaluate_mind_category(
    embeddings: np.ndarray,
    news_ids: List[str],
    news_df: pd.DataFrame,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
) -> Dict[str, float]:
    """Category retrieval using FAISS: do nearest neighbors share the same topic?"""
    id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    id_to_cat = dict(zip(news_df["news_id"], news_df["category"]))

    query_indices = list(range(len(news_ids)))
    if len(query_indices) > max_queries:
        rng = np.random.RandomState(42)
        query_indices = rng.choice(query_indices, size=max_queries, replace=False).tolist()

    max_k = max(k_values)
    faiss_index = build_faiss_index(embeddings, index_type="IndexFlatIP")

    query_embs = embeddings[query_indices].astype(np.float32)
    distances, nn_indices = faiss_index.search(query_embs, max_k + 1)

    scores_list, relevance_list = [], []
    for i, q_idx in enumerate(query_indices):
        q_nid = news_ids[q_idx]
        q_cat = id_to_cat.get(q_nid)
        if q_cat is None:
            continue

        row_idx = nn_indices[i]
        row_dist = distances[i]
        mask = row_idx != q_idx
        row_idx = row_idx[mask][:max_k]
        row_dist = row_dist[mask][:max_k]

        relevance = np.array([
            1 if id_to_cat.get(news_ids[idx]) == q_cat else 0
            for idx in row_idx
        ])
        scores_list.append(row_dist)
        relevance_list.append(relevance)

    return evaluate_ranking_batch(scores_list, relevance_list, k_values=k_values)


def evaluate_mind_coclick(
    embeddings: np.ndarray,
    news_ids: List[str],
    coclick_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
) -> Dict[str, float]:
    """Co-click retrieval using FAISS: are behaviorally related articles closer?"""
    id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    query_nids = [nid for nid in news_ids
                  if nid in coclick_map and len(coclick_map[nid]) > 0]
    if len(query_nids) > max_queries:
        rng = np.random.RandomState(42)
        query_nids = list(rng.choice(query_nids, size=max_queries, replace=False))

    max_k = max(k_values)
    faiss_index = build_faiss_index(embeddings, index_type="IndexFlatIP")

    query_indices = np.array([id_to_idx[nid] for nid in query_nids])
    query_embs = embeddings[query_indices].astype(np.float32)
    distances, nn_indices = faiss_index.search(query_embs, max_k + 1)

    scores_list, relevance_list = [], []
    for i, (q_idx, q_nid) in enumerate(zip(query_indices, query_nids)):
        row_idx = nn_indices[i]
        row_dist = distances[i]
        mask = row_idx != q_idx
        row_idx = row_idx[mask][:max_k]
        row_dist = row_dist[mask][:max_k]

        partners = coclick_map[q_nid]
        partner_idx_set = {id_to_idx[p] for p in partners if p in id_to_idx}

        relevance = np.array([1 if idx in partner_idx_set else 0 for idx in row_idx])
        scores_list.append(row_dist)
        relevance_list.append(relevance)

    metrics = evaluate_ranking_batch(scores_list, relevance_list, k_values=k_values)
    metrics["num_queries"] = len(query_nids)
    return metrics


# ============================================================================
# Head-to-head comparison
# ============================================================================

def find_finetuned_model(dataset_name: str, base_model: str) -> Optional[Path]:
    """Locate the fine-tuned checkpoint directory."""
    safe_base = base_model.replace("/", "_")
    ds_tag = "amazon" if "amazon" in dataset_name else dataset_name
    candidate = MODELS_DIR / "contrastive_finetuned" / f"{safe_base}_{ds_tag}"
    if candidate.exists():
        return candidate
    return None


def compare_amazon(
    base_model: str,
    finetuned_path: Path,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 20000,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Run Amazon KDD evaluation for both zero-shot and fine-tuned."""
    config = DEFAULT_AMAZON_KDD_CONFIG

    logger.info("Loading Amazon KDD dataset...")
    dataset = AmazonKDDDataset(data_dir=config.data_dir, locale=config.locale)
    dataset.load_products()
    product_ids = set(dataset.products_df["id"])
    dataset.load_sessions(product_ids=product_ids)
    next_item_map = dataset.get_next_item_pairs(min_prev_items=config.min_session_length - 1)

    item_texts = dataset.get_item_texts(
        use_title=config.use_title,
        use_brand=config.use_brand,
        use_description=config.use_description,
        template=config.text_template,
    )

    results_rows = []

    # --- Zero-shot baseline ---
    safe_base = _safe_model_tag(base_model)
    zs_tag = f"amazon_kdd_embeddings_{safe_base}"
    zs_emb, zs_ids = encode_items(base_model, item_texts, zs_tag,
                                   force_recompute=force_recompute)

    zs_metrics = evaluate_amazon(zs_emb, zs_ids, next_item_map,
                                  k_values=k_values, max_queries=max_queries)
    print_ranking_metrics(zs_metrics, f"Zero-Shot — {base_model}")
    zs_metrics["model"] = f"zero-shot ({base_model.split('/')[-1]})"
    zs_metrics["dim"] = zs_emb.shape[1]
    results_rows.append(zs_metrics)

    # --- Fine-tuned ---
    ft_tag = f"amazon_kdd_embeddings_finetuned_{safe_base}"
    ft_emb, ft_ids = encode_items(str(finetuned_path), item_texts, ft_tag,
                                   force_recompute=force_recompute)

    ft_metrics = evaluate_amazon(ft_emb, ft_ids, next_item_map,
                                  k_values=k_values, max_queries=max_queries)
    print_ranking_metrics(ft_metrics, f"Fine-Tuned — {finetuned_path.name}")
    ft_metrics["model"] = f"fine-tuned ({finetuned_path.name})"
    ft_metrics["dim"] = ft_emb.shape[1]
    results_rows.append(ft_metrics)

    df = pd.DataFrame(results_rows)
    return df


def compare_mind(
    base_model: str,
    finetuned_path: Path,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 2000,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Run MIND evaluation for both zero-shot and fine-tuned."""
    from collections import defaultdict

    config = DEFAULT_MIND_CONFIG

    logger.info("Loading MIND dataset...")
    dataset = MINDDataset(data_dir=config.data_dir)
    news_df = dataset.load_news()
    behaviors_df = dataset.load_behaviors()

    item_texts = dataset.get_item_texts(
        use_title=config.use_title,
        use_abstract=config.use_abstract,
        use_category=config.use_category,
        use_subcategory=config.use_subcategory,
        template=config.text_template,
    )

    # Build co-click ground truth
    valid_ids = set(item_texts.keys())
    coclick_map = defaultdict(set)
    for history in behaviors_df["history_list"]:
        valid_h = [nid for nid in history if nid in valid_ids]
        if len(valid_h) < 3:
            continue
        for i, a in enumerate(valid_h):
            for b in valid_h[i + 1:]:
                coclick_map[a].add(b)
                coclick_map[b].add(a)

    # Cap per-article to keep evaluation balanced
    rng = np.random.RandomState(42)
    for nid in coclick_map:
        partners = coclick_map[nid]
        if len(partners) > 20:
            coclick_map[nid] = set(rng.choice(list(partners), 20, replace=False))

    results_rows = []

    for label, model_ref in [
        (f"zero-shot ({base_model.split('/')[-1]})", base_model),
        (f"fine-tuned ({finetuned_path.name})", str(finetuned_path)),
    ]:
        safe = _safe_model_tag(model_ref)
        tag = f"mind_embeddings_{safe}"
        emb, ids = encode_items(model_ref, item_texts, tag,
                                 force_recompute=force_recompute)

        cat_m = evaluate_mind_category(emb, ids, news_df,
                                        k_values=k_values, max_queries=max_queries)
        cc_m = evaluate_mind_coclick(emb, ids, coclick_map,
                                      k_values=k_values, max_queries=max_queries)

        print_ranking_metrics(cat_m, f"Category Retrieval — {label}")
        print_ranking_metrics(cc_m, f"Co-Click Retrieval — {label}")

        row = {"model": label, "dim": emb.shape[1]}
        for k, v in cat_m.items():
            row[f"cat_{k}"] = v
        for k, v in cc_m.items():
            row[f"cc_{k}"] = v
        results_rows.append(row)

    return pd.DataFrame(results_rows)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section 10.3 — Compare zero-shot vs fine-tuned embeddings"
    )
    parser.add_argument(
        "--dataset", type=str, default="amazon",
        choices=["amazon", "amazon_kdd", "mind"],
    )
    parser.add_argument("--finetuned_path", type=str, default=None,
                        help="Path to fine-tuned model dir (auto-detected if omitted)")
    parser.add_argument("--base_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--max_queries", type=int, default=20000)
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore cached embeddings and re-encode")

    args = parser.parse_args()

    # Resolve fine-tuned path
    if args.finetuned_path:
        ft_path = Path(args.finetuned_path)
    else:
        ft_path = find_finetuned_model(args.dataset, args.base_model)

    if ft_path is None or not ft_path.exists():
        print(f"ERROR: Fine-tuned model not found. Expected at:")
        print(f"  {ft_path or find_finetuned_model(args.dataset, args.base_model)}")
        print(f"\nRun training first:")
        print(f"  python finetune_contrastive.py --dataset {args.dataset}")
        sys.exit(1)

    logger.info(f"Fine-tuned model: {ft_path}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset in ("amazon", "amazon_kdd"):
        df = compare_amazon(
            base_model=args.base_model,
            finetuned_path=ft_path,
            k_values=args.k_values,
            max_queries=args.max_queries,
            force_recompute=args.force_recompute,
        )
    elif args.dataset == "mind":
        df = compare_mind(
            base_model=args.base_model,
            finetuned_path=ft_path,
            k_values=args.k_values,
            max_queries=args.max_queries,
            force_recompute=args.force_recompute,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Print summary table
    print("\n" + "=" * 110)
    print(f"ZERO-SHOT vs FINE-TUNED COMPARISON ({args.dataset.upper()})")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)

    # Compute deltas
    if len(df) == 2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        deltas = df[numeric_cols].iloc[1] - df[numeric_cols].iloc[0]
        print("\nIMPROVEMENT (fine-tuned minus zero-shot):")
        for col in numeric_cols:
            d = deltas[col]
            if abs(d) > 1e-6:
                sign = "+" if d > 0 else ""
                print(f"  {col:.<40} {sign}{d:.4f}")

    # Save
    csv_path = METRICS_DIR / f"finetuned_comparison_{args.dataset}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison saved to {csv_path}")

    # Save as JSON too
    json_path = METRICS_DIR / f"finetuned_comparison_{args.dataset}_{ts}.json"
    records = df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
