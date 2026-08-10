"""
Evaluate Zero-Shot Item Embeddings on Amazon KDD 2023 (E-Commerce)

Section 10.1: Pre-trained Text Encoder Baselines — E-Commerce Domain

Evaluates embedding quality through two complementary lenses:
1. Next-Item Retrieval  — Given the last viewed item, is the next engaged
                          item nearby in embedding space?  Respects the
                          temporal ordering within sessions.
2. Qualitative NN       — Human-inspectable nearest neighbor examples

This is where the RexBERT comparison is meaningful: the products have real
English text (titles, descriptions) from a retail domain that matches
RexBERT's pre-training corpus.

Usage:
    python evaluate_amazon.py --compare_models
    python evaluate_amazon.py --model rexbert-base
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
    DEFAULT_AMAZON_KDD_CONFIG, EMBEDDINGS_DIR, METRICS_DIR
)
from data.amazon_kdd_dataset import AmazonKDDDataset
from models import create_encoder
from utils import evaluate_ranking_batch
from utils.faiss_index import build_faiss_index
from utils.metrics import print_ranking_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Cache
# ============================================================================

def _embedding_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_").replace(" ", "_")
    return EMBEDDINGS_DIR / f"amazon_kdd_embeddings_{safe}.npz"


def load_or_encode(
    model_name: str,
    item_texts: Dict[str, str],
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (embeddings, item_ids).

    If a cached .npz exists on disk for this model, load it.
    Otherwise encode all items and save to disk for future reuse.
    """
    emb_file = _embedding_path(model_name)

    expected_count = len(item_texts)

    if emb_file.exists():
        logger.info(f"Found cached embeddings at {emb_file}")
        data = np.load(emb_file, allow_pickle=True)
        embeddings = data["embeddings"]
        item_ids = list(data["item_ids"])

        if embeddings.shape[0] == expected_count:
            logger.info(f"  Cache valid: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
            return embeddings, item_ids
        else:
            logger.warning(f"  Cache stale ({embeddings.shape[0]} != {expected_count} expected). Re-encoding.")

    logger.info(f"No cache found — encoding {len(item_texts)} products with {model_name}")
    encoder = create_encoder(
        model_name=model_name,
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
        model_name=model_name,
        embedding_dim=embeddings.shape[1],
    )
    logger.info(f"Saved embeddings to {emb_file}  ({embeddings.shape})")
    return embeddings, item_ids


# ============================================================================
# Evaluation 1: Next-Item Retrieval (temporally ordered, FAISS)
# ============================================================================

def evaluate_next_item_retrieval(
    embeddings: np.ndarray,
    item_ids: List[str],
    next_item_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 5000,
) -> Dict[str, float]:
    """
    For each query item (last item viewed in a session), check whether
    the actual next item the user engaged with appears among the nearest
    neighbors in embedding space.

    Uses FAISS for memory-efficient search over 500K+ items.
    """
    logger.info(f"Evaluating next-item retrieval (max_queries={max_queries})...")

    id_to_idx = {pid: i for i, pid in enumerate(item_ids)}

    query_ids = [pid for pid in item_ids
                 if pid in next_item_map and len(next_item_map[pid]) > 0]
    if len(query_ids) > max_queries:
        rng = np.random.RandomState(42)
        query_ids = list(rng.choice(query_ids, size=max_queries, replace=False))

    logger.info(f"  {len(query_ids)} query items with next-item targets")

    if len(query_ids) == 0:
        logger.warning("  No query items found — skipping next-item retrieval")
        return {"avg_next_items": 0, "num_queries": 0}

    max_k = max(k_values)
    fetch_k = max_k + 1  # +1 so we can drop self-hit

    # Build FAISS index (IndexFlatIP for cosine sim on normalised vectors)
    logger.info("  Building FAISS index...")
    faiss_index = build_faiss_index(embeddings, index_type="IndexFlatIP")

    query_indices = np.array([id_to_idx[pid] for pid in query_ids])
    query_embs = embeddings[query_indices].astype(np.float32)

    logger.info(f"  Searching FAISS (k={fetch_k})...")
    distances, nn_indices = faiss_index.search(query_embs, fetch_k)

    scores_list = []
    relevance_list = []

    for i, q_id in enumerate(query_ids):
        q_idx = query_indices[i]
        targets = next_item_map[q_id]
        target_idx_set = {id_to_idx[p] for p in targets if p in id_to_idx}

        # Drop self from FAISS results
        row_indices = nn_indices[i]
        row_dists = distances[i]
        mask = row_indices != q_idx
        row_indices = row_indices[mask][:max_k]
        row_dists = row_dists[mask][:max_k]

        relevance = np.array([
            1 if idx in target_idx_set else 0
            for idx in row_indices
        ])

        scores_list.append(row_dists)
        relevance_list.append(relevance)

    metrics = evaluate_ranking_batch(
        scores_list, relevance_list, k_values=k_values
    )

    avg_targets = np.mean([len(next_item_map[pid]) for pid in query_ids])
    metrics["avg_next_items"] = avg_targets
    metrics["num_queries"] = len(query_ids)

    return metrics


# ============================================================================
# Evaluation 2: Qualitative Nearest Neighbors (FAISS)
# ============================================================================

def print_qualitative_neighbors(
    embeddings: np.ndarray,
    item_ids: List[str],
    products_df: pd.DataFrame,
    faiss_index,
    num_examples: int = 6,
    top_k: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """Print nearest neighbor examples for human inspection."""
    logger.info("Generating qualitative nearest-neighbor examples...")

    id_to_idx = {pid: i for i, pid in enumerate(item_ids)}
    id_to_row = {row["id"]: row for _, row in products_df.iterrows()
                 if row["id"] in id_to_idx}

    safe = lambda s: s.encode("ascii", errors="replace").decode() if isinstance(s, str) else str(s)

    valid_ids = [pid for pid in item_ids if pid in id_to_row]
    if not valid_ids:
        logger.warning("No valid products for qualitative examples")
        return []

    rng = np.random.RandomState(seed)
    sample_ids = rng.choice(valid_ids, size=min(num_examples, len(valid_ids)), replace=False)

    # Batch FAISS search
    sample_indices = np.array([id_to_idx[pid] for pid in sample_ids])
    query_embs = embeddings[sample_indices].astype(np.float32)
    dists, nn_indices = faiss_index.search(query_embs, top_k + 1)

    examples = []
    print("\n" + "=" * 100)
    print("QUALITATIVE NEAREST NEIGHBORS (Amazon KDD)")
    print("=" * 100)

    for i, query_id in enumerate(sample_ids):
        q_idx = sample_indices[i]
        q_row = id_to_row.get(query_id, {})

        row_idx = nn_indices[i]
        row_dist = dists[i]
        mask = row_idx != q_idx
        row_idx = row_idx[mask][:top_k]
        row_dist = row_dist[mask][:top_k]

        q_title = safe(str(q_row.get("title", "N/A"))[:90])
        q_brand = safe(str(q_row.get("brand", "N/A"))[:40])
        q_price = q_row.get("price", "N/A")

        print(f"\nQuery: {q_title}")
        print(f"  Brand: {q_brand} | Price: {q_price}")
        print("-" * 100)

        neighbors = []
        for rank, (idx, sim) in enumerate(zip(row_idx, row_dist), 1):
            n_id = item_ids[idx]
            n_row = id_to_row.get(n_id, {})
            n_title = safe(str(n_row.get("title", "N/A"))[:80])
            n_brand = safe(str(n_row.get("brand", "?"))[:30])
            same_brand = "Y" if n_row.get("brand") == q_row.get("brand") else "N"

            print(f"  #{rank} (sim={float(sim):.4f}, same_brand={same_brand}) "
                  f"[{n_brand}] {n_title}")

            neighbors.append({
                "rank": rank, "product_id": n_id,
                "title": str(n_row.get("title", "")),
                "brand": str(n_row.get("brand", "")),
                "similarity": float(sim), "same_brand": same_brand == "Y",
            })

        examples.append({
            "query_id": query_id,
            "query_title": str(q_row.get("title", "")),
            "query_brand": str(q_row.get("brand", "")),
            "neighbors": neighbors,
        })

    print("=" * 100 + "\n")
    return examples


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

def run_full_evaluation(
    model_name: str,
    dataset: AmazonKDDDataset,
    next_item_map: Dict[str, set],
    config,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 5000,
    show_qualitative: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete evaluation for a single model on Amazon KDD.

    The dataset and next_item_map are passed in so they are loaded once
    and shared across models.
    """
    logger.info("=" * 80)
    logger.info(f"Evaluating: {model_name} on Amazon KDD 2023 (UK)")
    logger.info("=" * 80)

    item_texts = dataset.get_item_texts(
        use_title=config.use_title,
        use_brand=config.use_brand,
        use_description=config.use_description,
        use_color=config.use_color,
        use_material=config.use_material,
        template=config.text_template,
    )

    embeddings, item_ids = load_or_encode(model_name, item_texts)

    results: Dict[str, Any] = {
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_products": len(embeddings),
    }

    # --- Eval 1: Next-item retrieval (FAISS) ---
    next_metrics = evaluate_next_item_retrieval(
        embeddings, item_ids, next_item_map,
        k_values=k_values, max_queries=max_queries,
    )
    for k, v in next_metrics.items():
        results[f"next_{k}"] = v

    print_ranking_metrics(next_metrics, f"Next-Item Retrieval — {model_name}")

    # --- Eval 2: Qualitative neighbors (FAISS) ---
    if show_qualitative:
        faiss_index = build_faiss_index(embeddings, index_type="IndexFlatIP")
        examples = print_qualitative_neighbors(
            embeddings, item_ids, dataset.products_df,
            faiss_index=faiss_index,
            num_examples=6, top_k=5,
        )
        results["qualitative_examples"] = examples

    return results


def compare_models(
    models: List[str],
    dataset: AmazonKDDDataset,
    next_item_map: Dict[str, set],
    config,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 5000,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Compare multiple models (dataset loaded once, shared)."""
    logger.info("=" * 80)
    logger.info("Amazon KDD Model Comparison (Section 10.1 — E-Commerce)")
    logger.info("=" * 80)

    all_results = []
    for model_name in models:
        results = run_full_evaluation(
            model_name=model_name,
            dataset=dataset,
            next_item_map=next_item_map,
            config=config,
            k_values=k_values,
            max_queries=max_queries,
            show_qualitative=(model_name == models[0]),
        )
        all_results.append(results)

    flat_results = []
    for r in all_results:
        flat_results.append({k: v for k, v in r.items() if not isinstance(v, (list, dict))})

    df = pd.DataFrame(flat_results)

    info_cols = ["model_name", "embedding_dim", "num_products"]
    next_cols = sorted([c for c in df.columns if c.startswith("next_")])
    remaining = [c for c in df.columns if c not in info_cols + next_cols]
    df = df[info_cols + next_cols + remaining]

    return df, all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot item embeddings on Amazon KDD 2023 (Section 10.1)"
    )
    parser.add_argument("--model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[1, 5, 10, 20, 50])
    parser.add_argument("--max_queries", type=int, default=5000)
    parser.add_argument("--compare_models", action="store_true",
                        help="Compare SBERT + RexBERT models")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_AMAZON_KDD_CONFIG

    # ---- Load dataset once (shared across all models) ----
    logger.info("Loading Amazon KDD dataset (all UK products)...")
    dataset = AmazonKDDDataset(data_dir=config.data_dir, locale=config.locale)
    dataset.load_products()
    dataset.print_stats()

    product_id_set = set(dataset.products_df["id"])
    dataset.load_sessions(product_ids=product_id_set)

    next_item_map = dataset.get_next_item_pairs(
        min_prev_items=config.min_session_length - 1,
    )

    # ---- Run evaluation(s) ----
    if args.compare_models:
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "rexbert-base",
        ]

        df, all_results = compare_models(
            models=models,
            dataset=dataset,
            next_item_map=next_item_map,
            config=config,
            k_values=args.k_values,
            max_queries=args.max_queries,
        )

        # Print summary
        print("\n" + "=" * 110)
        print("MODEL COMPARISON SUMMARY (Amazon KDD 2023 — E-Commerce)")
        print("=" * 110)

        summary_cols = ["model_name", "embedding_dim"]
        for k in [5, 10, 20]:
            col = f"next_precision@{k}"
            if col in df.columns:
                summary_cols.append(col)
        for extra in ["next_mrr", "next_avg_next_items", "next_num_queries"]:
            if extra in df.columns:
                summary_cols.append(extra)

        available = [c for c in summary_cols if c in df.columns]
        print(df[available].to_string(index=False))
        print("=" * 110)

        # Save comparison CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"amazon_kdd_model_comparison_{ts}.csv"
        df.to_csv(csv_path, index=False)

        # Save detailed JSON
        json_path = output_dir / f"amazon_kdd_detailed_results_{ts}.json"
        serializable = []
        for r in all_results:
            serializable.append({k: v for k, v in r.items()
                                 if isinstance(v, (int, float, str, list, dict))})
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Comparison CSV  -> {csv_path}")
        logger.info(f"Detailed JSON   -> {json_path}")
        return

    # ---- Single model ----
    results = run_full_evaluation(
        model_name=args.model,
        dataset=dataset,
        next_item_map=next_item_map,
        config=config,
        k_values=args.k_values,
        max_queries=args.max_queries,
        show_qualitative=True,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.model.replace("/", "_")
    out_file = output_dir / f"amazon_kdd_eval_{safe_name}_{ts}.json"
    serializable = {k: v for k, v in results.items()
                    if isinstance(v, (int, float, str, list, dict))}
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
