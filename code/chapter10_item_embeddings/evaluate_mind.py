"""
Evaluate Zero-Shot Item Embeddings on MIND News Dataset

Section 10.1: Pre-trained Text Encoder Baselines

Evaluates embedding quality through three complementary lenses:
  1. Category Retrieval  — Do nearest neighbors share the same topic?
  2. Co-Click Retrieval  — Are behaviorally related articles closer?
  3. Qualitative NN      — Human-inspectable nearest neighbor examples

Usage:
    python evaluate_mind.py --compare_models
    python evaluate_mind.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_TEXT_ENCODER, DEFAULT_MIND_CONFIG,
    EMBEDDINGS_DIR, METRICS_DIR
)
from data import MINDDataset
from models import create_encoder
from utils import evaluate_ranking_batch
from utils.metrics import print_ranking_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation 1: Category-Based Retrieval
# ============================================================================

def evaluate_category_retrieval(
    embeddings: np.ndarray,
    news_ids: List[str],
    news_df: pd.DataFrame,
    k_values: List[int] = [1, 5, 10, 20, 50],
    level: str = "category",
    max_queries: int = 2000
) -> Dict[str, float]:
    """
    For each article, retrieve nearest neighbors and measure what fraction
    share the same category (or subcategory).

    This is an intrinsic evaluation: good text embeddings should place
    articles about the same topic close together without any fine-tuning.

    Args:
        embeddings: (N, D) normalized embedding matrix
        news_ids: ordered list of news IDs matching rows of embeddings
        news_df: DataFrame with news_id, category, subcategory columns
        k_values: K values for Precision@K
        level: "category" or "subcategory"
        max_queries: cap on the number of query articles (for speed)

    Returns:
        Dictionary of aggregated metrics
    """
    logger.info(f"Evaluating {level}-based retrieval (max_queries={max_queries})...")

    id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    id_to_label = dict(zip(news_df["news_id"], news_df[level]))

    query_indices = list(range(len(news_ids)))
    if len(query_indices) > max_queries:
        rng = np.random.RandomState(42)
        query_indices = rng.choice(query_indices, size=max_queries, replace=False).tolist()

    max_k = max(k_values)

    # Compute similarity and get top-K+1 (exclude self)
    logger.info("Computing nearest neighbors...")
    sim_matrix = embeddings[query_indices] @ embeddings.T  # (Q, N)

    scores_list = []
    relevance_list = []

    for i, q_idx in enumerate(query_indices):
        q_nid = news_ids[q_idx]
        q_label = id_to_label.get(q_nid)
        if q_label is None:
            continue

        sims = sim_matrix[i].copy()
        sims[q_idx] = -np.inf  # exclude self

        top_indices = np.argpartition(-sims, max_k)[:max_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        top_scores = sims[top_indices]

        relevance = np.array([
            1 if id_to_label.get(news_ids[idx]) == q_label else 0
            for idx in top_indices
        ])

        scores_list.append(top_scores)
        relevance_list.append(relevance)

    logger.info(f"Evaluated {len(scores_list)} queries")

    metrics = evaluate_ranking_batch(
        scores_list, relevance_list, k_values=k_values
    )

    # Also compute category-specific precision (more intuitive name)
    for k in k_values:
        prec_key = f"precision@{k}"
        if prec_key in metrics:
            metrics[f"{level}_precision@{k}"] = metrics[prec_key]

    return metrics


# ============================================================================
# Evaluation 2: Co-Click Retrieval
# ============================================================================

def build_coclick_ground_truth(
    behaviors_df: pd.DataFrame,
    valid_news_ids: set,
    min_history_length: int = 3,
    max_pairs_per_article: int = 20
) -> Dict[str, set]:
    """
    Build ground truth: for each article, the set of articles co-clicked
    by at least one user in the same session/history.

    Args:
        behaviors_df: behaviors DataFrame with history_list column
        valid_news_ids: set of news IDs that have embeddings
        min_history_length: ignore very short histories
        max_pairs_per_article: cap to prevent popular articles dominating

    Returns:
        Dict mapping news_id -> set of co-clicked news_ids
    """
    logger.info("Building co-click ground truth from user histories...")

    coclick_map = defaultdict(set)

    for history in behaviors_df["history_list"]:
        valid_history = [nid for nid in history if nid in valid_news_ids]
        if len(valid_history) < min_history_length:
            continue

        for i, nid_a in enumerate(valid_history):
            for nid_b in valid_history[i+1:]:
                coclick_map[nid_a].add(nid_b)
                coclick_map[nid_b].add(nid_a)

    # Cap per-article pairs for balanced evaluation
    if max_pairs_per_article:
        rng = np.random.RandomState(42)
        for nid in coclick_map:
            partners = coclick_map[nid]
            if len(partners) > max_pairs_per_article:
                coclick_map[nid] = set(rng.choice(list(partners), max_pairs_per_article, replace=False))

    articles_with_coclicks = sum(1 for v in coclick_map.values() if len(v) > 0)
    total_pairs = sum(len(v) for v in coclick_map.values()) // 2
    logger.info(f"Co-click ground truth: {articles_with_coclicks} articles, {total_pairs} unique pairs")

    return coclick_map


def evaluate_coclick_retrieval(
    embeddings: np.ndarray,
    news_ids: List[str],
    coclick_map: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 2000
) -> Dict[str, float]:
    """
    For each article with co-click partners, check if nearest neighbors
    include the co-clicked articles.

    Args:
        embeddings: (N, D) normalized embedding matrix
        news_ids: ordered list of news IDs
        coclick_map: maps news_id -> set of co-clicked news_ids
        k_values: K values for evaluation
        max_queries: cap on queries

    Returns:
        Dictionary of aggregated metrics
    """
    logger.info(f"Evaluating co-click retrieval (max_queries={max_queries})...")

    id_to_idx = {nid: i for i, nid in enumerate(news_ids)}

    # Only query articles that have co-click partners
    query_nids = [nid for nid in news_ids if nid in coclick_map and len(coclick_map[nid]) > 0]
    if len(query_nids) > max_queries:
        rng = np.random.RandomState(42)
        query_nids = list(rng.choice(query_nids, size=max_queries, replace=False))

    logger.info(f"Querying {len(query_nids)} articles with co-click partners")

    max_k = max(k_values)
    query_indices = [id_to_idx[nid] for nid in query_nids]

    sim_matrix = embeddings[query_indices] @ embeddings.T

    scores_list = []
    relevance_list = []

    for i, (q_idx, q_nid) in enumerate(zip(query_indices, query_nids)):
        sims = sim_matrix[i].copy()
        sims[q_idx] = -np.inf  # exclude self

        top_indices = np.argpartition(-sims, max_k)[:max_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        top_scores = sims[top_indices]

        partners = coclick_map[q_nid]
        partner_indices = {id_to_idx[p] for p in partners if p in id_to_idx}

        relevance = np.array([
            1 if idx in partner_indices else 0
            for idx in top_indices
        ])

        scores_list.append(top_scores)
        relevance_list.append(relevance)

    metrics = evaluate_ranking_batch(
        scores_list, relevance_list, k_values=k_values
    )

    # Compute average number of co-click partners per query (context for interpreting recall)
    avg_partners = np.mean([len(coclick_map[nid]) for nid in query_nids])
    metrics["avg_coclick_partners"] = avg_partners
    metrics["num_coclick_queries"] = len(query_nids)

    return metrics


# ============================================================================
# Evaluation 3: Qualitative Nearest Neighbors
# ============================================================================

def print_qualitative_neighbors(
    embeddings: np.ndarray,
    news_ids: List[str],
    news_df: pd.DataFrame,
    num_examples: int = 5,
    top_k: int = 5,
    seed: int = 42
) -> List[Dict]:
    """
    Print nearest neighbor examples for human inspection.

    Picks one article from each of several categories and shows its
    top-K neighbors with titles and categories.

    Returns:
        List of example dictionaries (for saving to file)
    """
    logger.info("Generating qualitative nearest-neighbor examples...")

    id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    id_to_row = {row["news_id"]: row for _, row in news_df.iterrows()}

    # Pick one article per major category
    top_categories = news_df["category"].value_counts().head(num_examples).index.tolist()
    rng = np.random.RandomState(seed)

    examples = []
    print("\n" + "=" * 90)
    print("QUALITATIVE NEAREST NEIGHBORS")
    print("=" * 90)

    for cat in top_categories:
        cat_articles = news_df[news_df["category"] == cat]["news_id"].tolist()
        cat_articles = [nid for nid in cat_articles if nid in id_to_idx]
        if not cat_articles:
            continue

        query_nid = rng.choice(cat_articles)
        q_idx = id_to_idx[query_nid]
        q_row = id_to_row.get(query_nid, {})

        sims = (embeddings[q_idx:q_idx+1] @ embeddings.T).flatten()
        sims[q_idx] = -np.inf

        top_indices = np.argsort(-sims)[:top_k]

        print(f"\nQuery [{cat}]: {q_row.get('title', 'N/A')}")
        print(f"  Subcategory: {q_row.get('subcategory', 'N/A')}")
        print("-" * 90)

        neighbors = []
        for rank, idx in enumerate(top_indices, 1):
            n_nid = news_ids[idx]
            n_row = id_to_row.get(n_nid, {})
            sim = float(sims[idx])
            same_cat = "Y" if n_row.get("category") == cat else "N"

            print(f"  #{rank} (sim={sim:.4f}, same_cat={same_cat}) "
                  f"[{n_row.get('category', '?')}/{n_row.get('subcategory', '?')}] "
                  f"{str(n_row.get('title', 'N/A'))[:80]}")

            neighbors.append({
                "rank": rank, "news_id": n_nid,
                "title": str(n_row.get("title", "")),
                "category": str(n_row.get("category", "")),
                "subcategory": str(n_row.get("subcategory", "")),
                "similarity": sim, "same_category": same_cat == "Y"
            })

        examples.append({
            "query_news_id": query_nid,
            "query_title": str(q_row.get("title", "")),
            "query_category": cat,
            "neighbors": neighbors
        })

    print("=" * 90 + "\n")
    return examples


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

def run_full_evaluation(
    model_name: str,
    sample_size: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 2000,
    show_qualitative: bool = True
) -> Dict[str, Any]:
    """
    Run the complete Section 10.1 evaluation for a single model on MIND.

    Returns:
        Dictionary containing all metrics and metadata
    """
    logger.info("=" * 80)
    logger.info(f"Evaluating: {model_name}")
    logger.info("=" * 80)

    # --- Load dataset ---
    dataset = MINDDataset(
        data_dir=DEFAULT_MIND_CONFIG.data_dir,
        sample_size=sample_size,
        random_seed=42
    )
    news_df = dataset.load_news()
    behaviors_df = dataset.load_behaviors()
    dataset.print_stats()

    # --- Generate text and encode ---
    item_texts = dataset.get_item_texts(
        use_title=DEFAULT_MIND_CONFIG.use_title,
        use_abstract=DEFAULT_MIND_CONFIG.use_abstract,
        use_category=DEFAULT_MIND_CONFIG.use_category,
        use_subcategory=DEFAULT_MIND_CONFIG.use_subcategory,
        template=DEFAULT_MIND_CONFIG.text_template
    )

    encoder = create_encoder(
        model_name=model_name,
        max_seq_length=128,
        normalize_embeddings=True
    )

    logger.info("Encoding news articles...")
    embedding_dict = encoder.encode_dict(item_texts, batch_size=64, show_progress=True)

    news_ids = list(embedding_dict.keys())
    embeddings = np.stack([embedding_dict[nid] for nid in news_ids])
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # --- Save embeddings ---
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    emb_file = EMBEDDINGS_DIR / f"mind_embeddings_{safe_name}.npz"
    np.savez_compressed(emb_file, embeddings=embeddings, news_ids=news_ids,
                        model_name=model_name, embedding_dim=embeddings.shape[1])
    logger.info(f"Saved embeddings to {emb_file}")

    results = {
        "model_name": model_name,
        "embedding_dim": embeddings.shape[1],
        "num_articles": len(embeddings),
    }

    # --- Eval 1: Category retrieval ---
    cat_metrics = evaluate_category_retrieval(
        embeddings, news_ids, news_df,
        k_values=k_values, level="category", max_queries=max_queries
    )
    for k, v in cat_metrics.items():
        results[f"cat_{k}"] = v

    print_ranking_metrics(
        cat_metrics,
        f"Category Retrieval — {model_name}"
    )

    # --- Eval 1b: Subcategory retrieval ---
    subcat_metrics = evaluate_category_retrieval(
        embeddings, news_ids, news_df,
        k_values=k_values, level="subcategory", max_queries=max_queries
    )
    for k, v in subcat_metrics.items():
        results[f"subcat_{k}"] = v

    print_ranking_metrics(
        subcat_metrics,
        f"Subcategory Retrieval — {model_name}"
    )

    # --- Eval 2: Co-click retrieval ---
    valid_ids = set(news_ids)
    coclick_map = build_coclick_ground_truth(
        behaviors_df, valid_ids,
        min_history_length=3,
        max_pairs_per_article=20
    )

    coclick_metrics = evaluate_coclick_retrieval(
        embeddings, news_ids, coclick_map,
        k_values=k_values, max_queries=max_queries
    )
    for k, v in coclick_metrics.items():
        results[f"coclick_{k}"] = v

    print_ranking_metrics(
        coclick_metrics,
        f"Co-Click Retrieval — {model_name}"
    )

    # --- Eval 3: Qualitative neighbors ---
    if show_qualitative:
        examples = print_qualitative_neighbors(
            embeddings, news_ids, news_df,
            num_examples=5, top_k=5
        )
        results["qualitative_examples"] = examples

    return results


def compare_models(
    models: List[str],
    sample_size: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    max_queries: int = 2000
) -> pd.DataFrame:
    """
    Compare multiple models on the MIND evaluation suite.

    Returns:
        DataFrame with side-by-side comparison
    """
    logger.info("=" * 80)
    logger.info("MIND Model Comparison (Section 10.1)")
    logger.info("=" * 80)

    all_results = []

    for model_name in models:
        results = run_full_evaluation(
            model_name=model_name,
            sample_size=sample_size,
            k_values=k_values,
            max_queries=max_queries,
            show_qualitative=(model_name == models[0])  # only for first model
        )
        all_results.append(results)

    # Build comparison DataFrame (exclude nested dicts)
    flat_results = []
    for r in all_results:
        flat = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        flat_results.append(flat)

    df = pd.DataFrame(flat_results)

    # Reorder: model info first, then category, subcat, coclick metrics
    info_cols = ["model_name", "embedding_dim", "num_articles"]
    cat_cols = sorted([c for c in df.columns if c.startswith("cat_")])
    subcat_cols = sorted([c for c in df.columns if c.startswith("subcat_")])
    coclick_cols = sorted([c for c in df.columns if c.startswith("coclick_")])
    remaining = [c for c in df.columns if c not in info_cols + cat_cols + subcat_cols + coclick_cols]
    df = df[info_cols + cat_cols + subcat_cols + coclick_cols + remaining]

    return df, all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot item embeddings on MIND (Section 10.1)"
    )
    parser.add_argument(
        "--model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder model name"
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="Number of news articles to use (None = all)"
    )
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 50],
        help="K values for Precision@K, Recall@K, NDCG@K"
    )
    parser.add_argument(
        "--max_queries", type=int, default=2000,
        help="Max queries per evaluation (for speed)"
    )
    parser.add_argument(
        "--compare_models", action="store_true",
        help="Compare multiple models"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_models:
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            # RexBERT excluded: it's an e-commerce model, not suited for news.
            # See evaluate_amazon.py for the RexBERT comparison.
        ]

        df, all_results = compare_models(
            models=models,
            sample_size=args.sample_size,
            k_values=args.k_values,
            max_queries=args.max_queries
        )

        # Print summary table
        print("\n" + "=" * 100)
        print("MODEL COMPARISON SUMMARY (MIND Dataset)")
        print("=" * 100)

        summary_cols = ["model_name", "embedding_dim"]
        for prefix, label in [("cat_", "Category"), ("coclick_", "CoClick")]:
            for k in [5, 10, 20]:
                col = f"{prefix}precision@{k}"
                if col in df.columns:
                    summary_cols.append(col)
            mrr_col = f"{prefix}mrr"
            if mrr_col in df.columns:
                summary_cols.append(mrr_col)

        available = [c for c in summary_cols if c in df.columns]
        print(df[available].to_string(index=False))
        print("=" * 100)

        # Save full results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(output_dir / f"mind_model_comparison_{ts}.csv", index=False)

        # Save detailed results (without non-serializable objects)
        serializable = []
        for r in all_results:
            clean = {k: v for k, v in r.items()
                     if isinstance(v, (int, float, str, list, dict))}
            serializable.append(clean)
        with open(output_dir / f"mind_detailed_results_{ts}.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")
        return

    # Single model evaluation
    results = run_full_evaluation(
        model_name=args.model,
        sample_size=args.sample_size,
        k_values=args.k_values,
        max_queries=args.max_queries,
        show_qualitative=True
    )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.model.replace("/", "_")
    out_file = output_dir / f"mind_eval_{safe_name}_{ts}.json"
    serializable = {k: v for k, v in results.items()
                    if isinstance(v, (int, float, str, list, dict))}
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
