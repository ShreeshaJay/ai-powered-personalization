"""
Evaluation Script for Section 11.1: Aggregation Baselines

Evaluates five aggregation methods for constructing user/session embeddings
from item embeddings, on two datasets:

1. Amazon KDD (session-based): Can a session embedding built from browsing
   history retrieve the next item the user clicks?
2. MIND (user-based): Can a user embedding built from reading history
   predict which article the user clicks next?

Comparison baselines from Chapter 10:
- "Last-item only" (Chapter 10 baseline): query = embedding of the single
  last item viewed. MRR=0.319 on Amazon KDD with fine-tuned SBERT.

Key result to watch: Does aggregating multiple items beat using just the
last item? For short sessions (Amazon), the answer may be nuanced.
For long histories (MIND), aggregation should clearly help.

Usage:
    python evaluate_aggregation.py [--dataset amazon|mind|both] [--quick]
"""

import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from config import (
    DEFAULT_AMAZON_SESSION_CONFIG,
    DEFAULT_MIND_USER_CONFIG,
    DEFAULT_AGGREGATION_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    METRICS_DIR,
    EMBEDDINGS_DIR,
)
from data.amazon_session_loader import AmazonSessionDataset
from data.mind_user_loader import MINDUserDataset
from models.aggregators import (
    SimpleMeanAggregator,
    LastKMeanAggregator,
    ExponentialDecayAggregator,
    TFIDFWeightedAggregator,
    TFIDFRecencyAggregator,
    compute_idf_scores,
    create_aggregators,
)
from utils.faiss_index import build_faiss_index, FAISS_AVAILABLE, brute_force_search
from utils.metrics import (
    evaluate_ranking,
    print_ranking_metrics,
    mean_reciprocal_rank,
)


# ============================================================================
# Amazon KDD Evaluation (Session-Based)
# ============================================================================

def evaluate_amazon(
    max_sessions: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
    include_item2vec: bool = True,
) -> Dict:
    """Evaluate aggregation baselines on Amazon KDD next-item retrieval.

    Protocol:
    1. Load sessions with >= 3 prev_items and full embedding coverage.
    2. For each session: aggregate prev_items embeddings -> session embedding.
    3. Build FAISS index over all 500K product embeddings.
    4. Retrieve top-K items using session embedding as query.
    5. Evaluate: is the next_item in the top-K?
    6. Compare against "last-item only" baseline from Chapter 10.

    Note on leave-last-out bias: see Chapter11_Design_Notes.md.
    """
    cfg = DEFAULT_AMAZON_SESSION_CONFIG
    agg_cfg = DEFAULT_AGGREGATION_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_sessions is None:
        max_sessions = eval_cfg.max_eval_sessions_amazon

    logger.info("=" * 80)
    logger.info("AMAZON KDD: Session Embedding Evaluation (Section 11.1)")
    logger.info("=" * 80)

    # --- Load data ---
    logger.info("\n[1/5] Loading dataset and cached item embeddings...")
    dataset = AmazonSessionDataset(
        data_dir=cfg.data_dir,
        locale=cfg.locale,
        min_session_length=cfg.min_session_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()
    dataset.print_stats()

    # --- Get evaluation sessions ---
    logger.info("\n[2/5] Preparing evaluation sessions...")
    sessions = dataset.get_evaluation_sessions(max_sessions=max_sessions, seed=seed)
    logger.info(f"  Evaluation sessions: {len(sessions):,}")

    # --- Compute IDF scores ---
    logger.info("\n[3/5] Computing IDF scores from session data...")
    item_freq = dataset.get_item_frequency()
    total_sessions = len(dataset.sessions_df)
    idf_scores = compute_idf_scores(item_freq, total_sessions, smooth=agg_cfg.idf_smooth)
    logger.info(f"  IDF vocabulary size: {len(idf_scores):,}")

    # Compute median IDF for default
    idf_values = np.array(list(idf_scores.values()))
    default_idf = float(np.median(idf_values))
    logger.info(f"  Median IDF (default for unseen items): {default_idf:.3f}")

    # --- Determine data-driven hyperparameters ---
    session_stats = dataset.get_session_length_stats()
    median_len = session_stats["median"]
    decay_half_life = max(1.0, median_len - 1)

    # Last-K values: grounded in percentiles of session length distribution
    last_k_values = sorted(set([
        max(2, int(session_stats["p25"])),
        max(2, int(median_len)),
        max(3, int(session_stats["p75"])),
    ]))

    logger.info(f"\n  Data-driven hyperparameters:")
    logger.info(f"    Median session length: {median_len}")
    logger.info(f"    Last-K values: {last_k_values}")
    logger.info(f"    Decay half-life (positions): {decay_half_life:.1f}")

    # --- Build FAISS index ---
    logger.info("\n[4/5] Building FAISS index over item catalog...")
    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}

    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(all_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None
        logger.warning("FAISS not available, using brute-force search (slower)")

    # --- Create aggregators ---
    aggregators = create_aggregators(
        idf_scores=idf_scores,
        last_k_values=last_k_values,
        decay_half_life=decay_half_life,
        decay_mode="position",
    )

    # --- Evaluate each aggregator ---
    logger.info(f"\n[5/5] Evaluating {len(aggregators)} aggregation methods...")
    all_results = []

    for agg in aggregators:
        logger.info(f"\n  --- {agg.name} ---")
        t0 = time.time()

        # Build user/session embeddings
        mrr_values = []
        all_query_metrics = []

        for session in sessions:
            # Aggregate item embeddings into session embedding
            user_emb = agg.aggregate(
                item_embeddings=session["history_embeddings"],
                item_ids=session.get("history_item_ids"),
            )
            user_emb = user_emb.reshape(1, -1).astype(np.float32)

            # Retrieve top-K items
            max_k = max(k_values) + 1  # +1 to handle potential self-hit
            if faiss_index is not None:
                distances, indices = faiss_index.search(user_emb, max_k)
            else:
                distances, indices = brute_force_search(
                    all_embeddings, user_emb, k=max_k, metric="cosine"
                )

            retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
            target_id = session["target_item_id"]

            # Build relevance vector (exclude items in the session history
            # to avoid trivially retrieving what we already know)
            history_set = set(session["history_item_ids"])
            filtered_ids = [rid for rid in retrieved_ids if rid not in history_set]
            # Take only max(k_values) after filtering
            filtered_ids = filtered_ids[:max(k_values)]

            relevance = np.array([
                1.0 if rid == target_id else 0.0
                for rid in filtered_ids
            ])

            # Compute metrics
            query_metrics = evaluate_ranking(
                scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
                relevance=relevance,
                k_values=k_values,
            )
            all_query_metrics.append(query_metrics)
            mrr_values.append(query_metrics["mrr"])

        # Aggregate metrics across all sessions
        agg_metrics = {}
        for metric_name in all_query_metrics[0].keys():
            values = [m[metric_name] for m in all_query_metrics]
            agg_metrics[metric_name] = float(np.mean(values))

        elapsed = time.time() - t0
        agg_metrics["num_queries"] = len(sessions)
        agg_metrics["method"] = agg.name
        agg_metrics["time_seconds"] = round(elapsed, 2)

        print_ranking_metrics(agg_metrics, f"Amazon KDD - {agg.name}")
        all_results.append(agg_metrics)

    # --- Add Chapter 10 baseline for comparison ---
    logger.info("\n  --- Last-Item Only (Chapter 10 Baseline) ---")
    t0 = time.time()
    mrr_values = []
    all_query_metrics = []

    for session in sessions:
        # Use only the LAST item in prev_items as query (Chapter 10 approach)
        last_item_emb = session["history_embeddings"][-1:].astype(np.float32)

        max_k = max(k_values) + 1
        if faiss_index is not None:
            distances, indices = faiss_index.search(last_item_emb, max_k)
        else:
            distances, indices = brute_force_search(
                all_embeddings, last_item_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
        target_id = session["target_item_id"]

        # Exclude the query item itself (self-hit)
        last_item_id = session["history_item_ids"][-1]
        filtered_ids = [rid for rid in retrieved_ids if rid != last_item_id]
        filtered_ids = filtered_ids[:max(k_values)]

        relevance = np.array([
            1.0 if rid == target_id else 0.0
            for rid in filtered_ids
        ])

        query_metrics = evaluate_ranking(
            scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
            relevance=relevance,
            k_values=k_values,
        )
        all_query_metrics.append(query_metrics)

    baseline_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        baseline_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    baseline_metrics["num_queries"] = len(sessions)
    baseline_metrics["method"] = "Last-Item Only (Ch10 Baseline)"
    baseline_metrics["time_seconds"] = round(elapsed, 2)

    print_ranking_metrics(baseline_metrics, "Amazon KDD - Last-Item Only (Ch10 Baseline)")
    all_results.append(baseline_metrics)

    # --- Optional: Item2Vec comparison ---
    if include_item2vec and DEFAULT_AMAZON_SESSION_CONFIG.item2vec_embedding_file.exists():
        logger.info("\n  --- Simple Mean (Item2Vec embeddings) ---")
        i2v_results = _evaluate_item2vec_baseline(
            sessions_df=dataset.sessions_df,
            item2vec_file=cfg.item2vec_embedding_file,
            max_sessions=max_sessions,
            k_values=k_values,
            seed=seed,
        )
        if i2v_results is not None:
            print_ranking_metrics(i2v_results, "Amazon KDD - Simple Mean (Item2Vec)")
            all_results.append(i2v_results)

    # --- Summary comparison table ---
    _print_comparison_table(all_results, "Amazon KDD Session Embedding Results")

    return {
        "dataset": "amazon_kdd",
        "results": all_results,
        "session_stats": dataset.get_session_length_stats(),
        "hyperparameters": {
            "last_k_values": last_k_values,
            "decay_half_life": decay_half_life,
            "idf_vocabulary_size": len(idf_scores),
            "default_idf": default_idf,
        },
    }


def _evaluate_item2vec_baseline(
    sessions_df,
    item2vec_file: Path,
    max_sessions: int,
    k_values: List[int],
    seed: int,
) -> Optional[Dict]:
    """Evaluate simple mean using Item2Vec embeddings (128-dim).

    This is a footnote comparison: same aggregation method (simple mean),
    different underlying item embeddings. Shows that user embedding quality
    is bounded by item embedding quality.
    """
    try:
        data = np.load(item2vec_file, allow_pickle=True)
        i2v_embeddings = data["embeddings"]
        # Handle different key names across Chapter 10 NPZ files
        if "item_ids" in data:
            i2v_ids = list(data["item_ids"])
        elif "ids" in data:
            i2v_ids = list(data["ids"])
        elif "news_ids" in data:
            i2v_ids = list(data["news_ids"])
        else:
            raise KeyError(f"No ID key found in {item2vec_file.name}. Keys: {list(data.keys())}")
        i2v_id_to_idx = {iid: idx for idx, iid in enumerate(i2v_ids)}
        i2v_valid = set(i2v_ids)
    except Exception as e:
        logger.warning(f"Could not load Item2Vec embeddings: {e}")
        return None

    # Normalize Item2Vec embeddings for cosine similarity
    norms = np.linalg.norm(i2v_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    i2v_embeddings = i2v_embeddings / norms

    if FAISS_AVAILABLE:
        i2v_index = build_faiss_index(i2v_embeddings, index_type="IndexFlatIP")
    else:
        i2v_index = None

    agg = SimpleMeanAggregator(normalize=True)
    t0 = time.time()

    # Filter sessions to those with Item2Vec coverage
    all_query_metrics = []
    count = 0

    for row in sessions_df.sample(n=min(max_sessions * 2, len(sessions_df)), seed=seed).iter_rows(named=True):
        if count >= max_sessions:
            break

        prev_items = row["prev_items_list"]
        next_item = row["next_item"]

        # Check coverage
        if next_item not in i2v_valid:
            continue
        if not all(iid in i2v_valid for iid in prev_items):
            continue

        # Build session embedding
        history_embs = np.stack([i2v_embeddings[i2v_id_to_idx[iid]] for iid in prev_items])
        user_emb = agg.aggregate(history_embs).reshape(1, -1).astype(np.float32)

        max_k = max(k_values) + 1
        if i2v_index is not None:
            distances, indices = i2v_index.search(user_emb, max_k)
        else:
            distances, indices = brute_force_search(
                i2v_embeddings, user_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [i2v_ids[idx] for idx in indices[0]]
        history_set = set(prev_items)
        filtered_ids = [rid for rid in retrieved_ids if rid not in history_set]
        filtered_ids = filtered_ids[:max(k_values)]

        relevance = np.array([1.0 if rid == next_item else 0.0 for rid in filtered_ids])
        query_metrics = evaluate_ranking(
            scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
            relevance=relevance,
            k_values=k_values,
        )
        all_query_metrics.append(query_metrics)
        count += 1

    if not all_query_metrics:
        logger.warning("No valid Item2Vec sessions found")
        return None

    metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    metrics["num_queries"] = count
    metrics["method"] = "Simple Mean (Item2Vec 128d)"
    metrics["time_seconds"] = round(elapsed, 2)
    metrics["embedding_dim"] = 128

    return metrics


# ============================================================================
# MIND Evaluation (User-Based)
# ============================================================================

def evaluate_mind(
    max_users: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
) -> Dict:
    """Evaluate aggregation baselines on MIND next-click prediction.

    Protocol:
    1. Load users with >= 5 clicks in their history.
    2. Temporal split: history = all clicks before last impression,
       target = articles clicked in last impression.
    3. For each user: aggregate history article embeddings -> user embedding.
    4. Build FAISS index over all 51K article embeddings.
    5. Retrieve top-K articles using user embedding as query.
    6. Evaluate: are the target articles in the top-K?
    """
    cfg = DEFAULT_MIND_USER_CONFIG
    agg_cfg = DEFAULT_AGGREGATION_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_users is None:
        max_users = eval_cfg.max_eval_users_mind

    logger.info("=" * 80)
    logger.info("MIND: User Embedding Evaluation (Section 11.1)")
    logger.info("=" * 80)

    # --- Load data ---
    logger.info("\n[1/5] Loading dataset and cached article embeddings...")
    dataset = MINDUserDataset(
        data_dir=cfg.data_dir,
        news_file=cfg.news_file,
        behaviors_file=cfg.behaviors_file,
        min_history_length=cfg.min_history_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()
    dataset.print_stats()

    # --- Get evaluation users ---
    logger.info("\n[2/5] Preparing evaluation users...")
    eval_users = dataset.get_evaluation_users(max_users=max_users, seed=seed)
    logger.info(f"  Evaluation users: {len(eval_users):,}")

    # --- Compute IDF scores ---
    logger.info("\n[3/5] Computing IDF scores from user histories...")
    item_freq = dataset.get_item_frequency()
    total_users = len(eval_users)
    idf_scores = compute_idf_scores(item_freq, total_users, smooth=agg_cfg.idf_smooth)
    logger.info(f"  IDF vocabulary size: {len(idf_scores):,}")

    idf_values = np.array(list(idf_scores.values()))
    default_idf = float(np.median(idf_values))

    # --- Determine data-driven hyperparameters ---
    history_stats = dataset.get_history_length_stats()
    median_len = history_stats["median"]

    last_k_values = sorted(set([
        max(5, int(history_stats["p25"])),
        max(5, int(median_len)),
        max(10, int(history_stats["p75"])),
        max(20, int(history_stats["p90"])),
    ]))

    # For MIND, position-based decay (no per-click timestamps in history field)
    decay_half_life = max(1.0, median_len / 2)

    logger.info(f"\n  Data-driven hyperparameters:")
    logger.info(f"    Median history length: {median_len}")
    logger.info(f"    Last-K values: {last_k_values}")
    logger.info(f"    Decay half-life (positions): {decay_half_life:.1f}")

    # --- Build FAISS index ---
    logger.info("\n[4/5] Building FAISS index over article catalog...")
    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}

    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(all_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None
        logger.warning("FAISS not available, using brute-force search")

    # --- Create aggregators ---
    aggregators = create_aggregators(
        idf_scores=idf_scores,
        last_k_values=last_k_values,
        decay_half_life=decay_half_life,
        decay_mode="position",
    )

    # --- Evaluate each aggregator ---
    logger.info(f"\n[5/5] Evaluating {len(aggregators)} aggregation methods...")
    all_results = []

    for agg in aggregators:
        logger.info(f"\n  --- {agg.name} ---")
        t0 = time.time()

        all_query_metrics = []

        for user in eval_users:
            # Aggregate history embeddings into user embedding
            user_emb = agg.aggregate(
                item_embeddings=user["history_embeddings"],
                item_ids=user.get("history_item_ids"),
            )
            user_emb = user_emb.reshape(1, -1).astype(np.float32)

            # Retrieve top-K articles
            max_k = max(k_values) + 10  # Extra buffer for filtering
            if faiss_index is not None:
                distances, indices = faiss_index.search(user_emb, max_k)
            else:
                distances, indices = brute_force_search(
                    all_embeddings, user_emb, k=max_k, metric="cosine"
                )

            retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
            target_ids = set(user["target_item_ids"])

            # Exclude items already in the user's history
            history_set = set(user["history_item_ids"])
            filtered_ids = [rid for rid in retrieved_ids if rid not in history_set]
            filtered_ids = filtered_ids[:max(k_values)]

            # Relevance: 1 if retrieved article is in the target set
            relevance = np.array([
                1.0 if rid in target_ids else 0.0
                for rid in filtered_ids
            ])

            query_metrics = evaluate_ranking(
                scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
                relevance=relevance,
                k_values=k_values,
            )
            all_query_metrics.append(query_metrics)

        # Aggregate metrics
        agg_metrics = {}
        for metric_name in all_query_metrics[0].keys():
            values = [m[metric_name] for m in all_query_metrics]
            agg_metrics[metric_name] = float(np.mean(values))

        elapsed = time.time() - t0
        agg_metrics["num_queries"] = len(eval_users)
        agg_metrics["method"] = agg.name
        agg_metrics["time_seconds"] = round(elapsed, 2)

        print_ranking_metrics(agg_metrics, f"MIND - {agg.name}")
        all_results.append(agg_metrics)

    # --- Add single-last-click baseline ---
    logger.info("\n  --- Last-Click Only Baseline ---")
    t0 = time.time()
    all_query_metrics = []

    for user in eval_users:
        last_click_emb = user["history_embeddings"][-1:].astype(np.float32)

        max_k = max(k_values) + 10
        if faiss_index is not None:
            distances, indices = faiss_index.search(last_click_emb, max_k)
        else:
            distances, indices = brute_force_search(
                all_embeddings, last_click_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
        target_ids = set(user["target_item_ids"])

        last_click_id = user["history_item_ids"][-1]
        filtered_ids = [rid for rid in retrieved_ids if rid != last_click_id]
        filtered_ids = filtered_ids[:max(k_values)]

        relevance = np.array([
            1.0 if rid in target_ids else 0.0
            for rid in filtered_ids
        ])

        query_metrics = evaluate_ranking(
            scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
            relevance=relevance,
            k_values=k_values,
        )
        all_query_metrics.append(query_metrics)

    baseline_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        baseline_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    baseline_metrics["num_queries"] = len(eval_users)
    baseline_metrics["method"] = "Last-Click Only Baseline"
    baseline_metrics["time_seconds"] = round(elapsed, 2)

    print_ranking_metrics(baseline_metrics, "MIND - Last-Click Only Baseline")
    all_results.append(baseline_metrics)

    _print_comparison_table(all_results, "MIND User Embedding Results")

    return {
        "dataset": "mind",
        "results": all_results,
        "history_stats": dataset.get_history_length_stats(),
        "hyperparameters": {
            "last_k_values": last_k_values,
            "decay_half_life": decay_half_life,
            "idf_vocabulary_size": len(idf_scores),
            "default_idf": default_idf,
        },
    }


# ============================================================================
# Comparison Table & Output
# ============================================================================

def _print_comparison_table(results: List[Dict], title: str):
    """Print a concise comparison table of all methods."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = f"  {'Method':<35} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'R@50':>8} {'Time':>7}"
    print(header)
    print(f"  {'-' * 95}")

    for r in results:
        method = r.get("method", "Unknown")
        mrr = r.get("mrr", 0)
        r1 = r.get("recall@1", 0)
        r5 = r.get("recall@5", 0)
        r10 = r.get("recall@10", 0)
        r20 = r.get("recall@20", 0)
        r50 = r.get("recall@50", 0)
        t = r.get("time_seconds", 0)
        print(
            f"  {method:<35} {mrr:>8.4f} {r1:>8.4f} {r5:>8.4f} "
            f"{r10:>8.4f} {r20:>8.4f} {r50:>8.4f} {t:>6.1f}s"
        )

    print(f"{'=' * 100}\n")


def save_results(results: Dict, filename: str):
    """Save evaluation results to JSON."""
    out_path = METRICS_DIR / filename
    # Make numpy types JSON-serializable
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"Results saved to {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate aggregation-based user embeddings (Section 11.1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["amazon", "mind", "both"],
        help="Which dataset to evaluate on",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: evaluate on fewer sessions/users (2K instead of 20K/10K)",
    )
    parser.add_argument(
        "--no-item2vec",
        action="store_true",
        help="Skip Item2Vec comparison on Amazon",
    )
    args = parser.parse_args()

    quick_amazon = 2_000 if args.quick else None
    quick_mind = 2_000 if args.quick else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    if args.dataset in ("amazon", "both"):
        amazon_results = evaluate_amazon(
            max_sessions=quick_amazon,
            include_item2vec=not args.no_item2vec,
        )
        all_results["amazon_kdd"] = amazon_results
        save_results(
            amazon_results,
            f"aggregation_amazon_{timestamp}.json",
        )

    if args.dataset in ("mind", "both"):
        mind_results = evaluate_mind(max_users=quick_mind)
        all_results["mind"] = mind_results
        save_results(
            mind_results,
            f"aggregation_mind_{timestamp}.json",
        )

    # Save combined comparison
    if len(all_results) == 2:
        save_results(
            all_results,
            f"aggregation_comparison_{timestamp}.json",
        )

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
