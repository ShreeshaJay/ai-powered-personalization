"""
Evaluation framework for hybrid search comparison.

Provides two evaluation types:
  A. Full-catalog retrieval metrics (Recall@K, label distribution)
  B. Score-based analysis (re-ranking NDCG, boxplots, separability AUC)

Also generates comparison plots: score distribution boxplots per ESCI label,
and stacked bar charts for label distribution in top-K.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import json
import logging

from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    ESCI_GAINS, ESCI_GAINS_QDRANT, ESCI_LABELS_ORDERED, ESCI_COLORS,
    ESCI_LABEL_NAMES, EvalConfig, DEFAULT_EVAL_CONFIG, METRICS_DIR, PLOTS_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Full-Catalog Retrieval Metrics
# =============================================================================

def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int
) -> float:
    """Recall@K: fraction of relevant items found in top-K retrieved results."""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    found = top_k & relevant_ids
    return len(found) / len(relevant_ids)


def compute_label_distribution_topk(
    retrieved_ids: List[str],
    id_to_label: Dict[str, str],
    k: int
) -> Dict[str, float]:
    """Compute ESCI label distribution among top-K retrieved results.

    Returns dict with keys E, S, C, I, unlabeled as fractions summing to 1.0.
    """
    top_k = retrieved_ids[:k]
    if len(top_k) == 0:
        return {label: 0.0 for label in ESCI_LABELS_ORDERED + ["unlabeled"]}

    counts = defaultdict(int)
    for pid in top_k:
        label = id_to_label.get(pid, "unlabeled")
        counts[label] += 1

    total = len(top_k)
    dist = {}
    for label in ESCI_LABELS_ORDERED:
        dist[label] = counts[label] / total
    dist["unlabeled"] = counts["unlabeled"] / total
    return dist


def evaluate_retrieval(
    retrieved_per_query: Dict[Any, List[str]],
    eval_queries: Dict[Any, Dict[str, Any]],
    k_values: List[int] = [10, 20, 50]
) -> Dict[str, Any]:
    """Aggregate full-catalog retrieval metrics across all queries.

    Args:
        retrieved_per_query: {query_id: [product_ids in ranked order]}
        eval_queries: from ESCIDataset.get_evaluation_queries()
        k_values: K values for Recall@K

    Returns:
        Dict with aggregated recall and label distribution metrics.
    """
    recall_exact = {k: [] for k in k_values}
    recall_exact_sub = {k: [] for k in k_values}
    label_dists = {k: [] for k in k_values}

    for qid, retrieved_ids in retrieved_per_query.items():
        if qid not in eval_queries:
            continue

        qdata = eval_queries[qid]

        for k in k_values:
            # Recall@K for Exact products
            r_e = compute_recall_at_k(retrieved_ids, qdata["exact_ids"], k)
            recall_exact[k].append(r_e)

            # Recall@K for Exact + Substitute products
            r_es = compute_recall_at_k(retrieved_ids, qdata["exact_substitute_ids"], k)
            recall_exact_sub[k].append(r_es)

            # Label distribution
            dist = compute_label_distribution_topk(
                retrieved_ids, qdata["all_labeled_ids"], k
            )
            label_dists[k].append(dist)

    results = {"num_queries": len(recall_exact[k_values[0]])}

    for k in k_values:
        results[f"recall_exact@{k}"] = float(np.mean(recall_exact[k]))
        results[f"recall_exact_sub@{k}"] = float(np.mean(recall_exact_sub[k]))

        # Average label distribution
        avg_dist = {}
        for label in ESCI_LABELS_ORDERED + ["unlabeled"]:
            avg_dist[label] = float(np.mean([d[label] for d in label_dists[k]]))
        results[f"label_dist@{k}"] = avg_dist

    return results


# =============================================================================
# Curated Pool Evaluation (Qdrant-style)
# =============================================================================

CURATED_POOL_CAVEAT = (
    "NOTE: Curated pool evaluation treats unlabeled products as Irrelevant "
    "(gain=0.0). This assumption is standard (used by Qdrant's SPLADE "
    "evaluation) but may underestimate methods that retrieve truly relevant "
    "but unlabeled products."
)


def evaluate_curated_pool(
    retrieved_per_query: Dict[Any, List[str]],
    query_data: Dict[Any, Dict[str, Any]],
    k_values: List[int] = [10, 20, 50],
    gains: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval on a curated pool with dense label coverage.

    Unlike full-catalog evaluation, this assumes unlabeled products are
    Irrelevant (I=0.0). The curated pool is small enough that most retrieved
    products have ESCI labels, making this assumption reasonable.

    Args:
        retrieved_per_query: {query_id: [product_ids in ranked order]}
        query_data: {query_id: {query_text, labeled_products: {pid: label}}}
        k_values: K values for NDCG@K, MRR@K, Precision@K
        gains: Relevance gains per label (default: Qdrant-compatible E=1.0, S=0.7, C=0.5, I=0.0)

    Returns:
        Dict with NDCG, MRR, Precision, Recall metrics and the caveat note.
    """
    if gains is None:
        gains = dict(ESCI_GAINS_QDRANT)

    ndcg_scores = {k: [] for k in k_values}
    mrr_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    recall_e_scores = {k: [] for k in k_values}
    recall_es_scores = {k: [] for k in k_values}

    n_evaluated = 0
    for qid, retrieved_ids in retrieved_per_query.items():
        if qid not in query_data:
            continue

        labeled_products = query_data[qid]["labeled_products"]
        exact_ids = {pid for pid, lbl in labeled_products.items() if lbl == "E"}
        es_ids = {pid for pid, lbl in labeled_products.items() if lbl in ("E", "S")}

        # Build gain array for retrieved products
        # Unlabeled products get I=0.0 (the key assumption)
        retrieved_gains = np.array([
            gains.get(labeled_products.get(pid, "I"), 0.0)
            for pid in retrieved_ids
        ], dtype=np.float64)

        # Also need ideal gains (all labeled products sorted by gain)
        all_gains = sorted(
            [gains.get(lbl, 0.0) for lbl in labeled_products.values()],
            reverse=True,
        )
        all_gains = np.array(all_gains, dtype=np.float64)

        for k in k_values:
            # NDCG@K
            k_actual = min(k, len(retrieved_gains))
            if k_actual == 0:
                continue

            positions = np.arange(1, k_actual + 1)
            discounts = np.log2(positions + 1)
            dcg = np.sum(retrieved_gains[:k_actual] / discounts)

            ideal_k = min(k, len(all_gains))
            ideal_positions = np.arange(1, ideal_k + 1)
            ideal_discounts = np.log2(ideal_positions + 1)
            idcg = np.sum(all_gains[:ideal_k] / ideal_discounts)

            ndcg = float(dcg / idcg) if idcg > 0 else 0.0
            ndcg_scores[k].append(ndcg)

            # MRR@K (first item with gain > 0)
            mrr = 0.0
            for i in range(k_actual):
                if retrieved_gains[i] > 0:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores[k].append(mrr)

            # Precision@K (fraction of top-K with gain > 0)
            prec = float(np.sum(retrieved_gains[:k_actual] > 0) / k_actual)
            precision_scores[k].append(prec)

            # Recall@K
            top_k_ids = set(retrieved_ids[:k])
            r_e = len(top_k_ids & exact_ids) / len(exact_ids) if exact_ids else 0.0
            r_es = len(top_k_ids & es_ids) / len(es_ids) if es_ids else 0.0
            recall_e_scores[k].append(r_e)
            recall_es_scores[k].append(r_es)

        n_evaluated += 1

    results = {
        "num_queries": n_evaluated,
        "gains_used": gains,
        "caveat": CURATED_POOL_CAVEAT,
    }

    for k in k_values:
        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
        results[f"mrr@{k}"] = float(np.mean(mrr_scores[k])) if mrr_scores[k] else 0.0
        results[f"precision@{k}"] = float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
        results[f"recall_exact@{k}"] = float(np.mean(recall_e_scores[k])) if recall_e_scores[k] else 0.0
        results[f"recall_exact_sub@{k}"] = float(np.mean(recall_es_scores[k])) if recall_es_scores[k] else 0.0

    return results


def print_curated_pool_results(
    method_results: Dict[str, Dict[str, Any]],
    k_values: List[int] = [10, 20, 50],
) -> None:
    """Pretty-print curated pool evaluation results for all methods."""
    print(f"\n{'='*90}")
    print(f"  CURATED POOL EVALUATION (Qdrant-style)")
    print(f"{'='*90}")

    # Print caveat
    first_result = next(iter(method_results.values()))
    if "caveat" in first_result:
        print(f"\n  {first_result['caveat']}")

    # Header
    metrics = [f"ndcg@{k}" for k in k_values] + [f"mrr@{k_values[0]}"]
    header = f"\n  {'Method':<25s}"
    for m in metrics:
        header += f" {m:>10s}"
    header += f" {'queries':>8s}"
    print(header)
    print(f"  {'-'*85}")

    for method_name, result in method_results.items():
        row = f"  {method_name:<25s}"
        for m in metrics:
            row += f" {result.get(m, 0):>10.4f}"
        row += f" {result['num_queries']:>8d}"
        print(row)


# =============================================================================
# Score-Based Analysis Metrics
# =============================================================================

def ndcg_at_k_graded(gains: np.ndarray, k: int) -> float:
    """NDCG@K with graded relevance (ESCI gains: E=3, S=2, C=1, I=0).

    DCG@K  = sum_{i=1}^{K} gain_i / log2(i + 1)
    IDCG@K = same formula but gains sorted descending (ideal ranking)
    NDCG@K = DCG@K / IDCG@K
    """
    k = min(k, len(gains))
    if k == 0:
        return 0.0

    # DCG
    positions = np.arange(1, k + 1)
    discounts = np.log2(positions + 1)
    dcg = np.sum(gains[:k] / discounts)

    # IDCG (ideal: gains sorted descending)
    ideal_gains = np.sort(gains)[::-1]
    idcg = np.sum(ideal_gains[:k] / discounts)

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def mrr_at_k(gains: np.ndarray, k: int, min_gain: float = 1.0) -> float:
    """MRR@K: reciprocal rank of first item with gain >= min_gain."""
    k = min(k, len(gains))
    for i in range(k):
        if gains[i] >= min_gain:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(gains: np.ndarray, k: int, min_gain: float = 1.0) -> float:
    """Precision@K: fraction of top-K with gain >= min_gain."""
    k = min(k, len(gains))
    if k == 0:
        return 0.0
    return float(np.sum(gains[:k] >= min_gain) / k)


def compute_separability_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    positive_labels: List[str] = None,
    negative_labels: List[str] = None,
) -> float:
    """ROC AUC for separating relevant (E+S+C) from irrelevant (I) using raw scores.

    A higher AUC means the method's scores better distinguish relevant from
    irrelevant items, i.e., a simple threshold could filter out junk.
    """
    if positive_labels is None:
        positive_labels = ["E", "S", "C"]
    if negative_labels is None:
        negative_labels = ["I"]

    # Create binary labels: 1 = relevant, 0 = irrelevant
    binary = np.zeros(len(labels), dtype=np.int32)
    for i, label in enumerate(labels):
        if label in positive_labels:
            binary[i] = 1
        elif label in negative_labels:
            binary[i] = 0
        else:
            binary[i] = -1  # skip unknown labels

    # Filter to only positive/negative labels
    mask = binary >= 0
    binary = binary[mask]
    filtered_scores = scores[mask]

    if len(np.unique(binary)) < 2:
        logger.warning("Cannot compute AUC: only one class present")
        return 0.0

    return float(roc_auc_score(binary, filtered_scores))


def evaluate_candidate_scoring(
    query_candidates: Dict[Any, Dict[str, Any]],
    scoring_fn: Callable,
    method_name: str,
    eval_config: EvalConfig = DEFAULT_EVAL_CONFIG,
) -> Dict[str, Any]:
    """Score-based evaluation on ESCI annotated candidates.

    For each query, scores its ~16 labeled candidates using scoring_fn,
    then computes re-ranking NDCG, MRR, Precision, and collects
    per-label score distributions for boxplots.

    Args:
        query_candidates: from ESCIDataset.get_query_candidates()
        scoring_fn: Callable(query_text, candidate_ids) -> np.ndarray of scores
        method_name: identifier for this method
        eval_config: evaluation configuration

    Returns:
        Dict with ranking metrics, per-label scores, and separability AUC.
    """
    from tqdm import tqdm

    k_values = eval_config.k_values

    # Accumulators
    ndcg_scores = {k: [] for k in k_values}
    mrr_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    label_dists = {k: [] for k in k_values}

    # Per-label score accumulation for boxplots
    per_label_scores = {label: [] for label in ESCI_LABELS_ORDERED}

    # All scores and labels for separability AUC
    all_scores = []
    all_labels = []

    for qid, qdata in tqdm(query_candidates.items(),
                           desc=f"Scoring [{method_name}]",
                           total=len(query_candidates)):
        query_text = qdata["query_text"]
        candidate_ids = qdata["candidate_ids"]
        labels = qdata["labels"]

        # Score candidates
        scores = scoring_fn(query_text, candidate_ids)
        if scores is None or len(scores) == 0:
            continue

        scores = np.asarray(scores, dtype=np.float64)

        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        sorted_gains = np.array([ESCI_GAINS.get(labels[i], 0) for i in sorted_indices],
                                dtype=np.float32)
        sorted_labels = [labels[i] for i in sorted_indices]

        # Compute ranking metrics
        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k_graded(sorted_gains, k))
            mrr_scores[k].append(mrr_at_k(sorted_gains, k))
            precision_scores[k].append(precision_at_k(sorted_gains, k))

            # Label distribution in top-K
            top_k_labels = sorted_labels[:k]
            dist = {}
            for label in ESCI_LABELS_ORDERED:
                dist[label] = top_k_labels.count(label) / min(k, len(top_k_labels)) if top_k_labels else 0.0
            label_dists[k].append(dist)

        # Collect per-label scores for boxplots
        for score, label in zip(scores, labels):
            per_label_scores[label].append(float(score))

        all_scores.extend(scores.tolist())
        all_labels.extend(labels)

    # Aggregate
    results = {
        "method": method_name,
        "num_queries": len(ndcg_scores[k_values[0]]),
        "ranking_metrics": {},
        "label_distribution": {},
        "per_label_scores": {},
        "separability_auc": 0.0,
    }

    for k in k_values:
        results["ranking_metrics"][f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))
        results["ranking_metrics"][f"mrr@{k}"] = float(np.mean(mrr_scores[k]))
        results["ranking_metrics"][f"precision@{k}"] = float(np.mean(precision_scores[k]))

        # Average label distribution
        avg_dist = {}
        for label in ESCI_LABELS_ORDERED:
            avg_dist[label] = float(np.mean([d[label] for d in label_dists[k]]))
        results["label_distribution"][f"K={k}"] = avg_dist

    # Convert per-label scores to numpy
    for label in ESCI_LABELS_ORDERED:
        results["per_label_scores"][label] = np.array(per_label_scores[label])

    # Separability AUC
    if len(all_scores) > 0:
        results["separability_auc"] = compute_separability_auc(
            np.array(all_scores),
            np.array(all_labels),
            eval_config.separability_positive_labels,
            eval_config.separability_negative_labels,
        )

    return results


# =============================================================================
# Printing Functions
# =============================================================================

def print_evaluation_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results for a single method."""
    method = results["method"]
    n_queries = results["num_queries"]

    print(f"\n{'='*60}")
    print(f"  {method} ({n_queries:,} queries)")
    print(f"{'='*60}")

    # Ranking metrics
    print(f"\n  Ranking Metrics:")
    for metric, value in sorted(results["ranking_metrics"].items()):
        print(f"    {metric:<20s} {value:.4f}")

    # Separability AUC
    print(f"\n  Separability AUC (E+S+C vs I): {results['separability_auc']:.4f}")

    # Label distribution
    print(f"\n  Label Distribution in Top-K:")
    for k_label, dist in sorted(results["label_distribution"].items()):
        parts = [f"{l}={100*v:.1f}%" for l, v in dist.items()]
        print(f"    {k_label}: {', '.join(parts)}")

    # Per-label score stats
    print(f"\n  Score Statistics by Label:")
    for label in ESCI_LABELS_ORDERED:
        scores = results["per_label_scores"].get(label)
        if scores is not None and len(scores) > 0:
            if isinstance(scores, list):
                scores = np.array(scores)
            print(f"    {label} ({ESCI_LABEL_NAMES[label]}): "
                  f"mean={np.mean(scores):.4f}, "
                  f"median={np.median(scores):.4f}, "
                  f"std={np.std(scores):.4f}, "
                  f"n={len(scores):,}")


def print_comparison_table(all_results: List[Dict[str, Any]]) -> None:
    """Print side-by-side comparison of all methods."""
    print(f"\n{'='*90}")
    print(f"  METHOD COMPARISON")
    print(f"{'='*90}")

    # Header
    header = f"  {'Method':<20s}"
    metrics_to_show = ["ndcg@10", "mrr@10", "precision@10"]
    for m in metrics_to_show:
        header += f" {m:>12s}"
    header += f" {'%E@10':>8s} {'%I@10':>8s} {'Sep.AUC':>8s}"
    print(header)
    print(f"  {'-'*86}")

    for r in all_results:
        row = f"  {r['method']:<20s}"
        for m in metrics_to_show:
            val = r["ranking_metrics"].get(m, 0)
            row += f" {val:>12.4f}"

        # Label distribution at K=10
        dist_10 = r["label_distribution"].get("K=10", {})
        row += f" {100*dist_10.get('E', 0):>7.1f}%"
        row += f" {100*dist_10.get('I', 0):>7.1f}%"
        row += f" {r['separability_auc']:>8.4f}"
        print(row)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_score_boxplots(
    all_results: List[Dict[str, Any]],
    output_path: Path = PLOTS_DIR / "score_boxplots.png",
    max_points_per_label: int = 50000,
) -> None:
    """Generate score distribution boxplots per ESCI label.

    Creates 1 x N subplots (one per method). Each subplot has:
      X-axis: ESCI labels (E, S, C, I)
      Y-axis: Raw scores from the method

    Uses seaborn boxenplot (letter-value plot) for better visualization
    of large distributions.
    """
    n_methods = len(all_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    rng = np.random.RandomState(42)

    for ax, result in zip(axes, all_results):
        method_name = result["method"]
        data_rows = []

        for label in ESCI_LABELS_ORDERED:
            scores = result["per_label_scores"].get(label, np.array([]))
            if isinstance(scores, list):
                scores = np.array(scores)
            if len(scores) == 0:
                continue

            # Subsample for plotting speed
            if len(scores) > max_points_per_label:
                indices = rng.choice(len(scores), size=max_points_per_label, replace=False)
                scores = scores[indices]

            for s in scores:
                data_rows.append({"ESCI Label": label, "Score": s})

        if not data_rows:
            continue

        df_plot = pd.DataFrame(data_rows)
        palette = [ESCI_COLORS[l] for l in ESCI_LABELS_ORDERED if l in df_plot["ESCI Label"].unique()]

        sns.boxenplot(
            data=df_plot,
            x="ESCI Label",
            y="Score",
            ax=ax,
            palette=palette,
            order=ESCI_LABELS_ORDERED,
        )
        ax.set_title(method_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("ESCI Label", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Score boxplots saved to {output_path}")


def plot_label_distribution_bars(
    all_results: List[Dict[str, Any]],
    k_values: List[int] = [10, 20, 50],
    output_path: Path = PLOTS_DIR / "label_distribution.png",
) -> None:
    """Stacked bar chart of ESCI label distribution in top-K.

    One group of bars per K value, one bar per method.
    Stacked segments: E (green), S (blue), C (orange), I (red).
    """
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 5))
    if n_k == 1:
        axes = [axes]

    method_names = [r["method"] for r in all_results]

    for ax, k in zip(axes, k_values):
        bottoms = np.zeros(len(method_names))
        k_label = f"K={k}"

        for label in ESCI_LABELS_ORDERED:
            values = []
            for r in all_results:
                dist = r["label_distribution"].get(k_label, {})
                values.append(100 * dist.get(label, 0))

            values = np.array(values)
            ax.bar(
                method_names, values, bottom=bottoms,
                label=f"{label} ({ESCI_LABEL_NAMES[label]})",
                color=ESCI_COLORS[label],
                edgecolor="white", linewidth=0.5,
            )
            bottoms += values

        ax.set_title(f"Top-{k} Label Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30)
        if ax == axes[-1]:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Label distribution chart saved to {output_path}")


# =============================================================================
# Output Functions
# =============================================================================

def save_results_json(
    all_results: List[Dict[str, Any]],
    output_path: Path = METRICS_DIR / "hybrid_search_results.json",
) -> None:
    """Save evaluation results to JSON.

    Converts numpy arrays to lists for JSON serialization.
    """
    serializable = []
    for r in all_results:
        r_copy = {}
        for key, value in r.items():
            if key == "per_label_scores":
                # Convert numpy arrays to summary stats (not full arrays)
                r_copy[key] = {}
                for label, scores in value.items():
                    if isinstance(scores, np.ndarray):
                        r_copy[key][label] = {
                            "count": int(len(scores)),
                            "mean": float(np.mean(scores)) if len(scores) > 0 else 0,
                            "median": float(np.median(scores)) if len(scores) > 0 else 0,
                            "std": float(np.std(scores)) if len(scores) > 0 else 0,
                            "min": float(np.min(scores)) if len(scores) > 0 else 0,
                            "max": float(np.max(scores)) if len(scores) > 0 else 0,
                        }
                    else:
                        r_copy[key][label] = value
            elif isinstance(value, (np.floating, np.integer)):
                r_copy[key] = float(value)
            else:
                r_copy[key] = value
        serializable.append(r_copy)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


def save_comparison_csv(
    all_results: List[Dict[str, Any]],
    output_path: Path = METRICS_DIR / "method_comparison.csv",
) -> None:
    """Save comparison table as CSV."""
    rows = []
    for r in all_results:
        row = {"method": r["method"], "num_queries": r["num_queries"]}
        row.update(r["ranking_metrics"])
        row["separability_auc"] = r["separability_auc"]

        for k_label, dist in r["label_distribution"].items():
            for label, val in dist.items():
                row[f"{k_label}_{label}%"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison CSV saved to {output_path}")
