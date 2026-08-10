"""
Evaluation Metrics for User Embeddings (Chapter 11)

Reused from Chapter 10 with minor adaptations for user-level evaluation.
Implements standard retrieval and ranking metrics:
- Precision@K
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
"""

import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """Precision@K = (# relevant items in top-K) / K."""
    if k <= 0 or len(relevance_scores) == 0:
        return 0.0
    top_k = relevance_scores[:k]
    return float(np.sum(top_k) / k)


def recall_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """Recall@K = (# relevant items in top-K) / (total # relevant items)."""
    total_relevant = np.sum(relevance_scores)
    if total_relevant == 0 or k <= 0:
        return 0.0
    top_k = relevance_scores[:k]
    return float(np.sum(top_k) / total_relevant)


def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """DCG@K = sum_{i=1}^{K} (rel_i / log2(i+1))."""
    if k <= 0 or len(relevance_scores) == 0:
        return 0.0
    top_k = relevance_scores[:k]
    positions = np.arange(1, len(top_k) + 1)
    discounts = 1.0 / np.log2(positions + 1)
    return float(np.sum(top_k * discounts))


def ndcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K."""
    dcg = dcg_at_k(relevance_scores, k)
    ideal_relevance = np.sort(relevance_scores)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(relevance_scores: np.ndarray) -> float:
    """MRR = 1 / (rank of first relevant item)."""
    relevant_positions = np.where(relevance_scores > 0)[0]
    if len(relevant_positions) == 0:
        return 0.0
    first_relevant_rank = relevant_positions[0] + 1
    return 1.0 / first_relevant_rank


def evaluate_ranking(
    scores: np.ndarray,
    relevance: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20, 50],
) -> Dict[str, float]:
    """Evaluate ranking for a single query across multiple K values."""
    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices]

    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(sorted_relevance, k)
        metrics[f"recall@{k}"] = recall_at_k(sorted_relevance, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(sorted_relevance, k)
    metrics["mrr"] = mean_reciprocal_rank(sorted_relevance)
    return metrics


def evaluate_ranking_batch(
    scores_list: List[np.ndarray],
    relevance_list: List[np.ndarray],
    k_values: List[int] = [1, 5, 10, 20, 50],
) -> Dict[str, float]:
    """Evaluate ranking for multiple queries and aggregate (mean)."""
    if len(scores_list) != len(relevance_list):
        raise ValueError("scores_list and relevance_list must have same length")

    num_queries = len(scores_list)
    if num_queries == 0:
        return {}

    all_metrics = []
    for scores, relevance in zip(scores_list, relevance_list):
        metrics = evaluate_ranking(scores, relevance, k_values)
        all_metrics.append(metrics)

    aggregated = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        aggregated[metric_name] = float(np.mean(values))

    return aggregated


def print_ranking_metrics(
    metrics: Dict[str, float],
    title: str = "Ranking Metrics",
):
    """Pretty-print ranking metrics grouped by type."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    precision_metrics = {k: v for k, v in metrics.items() if "precision" in k}
    recall_metrics = {k: v for k, v in metrics.items() if "recall" in k}
    ndcg_metrics = {k: v for k, v in metrics.items() if "ndcg" in k}
    other_metrics = {
        k: v for k, v in metrics.items()
        if k not in precision_metrics
        and k not in recall_metrics
        and k not in ndcg_metrics
    }

    if precision_metrics:
        print("\nPrecision:")
        for k, v in sorted(precision_metrics.items()):
            print(f"  {k:.<40} {v:.4f}")

    if recall_metrics:
        print("\nRecall:")
        for k, v in sorted(recall_metrics.items()):
            print(f"  {k:.<40} {v:.4f}")

    if ndcg_metrics:
        print("\nNDCG:")
        for k, v in sorted(ndcg_metrics.items()):
            print(f"  {k:.<40} {v:.4f}")

    if other_metrics:
        print("\nOther:")
        for k, v in sorted(other_metrics.items()):
            if isinstance(v, (int, float)):
                print(f"  {k:.<40} {v:.4f}")
            else:
                print(f"  {k:.<40} {v}")

    print(f"{'=' * 80}\n")
