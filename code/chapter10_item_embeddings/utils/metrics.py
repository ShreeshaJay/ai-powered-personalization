"""
Evaluation Metrics for Item Embeddings

Implements standard retrieval and ranking metrics for bi-encoder evaluation:
- Precision@K
- Recall@K  
- NDCG@K
- MRR (Mean Reciprocal Rank)

Used for Section 10.1 bi-encoder evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(
    relevance_scores: np.ndarray,
    k: int
) -> float:
    """
    Compute Precision@K for a single query.
    
    Precision@K = (# relevant items in top-K) / K
    
    Args:
        relevance_scores: Binary relevance array (1=relevant, 0=not relevant)
                         in ranked order (most relevant first)
        k: Cut-off position
    
    Returns:
        Precision@K score
    """
    if k <= 0 or len(relevance_scores) == 0:
        return 0.0
    
    # Take top K
    top_k = relevance_scores[:k]
    
    # Count relevant items
    num_relevant = np.sum(top_k)
    
    return num_relevant / k


def recall_at_k(
    relevance_scores: np.ndarray,
    k: int
) -> float:
    """
    Compute Recall@K for a single query.
    
    Recall@K = (# relevant items in top-K) / (total # relevant items)
    
    Args:
        relevance_scores: Binary relevance array in ranked order
        k: Cut-off position
    
    Returns:
        Recall@K score (0.0 if no relevant items exist)
    """
    total_relevant = np.sum(relevance_scores)
    
    if total_relevant == 0:
        return 0.0
    
    if k <= 0:
        return 0.0
    
    # Take top K
    top_k = relevance_scores[:k]
    num_relevant_in_k = np.sum(top_k)
    
    return num_relevant_in_k / total_relevant


def dcg_at_k(
    relevance_scores: np.ndarray,
    k: int
) -> float:
    """
    Compute Discounted Cumulative Gain (DCG) at position K.
    
    DCG@K = sum_{i=1}^{K} (rel_i / log2(i+1))
    
    Args:
        relevance_scores: Relevance scores (can be binary or graded) in ranked order
        k: Cut-off position
    
    Returns:
        DCG@K score
    """
    if k <= 0 or len(relevance_scores) == 0:
        return 0.0
    
    # Take top K
    top_k = relevance_scores[:k]
    
    # Compute discounts: 1/log2(i+1) for i=1,2,...,K
    # i+1 because rank starts at 1, not 0
    positions = np.arange(1, len(top_k) + 1)
    discounts = 1.0 / np.log2(positions + 1)
    
    # DCG = sum of relevance * discount
    dcg = np.sum(top_k * discounts)
    
    return dcg


def ndcg_at_k(
    relevance_scores: np.ndarray,
    k: int
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at position K.
    
    NDCG@K = DCG@K / IDCG@K
    
    where IDCG is the ideal DCG (ranking by decreasing relevance).
    
    Args:
        relevance_scores: Relevance scores in ranked order
        k: Cut-off position
    
    Returns:
        NDCG@K score (0.0 if no relevant items)
    """
    # Compute DCG for current ranking
    dcg = dcg_at_k(relevance_scores, k)
    
    # Compute ideal DCG (sort by decreasing relevance)
    ideal_relevance = np.sort(relevance_scores)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def mean_reciprocal_rank(
    relevance_scores: np.ndarray
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.
    
    MRR = 1 / (rank of first relevant item)
    
    Args:
        relevance_scores: Binary relevance array in ranked order
    
    Returns:
        MRR score (0.0 if no relevant items)
    """
    # Find first relevant item
    relevant_positions = np.where(relevance_scores > 0)[0]
    
    if len(relevant_positions) == 0:
        return 0.0
    
    # Rank is position + 1 (1-indexed)
    first_relevant_rank = relevant_positions[0] + 1
    
    return 1.0 / first_relevant_rank


def evaluate_ranking(
    scores: np.ndarray,
    relevance: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate ranking for a single query across multiple K values.
    
    Args:
        scores: Predicted scores for items (higher = more relevant)
        relevance: Ground truth relevance (binary: 1=relevant, 0=not relevant)
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary of metric_name -> score
    """
    # Sort items by descending score
    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices]
    
    metrics = {}
    
    # Compute metrics for each K
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(sorted_relevance, k)
        metrics[f"recall@{k}"] = recall_at_k(sorted_relevance, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(sorted_relevance, k)
    
    # MRR (not K-dependent)
    metrics["mrr"] = mean_reciprocal_rank(sorted_relevance)
    
    return metrics


def evaluate_ranking_batch(
    scores_list: List[np.ndarray],
    relevance_list: List[np.ndarray],
    k_values: List[int] = [1, 5, 10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate ranking for multiple queries and aggregate.
    
    Args:
        scores_list: List of score arrays, one per query
        relevance_list: List of relevance arrays, one per query
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary of aggregated metrics (averaged over queries)
    """
    if len(scores_list) != len(relevance_list):
        raise ValueError("scores_list and relevance_list must have same length")
    
    num_queries = len(scores_list)
    
    if num_queries == 0:
        return {}
    
    # Evaluate each query
    all_metrics = []
    for scores, relevance in zip(scores_list, relevance_list):
        metrics = evaluate_ranking(scores, relevance, k_values)
        all_metrics.append(metrics)
    
    # Aggregate: compute mean for each metric
    aggregated = {}
    metric_names = all_metrics[0].keys()
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated[metric_name] = np.mean(values)
    
    return aggregated


def print_ranking_metrics(
    metrics: Dict[str, float],
    title: str = "Ranking Metrics"
):
    """
    Pretty print ranking metrics.
    
    Args:
        metrics: Dictionary of metric_name -> score
        title: Title to print
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Group by metric type
    precision_metrics = {k: v for k, v in metrics.items() if "precision" in k}
    recall_metrics = {k: v for k, v in metrics.items() if "recall" in k}
    ndcg_metrics = {k: v for k, v in metrics.items() if "ndcg" in k}
    other_metrics = {k: v for k, v in metrics.items() 
                    if k not in precision_metrics and k not in recall_metrics and k not in ndcg_metrics}
    
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
    
    print("="*80 + "\n")


# ============================================================================
# Main: Test Metrics
# ============================================================================

if __name__ == "__main__":
    print("Testing Ranking Metrics")
    print("="*80)
    
    # Sample data: 10 items with predicted scores and ground truth relevance
    np.random.seed(42)
    
    scores = np.array([0.9, 0.7, 0.85, 0.3, 0.6, 0.4, 0.75, 0.2, 0.5, 0.1])
    relevance = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])  # 4 relevant items
    
    print("\nPredicted Scores:")
    print(scores)
    print("\nGround Truth Relevance:")
    print(relevance)
    
    # Sort by score to see ranking
    sorted_indices = np.argsort(-scores)
    print("\nRanking (by descending score):")
    for rank, idx in enumerate(sorted_indices, 1):
        rel_label = "✓" if relevance[idx] == 1 else "✗"
        print(f"  Rank {rank}: Item {idx}, Score={scores[idx]:.2f}, Relevant={rel_label}")
    
    # Evaluate metrics
    print("\n" + "="*80)
    metrics = evaluate_ranking(scores, relevance, k_values=[1, 3, 5, 10])
    print_ranking_metrics(metrics, "Single Query Metrics")
    
    # Test batch evaluation
    print("="*80)
    print("Testing Batch Evaluation")
    print("="*80)
    
    # Generate 5 random queries
    num_queries = 5
    num_items = 20
    
    scores_list = []
    relevance_list = []
    
    for i in range(num_queries):
        scores = np.random.rand(num_items)
        # Random 10-20% of items are relevant
        num_relevant = np.random.randint(2, 5)
        relevance = np.zeros(num_items)
        relevant_indices = np.random.choice(num_items, num_relevant, replace=False)
        relevance[relevant_indices] = 1
        
        scores_list.append(scores)
        relevance_list.append(relevance)
    
    # Aggregate metrics
    aggregated_metrics = evaluate_ranking_batch(
        scores_list,
        relevance_list,
        k_values=[1, 5, 10, 20]
    )
    
    print_ranking_metrics(aggregated_metrics, "Aggregated Metrics (5 Queries)")
