"""
Evaluation Metrics for Value Functions and Diversity

This module provides metrics to evaluate the ordering pipeline:
1. Ranking quality (NDCG, MRR, Hit Rate)
2. Diversity metrics (ILD, Coverage)
3. Business metrics (artist diversity, engagement)
4. Relevance-diversity tradeoff analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Ranking Quality Metrics
# ============================================================================

def compute_ndcg(
    predicted_ranking: List[int],
    relevant_items: List[int],
    k: int = 10,
    relevance_scores: Optional[Dict[int, float]] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Args:
        predicted_ranking: List of item IDs in predicted order
        relevant_items: List of relevant item IDs (ground truth)
        k: Cutoff position
        relevance_scores: Optional dict of item_id -> relevance score
                         If None, binary relevance (1 if relevant, 0 otherwise)
    
    Returns:
        NDCG@k score in [0, 1]
    """
    relevant_set = set(relevant_items)
    
    # Compute DCG
    dcg = 0.0
    for i, item_id in enumerate(predicted_ranking[:k]):
        if relevance_scores is not None:
            rel = relevance_scores.get(item_id, 0.0)
        else:
            rel = 1.0 if item_id in relevant_set else 0.0
        
        dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
    
    # Compute ideal DCG
    if relevance_scores is not None:
        sorted_rels = sorted(
            [relevance_scores.get(item, 0.0) for item in relevant_items],
            reverse=True,
        )
    else:
        sorted_rels = [1.0] * len(relevant_items)
    
    idcg = 0.0
    for i, rel in enumerate(sorted_rels[:k]):
        idcg += rel / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_mrr(
    predicted_ranking: List[int],
    relevant_items: List[int],
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        predicted_ranking: List of item IDs in predicted order
        relevant_items: List of relevant item IDs
    
    Returns:
        MRR score
    """
    relevant_set = set(relevant_items)
    
    for i, item_id in enumerate(predicted_ranking):
        if item_id in relevant_set:
            return 1.0 / (i + 1)
    
    return 0.0


def compute_hit_rate(
    predicted_ranking: List[int],
    relevant_items: List[int],
    k: int = 10,
) -> float:
    """
    Compute Hit Rate at k.
    
    Args:
        predicted_ranking: List of item IDs in predicted order
        relevant_items: List of relevant item IDs
        k: Cutoff position
    
    Returns:
        1.0 if any relevant item in top-k, else 0.0
    """
    relevant_set = set(relevant_items)
    top_k = set(predicted_ranking[:k])
    
    return 1.0 if relevant_set & top_k else 0.0


# ============================================================================
# Diversity Metrics
# ============================================================================

def compute_intra_list_diversity(
    item_ids: List[int],
    embeddings: Dict[int, np.ndarray],
    metric: str = 'cosine',
) -> float:
    """
    Compute Intra-List Diversity (ILD).
    
    ILD = average pairwise dissimilarity among items
    
    Args:
        item_ids: List of item IDs
        embeddings: item_id -> embedding vector mapping
        metric: Distance metric ('cosine', 'euclidean')
    
    Returns:
        ILD score in [0, 1] (for cosine) or [0, inf) (for euclidean)
    """
    # Get embeddings for items
    embeds = [embeddings[i] for i in item_ids if i in embeddings]
    
    if len(embeds) < 2:
        return 0.0
    
    embed_matrix = np.vstack(embeds)
    n = len(embeds)
    
    if metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embed_matrix)
        
        total_dissim = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dissim += 1 - sim_matrix[i, j]
                count += 1
        
        return total_dissim / count if count > 0 else 0.0
    
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(embed_matrix)
        
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += dist_matrix[i, j]
                count += 1
        
        return total_dist / count if count > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_coverage(
    predicted_rankings: List[List[int]],
    catalog_size: int,
    k: int = 10,
) -> float:
    """
    Compute catalog coverage across multiple users.
    
    Args:
        predicted_rankings: List of rankings (one per user)
        catalog_size: Total number of items in catalog
        k: Cutoff position
    
    Returns:
        Fraction of catalog covered in top-k across all users
    """
    covered_items = set()
    
    for ranking in predicted_rankings:
        covered_items.update(ranking[:k])
    
    return len(covered_items) / catalog_size


def compute_artist_diversity(
    item_ids: List[int],
    artist_mapping: Dict[int, int],
) -> Dict[str, float]:
    """
    Compute artist diversity metrics.
    
    Args:
        item_ids: List of item IDs in ranking
        artist_mapping: item_id -> artist_id mapping
    
    Returns:
        Dictionary with diversity metrics
    """
    artists = []
    for item_id in item_ids:
        artist_id = artist_mapping.get(item_id)
        if artist_id is not None:
            artists.append(artist_id)
    
    if not artists:
        return {
            'unique_artists': 0,
            'artist_entropy': 0.0,
            'artist_gini': 0.0,
        }
    
    unique_artists = len(set(artists))
    
    # Compute entropy
    from collections import Counter
    artist_counts = Counter(artists)
    probs = np.array(list(artist_counts.values())) / len(artists)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Compute Gini coefficient
    sorted_counts = np.sort(list(artist_counts.values()))
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    
    return {
        'unique_artists': unique_artists,
        'artist_entropy': entropy,
        'artist_gini': gini,
    }


# ============================================================================
# Combined Evaluation
# ============================================================================

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    ndcg_5: float
    ndcg_10: float
    mrr: float
    hit_rate_10: float
    ild: float
    artist_diversity: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'ndcg@5': self.ndcg_5,
            'ndcg@10': self.ndcg_10,
            'mrr': self.mrr,
            'hit_rate@10': self.hit_rate_10,
            'ild': self.ild,
            **{f'artist_{k}': v for k, v in self.artist_diversity.items()},
        }


def evaluate_ranking(
    predicted_ranking: List[int],
    relevant_items: List[int],
    embeddings: Optional[Dict[int, np.ndarray]] = None,
    artist_mapping: Optional[Dict[int, int]] = None,
) -> EvaluationResult:
    """
    Comprehensive evaluation of a single ranking.
    
    Args:
        predicted_ranking: List of item IDs in predicted order
        relevant_items: List of relevant item IDs (ground truth)
        embeddings: Optional embeddings for ILD computation
        artist_mapping: Optional artist mapping for artist diversity
    
    Returns:
        EvaluationResult with all metrics
    """
    ndcg_5 = compute_ndcg(predicted_ranking, relevant_items, k=5)
    ndcg_10 = compute_ndcg(predicted_ranking, relevant_items, k=10)
    mrr = compute_mrr(predicted_ranking, relevant_items)
    hit_rate_10 = compute_hit_rate(predicted_ranking, relevant_items, k=10)
    
    ild = 0.0
    if embeddings:
        ild = compute_intra_list_diversity(predicted_ranking[:10], embeddings)
    
    artist_div = {}
    if artist_mapping:
        artist_div = compute_artist_diversity(predicted_ranking[:10], artist_mapping)
    
    return EvaluationResult(
        ndcg_5=ndcg_5,
        ndcg_10=ndcg_10,
        mrr=mrr,
        hit_rate_10=hit_rate_10,
        ild=ild,
        artist_diversity=artist_div,
    )


def evaluate_pipeline_batch(
    rankings: List[List[int]],
    ground_truths: List[List[int]],
    embeddings: Optional[Dict[int, np.ndarray]] = None,
    artist_mapping: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """
    Evaluate pipeline across multiple users.
    
    Args:
        rankings: List of predicted rankings (one per user)
        ground_truths: List of ground truth relevant items (one per user)
        embeddings: Optional embeddings for ILD
        artist_mapping: Optional artist mapping
    
    Returns:
        Dictionary of averaged metrics
    """
    results = []
    
    for ranking, gt in zip(rankings, ground_truths):
        result = evaluate_ranking(ranking, gt, embeddings, artist_mapping)
        results.append(result)
    
    # Average all metrics
    avg_metrics = defaultdict(float)
    for result in results:
        for key, value in result.to_dict().items():
            avg_metrics[key] += value
    
    n = len(results)
    return {key: value / n for key, value in avg_metrics.items()}


# ============================================================================
# Tradeoff Analysis
# ============================================================================

def analyze_relevance_diversity_tradeoff(
    candidates: List[Tuple[int, float]],
    embeddings: Dict[int, np.ndarray],
    relevant_items: List[int],
    lambda_values: List[float] = None,
) -> List[Dict]:
    """
    Analyze relevance-diversity tradeoff across lambda values.
    
    Args:
        candidates: List of (item_id, score) tuples
        embeddings: item_id -> embedding mapping
        relevant_items: Ground truth relevant items
        lambda_values: List of lambda values to test
    
    Returns:
        List of dicts with metrics for each lambda
    """
    from .mmr_diversity import MMRReranker, DiversityConfig
    
    lambda_values = lambda_values or [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    results = []
    
    for lam in lambda_values:
        config = DiversityConfig(lambda_param=lam, top_k=10)
        reranker = MMRReranker(embeddings, config)
        
        ranking = reranker.rerank(candidates)
        
        ndcg = compute_ndcg(ranking, relevant_items, k=10)
        ild = compute_intra_list_diversity(ranking, embeddings)
        
        results.append({
            'lambda': lam,
            'ndcg@10': ndcg,
            'ild': ild,
        })
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Metrics Demonstration")
    print("=" * 60)
    
    # Synthetic example
    np.random.seed(42)
    
    # Ground truth: items 0-4 are relevant
    relevant_items = [0, 1, 2, 3, 4]
    
    # Different rankings to compare
    rankings = {
        'Perfect': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Partial': [0, 5, 1, 6, 2, 7, 8, 9, 3, 4],
        'Poor': [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
    }
    
    print("\nRanking Quality Metrics:")
    print("-" * 40)
    
    for name, ranking in rankings.items():
        ndcg = compute_ndcg(ranking, relevant_items, k=10)
        mrr = compute_mrr(ranking, relevant_items)
        hr = compute_hit_rate(ranking, relevant_items, k=5)
        
        print(f"\n{name} ranking: {ranking[:10]}")
        print(f"  NDCG@10: {ndcg:.4f}")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Hit Rate@5: {hr:.4f}")
    
    # Diversity metrics with synthetic embeddings
    print("\n" + "=" * 60)
    print("Diversity Metrics")
    print("=" * 60)
    
    # Create embeddings (cluster structure)
    embeddings = {}
    for i in range(20):
        cluster = i % 4
        center = np.zeros(64)
        center[cluster * 16:(cluster + 1) * 16] = 1.0
        noise = np.random.randn(64) * 0.1
        embeddings[i] = (center + noise) / np.linalg.norm(center + noise)
    
    diverse_ranking = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6]  # Mix of clusters
    homogeneous_ranking = [0, 1, 2, 3, 16, 17, 18, 19, 4, 5]  # Same cluster
    
    print(f"\nDiverse ranking ILD: {compute_intra_list_diversity(diverse_ranking, embeddings):.4f}")
    print(f"Homogeneous ranking ILD: {compute_intra_list_diversity(homogeneous_ranking, embeddings):.4f}")

