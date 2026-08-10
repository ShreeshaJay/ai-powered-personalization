"""
Maximal Marginal Relevance (MMR) for Diversity Optimization

This module implements MMR reranking to balance relevance and diversity
in recommendation lists, preventing homogeneous results.

Key Concepts:
- Pointwise relevance vs. listwise diversity
- Embedding-based similarity for measuring redundancy
- Greedy selection with marginal gain

Reference: Carbonell & Goldstein (1998) - "The Use of MMR, Diversity-Based 
Reranking for Reordering Documents and Producing Summaries"
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for MMR diversity reranking.
    
    Attributes:
        lambda_param: Trade-off between relevance and diversity.
            - 1.0 = pure relevance (no diversity)
            - 0.0 = pure diversity (ignores relevance)
            - Typical values: 0.5 - 0.8
        top_k: Number of items to select in final ranking
        similarity_metric: Similarity function ('cosine', 'euclidean', 'dot')
        normalize_relevance: Whether to normalize relevance scores to [0, 1]
        min_similarity_threshold: Skip diversity penalty if similarity < threshold
    """
    lambda_param: float = 0.7
    top_k: int = 20
    similarity_metric: str = 'cosine'
    normalize_relevance: bool = True
    min_similarity_threshold: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.lambda_param <= 1:
            raise ValueError(f"lambda_param must be in [0, 1], got {self.lambda_param}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


class MMRReranker:
    """
    Maximal Marginal Relevance reranker using item embeddings.
    
    MMR selects items greedily, at each step choosing the item that
    maximizes:
    
        MMR(i) = λ * Relevance(i) - (1-λ) * max_{j ∈ S} Similarity(i, j)
    
    Where:
        - Relevance(i) = normalized value score for item i
        - S = set of already selected items
        - Similarity = embedding similarity (e.g., cosine)
    
    This balances showing relevant items while avoiding redundancy.
    
    Example:
        >>> embeddings = load_embeddings("embeddings.parquet")
        >>> reranker = MMRReranker(embeddings, DiversityConfig(lambda_param=0.7))
        >>> candidates = [(item_id, score), ...]  # From value function
        >>> diverse_ranking = reranker.rerank(candidates, top_k=20)
    """
    
    def __init__(
        self,
        embeddings: Dict[int, np.ndarray],
        config: Optional[DiversityConfig] = None,
    ):
        """
        Initialize MMR reranker with item embeddings.
        
        Args:
            embeddings: Dictionary mapping item_id -> embedding vector
            config: DiversityConfig instance. Uses defaults if None.
        """
        self.embeddings = embeddings
        self.config = config or DiversityConfig()
        self.embedding_dim = self._infer_embedding_dim()
        logger.info(
            f"Initialized MMRReranker with {len(embeddings)} embeddings "
            f"(dim={self.embedding_dim}), lambda={self.config.lambda_param}"
        )
    
    def _infer_embedding_dim(self) -> int:
        """Infer embedding dimension from first embedding."""
        if not self.embeddings:
            return 0
        sample_embed = next(iter(self.embeddings.values()))
        return len(sample_embed)
    
    def rerank(
        self,
        candidates: List[Tuple[int, float]],
        top_k: Optional[int] = None,
        lambda_param: Optional[float] = None,
    ) -> List[int]:
        """
        Rerank candidates using MMR to balance relevance and diversity.
        
        Args:
            candidates: List of (item_id, relevance_score) tuples
            top_k: Override config.top_k for this call
            lambda_param: Override config.lambda_param for this call
        
        Returns:
            List of item_ids in MMR-reranked order
        """
        if not candidates:
            return []
        
        top_k = top_k or self.config.top_k
        lambda_param = lambda_param if lambda_param is not None else self.config.lambda_param
        
        # Extract scores and normalize
        item_ids = [item_id for item_id, _ in candidates]
        scores = np.array([score for _, score in candidates], dtype=np.float32)
        
        if self.config.normalize_relevance:
            scores = self._normalize_scores(scores)
        
        # Build score lookup
        score_map = {item_id: score for item_id, score in zip(item_ids, scores)}
        
        # Greedy MMR selection
        selected = []
        remaining = set(item_ids)
        
        while len(selected) < top_k and remaining:
            best_item = None
            best_mmr = -np.inf
            
            for item_id in remaining:
                relevance = score_map[item_id]
                
                if not selected:
                    # First item: no diversity penalty
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = self._compute_max_similarity(item_id, selected)
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_item = item_id
            
            if best_item is not None:
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                break
        
        return selected
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score < 1e-8:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)
    
    def _compute_max_similarity(
        self,
        item_id: int,
        selected_items: List[int],
    ) -> float:
        """
        Compute maximum similarity between item and any selected item.
        
        Returns 0 if item has no embedding (graceful degradation).
        """
        item_embed = self.embeddings.get(item_id)
        if item_embed is None:
            return 0.0
        
        item_embed = np.asarray(item_embed).reshape(1, -1)
        
        max_sim = 0.0
        for sel_id in selected_items:
            sel_embed = self.embeddings.get(sel_id)
            if sel_embed is None:
                continue
            
            sel_embed = np.asarray(sel_embed).reshape(1, -1)
            
            if self.config.similarity_metric == 'cosine':
                sim = cosine_similarity(item_embed, sel_embed)[0, 0]
            elif self.config.similarity_metric == 'dot':
                sim = np.dot(item_embed.flatten(), sel_embed.flatten())
            elif self.config.similarity_metric == 'euclidean':
                # Convert distance to similarity
                dist = np.linalg.norm(item_embed - sel_embed)
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown similarity metric: {self.config.similarity_metric}")
            
            if sim > max_sim:
                max_sim = sim
        
        return max_sim
    
    def rerank_with_scores(
        self,
        candidates: List[Tuple[int, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float, float]]:
        """
        Rerank with detailed scores for debugging.
        
        Returns:
            List of (item_id, original_score, mmr_score) tuples
        """
        if not candidates:
            return []
        
        top_k = top_k or self.config.top_k
        lambda_param = self.config.lambda_param
        
        item_ids = [item_id for item_id, _ in candidates]
        scores = np.array([score for _, score in candidates], dtype=np.float32)
        
        if self.config.normalize_relevance:
            normalized_scores = self._normalize_scores(scores)
        else:
            normalized_scores = scores
        
        score_map = {item_id: (orig, norm) for item_id, orig, norm 
                     in zip(item_ids, scores, normalized_scores)}
        
        selected = []
        remaining = set(item_ids)
        results = []
        
        while len(selected) < top_k and remaining:
            best_item = None
            best_mmr = -np.inf
            
            for item_id in remaining:
                _, norm_score = score_map[item_id]
                
                if not selected:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = self._compute_max_similarity(item_id, selected)
                
                mmr_score = lambda_param * norm_score - (1 - lambda_param) * diversity_penalty
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_item = item_id
            
            if best_item is not None:
                orig_score, _ = score_map[best_item]
                results.append((best_item, orig_score, best_mmr))
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                break
        
        return results


# ============================================================================
# Embedding Loading Utilities
# ============================================================================

def load_yambda_embeddings(
    embeddings_path: Union[str, Path],
    use_normalized: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Load Yambda audio embeddings from parquet file.
    
    Args:
        embeddings_path: Path to embeddings.parquet
        use_normalized: If True, use normalized_embed (L2-normalized).
                       If False, use raw embed.
    
    Returns:
        Dictionary mapping item_id -> embedding vector
    """
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    embed_col = 'normalized_embed' if use_normalized else 'embed'
    
    df = pl.read_parquet(embeddings_path)
    
    embeddings = {}
    for row in df.iter_rows(named=True):
        item_id = row['item_id']
        embed = row[embed_col]
        if embed is not None:
            embeddings[item_id] = np.array(embed, dtype=np.float32)
    
    logger.info(f"Loaded {len(embeddings)} embeddings")
    return embeddings


# ============================================================================
# Diversity Metrics
# ============================================================================

def compute_intra_list_diversity(
    item_ids: List[int],
    embeddings: Dict[int, np.ndarray],
) -> float:
    """
    Compute Intra-List Diversity (ILD) for a ranked list.
    
    ILD = average pairwise dissimilarity among items in list
    
    Args:
        item_ids: List of item IDs in ranking
        embeddings: item_id -> embedding mapping
    
    Returns:
        ILD score in [0, 1]. Higher = more diverse.
    """
    # Get embeddings for items that have them
    embeds = []
    for item_id in item_ids:
        if item_id in embeddings:
            embeds.append(embeddings[item_id])
    
    if len(embeds) < 2:
        return 0.0
    
    embed_matrix = np.vstack(embeds)
    sim_matrix = cosine_similarity(embed_matrix)
    
    # Average pairwise dissimilarity (excluding diagonal)
    n = len(embeds)
    total_dissim = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            total_dissim += 1 - sim_matrix[i, j]
            count += 1
    
    if count == 0:
        return 0.0
    
    return total_dissim / count


def compute_coverage(
    item_ids: List[int],
    artist_mapping: Dict[int, int],
) -> Dict[str, float]:
    """
    Compute coverage metrics for a ranked list.
    
    Args:
        item_ids: List of item IDs
        artist_mapping: item_id -> artist_id mapping
    
    Returns:
        Dictionary with coverage metrics
    """
    artists_in_list = set()
    items_with_artist = 0
    
    for item_id in item_ids:
        artist_id = artist_mapping.get(item_id)
        if artist_id is not None:
            artists_in_list.add(artist_id)
            items_with_artist += 1
    
    return {
        'unique_artists': len(artists_in_list),
        'artist_coverage_ratio': len(artists_in_list) / len(item_ids) if item_ids else 0,
        'items_with_artist': items_with_artist,
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    print("=" * 60)
    print("MMR Diversity Reranking Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create synthetic embeddings (3 clusters + noise)
    n_items = 50
    embed_dim = 64
    
    embeddings = {}
    cluster_centers = [
        np.random.randn(embed_dim),  # Cluster 1
        np.random.randn(embed_dim),  # Cluster 2
        np.random.randn(embed_dim),  # Cluster 3
    ]
    
    for i in range(n_items):
        cluster = i % 3
        noise = np.random.randn(embed_dim) * 0.3
        embed = cluster_centers[cluster] + noise
        embed = embed / np.linalg.norm(embed)  # Normalize
        embeddings[i] = embed
    
    # Create candidates with relevance scores
    # Higher scores for cluster 0 (to show MMR doesn't just pick top scores)
    candidates = []
    for i in range(n_items):
        cluster = i % 3
        if cluster == 0:
            score = np.random.uniform(0.8, 1.0)
        elif cluster == 1:
            score = np.random.uniform(0.5, 0.7)
        else:
            score = np.random.uniform(0.3, 0.5)
        candidates.append((i, score))
    
    # Sort by score (baseline: pure relevance)
    baseline = sorted(candidates, key=lambda x: -x[1])[:10]
    baseline_ids = [item_id for item_id, _ in baseline]
    
    print(f"\nBaseline (top 10 by relevance):")
    print(f"  Item IDs: {baseline_ids}")
    print(f"  Clusters: {[i % 3 for i in baseline_ids]}")
    print(f"  ILD: {compute_intra_list_diversity(baseline_ids, embeddings):.4f}")
    
    # MMR reranking with different lambda values
    for lambda_param in [1.0, 0.7, 0.5, 0.3]:
        config = DiversityConfig(lambda_param=lambda_param, top_k=10)
        reranker = MMRReranker(embeddings, config)
        mmr_ids = reranker.rerank(candidates)
        
        print(f"\nMMR (lambda={lambda_param}):")
        print(f"  Item IDs: {mmr_ids}")
        print(f"  Clusters: {[i % 3 for i in mmr_ids]}")
        print(f"  ILD: {compute_intra_list_diversity(mmr_ids, embeddings):.4f}")

