"""
Utilities for Chapter 10: Item Embeddings
"""

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    evaluate_ranking,
    evaluate_ranking_batch,
    print_ranking_metrics
)

from .faiss_index import (
    build_faiss_index,
    search_faiss_index,
    brute_force_search,
    FaissIndexWrapper
)

__all__ = [
    # Metrics
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "evaluate_ranking",
    "evaluate_ranking_batch",
    "print_ranking_metrics",
    # FAISS
    "build_faiss_index",
    "search_faiss_index",
    "brute_force_search",
    "FaissIndexWrapper",
]
