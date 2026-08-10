"""
Utility functions for Chapter 11: User/Customer Embeddings.

Metrics and FAISS index utilities are adapted from Chapter 10.
"""

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    evaluate_ranking,
    evaluate_ranking_batch,
    print_ranking_metrics,
)
from .faiss_index import (
    FaissIndexWrapper,
    build_faiss_index,
    search_faiss_index,
    brute_force_search,
    FAISS_AVAILABLE,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "evaluate_ranking",
    "evaluate_ranking_batch",
    "print_ranking_metrics",
    "FaissIndexWrapper",
    "build_faiss_index",
    "search_faiss_index",
    "brute_force_search",
    "FAISS_AVAILABLE",
]
