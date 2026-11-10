"""
Ranking metrics for a single relevant item per query.

Every evaluation in this chapter has exactly one held-out target per session, so the
metrics are functions of the target's 1-based rank r (r = inf when it is not retrieved):

    Recall@K = 1[r <= K]
    NDCG@K   = 1[r <= K] / log2(r + 1)      (the ideal DCG is 1)
    MRR      = 1 / r                         (0 when not retrieved)
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np


def metrics_from_rank(rank: float, k_values: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    found = math.isfinite(rank)
    for k in k_values:
        hit = found and rank <= k
        out[f"recall@{k}"] = 1.0 if hit else 0.0
        out[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
    out["mrr"] = 1.0 / rank if found else 0.0
    return out


def aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {"num_queries": 0}
    keys = per_query[0].keys()
    out = {k: float(np.mean([m[k] for m in per_query])) for k in keys}
    out["num_queries"] = len(per_query)
    return out


def rank_of_target(scores: np.ndarray, target_index: int) -> int:
    """1-based rank of ``target_index`` under descending scores; ties count against the target."""
    scores = np.asarray(scores, dtype=np.float64)
    target_score = scores[target_index]
    return int(np.sum(scores > target_score) + np.sum(scores[:target_index] == target_score)) + 1


def rank_in_list(ranked_ids: Iterable, target) -> float:
    """1-based position of ``target`` in a ranked list, or inf when absent."""
    for pos, item in enumerate(ranked_ids, start=1):
        if item == target:
            return float(pos)
    return math.inf


def random_slate_expectation(slate_size: int, k_values: Sequence[int]) -> Dict[str, float]:
    """Expected metrics when the target's position in a slate of size n is uniformly random."""
    n = slate_size
    out: Dict[str, float] = {}
    for k in k_values:
        kk = min(k, n)
        out[f"recall@{k}"] = kk / n
        out[f"ndcg@{k}"] = sum(1.0 / math.log2(r + 1) for r in range(1, kk + 1)) / n
    out["mrr"] = sum(1.0 / r for r in range(1, n + 1)) / n
    return out
