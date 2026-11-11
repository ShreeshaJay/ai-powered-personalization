"""Retrieval metrics, bootstrap intervals, and the pre-registered decision rule.

Single-target Recall@K equals hit rate. The Chapter 10 protocol aggregates
several targets per query, and there Recall@K is the fraction of those targets
retrieved. Both are computed from exact inner products. faiss.IndexFlatIP is
built on the same vectors and checked against the exact top neighbors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from config import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_REQUIRED_SEEDS,
    RETRIEVAL_TOP_K,
    UNIT_NORM_TOLERANCE,
)

METRIC_NAMES = ("recall_at_10", "recall_at_50", "mrr", "mrr_at_100", "ndcg_at_10")


def unit_norms(embeddings: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(embeddings, dtype=np.float64), axis=1)


def check_unit_norm(embeddings: np.ndarray, tolerance: float = UNIT_NORM_TOLERANCE) -> float:
    norms = unit_norms(embeddings)
    if norms.size == 0:
        return 0.0
    deviation = float(np.max(np.abs(norms - 1.0)))
    if deviation > tolerance:
        raise ValueError(f"Embedding norms deviate from 1 by {deviation:.3e}")
    return deviation


def ndcg_from_binary_rels(rels: Sequence[int], n_relevant: int, k: int) -> float:
    """NDCG@k for binary relevance. Discount is 1/log2(rank+1) with rank starting at 1."""
    if n_relevant <= 0 or k <= 0:
        return 0.0
    dcg = 0.0
    for index, rel in enumerate(rels[:k]):
        if rel:
            dcg += 1.0 / math.log2(index + 2)
    ideal = 0.0
    for index in range(min(n_relevant, k)):
        ideal += 1.0 / math.log2(index + 2)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def single_target_metrics(rank: int) -> dict[str, float]:
    """Metrics for one relevant item. rank is 1-based; 0 means the target is absent."""
    if rank <= 0:
        return {name: 0.0 for name in METRIC_NAMES}
    return {
        "recall_at_10": 1.0 if rank <= 10 else 0.0,
        "recall_at_50": 1.0 if rank <= 50 else 0.0,
        "mrr": 1.0 / rank,
        "mrr_at_100": (1.0 / rank) if rank <= 100 else 0.0,
        "ndcg_at_10": (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
    }


def multi_target_at_k(
    neighbor_ids: Sequence[int],
    target_ids: set[int],
    k_recall: tuple[int, ...] = (10, 50),
) -> dict[str, float]:
    """Recall, hit rate, MRR@100 and NDCG@10 from a self-excluded neighbor list."""
    n_relevant = len(target_ids)
    if n_relevant == 0:
        return {
            "recall_at_10": 0.0,
            "recall_at_50": 0.0,
            "hit_rate_at_10": 0.0,
            "hit_rate_at_50": 0.0,
            "mrr_at_100": 0.0,
            "ndcg_at_10": 0.0,
            "best_rank_in_list": 0,
        }
    trimmed = list(neighbor_ids[:RETRIEVAL_TOP_K])
    best = 0
    for rank, item in enumerate(trimmed, start=1):
        if item in target_ids:
            best = rank
            break
    hits = {}
    for k in k_recall:
        hits[k] = sum(1 for item in trimmed[:k] if item in target_ids)
    rels = [1 if item in target_ids else 0 for item in trimmed[:10]]
    return {
        "recall_at_10": hits[10] / n_relevant,
        "recall_at_50": hits[50] / n_relevant,
        "hit_rate_at_10": 1.0 if hits[10] > 0 else 0.0,
        "hit_rate_at_50": 1.0 if hits[50] > 0 else 0.0,
        "mrr_at_100": (1.0 / best) if 0 < best <= 100 else 0.0,
        "ndcg_at_10": ndcg_from_binary_rels(rels, n_relevant, 10),
        "best_rank_in_list": best,
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        means[draw] = values[rng.integers(0, n, size=n)].mean()
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high), "n": n}


def summarize_metric_matrix(
    per_query: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict]:
    summary = {}
    for name, values in per_query.items():
        series = np.asarray(values, dtype=np.float64)
        if mask is not None:
            series = series[mask]
        summary[name] = bootstrap_mean_interval(series, n_resamples=n_resamples, seed=seed)
    return summary


def supported_difference(
    per_seed_mean_diffs: Sequence[float],
    interval_low: float,
    interval_high: float,
    n_required: int = DECISION_REQUIRED_SEEDS,
) -> dict:
    """A difference is supported when the paired 95% interval excludes zero and every required seed agrees in sign."""
    diffs = [float(value) for value in per_seed_mean_diffs]
    agree_pos = len(diffs) >= n_required and all(value > 0 for value in diffs)
    agree_neg = len(diffs) >= n_required and all(value < 0 for value in diffs)
    excludes = interval_low > 0 or interval_high < 0
    if agree_pos:
        sign = "positive"
    elif agree_neg:
        sign = "negative"
    else:
        sign = "not_agreed"
    return {
        "supported": bool((agree_pos or agree_neg) and excludes),
        "sign": sign,
        "interval_excludes_zero": bool(excludes),
        "n_seeds": len(diffs),
        "n_seeds_required": n_required,
        "per_seed_mean_difference": diffs,
        "interval_low": float(interval_low),
        "interval_high": float(interval_high),
    }


def seed_averaged_difference(
    left_by_seed: Sequence[np.ndarray],
    right_by_seed: Sequence[np.ndarray],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired interval on the per-query mean across seeds, plus each seed's mean difference."""
    if len(left_by_seed) != len(right_by_seed) or not left_by_seed:
        raise ValueError("Seed stacks must be non-empty and aligned")
    per_seed = []
    for left, right in zip(left_by_seed, right_by_seed):
        per_seed.append(float(np.mean(np.asarray(left) - np.asarray(right))))
    avg_left = np.mean(np.stack([np.asarray(row, dtype=np.float64) for row in left_by_seed]), axis=0)
    avg_right = np.mean(np.stack([np.asarray(row, dtype=np.float64) for row in right_by_seed]), axis=0)
    interval = bootstrap_mean_interval(avg_left - avg_right, n_resamples=n_resamples, seed=seed)
    decision = supported_difference(per_seed, float(interval["low"]), float(interval["high"]))
    decision["seed_averaged_mean_difference"] = interval["mean"]
    decision["metric_interval"] = interval
    return decision


def _as_float32(embeddings: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def build_ip_index(embeddings: np.ndarray):
    import faiss

    matrix = _as_float32(embeddings)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def exact_ranks(
    embeddings: np.ndarray,
    query_rows: np.ndarray,
    target_rows: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """1-based rank of each target under exact inner product, with the query row removed."""
    matrix = _as_float32(embeddings)
    ranks = np.empty(len(query_rows), dtype=np.int32)
    for start in range(0, len(query_rows), batch_size):
        stop = min(start + batch_size, len(query_rows))
        q_idx = query_rows[start:stop]
        t_idx = target_rows[start:stop]
        sims = matrix[q_idx] @ matrix.T
        for local in range(stop - start):
            sims[local, int(q_idx[local])] = -np.inf
            score = sims[local, int(t_idx[local])]
            better = int(np.sum(sims[local] > score))
            ranks[start + local] = better + 1
    return ranks


def faiss_top_matches_exact(
    embeddings: np.ndarray,
    query_rows: np.ndarray,
    n_check: int = 8,
    k: int = 5,
) -> bool:
    """Confirm IndexFlatIP's top neighbors match an exact inner-product sort."""
    if len(query_rows) == 0 or embeddings.shape[0] < 2:
        return True
    matrix = _as_float32(embeddings)
    k_eff = min(k, matrix.shape[0] - 1)
    take = min(n_check, len(query_rows))
    chosen = query_rows[:take]
    index = build_ip_index(matrix)
    _, neighbors = index.search(matrix[chosen], k_eff + 1)
    for local, query_row in enumerate(chosen):
        sims = matrix @ matrix[int(query_row)]
        sims[int(query_row)] = -np.inf
        exact_scores = np.sort(sims)[-k_eff:][::-1]
        filtered = [int(item) for item in neighbors[local] if int(item) != int(query_row) and int(item) >= 0][:k_eff]
        if len(filtered) < k_eff:
            return False
        got = np.sort(sims[filtered])[::-1]
        if not np.allclose(got, exact_scores, rtol=0.0, atol=1e-3):
            return False
    return True


def ranks_for_single_targets(
    embeddings: np.ndarray,
    query_rows: np.ndarray,
    target_rows: np.ndarray,
    top_k: int = RETRIEVAL_TOP_K,
) -> np.ndarray:
    """1-based ranks. IndexFlatIP supplies the top-k; an exact scan covers targets beyond it."""
    if len(query_rows) == 0:
        return np.zeros(0, dtype=np.int32)
    matrix = _as_float32(embeddings)
    if not faiss_top_matches_exact(matrix, query_rows):
        raise RuntimeError("IndexFlatIP neighbors did not match the exact inner-product order")
    index = build_ip_index(matrix)
    search_k = min(top_k + 1, matrix.shape[0])
    _, neighbors = index.search(matrix[query_rows], search_k)
    ranks = np.full(len(query_rows), -1, dtype=np.int32)
    missed: list[int] = []
    for row in range(len(query_rows)):
        self_row = int(query_rows[row])
        target = int(target_rows[row])
        filtered = [int(item) for item in neighbors[row] if int(item) != self_row and int(item) >= 0][:top_k]
        if target in filtered:
            ranks[row] = filtered.index(target) + 1
        else:
            missed.append(row)
    if missed:
        missed_arr = np.asarray(missed, dtype=np.int64)
        ranks[missed_arr] = exact_ranks(matrix, query_rows[missed_arr], target_rows[missed_arr])
    return ranks


def per_query_from_ranks(ranks: np.ndarray) -> dict[str, np.ndarray]:
    out = {name: np.empty(len(ranks), dtype=np.float64) for name in METRIC_NAMES}
    for index, rank in enumerate(ranks.tolist()):
        metrics = single_target_metrics(int(rank))
        for name in METRIC_NAMES:
            out[name][index] = metrics[name]
    return out


def self_excluded_neighbors(neighbor_row: np.ndarray, self_row: int, top_k: int) -> list[int]:
    filtered = [int(item) for item in neighbor_row if int(item) != int(self_row) and int(item) >= 0]
    return filtered[:top_k]


def lookup_rows(ids: Sequence[str], id_to_row: dict[str, int]) -> tuple[list[int], list[int]]:
    """Return (kept local positions, row indices) for ids that are in the catalog index."""
    kept: list[int] = []
    rows: list[int] = []
    for position, product_id in enumerate(ids):
        row = id_to_row.get(product_id)
        if row is None:
            continue
        kept.append(position)
        rows.append(row)
    return kept, rows
