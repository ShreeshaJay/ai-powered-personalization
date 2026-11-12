"""Rank fusion and TREC run files.

Ranks are 1-based: the first row of a leg has rank 1, so its RRF term is
1 / (k + 1). Ties are already broken in the run file by score descending and
then product id ascending. A product that is absent from a leg contributes
nothing under RRF, and contributes 0 after normalization under convex
combination and DBSF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HitList = list[tuple[str, float]]
Run = dict[str, HitList]


def sort_hits(scores: dict[str, float]) -> HitList:
    items = list(scores.items())
    items.sort(key=lambda item: (-item[1], item[0]))
    return items


def truncate(hits: HitList, depth: int) -> HitList:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    return list(hits[:depth])


def _as_score_map(hits: HitList) -> dict[str, float]:
    """Keep the first, best, occurrence of each product."""
    found: dict[str, float] = {}
    for product_id, score in hits:
        if product_id not in found:
            found[product_id] = float(score)
    return found


def minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return {product_id: 0.0 for product_id in scores}
    span = hi - lo
    return {product_id: (score - lo) / span for product_id, score in scores.items()}


def zscore_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    mu = float(values.mean())
    sigma = float(values.std(ddof=0))
    if sigma == 0.0:
        return {product_id: 0.0 for product_id in scores}
    return {product_id: (score - mu) / sigma for product_id, score in scores.items()}


def dbsf_norm(scores: dict[str, float]) -> dict[str, float]:
    """Normalize by mean ± 3 sample standard deviations.

    This is the distribution-based fusion used by qdrant-client 1.20.0 local
    mode. One score, or a leg with zero sample variance, normalizes to 0.5.
    A product missing from the leg is absent here and contributes 0 when legs
    are summed. The sample variance divides by n - 1.
    """
    if not scores:
        return {}
    if len(scores) == 1:
        return {product_id: 0.5 for product_id in scores}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    mu = float(values.mean())
    sigma = float(values.std(ddof=1))
    if sigma == 0.0:
        return {product_id: 0.5 for product_id in scores}
    lo = mu - 3.0 * sigma
    span = 6.0 * sigma
    return {product_id: (score - lo) / span for product_id, score in scores.items()}


def _weighted_sum(normed_legs: list[dict[str, float]], weights: list[float]) -> HitList:
    if len(normed_legs) != len(weights):
        raise ValueError("Each leg needs one weight.")
    totals: dict[str, float] = {}
    for normed, weight in zip(normed_legs, weights):
        if weight == 0.0:
            continue
        for product_id, score in normed.items():
            totals[product_id] = totals.get(product_id, 0.0) + weight * score
    return sort_hits(totals)


def rrf(legs: list[HitList], k: float = 60.0, weights: list[float] | None = None) -> HitList:
    """Reciprocal rank fusion. ``k`` is the constant added to the 1-based rank."""
    if k < 0:
        raise ValueError("RRF k must be non-negative.")
    if weights is None:
        weights = [1.0] * len(legs)
    if len(weights) != len(legs):
        raise ValueError("Each leg needs one RRF weight.")
    totals: dict[str, float] = {}
    for hits, weight in zip(legs, weights):
        if weight == 0.0:
            continue
        seen: set[str] = set()
        for rank0, (product_id, _score) in enumerate(hits):
            if product_id in seen:
                continue
            seen.add(product_id)
            rank = rank0 + 1
            totals[product_id] = totals.get(product_id, 0.0) + weight / (k + rank)
    return sort_hits(totals)


def convex(legs: list[HitList], weights: list[float], normalizer: str = "minmax") -> HitList:
    if normalizer == "minmax":
        norm = minmax_norm
    elif normalizer == "zscore":
        norm = zscore_norm
    else:
        raise ValueError(f"Unknown normalizer {normalizer!r}.")
    return _weighted_sum([norm(_as_score_map(hits)) for hits in legs], weights)


def dbsf(legs: list[HitList], weights: list[float] | None = None) -> HitList:
    if weights is None:
        weights = [1.0] * len(legs)
    return _weighted_sum([dbsf_norm(_as_score_map(hits)) for hits in legs], weights)


def simplex_weights(step: float = 0.1, n_legs: int = 3) -> list[tuple[float, ...]]:
    """Nonnegative weights that sum to 1, on a grid of ``step``."""
    if n_legs != 3:
        raise ValueError("The pre-registered grid is the three-leg simplex.")
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1.")
    grid: list[tuple[float, ...]] = []
    for i in range(units + 1):
        for j in range(units - i + 1):
            k = units - i - j
            grid.append((i * step, j * step, k * step))
    return grid


def write_run(path: Path, run: Run, run_name: str, depth: int | None = None) -> int:
    """Write a TREC run. Returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    query_ids = sorted(run)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in query_ids:
            hits = run[query_id]
            if depth is not None:
                hits = hits[:depth]
            for rank0, (product_id, score) in enumerate(hits):
                if any(char.isspace() for char in f"{query_id}{product_id}{run_name}"):
                    raise ValueError("Run fields must not contain whitespace.")
                score_text = format(float(score), ".17g")
                handle.write(
                    f"{query_id} Q0 {product_id} {rank0 + 1} {score_text} {run_name}\n"
                )
                rows += 1
    return rows


def read_run(path: Path) -> Run:
    run: Run = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_number} does not have 6 fields.")
            query_id, unused, product_id, rank_text, score_text = parts[:5]
            if unused != "Q0":
                raise ValueError(f"{path}:{line_number} is missing the Q0 field.")
            rank = int(rank_text)
            expected = len(run.get(query_id, [])) + 1
            if rank != expected:
                raise ValueError(
                    f"{path}:{line_number} has rank {rank}, expected {expected}."
                )
            run.setdefault(query_id, []).append((product_id, float(score_text)))
    return run


def host_array(values, dtype=np.float64) -> np.ndarray:
    """Return a NumPy array on the CPU.

    A CUDA tensor has no NumPy view, and bfloat16 is not a NumPy dtype.
    Both are copied to CPU float32 before the requested dtype is applied.
    """
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(values, dtype=dtype)


def ranking_from_scores(product_ids: list[str], scores, depth: int | None = None) -> HitList:
    array = host_array(scores).reshape(-1)
    if len(product_ids) != len(array):
        raise ValueError("product ids and scores have different lengths.")
    order = sorted(
        range(len(product_ids)),
        key=lambda index: (-float(array[index]), product_ids[index]),
    )
    if depth is not None:
        order = order[:depth]
    return [(product_ids[index], float(array[index])) for index in order]
