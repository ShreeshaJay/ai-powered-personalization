"""
Index builders, search-time parameter sweeps, and Pareto analysis.

For each index and search-time setting the sweep measures:

* recall@K against exact inner-product search;
* next-item hit rate@K on session queries, when targets exist (Coveo);
* single-query latency at one thread (p50 and p99, repeated runs);
* batched throughput with all threads;
* index memory (serialized size) and build time.

A setting is *dominated* when another setting for the same dataset and K has
latency no higher and recall no lower, with at least one strictly better.
Only non-dominated settings are worth considering (Douze et al., 2024).
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from latency.timing import bootstrap_ci, repeat_ci_log


def default_nlist(n_items: int) -> int:
    """Power of two near 4 * sqrt(n), capped so each list gets at least 39 training points."""
    target = 4 * math.sqrt(n_items)
    nlist = 2 ** int(round(math.log2(max(target, 1))))
    while nlist > 1 and n_items / nlist < 39:
        nlist //= 2
    return max(nlist, 1)


def pq_subquantizers(dim: int) -> int:
    for m in (16, 10, 8, 5, 4, 2):
        if dim % m == 0 and dim // m >= 2:
            return m
    return 1


def build_index(kind: str, xb: np.ndarray, nlist: Optional[int] = None, hnsw_m: int = 32,
                ef_construction: int = 80, seed: int = 0) -> Tuple[faiss.Index, float]:
    """Build ``flat``, ``ivf_flat``, ``hnsw_flat``, or ``ivf_pq`` with inner-product metric."""
    d = xb.shape[1]
    t0 = time.perf_counter()
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind in ("ivf_flat", "ivf_pq"):
        nlist = nlist or default_nlist(len(xb))
        quantizer = faiss.IndexFlatIP(d)
        if kind == "ivf_flat":
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_subquantizers(d), 8, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = seed
        rng = np.random.default_rng(seed)
        n_train = min(len(xb), max(nlist * 64, 50_000))
        index.train(xb[rng.choice(len(xb), n_train, replace=False)])
    elif kind == "hnsw_flat":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
    else:
        raise ValueError(f"unknown index kind: {kind}")
    index.add(xb)
    return index, time.perf_counter() - t0


def index_memory_mb(index: faiss.Index) -> float:
    return faiss.serialize_index(index).nbytes / 2**20


def set_search_param(index: faiss.Index, kind: str, value: Optional[int]) -> None:
    if kind in ("ivf_flat", "ivf_pq"):
        index.nprobe = int(value)
    elif kind == "hnsw_flat":
        index.hnsw.efSearch = int(value)


def param_grid(kind: str, index: faiss.Index, k: int) -> List[Optional[int]]:
    if kind == "flat":
        return [None]
    if kind in ("ivf_flat", "ivf_pq"):
        nlist = index.nlist
        return [p for p in (2 ** i for i in range(0, 14)) if p <= min(nlist, 1024)]
    # HNSW search uses max(efSearch, k), so smaller settings repeat efSearch = k.
    values = sorted({k} | {2 ** i for i in range(4, 13) if 2 ** i > k})
    return values[:7]


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of the true top-K found in the approximate top-K."""
    k = truth.shape[1]
    hits = 0
    for row_f, row_t in zip(found, truth):
        hits += len(np.intersect1d(row_f[row_f >= 0], row_t, assume_unique=False))
    return hits / (k * len(truth))


def hit_rate_at_k(found: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean([t in row for row, t in zip(found, targets)]))


def single_query_latencies(index: faiss.Index, xq: np.ndarray, k: int, repeats: int,
                           seed: int) -> List[np.ndarray]:
    """Per-query search latency in ms at one thread, one array per repeat."""
    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(seed)
    for i in range(min(20, len(xq))):  # warmup
        index.search(xq[i:i + 1], k)
    runs = []
    for _ in range(repeats):
        order = rng.permutation(len(xq))
        lat = np.empty(len(xq))
        for j, qi in enumerate(order):
            q = xq[qi:qi + 1]
            t0 = time.perf_counter_ns()
            index.search(q, k)
            lat[j] = (time.perf_counter_ns() - t0) / 1e6
        runs.append(lat)
    return runs


def batched_qps(index: faiss.Index, xq: np.ndarray, k: int, repeats: int, threads: int) -> float:
    faiss.omp_set_num_threads(threads)
    index.search(xq[: min(64, len(xq))], k)
    qps = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        index.search(xq, k)
        qps.append(len(xq) / (time.perf_counter() - t0))
    return float(np.median(qps))


def sweep(name: str, xb: np.ndarray, xq_eval: np.ndarray, xq_lat: np.ndarray, ks: Sequence[int],
          kinds: Sequence[str], targets: Optional[np.ndarray], repeats: int, seed: int,
          log=print) -> List[Dict[str, object]]:
    threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(threads)
    kmax = max(ks)
    exact, flat_build = build_index("flat", xb)
    _, truth = exact.search(xq_eval, kmax)

    rows: List[Dict[str, object]] = []
    for kind in kinds:
        faiss.omp_set_num_threads(threads)
        if kind == "flat":
            index, build_s = exact, flat_build
        else:
            log(f"  [{name}] building {kind} over {len(xb):,} vectors ...")
            index, build_s = build_index(kind, xb, seed=seed)
        mem = index_memory_mb(index)
        for k in ks:
            for value in param_grid(kind, index, k):
                set_search_param(index, kind, value)
                faiss.omp_set_num_threads(threads)
                _, found = index.search(xq_eval, k)
                rec = recall_at_k(found, truth[:, :k])
                hit = hit_rate_at_k(found, targets[: len(found)]) if targets is not None else math.nan
                runs = single_query_latencies(index, xq_lat, k, repeats, seed)
                pooled = np.concatenate(runs)
                p99_point, p99_lo, p99_hi = bootstrap_ci(pooled, lambda v: np.percentile(v, 99),
                                                         n_boot=300, seed=seed)
                p99_gm, p99_rlo, p99_rhi = repeat_ci_log([np.percentile(r, 99) for r in runs])
                rows.append({
                    "dataset": name, "index": kind,
                    "param": {"ivf_flat": "nprobe", "ivf_pq": "nprobe", "hnsw_flat": "efSearch"}.get(kind, ""),
                    "value": value if value is not None else "", "K": k,
                    "recall_at_K": rec, "hit_rate_at_K": hit,
                    "p50_ms": float(np.percentile(pooled, 50)),
                    "p99_ms": p99_point, "p99_boot_lo": p99_lo, "p99_boot_hi": p99_hi,
                    "p99_repeat_gmean": p99_gm, "p99_repeat_lo": p99_rlo, "p99_repeat_hi": p99_rhi,
                    "batched_qps_all_threads": batched_qps(index, xq_eval, k, repeats=3, threads=threads),
                    "memory_mb": mem, "build_s": build_s,
                    "n_items": len(xb), "n_lat_queries": len(xq_lat), "repeats": repeats,
                })
                log(f"    {kind:9s} {rows[-1]['param']}={rows[-1]['value']!s:>5} K={k:<5} "
                    f"recall={rec:.3f} p99={p99_point:.3f} ms")
        if kind != "flat":
            del index
    return rows


def mark_pareto(rows: List[Dict[str, object]], latency_key: str = "p99_ms",
                quality_key: str = "recall_at_K") -> None:
    """Add a ``dominated`` flag per (dataset, K) group, in place."""
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["K"]), []).append(r)
    for members in groups.values():
        for r in members:
            r["dominated"] = any(
                o is not r
                and o[latency_key] <= r[latency_key] and o[quality_key] >= r[quality_key]
                and (o[latency_key] < r[latency_key] or o[quality_key] > r[quality_key])
                for o in members
            )


def select_operating_points(rows: List[Dict[str, object]], recall_floor: Optional[float],
                            latency_key: str = "p99_ms") -> None:
    """Flag the lowest-latency setting meeting ``recall_floor`` per (dataset, K), in place.

    Exact search is eligible: on a small catalog it can be the fastest
    setting that meets the floor, and then no approximate index is needed.
    """
    for r in rows:
        r["selected"] = False
    if recall_floor is None:
        return
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["K"]), []).append(r)
    for members in groups.values():
        ok = [r for r in members if r["recall_at_K"] >= recall_floor]
        if ok:
            min(ok, key=lambda r: r[latency_key])["selected"] = True


def index_resources(rows: List[Dict[str, object]], k: int) -> List[Dict[str, object]]:
    """One row per (dataset, index): memory, build time, and the recall range swept at ``k``.

    Compressed indexes rarely reach the latency and recall frontier; their case
    is memory, which this table shows next to the frontier.
    """
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for r in rows:
        if r["K"] != k:
            continue
        key = (r["dataset"], r["index"])
        e = out.setdefault(key, {"dataset": r["dataset"], "index": r["index"], "K": k,
                                 "memory_mb": r["memory_mb"], "build_s": r["build_s"],
                                 "min_recall": r["recall_at_K"], "max_recall": r["recall_at_K"],
                                 "min_p99_ms": r["p99_ms"]})
        e["min_recall"] = min(e["min_recall"], r["recall_at_K"])
        e["max_recall"] = max(e["max_recall"], r["recall_at_K"])
        e["min_p99_ms"] = min(e["min_p99_ms"], r["p99_ms"])
    return list(out.values())
