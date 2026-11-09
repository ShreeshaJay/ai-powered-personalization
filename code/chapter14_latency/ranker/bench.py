"""
Per-call latency and throughput of ranker variants across batch sizes.

Each (variant, threads, batch size) cell is warmed up, then timed for
``n_calls`` calls per repeat over several independent repeats. A call scores
one contiguous slice of real test rows, so inputs vary from call to call.
Repeat r of every variant scores the same slices, so speed-ups can be
computed from paired repeats (``speedups``).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from latency.timing import paired_ratio_ci, repeat_ci_log
from ranker.train import sparse_dict

Predictor = Callable[[np.ndarray, np.ndarray], object]


def torch_predictor(model) -> Predictor:
    def run(sparse: np.ndarray, dense: np.ndarray):
        with torch.inference_mode():
            return model(sparse_dict(sparse), torch.from_numpy(dense))
    return run


def ort_predictor(sess) -> Predictor:
    def run(sparse: np.ndarray, dense: np.ndarray):
        return sess.run(None, {"sparse": sparse, "dense": dense})
    return run


def time_calls(fn: Predictor, sparse: np.ndarray, dense: np.ndarray, batch: int, n_calls: int,
               repeats: int, warmup: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(sparse)
    if batch > n:
        raise ValueError(f"batch {batch} exceeds the {n} available rows")
    for _ in range(warmup):
        lo = int(rng.integers(0, n - batch + 1))
        fn(sparse[lo:lo + batch], dense[lo:lo + batch])
    runs = []
    for _ in range(repeats):
        offsets = rng.integers(0, n - batch + 1, n_calls)
        slices = [(np.ascontiguousarray(sparse[o:o + batch]), np.ascontiguousarray(dense[o:o + batch]))
                  for o in offsets]
        lat = np.empty(n_calls)
        for j, (s, d) in enumerate(slices):
            t0 = time.perf_counter_ns()
            fn(s, d)
            lat[j] = (time.perf_counter_ns() - t0) / 1e6
        runs.append(lat)
    return runs


def calls_for(batch: int, quick: bool) -> int:
    if quick:
        return max(10, min(60, 4_000 // batch))
    return max(40, min(400, 40_000 // batch))


def benchmark(variants: Dict[str, Dict[int, Predictor]], sparse: np.ndarray, dense: np.ndarray,
              batch_sizes: Sequence[int], repeats: int, quick: bool, seed: int,
              set_torch_threads: Callable[[int], None], log=print) -> List[Dict[str, object]]:
    """``variants[name][threads]`` is a predictor configured for that thread count."""
    rows = []
    for name, by_threads in variants.items():
        for threads, fn in by_threads.items():
            if name.startswith("torch"):
                set_torch_threads(threads)
            for b in batch_sizes:
                runs = time_calls(fn, sparse, dense, b, calls_for(b, quick), repeats,
                                  warmup=5 if quick else 20, seed=seed + b)
                pooled = np.concatenate(runs)
                p50 = [float(np.percentile(r, 50)) for r in runs]
                p99 = [float(np.percentile(r, 99)) for r in runs]
                m50, lo50, hi50 = repeat_ci_log(p50)
                m99, lo99, hi99 = repeat_ci_log(p99)
                rows.append({
                    "variant": name, "threads": threads, "batch": b, "calls": len(pooled),
                    "p50_ms": m50, "p50_lo": lo50, "p50_hi": hi50,
                    "p99_ms": m99, "p99_lo": lo99, "p99_hi": hi99,
                    "rows_per_s": b / (float(np.mean(pooled)) / 1000.0),
                    "_p50_runs": p50, "_p99_runs": p99,
                })
            log(f"  {name:22s} threads={threads:<3d} p50 at batch {batch_sizes[0]}: "
                f"{rows[-len(batch_sizes)]['p50_ms']:.3f} ms, at batch {batch_sizes[-1]}: {rows[-1]['p50_ms']:.3f} ms")
    return rows


def speedups(rows: List[Dict[str, object]], reference: str) -> None:
    """Add paired speed-up ratios against ``reference`` at the same threads and batch, in place.

    ``p50_speedup_vs_torch`` above 1 means the variant is faster. A cell counts
    as faster (or slower) only when the paired interval lies wholly above (or
    below) 1. The per-repeat lists are removed afterwards.
    """
    ref = {(r["threads"], r["batch"]): r for r in rows if r["variant"] == reference}
    for r in rows:
        base = ref[(r["threads"], r["batch"])]
        for q in ("p50", "p99"):
            s, lo, hi = paired_ratio_ci(base[f"_{q}_runs"], r[f"_{q}_runs"])
            r[f"{q}_speedup_vs_torch"], r[f"{q}_speedup_lo"], r[f"{q}_speedup_hi"] = s, lo, hi
        lo, hi = r["p50_speedup_lo"], r["p50_speedup_hi"]
        r["p50_faster"] = bool(np.isfinite(lo) and lo > 1.0)
        r["p50_slower"] = bool(np.isfinite(hi) and hi < 1.0)
    for r in rows:
        r.pop("_p50_runs", None)
        r.pop("_p99_runs", None)
