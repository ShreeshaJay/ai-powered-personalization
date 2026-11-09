"""
Exercise 2 Part A: choose an approximate-search operating point.

Builds exact, IVF-Flat, HNSW-Flat, and (optionally) IVF-PQ indexes over

* the Coveo SIGIR description vectors, with session queries whose next
  viewed item gives a downstream hit rate, and
* a synthetic clustered catalog (default one million vectors) for scale,

sweeps the search-time parameter (nprobe or efSearch) for candidate pool
sizes K, and writes recall@K, hit rate@K, single-query p50 and p99 at one
thread, batched throughput, memory, and build time per setting. Dominated
settings are flagged. With ``--recall-floor`` the script also flags the
lowest-p99 setting that meets the floor; there is deliberately no default
floor, because the right one comes from downstream quality.

Usage:
    python -m scripts.ex2a_ann_operating_points --recall-floor 0.95
    python -m scripts.ex2a_ann_operating_points --datasets synthetic --n-items 300000
    python -m scripts.ex2a_ann_operating_points --quick
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import config
from ann.data import load_catalog
from ann.sweep import index_resources, mark_pareto, select_operating_points, sweep
from latency.report import markdown_table, write_csv, write_run_info


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recall-floor", type=float, default=None,
                   help="flag the lowest-latency setting with recall@K at or above this value")
    p.add_argument("--datasets", default="auto",
                   help="comma-separated: coveo, synthetic; 'auto' uses both when Coveo is available")
    p.add_argument("--n-items", type=int, default=None, help="synthetic catalog size (default 1,000,000)")
    p.add_argument("--ks", default=None, help="candidate pool sizes (default 100,500,1000)")
    p.add_argument("--indexes", default=None,
                   help="comma-separated: flat, ivf_flat, hnsw_flat, ivf_pq (default all four)")
    p.add_argument("--n-eval-queries", type=int, default=None, help="queries for recall (default 2,000)")
    p.add_argument("--n-lat-queries", type=int, default=None, help="queries timed per repeat (default 200)")
    p.add_argument("--repeats", type=int, default=None, help="timing repeats (default 5)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--summary-dir", default=None)
    args = p.parse_args(argv)
    q = args.quick
    args.ks = [int(k) for k in (args.ks or ("100" if q else "100,500,1000")).split(",")]
    args.indexes = (args.indexes or ("flat,ivf_flat,hnsw_flat" if q else "flat,ivf_flat,hnsw_flat,ivf_pq")).split(",")
    args.n_eval_queries = args.n_eval_queries or (300 if q else 2_000)
    args.n_lat_queries = args.n_lat_queries or (50 if q else 200)
    args.repeats = args.repeats or (2 if q else 5)
    if args.datasets == "auto":
        args.datasets = ["coveo", "synthetic"] if config.sigir_dir() else ["synthetic"]
    else:
        args.datasets = args.datasets.split(",")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    out = config.summary_dir(args.quick, args.summary_dir)
    rows, notes = [], []
    for name in args.datasets:
        t0 = time.perf_counter()
        print(f"[{name}] loading ...")
        try:
            cat = load_catalog(name, config.sigir_dir(), args.quick, args.seed, args.n_items)
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
            notes.append(f"{name}: skipped ({exc})")
            continue
        rng = np.random.default_rng(args.seed)
        n_eval = min(args.n_eval_queries, len(cat.queries))
        eval_idx = rng.choice(len(cat.queries), n_eval, replace=False)
        lat_idx = eval_idx[: min(args.n_lat_queries, n_eval)]
        targets = cat.targets[eval_idx] if cat.targets is not None else None
        print(f"  {cat.n_items:,} vectors x {cat.dim} dims; {n_eval:,} evaluation queries")
        rows += sweep(cat.name, cat.vectors, cat.queries[eval_idx], cat.queries[lat_idx], args.ks,
                      args.indexes, targets, args.repeats, args.seed)
        notes.append(f"{name}: {cat.n_items:,} items, {cat.dim} dims, {n_eval:,} evaluation queries, "
                     f"{time.perf_counter() - t0:.0f} s")

    mark_pareto(rows)
    select_operating_points(rows, args.recall_floor)
    write_csv(rows, out / "ex2a_pareto.csv")

    cols = ["dataset", "index", "param", "value", "K", "recall_at_K", "hit_rate_at_K", "p50_ms", "p99_ms",
            "p99_repeat_lo", "p99_repeat_hi", "batched_qps_all_threads", "memory_mb", "build_s"]
    md = ["# Exercise 2 Part A: approximate-search operating points\n",
          "Single-query latency is measured at one thread; throughput uses all threads. Recall is "
          "against exact inner-product search. Hit rate is next-item hit rate on Coveo sessions. "
          "`p99_repeat_lo` and `p99_repeat_hi` are a Student-t interval on log latency across "
          f"{args.repeats} repeats.\n",
          "Datasets: " + "; ".join(notes) + "\n",
          f"Recall floor: {args.recall_floor if args.recall_floor is not None else 'not set'}\n",
          "## Non-dominated settings\n",
          markdown_table([r for r in rows if not r["dominated"]], cols)]
    chosen = [r for r in rows if r.get("selected")]
    if chosen:
        md += ["\n## Selected operating points (lowest p99 meeting the recall floor)\n",
               "Exact search (`flat`) is eligible; when it is selected, no approximate index is needed "
               "at that K.\n",
               markdown_table(chosen, cols)]
    md += [f"\n## Memory and build time per index (K = {args.ks[0]})\n",
           "Serialized index size, build time, and the recall and best p99 reached across the sweep. "
           "A compressed index can be dominated on latency and recall and still be the choice when "
           "memory is the constraint.\n",
           markdown_table(index_resources(rows, args.ks[0]),
                          ["dataset", "index", "memory_mb", "build_s", "min_recall", "max_recall", "min_p99_ms"])]
    md.append("\nAll settings, including dominated ones, are in `ex2a_pareto.csv`.\n")
    (out / "ex2a_summary.md").write_text("\n".join(md), encoding="utf-8")
    write_run_info(out, "scripts.ex2a_ann_operating_points", args,
                   {"runtime_s": time.perf_counter() - t_start})
    print(f"Done in {time.perf_counter() - t_start:.1f} s. Summaries in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
