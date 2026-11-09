"""
Exercise 1 Part B: attribute the tail of a recommendation request.

Runs the instrumented in-process pipeline (see ``pipeline/stages.py``) under
open-loop load at several fractions of its measured capacity, then:

* reports end-to-end p50, p95, and p99 per load level, from merged
  histograms, with intervals across repeated runs and a bootstrap interval;
* for requests above the target percentile, attributes critical-path time to
  stages S1 to S7 and the residual R, split into service and waiting time,
  and compares that with requests near the median;
* for each ``--lever``, reruns the same offered load with one change and
  compares the measured p99 change with the bound from the baseline's tail
  share of the spans that lever acts on;
* flags every load level at which the load generator fell behind its
  schedule (``--max-dispatch-lag-ms``), and marks lever comparisons that
  involve such a level as not valid.

Automatic garbage collection is paused during every measured window, after a
full collection, because the harness holds every trace until the window ends
(``latency.loadgen.gc_paused``). ``--gc-in-window`` leaves it on, which shows
how collection pauses of a growing heap stall the generator and the server.

The baseline flags (``--candidates``, ``--feature-cache``,
``--heavy-batching``, ``--mmr``, ``--popularity``, ``--hedge-feature-fetch``)
describe the system before any change. Each ``--lever`` names the changes to
try, for example ``--lever heavy-batching=batched``. Load levels for every
configuration are fractions of the *baseline* capacity, so all
configurations face the same offered request rate.

Usage:
    python -m scripts.ex1b_pipeline_attribution --load 0.3,0.5,0.7,0.85
    python -m scripts.ex1b_pipeline_attribution --load 0.5,0.7 \\
        --lever popularity=precomputed --lever heavy-batching=batched
    python -m scripts.ex1b_pipeline_attribution --quick
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import fields
from typing import Dict, List

import numpy as np

import config
from ann.data import load_catalog
from latency.histogram import LatencyHistogram
from latency.loadgen import (dispatch_lag_report, open_loop_schedule, run_closed_loop, run_open_loop,
                             throughput_per_s)
from latency.platform_check import GUIDANCE, run_check
from latency.report import (STAGE_LABELS, markdown_table, select_band, stage_shares, write_csv,
                            write_run_info)
from latency.timing import RESIDUAL_STAGE, bootstrap_ci, critical_path, paired_ratio_ci, repeat_ci_log
from pipeline.build_artifacts import build
from pipeline.stages import (DEFAULT_CONFIG, LEVER_SPANS, Pipeline, PipelineConfig, apply_levers,
                             configure_threads)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", default=None,
                   help="comma-separated load levels as fractions of measured capacity "
                        "(default 0.3,0.5,0.7,0.85; quick 0.3,0.7)")
    p.add_argument("--duration", type=float, default=None, help="seconds per repeat (default 15; quick 2)")
    p.add_argument("--repeats", type=int, default=None, help="independent runs per load (default 5; quick 2)")
    p.add_argument("--warmup", type=float, default=None, help="warmup seconds per configuration (default 3)")
    p.add_argument("--capacity-duration", type=float, default=None,
                   help="seconds of closed-loop saturation used to measure capacity (default 10)")
    p.add_argument("--workers", type=int, default=4, help="server worker threads")
    p.add_argument("--target-pct", type=float, default=99.0, help="percentile that defines the tail")
    p.add_argument("--attribution-load", type=float, default=None,
                   help="load level highlighted in the report (default: highest level not above 0.7)")
    p.add_argument("--process", choices=["poisson", "uniform"], default="poisson")
    p.add_argument("--max-dispatch-lag-ms", type=float, default=2.0,
                   help="flag a load level when the generator's p99 send lag exceeds this (default 2)")
    p.add_argument("--catalog", choices=["auto", "coveo", "synthetic"], default="auto")
    p.add_argument("--n-items", type=int, default=None, help="synthetic catalog size (default 200,000)")
    p.add_argument("--nprobe", type=int, default=16, help="IVF nprobe for the two-tower source")
    # Baseline configuration (plan entry-point flags).
    p.add_argument("--candidates", type=int, default=DEFAULT_CONFIG.candidates)
    p.add_argument("--feature-cache", action="store_true")
    p.add_argument("--heavy-batching", choices=["per-item", "batched"], default=DEFAULT_CONFIG.heavy_batching)
    p.add_argument("--mmr", choices=["naive", "vectorized"], default=DEFAULT_CONFIG.mmr)
    p.add_argument("--popularity", choices=["naive", "precomputed"], default=DEFAULT_CONFIG.popularity)
    p.add_argument("--hedge-feature-fetch", action="store_true")
    p.add_argument("--lever", action="append", default=[],
                   help="changes to compare against the baseline, e.g. 'mmr=vectorized' "
                        "(repeatable; comma-separate several changes in one lever)")
    p.add_argument("--gc-in-window", action="store_true",
                   help="leave automatic garbage collection on during measured windows")
    p.add_argument("--no-traces", action="store_true", help="do not write span-level Parquet traces")
    p.add_argument("--force", action="store_true", help="run even if timer granularity is too coarse")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--quick", action="store_true", help="small sizes for a smoke test")
    p.add_argument("--summary-dir", default=None)
    args = p.parse_args(argv)
    q = args.quick
    args.load = [float(x) for x in (args.load or ("0.3,0.7" if q else "0.3,0.5,0.7,0.85")).split(",")]
    args.duration = args.duration or (2.0 if q else 15.0)
    args.repeats = args.repeats or (2 if q else 5)
    args.warmup = args.warmup if args.warmup is not None else (0.5 if q else 3.0)
    args.capacity_duration = args.capacity_duration or (2.0 if q else 10.0)
    args.n_items = args.n_items or (20_000 if q else 200_000)
    if args.attribution_load is None:
        below = [x for x in args.load if x <= 0.7]
        args.attribution_load = max(below) if below else args.load[0]
    return args


# ----------------------------------------------------------- per request
def analyse(trace) -> Dict[str, object]:
    """Critical-path breakdowns of one trace."""
    spans = trace.spans
    by_id = {s.span_id: s for s in spans}
    root = trace.root
    stage_kind: Dict[tuple, int] = defaultdict(int)
    span_ns: Dict[str, int] = defaultdict(int)
    subtree_ns: Dict[str, int] = defaultdict(int)
    for span, ns in critical_path(spans, root):
        if ns <= 0:
            continue
        if span is root:
            stage_kind[(RESIDUAL_STAGE, "residual")] += ns
            span_ns["(residual)"] += ns
            continue
        stage_kind[(span.stage, span.kind)] += ns
        span_ns[span.name] += ns
        node = span
        while node is not None and node is not root:
            subtree_ns[node.name] += ns
            node = by_id.get(node.parent_id)
    durations = {s.name: s.duration_ns for s in spans if s.parent_id == root.span_id}
    for s in spans:
        if s.name in ("two_tower", "popularity"):
            durations[s.name] = s.duration_ns
    return {"e2e_ns": root.end_ns - root.start_ns, "stage_kind": dict(stage_kind),
            "span_ns": dict(span_ns), "subtree_ns": dict(subtree_ns), "durations": durations}


def trace_rows(trace, cfg_idx: int, load: float, rep: int) -> List[dict]:
    t0 = trace.root.start_ns
    return [{"config": cfg_idx, "load": load, "repeat": rep, "request": trace.request_id,
             "span_id": s.span_id, "parent_id": s.parent_id if s.parent_id is not None else -1,
             "name": s.name, "stage": s.stage, "kind": s.kind,
             "start_us": (s.start_ns - t0) / 1e3, "duration_us": s.duration_ns / 1e3} for s in trace.spans]


# -------------------------------------------------------------- running
def run_config(pipe: Pipeline, cfg_idx: int, args, rates: Dict[float, float], trace_dir,
               log=print) -> Dict[float, dict]:
    if args.warmup > 0:
        mid = rates[args.load[len(args.load) // 2]]
        run_open_loop(pipe.handle, open_loop_schedule(mid, args.warmup, seed=args.seed + 999), args.workers,
                      pause_gc=not args.gc_in_window)
    results: Dict[float, dict] = {}
    for li, load in enumerate(args.load):
        rate = rates[load]
        per_rep_p99, per_rep_rate, analyses, lags, errors = [], [], [], [], 0
        merged = LatencyHistogram()
        hedged = chunks = hits = cands = 0
        for rep in range(args.repeats):
            offsets = open_loop_schedule(rate, args.duration, seed=args.seed * 100 + li * 10 + rep,
                                         process=args.process)
            recs = run_open_loop(pipe.handle, offsets, args.workers, pause_gc=not args.gc_in_window)
            ok = [r for r in recs if r.error is None]
            errors += len(recs) - len(ok)
            if len(ok) < len(recs):
                log(f"      first error: {next(r.error for r in recs if r.error)}")
            if trace_dir is not None:
                import polars as pl
                rows = [row for r in ok for row in trace_rows(r.result["trace"], cfg_idx, load, rep)]
                pl.DataFrame(rows).write_parquet(trace_dir / f"config{cfg_idx}_load{load:g}_rep{rep}.parquet")
            lat = np.array([r.latency_ms for r in ok])
            hist = LatencyHistogram()
            hist.record_many(lat)
            merged.merge(hist)
            per_rep_p99.append(hist.percentile(args.target_pct))
            per_rep_rate.append(throughput_per_s(ok))
            lags.extend(r.dispatch_lag_ms for r in ok)
            for r in ok:
                analyses.append(analyse(r.result["trace"]))
                hedged += r.result["hedged"]
                chunks += r.result["chunks"]
                hits += r.result["cache_hits"]
                cands += r.result["candidates"]
        e2e = np.array([a["e2e_ns"] for a in analyses], dtype=np.float64)
        lag = dispatch_lag_report(lags, args.max_dispatch_lag_ms)
        results[load] = {"hist": merged, "per_rep_p99": per_rep_p99, "per_rep_rate": per_rep_rate,
                         "analyses": analyses, "e2e": e2e, "lag": lag, "errors": errors,
                         "rate": rate, "hedge_rate": hedged / max(chunks, 1),
                         "cache_hit_rate": hits / max(cands, 1)}
        log(f"    load {load:.2f} ({rate:.1f} req/s): p50={merged.percentile(50):.2f} "
            f"p99={merged.percentile(args.target_pct):.2f} ms over {merged.total} requests; "
            f"send lag p99 {lag['dispatch_lag_p99_ms']:.2f} ms")
        if lag["generator_behind"]:
            log(f"    WARNING: the load generator fell behind its schedule at load {load:g} "
                f"(send lag p99 {lag['dispatch_lag_p99_ms']:.2f} ms > {args.max_dispatch_lag_ms:g} ms). "
                "This level's latencies include the generator's own delay; it is excluded from "
                "lever comparisons.")
    return results


def band_rows(res: dict, idx: np.ndarray, label: str, cfg_name: str, load: float) -> List[dict]:
    sel = [res["analyses"][i] for i in idx]
    rows = stage_shares([a["stage_kind"] for a in sel], [a["e2e_ns"] for a in sel])
    for r in rows:
        r.update({"config": cfg_name, "load": load, "band": label, "requests": len(sel)})
    return rows


def span_rows(res: dict, idx: np.ndarray, label: str, cfg_name: str, load: float) -> List[dict]:
    sel = [res["analyses"][i] for i in idx]
    total = sum(a["e2e_ns"] for a in sel) or 1
    names = sorted({n for a in sel for n in a["span_ns"]})
    rows = []
    for n in names:
        vals = [a["span_ns"].get(n, 0) for a in sel]
        rows.append({"config": cfg_name, "load": load, "band": label, "span": n,
                     "mean_critical_ms": float(np.mean(vals)) / 1e6,
                     "share": float(np.sum(vals)) / total,
                     "critical_frequency": float(np.mean([v > 0 for v in vals]))})
    return sorted(rows, key=lambda r: -r["share"])


def changed_fields(cfg: PipelineConfig, base: PipelineConfig) -> List[str]:
    return [f.name for f in fields(cfg) if getattr(cfg, f.name) != getattr(base, f.name)]


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    out = config.summary_dir(args.quick, args.summary_dir)
    big = config.output_dir("ex1b", quick=args.quick)

    platform = run_check()
    print(f"Platform: {platform['platform']}; 1 ms sleep median overshoot "
          f"{platform['sleep_1ms']['median_overshoot_ms']:.2f} ms")
    if not platform["timer_ok"] and not args.force:
        print(GUIDANCE.format(limit=platform["max_granularity_ms"],
                              got=platform["sleep_1ms"]["median_overshoot_ms"]))
        return 2

    configure_threads()
    print(f"[1/4] Loading catalog ({args.catalog}) and building artifacts ...")
    catalog = load_catalog(args.catalog, config.sigir_dir(), args.quick, args.seed, args.n_items)
    print(f"  catalog: {catalog.name}, {catalog.n_items:,} items, {len(catalog.histories):,} request histories")
    art = build(catalog, args.seed, args.quick, nprobe=args.nprobe)
    configure_threads()

    base = PipelineConfig(candidates=args.candidates, feature_cache=args.feature_cache,
                          heavy_batching=args.heavy_batching, mmr=args.mmr, popularity=args.popularity,
                          hedge_feature_fetch=args.hedge_feature_fetch, seed=args.seed)
    configs = [base] + [apply_levers(base, spec) for spec in args.lever]
    names = ["baseline"] + [f"lever: {spec}" for spec in args.lever]

    print(f"[2/4] Measuring baseline capacity (closed loop, {args.workers} clients, "
          f"{args.capacity_duration:.0f} s) ...")
    pipe = Pipeline(art, base)
    run_closed_loop(pipe.handle, args.workers, min(1.0, args.capacity_duration / 2),
                    pause_gc=not args.gc_in_window)
    cap_records = run_closed_loop(pipe.handle, args.workers, args.capacity_duration,
                                  pause_gc=not args.gc_in_window)
    capacity = throughput_per_s(cap_records)
    rates = {load: load * capacity for load in args.load}
    print(f"  capacity: {capacity:.1f} requests/s")

    print("[3/4] Open-loop runs")
    trace_dir = None
    if not args.no_traces:
        trace_dir = big / "traces" / time.strftime("%Y%m%d_%H%M%S")
        trace_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ci, (cfg, name) in enumerate(zip(configs, names)):
        print(f"  configuration {ci}: {name}")
        if ci > 0:
            pipe.close()
            pipe = Pipeline(art, cfg)
        results.append(run_config(pipe, ci, args, rates, trace_dir))
    pipe.close()

    print("[4/4] Writing reports")
    curve, stages, spans, levers = [], [], [], []
    for ci, name in enumerate(names):
        for load, res in results[ci].items():
            h = res["hist"]
            p99_gm, p99_lo, p99_hi = repeat_ci_log(res["per_rep_p99"])
            _, b_lo, b_hi = bootstrap_ci(res["e2e"] / 1e6, lambda v: np.percentile(v, args.target_pct),
                                         n_boot=300, seed=args.seed)
            curve.append({
                "config": name, "load": load, "offered_rps": res["rate"],
                "achieved_rps": float(np.mean(res["per_rep_rate"])), "requests": h.total,
                "p50_ms": h.percentile(50), "p95_ms": h.percentile(95),
                f"p{args.target_pct:g}_ms": h.percentile(args.target_pct),
                f"p{args.target_pct:g}_repeat_gmean": p99_gm,
                f"p{args.target_pct:g}_repeat_lo": p99_lo, f"p{args.target_pct:g}_repeat_hi": p99_hi,
                f"p{args.target_pct:g}_boot_lo": b_lo, f"p{args.target_pct:g}_boot_hi": b_hi,
                "max_ms": h.max_ms,
                **res["lag"],
                "hedge_rate": res["hedge_rate"], "cache_hit_rate": res["cache_hit_rate"],
                "errors": res["errors"],
            })
            tail = select_band(res["e2e"], args.target_pct)
            median = select_band(res["e2e"], 40, 60)
            stages += band_rows(res, tail, f"above p{args.target_pct:g}", name, load)
            stages += band_rows(res, median, "p40 to p60", name, load)
            spans += span_rows(res, tail, f"above p{args.target_pct:g}", name, load)
            spans += span_rows(res, median, "p40 to p60", name, load)

    for ci in range(1, len(configs)):
        touched = changed_fields(configs[ci], base)
        span_names = sorted({s for f in touched for s in LEVER_SPANS.get(f, ())})
        for load in args.load:
            b, l = results[0][load], results[ci][load]
            tail = select_band(b["e2e"], args.target_pct)
            bound = float(np.mean([sum(b["analyses"][i]["subtree_ns"].get(s, 0) for s in span_names)
                                   for i in tail])) / 1e6
            bm, blo, bhi = repeat_ci_log(b["per_rep_p99"])
            lm, llo, lhi = repeat_ci_log(l["per_rep_p99"])
            # Repeat r of every configuration replays the same arrival schedule and
            # simulated delays, so the runs are paired.
            ratio, rlo, rhi = paired_ratio_ci(l["per_rep_p99"], b["per_rep_p99"])
            detected = bool(not math.isnan(rlo) and (rhi < 1.0 or rlo > 1.0))
            valid = not (b["lag"]["generator_behind"] or l["lag"]["generator_behind"])

            def own(res):
                return float(np.mean([sum(a["durations"].get(s, 0) for s in span_names)
                                      for a in res["analyses"]])) / 1e6
            levers.append({
                "lever": names[ci], "load": load, "spans": "+".join(span_names),
                "baseline_stage_mean_ms": own(b), "lever_stage_mean_ms": own(l),
                "bound_ms": bound,
                "baseline_p99_ms": bm, "baseline_p99_lo": blo, "baseline_p99_hi": bhi,
                "lever_p99_ms": lm, "lever_p99_lo": llo, "lever_p99_hi": lhi,
                "measured_p99_change_ms": lm - bm,
                "p99_ratio": ratio, "p99_ratio_lo": rlo, "p99_ratio_hi": rhi,
                "change_detected": detected,
                "gain_exceeds_bound": bool(detected and rhi < 1.0 and bm - lm > bound),
                "intervals_overlap": not (math.isnan(blo) or math.isnan(llo)) and (llo <= bhi and blo <= lhi),
                "lever_hedge_rate": l["hedge_rate"], "lever_cache_hit_rate": l["cache_hit_rate"],
                "valid": valid,
            })

    write_csv(curve, out / "ex1b_load_curve.csv")
    write_csv(stages, out / "ex1b_stage_shares.csv")
    write_csv(spans, out / "ex1b_span_shares.csv")
    write_csv(levers, out / "ex1b_levers.csv")

    write_markdown(out / "ex1b_attribution.md", args, platform, capacity, catalog.name, configs, names,
                   curve, stages, spans, levers, trace_dir)
    write_run_info(out, "scripts.ex1b_pipeline_attribution", args,
                   {"capacity_rps": capacity, "configs": [c.label() for c in configs],
                    "traces": str(trace_dir) if trace_dir else None,
                    "runtime_s": time.perf_counter() - t_start}, platform_report=platform)
    print(f"Done in {time.perf_counter() - t_start:.1f} s. Report: {out / 'ex1b_attribution.md'}")
    return 0


def write_markdown(path, args, platform, capacity, catalog_name, configs, names, curve, stages,
                   spans, levers, trace_path) -> None:
    pct = f"p{args.target_pct:g}"
    focus = args.attribution_load
    lines = [
        "# Exercise 1 Part B: tail attribution report\n",
        f"Catalog: {catalog_name}. Workers: {args.workers}. Arrival process: {args.process}. "
        f"Repeats per load: {args.repeats} x {args.duration:g} s. Tail: requests above {pct}.\n",
        f"Platform: {platform['platform']}, {platform['cpu'].get('logical_cpus')} logical CPUs; "
        f"1 ms sleep median overshoot {platform['sleep_1ms']['median_overshoot_ms']:.2f} ms.\n",
        f"Measured baseline capacity (closed loop): {capacity:.1f} requests/s. "
        "Every configuration below faces the same offered rates.\n",
        "## Configurations\n",
    ]
    lines += [f"* {n}: `{c.label()}`" for n, c in zip(names, configs)]
    behind = [r for r in curve if r["generator_behind"]]
    lines += ["\n## Latency against load\n",
              markdown_table(curve, ["config", "load", "offered_rps", "achieved_rps", "p50_ms", "p95_ms",
                                     f"{pct}_ms", f"{pct}_repeat_lo", f"{pct}_repeat_hi",
                                     "dispatch_lag_p99_ms", "generator_behind", "errors"]),
              f"\n`{pct}_repeat_lo` and `{pct}_repeat_hi` bound the {pct} across {args.repeats} independent "
              "repeats (Student-t interval on log latency, so it is asymmetric and never below zero).\n",
              "\n`dispatch_lag_p99_ms` is how late the load generator sent requests. Latency is timed from "
              "the intended send time, so the lateness is counted (as `generator_lag` in S1) instead of "
              "hidden, but it is the generator's delay, not the server's, and the late requests reach the "
              f"server in a burst. A level is flagged `generator_behind` when the lag p99 exceeds "
              f"{args.max_dispatch_lag_ms:g} ms; its latencies overstate the server's and it is excluded "
              "from lever comparisons.\n",
              ("Automatic garbage collection was left on during measured windows (`--gc-in-window`).\n"
               if args.gc_in_window else
               "Automatic garbage collection was paused during each measured window, after a full "
               "collection, because the harness holds every trace until the window ends.\n")]
    if behind:
        lines.append("**The generator fell behind at: " + "; ".join(
            f"{r['config']} at load {r['load']:g} (lag p99 {r['dispatch_lag_p99_ms']:.1f} ms)" for r in behind)
            + ".** Re-run on an otherwise idle machine before using these levels.\n")
    lines += [f"\n## Where the time goes at load {focus:g} (baseline)\n",
              "Critical-path milliseconds per stage, split into service and waiting time, for tail "
              "requests and for requests near the median. R is end-to-end time no span covers; "
              "an S1 row of kind `generator_lag` is time the load generator sent late.\n"]
    sel = [r for r in stages if r["config"] == "baseline" and r["load"] == focus]
    lines.append(markdown_table(sel, ["band", "stage", "stage_name", "kind", "mean_ms", "share"]))
    lines.append(f"\n### Spans on the critical path of tail requests (baseline, load {focus:g})\n")
    sp = [r for r in spans if r["config"] == "baseline" and r["load"] == focus and r["band"].startswith("above")]
    lines.append(markdown_table(sp[:15], ["span", "mean_critical_ms", "share", "critical_frequency"]))
    lines.append("\n`critical_frequency` is the share of tail requests in which the span was on the "
                 "critical path. A parallel branch with a frequency near zero cannot move the tail.\n")
    if levers:
        lines += ["\n## Levers: bound versus measured change\n",
                  "`bound_ms` is the baseline's mean tail critical-path time inside the spans the lever "
                  "acts on: the most the lever could remove directly from a tail request's own critical "
                  "path. Write it down before reading the measured change. Under load a lever that "
                  "shortens a stage holding a worker also shortens the queue (S1) for the requests behind, "
                  "so the measured gain can exceed the bound; `gain_exceeds_bound` marks rows where a "
                  "detected reduction is larger than the bound.\n",
                  f"`baseline_p99_ms` and `lever_p99_ms` are geometric means of the per-repeat {pct}, so they "
                  "differ slightly from the pooled values in the latency table. "
                  f"`p99_ratio` is the lever's {pct} divided by the baseline's, with a paired interval "
                  "across repeats (repeat r of every configuration replays the same arrivals). "
                  "`change_detected` is true when that interval excludes 1. `lever_hedge_rate` is the share "
                  "of feature-store calls that sent a backup, which is the extra load hedging costs. Rows "
                  "with `valid` false involve a level at which the generator fell behind.\n",
                  markdown_table(levers, ["lever", "load", "spans", "baseline_stage_mean_ms",
                                          "lever_stage_mean_ms", "bound_ms", "baseline_p99_ms",
                                          "lever_p99_ms", "measured_p99_change_ms", "p99_ratio",
                                          "p99_ratio_lo", "p99_ratio_hi", "change_detected",
                                          "gain_exceeds_bound", "lever_hedge_rate", "valid"])]
    lines.append("\nFull tables: `ex1b_load_curve.csv`, `ex1b_stage_shares.csv`, `ex1b_span_shares.csv`, "
                 "`ex1b_levers.csv`.")
    if trace_path:
        lines.append("Span-level traces (one Parquet file per configuration, load, and repeat) were "
                     "written to the large-output folder, outside the code folder.")
    lines.append("\nStage key: " + "; ".join(f"{k} {v}" for k, v in STAGE_LABELS.items()) + ".\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
