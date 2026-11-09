"""
Exercise 2 Part B: ranker inference levers and what they cost in quality.

1. Train the vendored Chapter 7 DCN-V2 on CPU (short Yambda window, or a
   deterministic synthetic dataset with planted feature crosses).
2. Export it to ONNX, rewrite Gemm as MatMul (identical arithmetic) so every
   dense layer reaches the quantizer as a MatMul, and produce a dynamically
   quantized int8 model.
3. Benchmark per-call latency (p50, p99 with intervals across repeats) and
   rows per second for batch sizes 1 to 1,024 at one thread and all threads:
     torch_eager_fp32            the reference
     ort_fp32_{basic,extended,all}  ONNX Runtime graph-optimization levels
     ort_int8_all                dynamic int8 with all optimizations
4. Check parity of each ONNX variant against the reference on the test set:
   scores, per-pseudo-request ranks, AUC, calibration, and slices.

Whether int8 is faster on a given processor is an outcome, not an
assumption. A variant that is faster but fails parity on any slice is
reported as rejected.

Usage:
    python -m scripts.ex2b_ranker_inference_parity --data yambda
    python -m scripts.ex2b_ranker_inference_parity --data synthetic
    python -m scripts.ex2b_ranker_inference_parity --quick
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnx
import torch

import config
from latency.report import markdown_table, write_csv, write_run_info
from ranker.bench import benchmark, ort_predictor, speedups, torch_predictor
from ranker.data import load_ranking_data
from ranker.export import export_onnx, file_sizes_mb, gemm_to_matmul, ort_predict, quantize_int8, session
from ranker.parity import Tolerances, parity_report, tolerances_dict
from ranker.train import predict, train


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["auto", "yambda", "synthetic"], default="auto",
                   help="auto uses Yambda when YAMBDA_DATA_DIR (or the sibling Dataset folder) has it")
    p.add_argument("--train-days", type=int, default=7, help="Yambda training window in days")
    p.add_argument("--epochs", type=int, default=None, help="training epochs (default 2; quick 1)")
    p.add_argument("--batch-size", type=int, default=4096, help="training batch size")
    p.add_argument("--train-threads", type=int, default=None,
                   help="PyTorch threads for training (default: half the logical CPUs, at most 8)")
    p.add_argument("--max-train-seconds", type=float, default=None,
                   help="stop training after this many seconds (default 900; quick 60)")
    p.add_argument("--batch-sizes", default=None, help="inference batch sizes (default 1,2,4,...,1024)")
    p.add_argument("--threads", default="1,all", help="thread counts to benchmark; 'all' = logical CPUs")
    p.add_argument("--repeats", type=int, default=None, help="timing repeats (default 5; quick 2)")
    p.add_argument("--max-auc-drop", type=float, default=Tolerances.max_auc_drop)
    p.add_argument("--min-top10-overlap", type=float, default=Tolerances.min_top10_overlap)
    p.add_argument("--max-abs-mean-shift", type=float, default=Tolerances.max_abs_mean_shift)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--summary-dir", default=None)
    args = p.parse_args(argv)
    q = args.quick
    args.epochs = args.epochs or (1 if q else 2)
    args.max_train_seconds = args.max_train_seconds or (60.0 if q else 900.0)
    args.batch_sizes = [int(b) for b in (args.batch_sizes or ("1,64,512" if q else
                                                               ",".join(str(2 ** i) for i in range(11)))).split(",")]
    args.repeats = args.repeats or (2 if q else 5)
    cpus = os.cpu_count() or 1
    args.threads = sorted({cpus if t.strip() == "all" else int(t) for t in args.threads.split(",")})
    args.train_threads = args.train_threads or max(1, min(8, cpus // 2))
    return args


def set_torch_threads(n: int) -> None:
    torch.set_num_threads(n)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    out = config.summary_dir(args.quick, args.summary_dir)
    big = config.output_dir("ex2b", quick=args.quick)
    torch.manual_seed(args.seed)
    tol = Tolerances(args.max_auc_drop, args.min_top10_overlap, args.max_abs_mean_shift)

    print(f"[1/5] Loading ranking data ({args.data}) ...")
    data = load_ranking_data(args.data, config.yambda_dir(), args.quick, args.seed, args.train_days)
    print(f"  {data.name}: {len(data.train.label):,} train rows, {len(data.test.label):,} test rows, "
          f"positive rate {data.train.label.mean():.3f}")

    print("[2/5] Training DCN-V2 on CPU ...")
    torch.set_num_threads(args.train_threads)
    model, train_metrics = train(data, args.epochs, args.batch_size, args.max_train_seconds, args.seed)
    print(f"  test AUC {train_metrics['test_auc']:.4f}; {train_metrics['n_params']:,} parameters")
    torch.save(model.state_dict(), big / "dcn_v2.pt")

    print("[3/5] Exporting to ONNX and quantizing ...")
    raw = export_onnx(model, data.test.dense.shape[1], big / "dcn_v2_fp32_gemm.onnx")
    fp32 = big / "dcn_v2_fp32.onnx"
    onnx.save(gemm_to_matmul(onnx.load(str(raw))), str(fp32))
    int8 = quantize_int8(fp32, big / "dcn_v2_int8.onnx")
    sizes = file_sizes_mb({"fp32_onnx": fp32, "int8_onnx": int8})
    print(f"  model files (MB): {json.dumps({k: round(v, 2) for k, v in sizes.items()})}")

    test = data.test
    sparse, dense = np.ascontiguousarray(test.sparse), np.ascontiguousarray(test.dense)
    variant_files = {"ort_fp32_basic": (fp32, "basic"), "ort_fp32_extended": (fp32, "extended"),
                     "ort_fp32_all": (fp32, "all"), "ort_int8_all": (int8, "all")}

    print("[4/5] Benchmarking inference ...")
    variants = {"torch_eager_fp32": {t: torch_predictor(model) for t in args.threads}}
    for name, (path, level) in variant_files.items():
        variants[name] = {t: ort_predictor(session(path, level, t)) for t in args.threads}
    lat_rows = benchmark(variants, sparse, dense, args.batch_sizes, args.repeats, args.quick,
                         args.seed, set_torch_threads)
    speedups(lat_rows, "torch_eager_fp32")
    write_csv(lat_rows, out / "ex2b_latency.csv")

    print("[5/5] Parity against the PyTorch reference ...")
    set_torch_threads(os.cpu_count() or 1)
    ref_scores = predict(model, sparse, dense)
    par_rows = []
    for name, (path, level) in variant_files.items():
        scores = ort_predict(session(path, level, os.cpu_count() or 1), sparse, dense)
        par_rows += parity_report(name, ref_scores, scores, test.label, test.group, test.user_history,
                                  test.item_plays, tol, seed=args.seed)
    write_csv(par_rows, out / "ex2b_parity.csv")

    write_summary(out / "ex2b_summary.md", args, data.name, train_metrics, sizes, lat_rows, par_rows, tol)
    write_run_info(out, "scripts.ex2b_ranker_inference_parity", args,
                   {"train": train_metrics, "model_files_mb": sizes, "tolerances": tolerances_dict(tol),
                    "artifacts": str(big), "runtime_s": time.perf_counter() - t_start})
    print(f"Done in {time.perf_counter() - t_start:.1f} s. Summaries in {out}")
    return 0


def write_summary(path, args, data_name, train_metrics, sizes, lat_rows, par_rows, tol) -> None:
    verdicts = []
    for name in sorted({r["variant"] for r in par_rows}):
        rows = [r for r in par_rows if r["variant"] == name]
        failed = [f"{r['slice']} ({r['failed_checks']})" for r in rows if not r["passes"]]
        mine = [r for r in lat_rows if r["variant"] == name]
        faster = [r for r in mine if r["p50_faster"]]
        slower = [r for r in mine if r["p50_slower"]]
        if failed:
            decision = "rejected (parity)"
        elif faster and not slower:
            decision = "candidate"
        elif faster:
            decision = "candidate where faster; slower elsewhere"
        else:
            decision = "no measured gain"
        verdicts.append({"variant": name, "parity": "pass" if not failed else "fail: " + "; ".join(failed),
                         "cells": len(mine), "faster_cells": len(faster), "slower_cells": len(slower),
                         "decision": decision})
    cols = ["variant", "threads", "batch", "p50_ms", "p50_lo", "p50_hi", "p99_ms", "rows_per_s",
            "p50_speedup_vs_torch", "p50_speedup_lo", "p50_speedup_hi", "p50_faster", "p50_slower"]
    pcols = ["variant", "slice", "rows", "max_abs_diff", "spearman", "top10_overlap", "top50_overlap",
             "kendall_tau", "rbo_p0.9", "auc_change", "label_rate", "mean_pred_ref", "mean_pred_shift",
             "passes", "failed_checks"]
    md = ["# Exercise 2 Part B: ranker inference and parity\n",
          f"Data: {data_name}. Training: {train_metrics['train_rows']:,} rows, test AUC "
          f"{train_metrics['test_auc']:.4f}, {train_metrics['train_seconds']:.0f} s"
          f"{' (stopped at the time budget)' if train_metrics['stopped_early'] else ''}.\n",
          f"Model files (MB): fp32 {sizes['fp32_onnx']:.2f}, int8 {sizes['int8_onnx']:.2f}.\n",
          f"Parity tolerances (set your own): AUC drop at most {tol.max_auc_drop}, top-10 overlap at least "
          f"{tol.min_top10_overlap}, absolute mean-prediction shift at most {tol.max_abs_mean_shift}.\n",
          "## Decisions\n",
          "A cell is one (threads, batch size) pair. It counts as faster or slower than PyTorch only "
          "when the paired interval of the p50 speed-up across repeats lies wholly above or below 1 "
          "(repeat r of every variant scores the same batches). Latency intervals are Student-t "
          "intervals on log latency.\n",
          markdown_table(verdicts),
          "\n## Latency by batch size\n", markdown_table(lat_rows, cols),
          "\n## Parity by slice\n",
          "`label_rate` is the observed positive rate in the slice; comparing it with `mean_pred_ref` "
          "shows the reference's own calibration, and `mean_pred_shift` is how far the variant moves "
          "the mean prediction. Top-10 overlap uses pseudo-requests with more than 10 rows.\n",
          markdown_table(par_rows, pcols, digits=4),
          "\nFull tables: `ex2b_latency.csv`, `ex2b_parity.csv`. Record your conclusions in `DESIGN_NOTES.md`.\n"]
    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
