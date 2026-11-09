"""
Run every Chapter 14 exercise in sequence, each in its own process.

Separate processes keep thread-pool settings (PyTorch, FAISS, ONNX Runtime)
from leaking between exercises.

Usage:
    python -m scripts.run_all --quick                 # smoke test, a few minutes
    python -m scripts.run_all                         # full run (see CLOUD_RUN.md)
    python -m scripts.run_all --data synthetic        # no licensed datasets needed
    python -m scripts.run_all --only ex1a,ex2a
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import config

FULL_LEVERS = ["popularity=precomputed", "hedge-feature-fetch=on", "feature-cache=on", "candidates=200"]
QUICK_LEVERS = ["popularity=precomputed", "hedge-feature-fetch=on"]


def commands(args) -> dict:
    synth = args.data == "synthetic"
    quick = ["--quick"] if args.quick else []
    ex1b = ["--catalog", "synthetic" if synth else "auto"]
    for lever in (QUICK_LEVERS if args.quick else FULL_LEVERS):
        ex1b += ["--lever", lever]
    if args.force_timer:
        ex1b.append("--force")
    ex2a = ["--datasets", "synthetic" if synth else "auto"]
    floor = args.recall_floor if args.recall_floor is not None else (0.9 if args.quick else None)
    if floor is not None:
        ex2a += ["--recall-floor", str(floor)]
    return {
        "ex1a": ["scripts.ex1a_queueing_sim"] + quick,
        "ex1b": ["scripts.ex1b_pipeline_attribution"] + quick + ex1b,
        "ex2a": ["scripts.ex2a_ann_operating_points"] + quick + ex2a,
        "ex2b": ["scripts.ex2b_ranker_inference_parity"] + quick + ["--data", "synthetic" if synth else "auto"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="reduced sizes for a smoke test")
    p.add_argument("--data", choices=["auto", "synthetic"], default="auto",
                   help="'synthetic' never reads Coveo or Yambda files")
    p.add_argument("--only", default=None, help="comma-separated subset of ex1a, ex1b, ex2a, ex2b")
    p.add_argument("--recall-floor", type=float, default=None, help="passed to ex2a (none by default)")
    p.add_argument("--force-timer", action="store_true", help="pass --force to ex1b")
    args = p.parse_args(argv)

    cmds = commands(args)
    selected = args.only.split(",") if args.only else list(cmds)
    status = {}
    t_all = time.perf_counter()
    for name in selected:
        cmd = [sys.executable, "-m"] + cmds[name]
        print(f"\n=== {name}: {' '.join(cmds[name])}", flush=True)
        t0 = time.perf_counter()
        code = subprocess.call(cmd, cwd=str(config.CODE_DIR))
        status[name] = {"command": " ".join(["python", "-m"] + cmds[name]), "exit_code": code,
                        "seconds": round(time.perf_counter() - t0, 1)}
        print(f"=== {name}: exit {code} after {status[name]['seconds']} s", flush=True)
    status["total_seconds"] = round(time.perf_counter() - t_all, 1)
    out = config.summary_dir(args.quick)
    (out / "run_all_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nTotal {status['total_seconds']} s. Status written to {out / 'run_all_status.json'}")
    return 0 if all(v["exit_code"] == 0 for k, v in status.items() if k != "total_seconds") else 1


if __name__ == "__main__":
    raise SystemExit(main())
