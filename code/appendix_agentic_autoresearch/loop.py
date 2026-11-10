"""Run the pre-registered arms and append every trial to the research ledger.

Scoring uses the validation labels. This script does not open the seal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

from harness.contract import (
    ROOT,
    apply_runtime_clip,
    artifacts_dir,
    bootstrap_replicates,
    load_contract,
    trial_budget,
)
from harness.evaluate import join_and_score
from harness.leakage import (
    check_coverage,
    check_feature_columns,
    check_metadata,
)
from harness.ledger import append_research_record, public_error
from harness.noise import build_noise_band, user_bootstrap_std
from harness.train import train_trial
from proposers.base import build_proposer, trial_seed


def run_search(
    run_dir: Path,
    *,
    synthetic: bool,
    in_process: bool | None = None,
    replay: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = load_contract()
    manifest = json.loads(
        (artifacts_dir(run_dir) / "split_manifest.json").read_text(encoding="utf-8")
    )
    mode = manifest["mode"]
    if synthetic and mode != "synthetic":
        raise RuntimeError("This harness was not built with --synthetic")
    if not synthetic and mode != "yambda":
        raise RuntimeError("Pass --synthetic when the harness is synthetic")
    if in_process is None:
        in_process = synthetic
    budget = trial_budget(contract, synthetic)
    records: list[dict[str, Any]] = []
    baseline_records = _run_baseline_seeds(
        run_dir, contract, synthetic=synthetic, in_process=in_process, records=records,
    )
    _write_noise_band(run_dir, contract, baseline_records, synthetic=synthetic)
    arms = ["random", "greedy"]
    if synthetic and contract["synthetic"].get("include_replay"):
        arms.append("replay")
    elif replay:
        arms.append("replay")
    for arm in arms:
        _run_arm(
            run_dir,
            contract,
            arm=arm,
            budget=budget,
            synthetic=synthetic,
            in_process=in_process,
            records=records,
        )
    winners = _winners(records)
    (run_dir / "winners.json").write_text(
        json.dumps(winners, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    summary = {
        "trials": len(records),
        "completed": sum(row["status"] == "completed" for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "timed_out": sum(row["status"] == "timed_out" for row in records),
        "seconds": round(elapsed, 3),
        "winners": winners,
    }
    (run_dir / "loop_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Loop finished in {elapsed:.1f}s: "
        f"{summary['completed']} completed, {summary['failed']} failed, "
        f"{summary['timed_out']} timed out"
    )
    return summary


def _run_baseline_seeds(run_dir, contract, *, synthetic, in_process, records) -> list[dict[str, Any]]:
    completed = []
    for index in range(int(contract["noise_band"]["baseline_seeds"])):
        trial_id = f"baseline-{index:03d}"
        proposed = dict(contract["baseline"])
        record = _execute(
            run_dir,
            contract,
            trial_id=trial_id,
            arm="baseline",
            proposer="baseline",
            proposed=proposed,
            synthetic=synthetic,
            in_process=in_process,
        )
        records.append(record)
        if record["status"] == "completed":
            completed.append(record)
    if not any(row["trial_id"] == "baseline-000" and row["status"] == "completed" for row in completed):
        raise RuntimeError("The pre-registered baseline trial did not complete")
    expected = int(contract["noise_band"]["baseline_seeds"])
    if len(completed) != expected:
        raise RuntimeError(f"The noise band needs all {expected} baseline seeds to complete")
    return completed


def _run_arm(run_dir, contract, *, arm, budget, synthetic, in_process, records) -> None:
    proposer = build_proposer(arm, contract=contract, seed=int(contract["seeds"]["master"]))
    history: list[dict[str, Any]] = []
    for index in range(budget):
        trial_id = f"{arm}-{index:03d}"
        try:
            proposed = proposer.propose(history)
        except (IndexError, ValueError, OSError) as exc:
            record = _failed_record(
                trial_id, arm, arm, {}, public_error(exc, run_dir),
            )
            append_research_record(run_dir, record)
            records.append(record)
            history.append(record)
            continue
        record = _execute(
            run_dir,
            contract,
            trial_id=trial_id,
            arm=arm,
            proposer=arm,
            proposed=proposed,
            synthetic=synthetic,
            in_process=in_process,
        )
        records.append(record)
        history.append(record)


def _execute(run_dir, contract, *, trial_id, arm, proposer, proposed, synthetic, in_process) -> dict[str, Any]:
    trained = apply_runtime_clip(dict(proposed), contract, synthetic)
    seed = trial_seed(int(contract["seeds"]["master"]), trial_id)
    trial_dir = run_dir / "trials" / trial_id
    started = time.perf_counter()
    timeout = int(contract["per_trial_timeout_seconds"])
    try:
        if in_process:
            train_trial(run_dir=run_dir, trial_dir=trial_dir, config=trained, seed=seed)
            state = "completed"
            note = ""
        else:
            state, note = _run_subprocess(run_dir, trial_dir, trained, seed, timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        state = "failed"
        note = public_error(exc, run_dir)
    elapsed = round(time.perf_counter() - started, 3)
    if state == "completed" and elapsed > timeout:
        state = "timed_out"
        note = "elapsed time exceeded the per-trial timeout"
    metrics: dict[str, Any] = {}
    if state == "completed":
        try:
            metrics = _score_validation(run_dir, trial_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            state = "failed"
            note = public_error(exc, run_dir)
            metrics = {}
    record = {
        "trial_id": trial_id,
        "arm": arm,
        "proposer": proposer,
        "status": state,
        "config": _jsonable(proposed),
        "config_trained": _jsonable(trained),
        "seed": seed,
        "validation_metrics": metrics,
        "runtime_seconds": elapsed,
        "trial_dir": f"trials/{trial_id}",
        "notes": note,
    }
    append_research_record(run_dir, record)
    return record


def _score_validation(run_dir: Path, trial_dir: Path) -> dict[str, Any]:
    manifest = json.loads((trial_dir / "submission_manifest.json").read_text(encoding="utf-8"))
    violations = check_feature_columns(manifest["feature_columns"])
    violations.extend(check_metadata(manifest["metadata"]))
    predictions = pl.read_parquet(trial_dir / "validation_predictions.parquet")
    labels = pl.read_parquet(artifacts_dir(run_dir) / "validation_labels.parquet")
    violations.extend(check_coverage(predictions, labels))
    if violations:
        raise ValueError("; ".join(violations))
    if "label" in predictions.columns:
        raise ValueError("Validation predictions must not carry a label column")
    return join_and_score(predictions, labels, split="validation")


def _write_noise_band(run_dir, contract, baseline_records, *, synthetic: bool) -> None:
    ordered = sorted(baseline_records, key=lambda row: row["trial_id"])
    aucs = [float(row["validation_metrics"]["auc_roc"]) for row in ordered]
    primary = next(row for row in ordered if row["trial_id"] == "baseline-000")
    predictions = pl.read_parquet(
        run_dir / primary["trial_dir"] / "validation_predictions.parquet"
    )
    labels = pl.read_parquet(artifacts_dir(run_dir) / "validation_labels.parquet")
    joined = labels.join(predictions.select(["row_id", "prediction"]), on="row_id", how="inner")
    replicates = bootstrap_replicates(contract, synthetic)
    spread = user_bootstrap_std(
        joined,
        replicates=replicates,
        seed=int(contract["seeds"]["master"]),
    )
    band = build_noise_band(aucs, spread)
    (run_dir / "noise_band.json").write_text(
        json.dumps(band, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_subprocess(run_dir: Path, trial_dir: Path, config: dict, seed: int, timeout: int) -> tuple[str, str]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config_trained.json"
    config_path.write_text(json.dumps(_jsonable(config), sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "run_trial.py"),
        "--run-dir",
        str(run_dir),
        "--trial-dir",
        str(trial_dir),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            command,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return "timed_out", "per-trial timeout"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "trial process failed").strip()
        return "failed", public_error(RuntimeError(detail), run_dir)
    return "completed", ""


def _winners(records: list[dict[str, Any]]) -> dict[str, str | None]:
    winners: dict[str, str | None] = {"baseline": None, "random": None, "greedy": None}
    for record in records:
        if record["trial_id"] == "baseline-000" and record["status"] == "completed":
            winners["baseline"] = record["trial_id"]
    for arm in ("random", "greedy"):
        done = [
            row for row in records
            if row["arm"] == arm and row["status"] == "completed"
        ]
        if done:
            best = max(done, key=lambda row: float(row["validation_metrics"]["auc_roc"]))
            winners[arm] = best["trial_id"]
    return winners


def _failed_record(trial_id, arm, proposer, config, note) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "arm": arm,
        "proposer": proposer,
        "status": "failed",
        "config": config,
        "config_trained": {},
        "seed": None,
        "validation_metrics": {},
        "runtime_seconds": 0.0,
        "trial_dir": f"trials/{trial_id}",
        "notes": note,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the guarded search loop")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Also run the replay arm")
    args = parser.parse_args(argv)
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = ROOT / "runs" / ("synthetic" if args.synthetic else "yambda")
    in_process = True if args.synthetic or args.in_process else False
    try:
        run_search(
            run_dir,
            synthetic=args.synthetic,
            in_process=in_process,
            replay=args.replay,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(public_error(exc, run_dir), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
