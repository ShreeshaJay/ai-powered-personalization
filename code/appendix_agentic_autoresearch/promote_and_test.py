"""Promotion gate. Hashes, leakage, the noise band, guardrails, then one test score.

The test budget is 3: the pre-registered baseline, the random-arm winner, and
the greedy-arm winner. A failed check is logged and does not spend the budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from harness.contract import artifacts_dir, contract_path, file_sha256, load_contract
from harness.evaluate import join_and_score
from harness.leakage import (
    check_coverage,
    check_feature_columns,
    check_metadata,
    check_reproduction,
)
from harness.ledger import public_error, read_jsonl, research_ledger_path
from harness.seal import (
    BudgetError,
    SealError,
    enforce_budget,
    record_rejection,
    record_spend,
    seal_dir,
    verify_hashes,
)
from harness.train import predict_with_saved_pipeline


def promote(run_dir: Path, candidate: str) -> dict[str, Any]:
    contract = load_contract()
    if candidate not in contract["test_candidates"]:
        raise ValueError(f"Candidate must be one of {contract['test_candidates']}")
    verify_hashes(run_dir, file_sha256(contract_path()))
    winners = json.loads((run_dir / "winners.json").read_text(encoding="utf-8"))
    trial_id = winners.get(candidate)
    if not trial_id:
        decision = _rejection(candidate, None, "no completed trial for this arm")
        record_rejection(run_dir, decision)
        print(f"{candidate}: rejected, no completed trial")
        return decision

    research = _research_record(run_dir, trial_id)
    trial_dir = run_dir / research["trial_dir"]
    manifest = json.loads((trial_dir / "submission_manifest.json").read_text(encoding="utf-8"))
    predictions = pl.read_parquet(trial_dir / "validation_predictions.parquet")
    labels = pl.read_parquet(artifacts_dir(run_dir) / "validation_labels.parquet")
    violations = check_feature_columns(manifest["feature_columns"])
    violations.extend(check_metadata(manifest["metadata"]))
    violations.extend(check_coverage(predictions, labels))
    if violations:
        decision = _rejection(candidate, trial_id, "; ".join(violations))
        record_rejection(run_dir, decision)
        print(f"{candidate}: rejected by the leakage checks")
        return decision

    recomputed = join_and_score(predictions, labels, split="validation")
    claimed = {
        key: research["validation_metrics"][key]
        for key in ("auc_roc", "log_loss", "expected_calibration_error")
        if key in research.get("validation_metrics", {})
    }
    violations = check_reproduction(claimed, recomputed)
    if violations:
        decision = _rejection(candidate, trial_id, "; ".join(violations))
        record_rejection(run_dir, decision)
        print(f"{candidate}: rejected because the stored metrics do not reproduce")
        return decision

    reasons = _threshold_reasons(contract, run_dir, candidate, recomputed)
    if reasons:
        decision = _rejection(candidate, trial_id, "; ".join(reasons))
        decision["validation_auc"] = recomputed["auc_roc"]
        record_rejection(run_dir, decision)
        print(f"{candidate}: rejected, {decision['reason']}")
        return decision

    ledger_ids = {row["trial_id"] for row in read_jsonl(research_ledger_path(run_dir))}
    if trial_id not in ledger_ids:
        decision = _rejection(candidate, trial_id, "trial is missing from the research ledger")
        record_rejection(run_dir, decision)
        return decision

    enforce_budget(run_dir, int(contract["test_budget"]))
    test_rows = pl.read_parquet(seal_dir(run_dir) / "test_rows.parquet")
    test_predictions = predict_with_saved_pipeline(trial_dir, test_rows)
    test_predictions.write_parquet(trial_dir / "test_predictions.parquet")
    test_labels = pl.read_parquet(seal_dir(run_dir) / "test_labels.parquet")
    coverage = check_coverage(test_predictions, test_labels)
    if coverage:
        raise SealError("; ".join(coverage))
    test_metrics = join_and_score(test_predictions, test_labels, split="test")
    validation_auc = float(recomputed["auc_roc"])
    optimism_gap = validation_auc - float(test_metrics["auc_roc"])
    evaluation = {
        "candidate": candidate,
        "trial_id": trial_id,
        "critic_verdict": "approve",
        "validation_auc": validation_auc,
        "test_metrics": test_metrics,
        "optimism_gap": optimism_gap,
        "reason": "gate checks passed",
    }
    record_spend(run_dir, evaluation)
    print(
        f"{candidate}: sealed test AUC {test_metrics['auc_roc']:.4f}, "
        f"optimism gap {optimism_gap:.4f}"
    )
    return evaluation


def _threshold_reasons(contract, run_dir, candidate, metrics) -> list[str]:
    reasons = []
    guardrails = contract["guardrails"]
    if float(metrics["log_loss"]) > float(guardrails["max_log_loss"]):
        reasons.append("log loss is above the guardrail")
    if float(metrics["expected_calibration_error"]) > float(guardrails["max_expected_calibration_error"]):
        reasons.append("expected calibration error is above the guardrail")
    if candidate != "baseline":
        band = json.loads((run_dir / "noise_band.json").read_text(encoding="utf-8"))
        gain = float(metrics["auc_roc"]) - float(band["baseline_mean_auc"])
        if not gain > float(band["noise_band"]):
            reasons.append("validation gain is inside the noise band")
    return reasons


def _research_record(run_dir: Path, trial_id: str) -> dict[str, Any]:
    matches = [
        row for row in read_jsonl(research_ledger_path(run_dir))
        if row.get("trial_id") == trial_id and row.get("status") == "completed"
    ]
    if not matches:
        raise ValueError(f"No completed research record for {trial_id}")
    return matches[-1]


def _rejection(candidate: str, trial_id: str | None, reason: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "trial_id": trial_id,
        "critic_verdict": "reject",
        "reason": reason,
        "spent": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one sealed test evaluation if the gate passes")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", required=True, choices=["baseline", "random", "greedy"])
    args = parser.parse_args(argv)
    try:
        promote(args.run_dir, args.candidate)
    except (SealError, BudgetError, ValueError, OSError, RuntimeError) as exc:
        print(public_error(exc, args.run_dir), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
