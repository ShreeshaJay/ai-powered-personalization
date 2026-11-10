"""Best-so-far curves, the random-arm expected maximum, and the optimism table."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from harness.ledger import read_jsonl, research_ledger_path
from harness.seal import test_ledger_path


def write_report(run_dir: Path) -> Path:
    research = read_jsonl(research_ledger_path(run_dir))
    test_rows = read_jsonl(test_ledger_path(run_dir))
    best = best_so_far(research)
    random_scores = [
        float(row["validation_metrics"]["auc_roc"])
        for row in research
        if row.get("arm") == "random" and row.get("status") == "completed"
    ]
    expected = expected_maximum(random_scores)
    optimism = optimism_table(test_rows)
    denominator = denominator_table(research)
    _write_csv(run_dir / "best_so_far.csv", best)
    _write_csv(run_dir / "expected_maximum_random.csv", expected)
    _write_csv(run_dir / "optimism.csv", optimism)
    _write_csv(run_dir / "denominator.csv", denominator)
    path = run_dir / "report.md"
    path.write_text(
        _markdown(best, expected, optimism, denominator),
        encoding="utf-8",
    )
    print(f"Wrote {path.name}")
    return path


def best_so_far(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ("baseline", "random", "greedy", "replay"):
        arm_rows = [row for row in records if row.get("arm") == arm]
        best = None
        for index, row in enumerate(arm_rows, start=1):
            score = (row.get("validation_metrics") or {}).get("auc_roc")
            if score is not None and (best is None or float(score) > best):
                best = float(score)
            rows.append(
                {
                    "arm": arm,
                    "trial_index": index,
                    "trial_id": row.get("trial_id"),
                    "status": row.get("status"),
                    "validation_auc": "" if score is None else float(score),
                    "best_so_far_validation_auc": "" if best is None else best,
                }
            )
    return rows


def expected_maximum(values: list[float]) -> list[dict[str, Any]]:
    """Mean of the maximum over subsets, the Dodge et al. (2019) curve."""

    if not values:
        return []
    rows = []
    count = len(values)
    for budget in range(1, count + 1):
        combinations = math.comb(count, budget)
        if count <= 16 and combinations <= 5000:
            total = sum(max(subset) for subset in itertools.combinations(values, budget))
            mean = total / combinations
        else:
            rng = np.random.default_rng(20261109)
            array = np.asarray(values, dtype=float)
            draws = [
                float(rng.choice(array, size=budget, replace=False).max())
                for _ in range(2000)
            ]
            mean = float(np.mean(draws))
        rows.append({"budget": budget, "expected_maximum_validation_auc": mean})
    return rows


def optimism_table(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spent = [row for row in test_rows if row.get("spent") is True]
    baseline_gap = next(
        (float(row["optimism_gap"]) for row in spent if row.get("candidate") == "baseline"),
        None,
    )
    rows = []
    for row in test_rows:
        if row.get("spent") is True:
            gap = float(row["optimism_gap"])
            adjusted = "" if baseline_gap is None else gap - baseline_gap
            rows.append(
                {
                    "candidate": row.get("candidate"),
                    "trial_id": row.get("trial_id"),
                    "decision": "tested",
                    "validation_auc": row.get("validation_auc"),
                    "test_auc": row["test_metrics"]["auc_roc"],
                    "optimism_gap": gap,
                    "shift_adjusted_optimism": adjusted,
                }
            )
        else:
            rows.append(
                {
                    "candidate": row.get("candidate"),
                    "trial_id": row.get("trial_id") or "",
                    "decision": "rejected",
                    "validation_auc": row.get("validation_auc", ""),
                    "test_auc": "",
                    "optimism_gap": "",
                    "shift_adjusted_optimism": "",
                    "reason": row.get("reason", ""),
                }
            )
    return rows


def denominator_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = ["baseline", "random", "greedy", "replay", "all"]
    rows = []
    for arm in arms:
        chosen = records if arm == "all" else [row for row in records if row.get("arm") == arm]
        if arm != "all" and not chosen:
            continue
        rows.append(
            {
                "arm": arm,
                "attempted": len(chosen),
                "completed": sum(row.get("status") == "completed" for row in chosen),
                "failed": sum(row.get("status") == "failed" for row in chosen),
                "timed_out": sum(row.get("status") == "timed_out" for row in chosen),
            }
        )
    return rows


def _markdown(best, expected, optimism, denominator) -> str:
    lines = [
        "# Appendix C search report",
        "",
        "Validation AUC is the selection metric. Test scores appear only for candidates the promotion gate evaluated.",
        "",
        "## Denominator",
        "",
        _table(denominator),
        "",
        "## Best validation AUC so far",
        "",
        _table(best),
        "",
        "## Expected maximum, random arm",
        "",
        "For a budget k, the value is the average maximum validation AUC over subsets of k completed random trials.",
        "",
        _table(expected) if expected else "No completed random trials.",
        "",
        "## Optimism",
        "",
        "The optimism gap is validation AUC minus sealed-test AUC. Shift-adjusted optimism subtracts the baseline gap.",
        "",
        _table(optimism) if optimism else "No promotion decisions were recorded.",
        "",
    ]
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in headers:
                headers.append(key)
    header = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        cells = [_cell(row.get(header_name, "")) for header_name in headers]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Appendix C result tables")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_report(args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
