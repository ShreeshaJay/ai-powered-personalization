"""Project 2,000-item reference-label cost from measured pilot usage."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from adjudicate_pilots import (
    DEFAULT_CACHE,
    PROVIDERS,
    TASK_PROMPT_VERSIONS,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "references" / "measured_cost_projection.json"


def project(
    cache: Path, target_items_per_task: int = 2000, audit_fraction: float = 0.05
) -> dict[str, Any]:
    connection = sqlite3.connect(cache)
    tasks: dict[str, Any] = {}
    total = 0.0
    for task, prompt_version in TASK_PROMPT_VERSIONS.items():
        task_result: dict[str, Any] = {}
        for provider, config in PROVIDERS.items():
            row = connection.execute(
                """
                SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd)
                FROM labels
                WHERE task = ? AND provider = ? AND model = ?
                  AND prompt_version = ? AND item_id NOT LIKE '%_ref_%'
                """,
                (task, provider, config["model"], prompt_version),
            ).fetchone()
            count = int(row[0] or 0)
            if count == 0:
                raise ValueError(
                    f"No pilot measurements for {task}/{provider}/{prompt_version}"
                )
            measured_cost = float(row[3])
            scale = target_items_per_task / count
            projected_cost = measured_cost * scale
            total += projected_cost
            task_result[provider] = {
                "measured_items": count,
                "measured_input_tokens": int(row[1]),
                "measured_output_tokens": int(row[2]),
                "measured_cost_usd": round(measured_cost, 6),
                "projected_items": target_items_per_task,
                "projected_cost_usd": round(projected_cost, 4),
            }
        tasks[task] = task_result
    connection.close()
    audit_cost = total * audit_fraction
    return {
        "target_items_per_task": target_items_per_task,
        "audit_fraction": audit_fraction,
        "tasks": tasks,
        "projected_base_cost_usd": round(total, 4),
        "projected_audit_cost_usd": round(audit_cost, 4),
        "projected_total_cost_usd": round(total + audit_cost, 4),
        "note": (
            "Projection uses standard synchronous API prices and measured pilot "
            "usage. Batch APIs would be approximately half this price."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--target-items-per-task", type=int, default=2000)
    parser.add_argument("--audit-fraction", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = project(
        args.cache, args.target_items_per_task, args.audit_fraction
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
