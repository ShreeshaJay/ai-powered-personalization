"""Dry-run TypeSafe Jev token and dollar estimates. Makes no API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adapters.jev import (
    INPUT_USD_PER_MILLION,
    build_systemone_payload,
    estimate_input_tokens,
    token_cost_usd,
)
from eval.datasets import load_examples


DEFAULT_TASKS = (
    "compatibility",
    "query_segmentation",
    "brand_category",
    "esci",
)


def estimate_task(
    task: str, dataset: str, max_items: int | None
) -> dict[str, Any]:
    examples = load_examples(task, dataset=dataset, max_items=max_items)
    tokens = [
        estimate_input_tokens(
            build_systemone_payload(task, example["adjudication_input"])
        )
        for example in examples
    ]
    total = sum(tokens)
    return {
        "task": task,
        "items": len(examples),
        "requests": len(examples),
        "estimated_input_tokens": total,
        "mean_input_tokens": round(total / len(tokens), 1) if tokens else 0.0,
        "estimated_cost_usd": round(token_cost_usd(total), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pilot", "reference"), default="reference")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    task_rows = [
        estimate_task(task, args.dataset, args.max_items or None) for task in tasks
    ]
    total_tokens = sum(row["estimated_input_tokens"] for row in task_rows)
    summary = {
        "model": "jev-1.13.0",
        "input_usd_per_million": INPUT_USD_PER_MILLION,
        "output_usd_per_million": 0.0,
        "tasks": task_rows,
        "total_items": sum(row["items"] for row in task_rows),
        "total_estimated_input_tokens": total_tokens,
        "total_estimated_cost_usd": round(token_cost_usd(total_tokens), 6),
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
