"""Estimate dual-adjudicator token usage and API cost without making API calls."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "reference_costs.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate(config: dict[str, Any], use_batch: bool) -> dict[str, Any]:
    repeat_fraction = float(config["audit_repeat_fraction"])
    mode = "batch" if use_batch else "standard"
    task_rows: dict[str, dict[str, float | int | str]] = {}
    total_items = 0
    total_input_tokens = 0.0
    base_output_tokens = 0.0

    for task_name, task in config["tasks"].items():
        items = int(task["items"])
        input_tokens = items * float(task["estimated_input_tokens_per_item"])
        output_tokens = items * float(task["estimated_output_tokens_per_item"])
        total_items += items
        total_input_tokens += input_tokens
        base_output_tokens += output_tokens
        task_rows[task_name] = {
            "items": items,
            "input_tokens": int(input_tokens),
            "base_output_tokens": int(output_tokens),
            "notes": str(task.get("notes", "")),
        }

    audit_factor = 1.0 + repeat_fraction
    total_input_tokens *= audit_factor
    base_output_tokens *= audit_factor
    scenarios: dict[str, Any] = {}

    for scenario_name, scenario in config["scenarios"].items():
        output_multiplier = float(scenario["output_token_multiplier"])
        output_tokens = base_output_tokens * output_multiplier
        model_costs: dict[str, Any] = {}
        total_cost = 0.0

        for model_name, model in config["models"].items():
            discount = float(model["batch_discount"]) if use_batch else 1.0
            input_cost = (
                total_input_tokens
                * float(model["input_usd_per_million"])
                * discount
                / 1_000_000
            )
            output_cost = (
                output_tokens
                * float(model["output_usd_per_million"])
                * discount
                / 1_000_000
            )
            cost = input_cost + output_cost
            total_cost += cost
            model_costs[model_name] = {
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
                "total_cost_usd": round(cost, 4),
            }

        budget = float(config["budget_usd"])
        scenarios[scenario_name] = {
            "description": str(scenario["description"]),
            "input_tokens_per_model": int(total_input_tokens),
            "output_tokens_per_model": int(output_tokens),
            "model_costs": model_costs,
            "combined_cost_usd": round(total_cost, 4),
            "remaining_budget_usd": round(budget - total_cost, 4),
            "within_budget": total_cost <= budget,
        }

    return {
        "mode": mode,
        "budget_usd": float(config["budget_usd"]),
        "audit_repeat_fraction": repeat_fraction,
        "items_per_request": int(config["items_per_request"]),
        "base_items": total_items,
        "estimated_requests_per_model": -(
            -int(total_items * audit_factor) // int(config["items_per_request"])
        ),
        "tasks": task_rows,
        "scenarios": scenarios,
        "warning": (
            "These are planning estimates. Claude thinking tokens and OpenAI reasoning "
            "tokens are billed output. Run a measured calibration before production."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Dual-Adjudicator Cost Estimate",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Base items: `{result['base_items']:,}`",
        f"- Repeat/order audit: `{result['audit_repeat_fraction']:.0%}`",
        f"- Estimated requests per model: `{result['estimated_requests_per_model']:,}`",
        f"- Hard budget: `${result['budget_usd']:.2f}`",
        "",
        "## Scenarios",
        "",
    ]
    for name, scenario in result["scenarios"].items():
        lines.extend(
            [
                f"### {name}",
                f"- {scenario['description']}",
                f"- Input tokens/model: `{scenario['input_tokens_per_model']:,}`",
                f"- Output tokens/model: `{scenario['output_tokens_per_model']:,}`",
            ]
        )
        for model_name, costs in scenario["model_costs"].items():
            lines.append(
                f"- `{model_name}`: `${costs['total_cost_usd']:.2f}` "
                f"(input `${costs['input_cost_usd']:.2f}`, "
                f"output `${costs['output_cost_usd']:.2f}`)"
            )
        lines.extend(
            [
                f"- Combined: `${scenario['combined_cost_usd']:.2f}`",
                f"- Remaining budget: `${scenario['remaining_budget_usd']:.2f}`",
                f"- Within budget: `{scenario['within_budget']}`",
                "",
            ]
        )
    lines.extend(["## Caveat", "", result["warning"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode", choices=("standard", "batch"), default="batch"
    )
    parser.add_argument(
        "--esci-items",
        type=int,
        help="Override the number of newly dual-adjudicated ESCI pairs.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deepcopy(load_config(args.config))
    if args.esci_items is not None:
        if args.esci_items < 0:
            raise ValueError("--esci-items must be non-negative")
        config["tasks"]["esci_new_enriched"]["items"] = args.esci_items

    result = estimate(config, use_batch=args.mode == "batch")
    rendered = render_markdown(result)
    print(rendered)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
