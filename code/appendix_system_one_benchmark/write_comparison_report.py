"""Compare frozen zero-shot models on shared evaluation slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmark" / "ZERO_SHOT_RESULTS.md"
DEFAULT_SUMMARIES = {
    "majority": ROOT / "outputs" / "benchmark" / "majority" / "summary.json",
    "laya": ROOT
    / "outputs"
    / "benchmark"
    / "laya_typed_decisions"
    / "summary.json",
    "jev": ROOT / "outputs" / "benchmark" / "jev_1_13_0" / "summary.json",
    "kev": ROOT / "outputs" / "benchmark" / "kev_0_8b" / "summary.json",
    "kev4": ROOT / "outputs" / "benchmark" / "kev_4b" / "summary.json",
    "jevlite": ROOT / "outputs" / "benchmark" / "jevlite" / "summary.json",
}

FIELD_ROWS = (
    ("compatibility", "label", "Compatibility"),
    ("query_segmentation", "goal", "Query goal"),
    ("query_segmentation", "object", "Query object"),
    ("query_segmentation", "specificity", "Query specificity"),
    ("query_segmentation", "commerce_scope", "Query commerce scope"),
    ("brand_category", "brand_intent", "Brand intent"),
    ("brand_category", "brand_choice", "Brand choice"),
    ("brand_category", "category_choice", "Category choice"),
    ("brand_category", "multi_target", "Multi-target"),
    ("esci", "esci_label", "ESCI (human slice)"),
)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def field_report(summary: dict[str, Any], task: str, field: str) -> dict[str, Any]:
    return ((summary.get("tasks") or {}).get(task) or {}).get("fields", {}).get(
        field, {}
    )


def cell(report: dict[str, Any], key: str) -> str:
    if not report or report.get(key) is None:
        return "n/a"
    value = report[key]
    if key == "accuracy":
        return f"{100 * value:.1f}%"
    return f"{value:.3f}"


def render(summaries: dict[str, dict[str, Any]]) -> str:
    models = list(summaries)
    header = "| Field | " + " | ".join(
        f"{model} acc | {model} macro-F1 | {model} ECE" for model in models
    ) + " |"
    divider = "|" + "---|" * (1 + 3 * len(models))
    lines = [
        "# Frozen Zero-Shot Comparison",
        "",
        "Laya is `convaiinnovations/laya` (`typed-decisions`). "
        "Jev is pinned `jev-1.13.0` via the TypeSafe API. "
        "Kev is `jaredpalmer/kev-0.8b` served locally over `/v1/systemone`. "
        "Majority is an a priori class prior and does not peek at evaluation labels. "
        "Compatibility, query, and brand/category use dual-judge field consensus; "
        "disagreements are withheld. ESCI is a balanced 8,000-pair human US test slice "
        "(2,000 per E/S/C/I).",
        "",
        header,
        divider,
    ]
    for task, field, title in FIELD_ROWS:
        cells = [title]
        for model in models:
            report = field_report(summaries[model], task, field)
            cells.extend(
                [
                    cell(report, "accuracy"),
                    cell(report, "macro_f1"),
                    cell(report, "ece"),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    serving_lines = []
    for model in models:
        for task, metrics in (summaries[model].get("tasks") or {}).items():
            latency = metrics.get("latency") or {}
            if not latency.get("items"):
                continue
            extra = ""
            memory = metrics.get("memory") or {}
            if memory.get("peak_reserved_mb") is not None:
                extra += f"; VRAM reserved {memory['peak_reserved_mb']:.0f} MB"
            if metrics.get("cost_usd") is not None:
                extra += f"; ${metrics['cost_usd']:.4f}"
            serving_lines.append(
                f"- **{model} {task}:** p50 {latency['p50_ms']:.1f} ms; "
                f"{latency['items_per_second']:.1f} items/s{extra}"
            )

    lines.extend(
        [
            "",
            "## Serving",
            "",
            *serving_lines,
            "",
            "## Reading the numbers",
            "",
            "- Majority wins accuracy on imbalanced tasks because `incompatible` and "
            "`multi_target=false` dominate the consensus labels.",
            "- Laya is fast locally but is not a strong frozen zero-shot classifier "
            "on these commerce schemas.",
            "- Jev is the hosted System One reference. Compare it to Laya and Kev on "
            "the same frozen questions and compact states, not to supervised controls.",
            "- Kev-0.8B is the first open TypeSafe-compatible checkpoint in this "
            "drop. It is much weaker than hosted Jev on compatibility, query axes, "
            "and ESCI, but brand and category choice are already usable.",
            "- JevLite and larger Kev checkpoints have adapters and a Colab notebook "
            "in this package, but those full-slice runs are not part of this "
            "published result drop.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        name: load_summary(path)
        for name, path in DEFAULT_SUMMARIES.items()
        if path.exists()
    }
    report = render(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
