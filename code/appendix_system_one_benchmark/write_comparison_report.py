"""Write outputs/benchmark/ZERO_SHOT_RESULTS.md from the committed summaries and the replay pack.

The Markdown file is generated: change this script, never the file. Run metadata
and serving values are read from each `summary.json`. Accuracy, macro-F1 and
calibration are computed from the replay pack in `replay/`; calibration is
top-label ECE from the probability vectors, and ECE on each adapter's reported
`confidence` field is printed as a comparison column only.
`python replay_scores.py verify` checks the summaries against the pack,
regenerates this file in memory and fails when the committed copy differs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from eval.calibration import DEFAULT_BINS, ece_equal_mass, ece_equal_width, macro_f1_union
from eval.replay_pack import DEFAULT_PACK_DIR, ReplayPack, field_view, is_degenerate, load_pack, majority_share

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmark" / "ZERO_SHOT_RESULTS.md"
MODEL_SLUGS = (
    "majority",
    "laya_typed_decisions",
    "jev_1_13_0",
    "kev_0_8b",
    "kev_4b",
    "jevlite",
)
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
NOT_RECORDED_GPU = "GPU type not recorded"
LAYA_NOTE = (
    "`laya_typed_decisions` ran with the adapter's `max_len=512` (`adapters/laya.py`), although the checkpoint "
    "accepts 1,024 tokens. A tokenizer check on 2026-11-09 found inputs cut at 512 tokens for 1,996 of the 2,000 "
    "brand/category items and 3 of the 2,000 compatibility items; no input exceeded 1,024 tokens. The checkpoint's "
    "256-token question limit kept 127 to 215 of the 452 instruction tokens in each query-segmentation question. "
    "ESCI inputs were not cut. Laya's results measure this configuration, not the checkpoint's best case."
)


def summary_path(slug: str, root: Path = ROOT) -> Path:
    return root / "outputs" / "benchmark" / slug / "summary.json"


def load_summaries(root: Path = ROOT, slugs: tuple[str, ...] = MODEL_SLUGS) -> dict[str, dict[str, Any]]:
    return {
        slug: json.loads(summary_path(slug, root).read_text(encoding="utf-8"))
        for slug in slugs
        if summary_path(slug, root).exists()
    }


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt(value: Any, kind: str = "3f") -> str:
    if is_missing(value):
        return "n/a"
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.3f}"


def gpu_type(summary: dict[str, Any]) -> str:
    if summary.get("device") == "cpu":
        return "n/a (CPU)"
    names = {summary.get("gpu_name")} | {
        (block.get("memory") or {}).get("gpu_name") for block in (summary.get("tasks") or {}).values()
    }
    names = sorted(name for name in names if name)
    return ", ".join(names) if names else NOT_RECORDED_GPU


def device_memory(summary: dict[str, Any]) -> str:
    totals = [
        (block.get("memory") or {}).get("device_total_mb")
        for block in (summary.get("tasks") or {}).values()
    ]
    totals = [value for value in totals if value is not None]
    return f"{max(totals):,.0f}" if totals else "not recorded"


def field_row(pack: ReplayPack, slug: str, task: str, field: str, bins: int) -> dict[str, float]:
    view = field_view(pack, slug, task, field)
    n = len(view)
    return {
        "n": n,
        "accuracy": sum(view.correct) / n if n else math.nan,
        "macro_f1": macro_f1_union(view.reference, view.predicted) if n else math.nan,
        "ece_top_label_width": ece_equal_width(view.top_label_probability, view.correct, bins),
        "ece_top_label_mass": ece_equal_mass(view.top_label_probability, view.correct, bins),
        "ece_confidence_width": ece_equal_width(view.reported_confidence, view.correct, bins),
        "mean_top_label_probability": sum(view.top_label_probability) / n if n else math.nan,
        "mean_confidence": sum(view.reported_confidence) / n if n else math.nan,
    }


def results_table(
    slugs: list[str],
    pack: ReplayPack,
    rows: list[tuple[str, str, str]],
    bins: int,
) -> list[str]:
    lines = [
        "| Field | Model | n | Accuracy | Macro-F1 "
        f"| Top-label ECE ({bins} equal-width) | Top-label ECE ({bins} equal-mass) "
        f"| ECE on reported `confidence` ({bins} equal-width; comparison only) "
        "| Mean top-label probability | Mean reported `confidence` |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, field, title in rows:
        for slug in slugs:
            row = field_row(pack, slug, task, field, bins)
            lines.append(
                f"| {title} | `{slug}` | {fmt(row['n'], 'int')} "
                f"| {fmt(row['accuracy'], 'pct')} | {fmt(row['macro_f1'])} "
                f"| {fmt(row['ece_top_label_width'])} | {fmt(row['ece_top_label_mass'])} "
                f"| {fmt(row['ece_confidence_width'])} | {fmt(row['mean_top_label_probability'])} "
                f"| {fmt(row['mean_confidence'])} |"
            )
    return lines


def serving_lines(summaries: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for slug, summary in summaries.items():
        for task, metrics in (summary.get("tasks") or {}).items():
            latency = metrics.get("latency") or {}
            if not latency.get("items"):
                continue
            extra = ""
            memory = metrics.get("memory") or {}
            if memory.get("peak_reserved_mb") is not None:
                extra += f"; VRAM reserved {memory['peak_reserved_mb']:,.0f} MB"
            if metrics.get("cost_usd") is not None:
                extra += f"; ${metrics['cost_usd']:.4f}"
            p50 = "n/a" if is_missing(latency.get("p50_ms")) else f"{latency['p50_ms']:.1f} ms"
            rate = latency.get("items_per_second")
            rate_text = "throughput n/a" if is_missing(rate) else f"{rate:.1f} items/s"
            lines.append(f"- **{slug} {task}:** p50 {p50}; {rate_text}{extra}")
    return lines


def render(summaries: dict[str, dict[str, Any]], pack: ReplayPack, bins: int = DEFAULT_BINS) -> str:
    headline = []
    degenerate = []
    for task, field, title in FIELD_ROWS:
        target = degenerate if is_degenerate(pack.examples[task], field) else headline
        target.append((task, field, title))

    lines = [
        "# Frozen Zero-Shot Comparison",
        "",
        "Generated by `write_comparison_report.py` from the committed `summary.json` files "
        "and the replay pack in `replay/`. Do not edit by hand; "
        "`python replay_scores.py verify` fails when this file differs from the script's output.",
        "",
        "## Runs",
        "",
        "| Model | Model ID | Prompt version | Recorded device | GPU type | Device memory (MB) | Batch size |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for slug, summary in summaries.items():
        versions = sorted(
            {block.get("prompt_version") for block in (summary.get("tasks") or {}).values()} - {None}
        )
        lines.append(
            f"| `{slug}` | `{summary.get('model_id')}` | {', '.join(f'`{v}`' for v in versions) or 'n/a'} "
            f"| `{summary.get('device')}` | {gpu_type(summary)} | {device_memory(summary)} "
            f"| {summary.get('batch_size', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "Recorded device, device memory and batch size are copied from each `summary.json`; for "
            "`jev_1_13_0`, a hosted API, the recorded device is the client machine that sent the requests. "
            "The `majority` adapter predicts the class listed for each field in `MAJORITY_PRIORS` "
            "(`adapters/schemas.py`) for every item.",
            "",
            LAYA_NOTE,
            "",
            "## Labels",
            "",
            "- Compatibility, query segmentation and brand/category: dual-judge field consensus. "
            "A field on which the two judges disagreed is withheld and not scored, so n differs by field.",
            "- ESCI: a class-balanced slice of 8,000 US test pairs (2,000 per E, S, C and I) "
            "scored against the ESCI human labels.",
            "",
            "## Definitions",
            "",
            "- **Accuracy** and **macro-F1** use the committed metric definitions, computed from the pack and "
            "rounded once for display; each `summary.json` stores the same values rounded to 4 decimals. "
            "Macro-F1 averages per-class F1 over the classes that occur in the reference or the predicted labels.",
            f"- **Top-label ECE** uses the probability each model assigned to the label it returned, "
            f"taken from its probability vector, and counts an item as correct when that label equals "
            f"the reference. Equal-width: {bins} bins on [0, 1], the first closed below. "
            f"Equal-mass: items stably sorted by that probability and split into {bins} groups of "
            f"near-equal size, using `numpy.array_split`, so the first n mod {bins} groups hold one extra item.",
            f"- **ECE on reported `confidence`** applies the same {bins}-bin equal-width estimator to the "
            "adapter's `confidence` field instead of the probability vector. Adapters define that field "
            "differently, so this column is not a calibration measure; it is printed for comparison only. "
            "`summary.json` stores the same quantity with 10 bins under `ece`.",
            "- Fields in which one class holds at least 99% of the labelled items are degenerate and are "
            "listed separately, below the main table.",
            "",
            "## Results",
            "",
            *results_table(list(summaries), pack, headline, bins),
            "",
        ]
    )
    for task, field, title in degenerate:
        share, top, labelled = majority_share(pack.examples[task], field)
        lines.extend(
            [
                f"## Degenerate field: {title}",
                "",
                f"One class holds {top:,} of the {labelled:,} labelled items ({100 * share:.2f}%), "
                "so accuracy on this field measures that class's prior. "
                "It is excluded from the main table.",
                "",
                *results_table(list(summaries), pack, [(task, field, title)], bins),
                "",
            ]
        )
    lines.extend(
        [
            "## Serving",
            "",
            "Each line is copied from one run's `summary.json`. The runs used different devices, "
            "batch sizes and concurrency, so these lines record each run and are not a latency comparison.",
            "",
            *serving_lines(summaries),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries()
    report = render(summaries, load_pack(args.pack_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
