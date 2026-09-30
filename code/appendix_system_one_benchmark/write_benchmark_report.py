"""Write a compact Markdown report from a benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def field_line(task: str, metrics: dict[str, Any]) -> list[str]:
    lines = []
    for field, report in metrics.get("fields", {}).items():
        if not report.get("items"):
            lines.append(f"- **{field}:** no labeled items")
            continue
        lines.append(
            f"- **{field}:** n={report['items']:,}; acc {percent(report['accuracy'])}; "
            f"macro-F1 {report['macro_f1']:.3f}; ECE {report['ece']:.3f}; "
            f"Brier {report['brier']:.3f}; NLL {report['nll']:.3f}"
        )
    exact = metrics.get("complete_exact_match") or {}
    if exact.get("items"):
        lines.append(
            f"- **complete exact match:** n={exact['items']:,}; "
            f"{percent(exact['exact_match'])}"
        )
    latency = metrics.get("latency") or {}
    if latency.get("items"):
        lines.append(
            f"- **latency:** p50 {latency['p50_ms']:.1f} ms; "
            f"p95 {latency['p95_ms']:.1f} ms; "
            f"{latency['items_per_second']:.1f} items/s"
        )
    memory = metrics.get("memory")
    if memory:
        lines.append(
            f"- **VRAM:** allocated {memory['peak_allocated_mb']:.0f} MB; "
            f"reserved {memory['peak_reserved_mb']:.0f} MB"
        )
    if metrics.get("cost_usd") is not None:
        lines.append(
            f"- **API cost:** ${metrics['cost_usd']:.6f} "
            f"({metrics.get('input_tokens', 0):,} input tokens)"
        )
    return lines


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# System One Benchmark Results",
        "",
        f"- Model: `{summary.get('model_id') or summary.get('model')}`.",
        f"- Dataset: `{summary.get('dataset')}`.",
        f"- Batch size: {summary.get('batch_size')}.",
        f"- Device: `{summary.get('device')}`.",
        "",
        "Primary labels for compatibility, query segmentation, and brand/category "
        "are dual-judge field consensus. Disagreements are withheld per field. "
        "ESCI is reported separately by `label_source` when present.",
        "",
    ]
    for task, metrics in summary.get("tasks", {}).items():
        lines.extend([f"## {task}", ""])
        lines.extend(field_line(task, metrics))
        lines.append("")
        esci_source = (
            (metrics.get("fields") or {}).get("esci_label") or {}
        ).get("accuracy_by_label_source")
        if esci_source:
            lines.append("Label-source accuracy:")
            for source, values in esci_source.items():
                lines.append(
                    f"- **{source}:** n={values['items']:,}; "
                    f"acc {percent(values['accuracy'])}"
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Consensus labels are synthetic references, not human ground truth.",
            "- This is a frozen zero-shot track. Do not compare it to supervised controls.",
            "- Latency is amortized per item for local batched models and per-request "
            "for hosted APIs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], output: Path) -> str:
    report = render_report(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    output = args.output or args.summary.with_name("BENCHMARK_RESULTS.md")
    print(write_report(summary, output))


if __name__ == "__main__":
    main()
