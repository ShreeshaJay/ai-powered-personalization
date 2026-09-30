"""Write a compact Markdown report for the scaled reference-label run."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from adjudicate_pilots import DEFAULT_CACHE


ROOT = Path(__file__).resolve().parent
DEFAULT_CONSENSUS = ROOT / "references" / "consensus" / "consensus_summary.json"
DEFAULT_VALIDATION = ROOT / "references" / "validation_summary.json"
DEFAULT_OUTPUT = ROOT / "references" / "REFERENCE_RESULTS.md"


def reference_usage(cache: Path) -> tuple[list[dict[str, Any]], float]:
    connection = sqlite3.connect(cache)
    rows = connection.execute(
        """
        SELECT task, provider, COUNT(1), SUM(input_tokens),
               SUM(output_tokens), SUM(cost_usd)
        FROM labels
        WHERE item_id LIKE '%_ref_%'
        GROUP BY task, provider
        ORDER BY task, provider
        """
    ).fetchall()
    result = [
        {
            "task": str(task),
            "provider": str(provider),
            "items": int(items),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": float(cost),
        }
        for task, provider, items, input_tokens, output_tokens, cost in rows
    ]
    error_counts = Counter(
        (str(task), str(provider))
        for task, provider in connection.execute(
            "SELECT task, provider FROM calls WHERE status = 'error'"
        )
    )
    connection.close()
    for row in result:
        row["discarded_calls_recorded"] = error_counts[
            (row["task"], row["provider"])
        ]
    return result, sum(row["cost_usd"] for row in result)


def domain_agreement(path: Path) -> dict[str, dict[str, Any]]:
    domains: dict[str, Counter[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source = row["sampling_context"]["source"]
            counts = domains.setdefault(source, Counter())
            counts["items"] += 1
            counts["complete"] += int(row["complete_agreement"])
            for field, agreed in row["field_agreement"].items():
                counts[f"field:{field}"] += int(agreed)
    output: dict[str, dict[str, Any]] = {}
    for source, counts in sorted(domains.items()):
        items = counts["items"]
        output[source] = {
            "items": items,
            "complete_agreement_rate": counts["complete"] / items,
            "field_agreement": {
                key.split(":", 1)[1]: value / items
                for key, value in counts.items()
                if key.startswith("field:")
            },
        }
    return output


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_report(
    consensus_path: Path, validation_path: Path, cache: Path
) -> str:
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    usage, total_cost = reference_usage(cache)
    query_domains = domain_agreement(
        consensus_path.parent / "query_segmentation_consensus.jsonl"
    )

    lines = [
        "# Scaled Reference-Label Results",
        "",
        "## Run status",
        "",
        "- 2,000 items per task.",
        "- Two independent labels per item: Claude Opus 5 and GPT-5.6 Sol.",
        "- 12,000 successfully persisted model labels.",
        f"- Recorded successful reference-label cost: `${total_cost:.6f}`.",
        "- Development pilots are excluded from the reference manifests.",
        "",
        "## Agreement",
        "",
        "| Task | Complete agreement | Consensus items |",
        "|---|---:|---:|",
    ]
    for task, data in consensus["tasks"].items():
        lines.append(
            f"| {task} | {percent(data['complete_agreement_rate'])} | "
            f"{data['complete_agreement_count']:,} |"
        )

    lines.extend(["", "### Field agreement", ""])
    for task, data in consensus["tasks"].items():
        fields = ", ".join(
            f"{field} {percent(values['rate'])}"
            for field, values in data["field_agreement"].items()
        )
        lines.append(f"- **{task}:** {fields}.")

    lines.extend(["", "### Query segmentation by source domain", ""])
    for source, data in query_domains.items():
        fields = ", ".join(
            f"{field} {percent(rate)}"
            for field, rate in data["field_agreement"].items()
        )
        lines.append(
            f"- **{source} ({data['items']:,}):** complete "
            f"{percent(data['complete_agreement_rate'])}; {fields}."
        )

    lines.extend(
        [
            "",
            "## API usage",
            "",
            "| Task | Provider | Items | Input tokens | Output tokens | Cost |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in usage:
        lines.append(
            f"| {row['task']} | {row['provider']} | {row['items']:,} | "
            f"{row['input_tokens']:,} | {row['output_tokens']:,} | "
            f"${row['cost_usd']:.4f} |"
        )

    lines.extend(["", "## Dataset validation", ""])
    for task, data in validation["datasets"].items():
        lines.append(
            f"- **{task}:** {data['rows']:,} rows; "
            f"input length mean {data['adjudication_input_chars']['mean']:.1f} "
            "characters."
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Consensus labels are synthetic references, not human ground truth.",
            "- ORCAS-I manual labels are retained only as source metadata; the two "
            "judges independently map queries to this benchmark's four-axis taxonomy.",
            "- Compatibility agreement is lowest for generated accessories and "
            "cross-sells, which reflects genuine missing fit evidence as well as "
            "candidate-map noise.",
            "- Two Anthropic brand/category responses were discarded before the "
            "normalization fix. Their provider-billed usage was not captured by the "
            "older failure ledger, so invoice cost may be slightly above the recorded "
            "successful-label cost.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.consensus, args.validation, args.cache)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
