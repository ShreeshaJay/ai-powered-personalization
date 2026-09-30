"""Export dual-judge consensus labels and calibration diagnostics."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from adjudicate_pilots import (
    DATASET_TASK_FILES,
    DEFAULT_CACHE,
    PROVIDERS,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "consensus"

AGREEMENT_FIELDS = {
    "compatibility": ["label"],
    "query_segmentation": ["goal", "object", "specificity", "commerce_scope"],
    "brand_category": [
        "brand_intent",
        "brand_choice",
        "category_choice",
        "multi_target",
    ],
}


def load_labels(
    connection: sqlite3.Connection, task: str
) -> dict[str, dict[str, dict[str, Any]]]:
    model_to_provider = {
        config["model"]: provider for provider, config in PROVIDERS.items()
    }
    rows = connection.execute(
        """
        SELECT item_id, model, label_json
        FROM labels
        WHERE task = ?
        ORDER BY created_at
        """,
        (task,),
    ).fetchall()
    labels: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item_id, model, label_json in rows:
        provider = model_to_provider.get(str(model), str(model))
        labels[str(item_id)][provider] = json.loads(label_json)
    return dict(labels)


def call_diagnostics(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT task, provider,
               SUM(item_count), SUM(input_tokens), SUM(output_tokens),
               SUM(cost_usd), SUM(elapsed_seconds), COUNT(*)
        FROM calls
        WHERE status = 'ok'
        GROUP BY task, provider
        ORDER BY task, provider
        """
    ).fetchall()
    result: dict[str, Any] = {}
    for (
        task,
        provider,
        item_count,
        input_tokens,
        output_tokens,
        cost_usd,
        elapsed_seconds,
        call_count,
    ) in rows:
        result.setdefault(str(task), {})[str(provider)] = {
            "items": int(item_count),
            "calls": int(call_count),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": round(float(cost_usd), 6),
            "elapsed_seconds": round(float(elapsed_seconds), 3),
        }
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def task_summary(
    task: str, manifest: list[dict[str, Any]], labels: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = AGREEMENT_FIELDS[task]
    field_matches = Counter()
    complete_matches = 0
    provider_distributions: dict[str, dict[str, Counter[Any]]] = {
        provider: {field: Counter() for field in fields}
        for provider in PROVIDERS
    }
    consensus_distributions = {field: Counter() for field in fields}
    stratum_stats: dict[str, Counter[str]] = defaultdict(Counter)
    exports: list[dict[str, Any]] = []
    disagreement_examples: list[dict[str, Any]] = []

    for row in manifest:
        item_id = row["item_id"]
        item_labels = labels.get(item_id, {})
        missing = set(PROVIDERS) - set(item_labels)
        if missing:
            raise ValueError(f"{task}/{item_id}: missing providers {sorted(missing)}")
        anthropic = item_labels["anthropic"]
        openai = item_labels["openai"]
        matches = {
            field: anthropic[field] == openai[field] for field in fields
        }
        all_match = all(matches.values())
        complete_matches += int(all_match)
        for field, matches_field in matches.items():
            field_matches[field] += int(matches_field)
            provider_distributions["anthropic"][field][anthropic[field]] += 1
            provider_distributions["openai"][field][openai[field]] += 1
            if matches_field:
                consensus_distributions[field][anthropic[field]] += 1

        stratum = row["sampling_context"]["source_stratum"]
        stratum_stats[stratum]["items"] += 1
        stratum_stats[stratum]["complete_agreement"] += int(all_match)
        field_consensus = {
            field: anthropic[field] if matches[field] else None
            for field in fields
        }
        consensus = (
            {field: anthropic[field] for field in fields} if all_match else None
        )
        exported = {
            "item_id": item_id,
            "task": row["task"],
            "complete_agreement": all_match,
            "field_agreement": matches,
            "field_consensus": field_consensus,
            "consensus": consensus,
            "anthropic": anthropic,
            "openai": openai,
            "sampling_context": row["sampling_context"],
        }
        exports.append(exported)
        if not all_match and len(disagreement_examples) < 20:
            disagreement_examples.append(exported)

    count = len(manifest)
    summary = {
        "items": count,
        "complete_agreement_count": complete_matches,
        "complete_agreement_rate": round(complete_matches / count, 4),
        "field_agreement": {
            field: {
                "count": field_matches[field],
                "rate": round(field_matches[field] / count, 4),
            }
            for field in fields
        },
        "provider_distributions": {
            provider: {
                field: dict(sorted(counter.items(), key=lambda pair: str(pair[0])))
                for field, counter in field_counters.items()
            }
            for provider, field_counters in provider_distributions.items()
        },
        "consensus_distributions": {
            field: dict(sorted(counter.items(), key=lambda pair: str(pair[0])))
            for field, counter in consensus_distributions.items()
        },
        "agreement_by_source_stratum": {
            stratum: {
                "items": counts["items"],
                "complete_agreement_count": counts["complete_agreement"],
                "complete_agreement_rate": round(
                    counts["complete_agreement"] / counts["items"], 4
                ),
            }
            for stratum, counts in sorted(stratum_stats.items())
        },
        "disagreement_examples": disagreement_examples,
    }
    return exports, summary


def export_all(cache: Path, output_dir: Path, dataset: str = "pilot") -> dict[str, Any]:
    connection = sqlite3.connect(cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_files = DATASET_TASK_FILES[dataset]
    summary: dict[str, Any] = {
        "dataset": dataset,
        "models": {
            provider: config["model"] for provider, config in PROVIDERS.items()
        },
        "tasks": {},
        "api_usage": call_diagnostics(connection),
    }
    for task, path in task_files.items():
        manifest = read_jsonl(path)
        labels = load_labels(connection, task)
        exports, diagnostics = task_summary(task, manifest, labels)
        write_jsonl(output_dir / f"{task}_consensus.jsonl", exports)
        summary["tasks"][task] = diagnostics
    total_cost = sum(
        provider["cost_usd"]
        for task in summary["api_usage"].values()
        for provider in task.values()
    )
    summary["total_cost_usd"] = round(total_cost, 6)
    summary_path = output_dir / "consensus_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    connection.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_TASK_FILES), default="pilot"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_all(args.cache, args.output_dir, args.dataset)
    compact = {
        "total_cost_usd": summary["total_cost_usd"],
        "task_agreement": {
            task: {
                "complete_agreement_rate": diagnostics[
                    "complete_agreement_rate"
                ],
                "field_agreement": diagnostics["field_agreement"],
            }
            for task, diagnostics in summary["tasks"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
