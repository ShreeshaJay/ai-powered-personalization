"""Load joined evaluation examples from manifests and consensus labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adjudicate_pilots import DATASET_TASK_FILES, read_jsonl


ROOT = Path(__file__).resolve().parent.parent
CONSENSUS_DIRS = {
    "pilot": ROOT / "outputs" / "consensus",
    "reference": ROOT / "references" / "consensus",
}
ESCI_SLICE_PATH = ROOT / "references" / "esci_eval_slice.jsonl"


def consensus_path(task: str, dataset: str) -> Path:
    return CONSENSUS_DIRS[dataset] / f"{task}_consensus.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_examples(
    task: str,
    dataset: str = "reference",
    require_complete: bool = False,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    if task == "esci":
        rows = load_jsonl(ESCI_SLICE_PATH)
        if max_items:
            rows = rows[:max_items]
        return rows

    manifest = load_jsonl(DATASET_TASK_FILES[dataset][task])
    consensus_rows = {
        row["item_id"]: row for row in load_jsonl(consensus_path(task, dataset))
    }
    examples = []
    for row in manifest:
        item_id = row["item_id"]
        if item_id not in consensus_rows:
            raise ValueError(f"{task}/{item_id}: missing consensus row")
        consensus = consensus_rows[item_id]
        if require_complete and not consensus["complete_agreement"]:
            continue
        examples.append(
            {
                "item_id": item_id,
                "task": task,
                "adjudication_input": row["adjudication_input"],
                "labels": consensus.get("field_consensus") or {},
                "complete_labels": consensus.get("consensus"),
                "complete_agreement": bool(consensus["complete_agreement"]),
                "sampling_context": row.get("sampling_context") or {},
                "label_source": "dual_judge_consensus",
            }
        )
        if max_items and len(examples) >= max_items:
            break
    return examples


def labeled_pairs(
    examples: list[dict[str, Any]],
    predictions: list[Any],
    field: str,
) -> list[tuple[Any, Any, dict[str, float], float, dict[str, Any]]]:
    pred_by_id = {prediction.item_id: prediction for prediction in predictions}
    pairs = []
    for example in examples:
        reference = example["labels"].get(field)
        if reference is None:
            continue
        prediction = pred_by_id[example["item_id"]]
        field_pred = prediction.fields[field]
        pairs.append(
            (
                reference,
                field_pred.predicted,
                field_pred.probabilities,
                field_pred.confidence,
                example,
            )
        )
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
