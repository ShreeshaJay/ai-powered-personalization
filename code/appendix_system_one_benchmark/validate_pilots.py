"""Validate generated pilot manifests without invoking any model APIs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PILOT_DIR = ROOT / "pilots"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def validate_common(rows: list[dict[str, Any]], task: str) -> None:
    if len(rows) != 100:
        raise ValueError(f"{task}: expected 100 rows, found {len(rows)}")
    item_ids = [str(row["item_id"]) for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"{task}: duplicate item IDs")
    for row in rows:
        if row["task"] != task:
            raise ValueError(f"{row['item_id']}: unexpected task {row['task']}")
        if row["rubric_version"] != "v1":
            raise ValueError(f"{row['item_id']}: unexpected rubric version")
        if row["adjudication_input"]["item_id"] != row["item_id"]:
            raise ValueError(f"{row['item_id']}: nested item ID mismatch")
        if "sampling_context" in row["adjudication_input"]:
            raise ValueError(f"{row['item_id']}: leaked sampling context")


def validate_compatibility(rows: list[dict[str, Any]]) -> None:
    validate_common(rows, "product_accessory_compatibility")
    for row in rows:
        inputs = row["adjudication_input"]
        anchor = inputs["anchor_product"]
        candidate = inputs["candidate_product"]
        if anchor["product_id"] == candidate["product_id"]:
            raise ValueError(f"{row['item_id']}: anchor equals candidate")
        if not anchor["title"] or not candidate["title"]:
            raise ValueError(f"{row['item_id']}: missing product title")


def validate_intent(rows: list[dict[str, Any]]) -> None:
    validate_common(rows, "commerce_query_segmentation")
    for row in rows:
        if not str(row["adjudication_input"]["query"]).strip():
            raise ValueError(f"{row['item_id']}: empty query")


def validate_candidates(
    item_id: str, candidates: list[dict[str, str]], expected_count: int
) -> None:
    if len(candidates) != expected_count:
        raise ValueError(
            f"{item_id}: expected {expected_count} candidates, found {len(candidates)}"
        )
    ids = [candidate["id"] for candidate in candidates]
    values = [candidate["value"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{item_id}: duplicate candidate IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{item_id}: duplicate candidate values")


def validate_brand_category(rows: list[dict[str, Any]]) -> None:
    validate_common(rows, "query_to_brand_category")
    for row in rows:
        inputs = row["adjudication_input"]
        validate_candidates(row["item_id"], inputs["brand_candidates"], 9)
        validate_candidates(row["item_id"], inputs["category_candidates"], 9)
        if inputs["brand_candidates"][-1]["id"] != "none_or_other":
            raise ValueError(f"{row['item_id']}: missing brand escape candidate")
        if inputs["category_candidates"][-1]["id"] != "other_or_ambiguous":
            raise ValueError(f"{row['item_id']}: missing category escape candidate")

        weak_path = row["sampling_context"]["matched_category_path_weak"]
        offered_paths = {
            candidate["value"] for candidate in inputs["category_candidates"]
        }
        if weak_path and weak_path not in offered_paths:
            raise ValueError(f"{row['item_id']}: weak category absent from candidates")

        explicit_brands = row["sampling_context"]["explicit_brand_matches_weak"]
        offered_brands = {
            candidate["value"] for candidate in inputs["brand_candidates"]
        }
        if any(brand not in offered_brands for brand in explicit_brands):
            raise ValueError(f"{row['item_id']}: explicit brand absent from candidates")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    serialized_lengths = [
        len(json.dumps(row["adjudication_input"], ensure_ascii=False))
        for row in rows
    ]
    return {
        "rows": len(rows),
        "source_strata": dict(
            sorted(
                Counter(
                    row["sampling_context"]["source_stratum"] for row in rows
                ).items()
            )
        ),
        "adjudication_input_chars": {
            "min": min(serialized_lengths),
            "mean": round(sum(serialized_lengths) / len(serialized_lengths), 1),
            "max": max(serialized_lengths),
        },
    }


def validate_all(pilot_dir: Path) -> dict[str, Any]:
    paths = {
        "compatibility": pilot_dir / "compatibility_pilot_100.jsonl",
        "query_segmentation": pilot_dir / "query_segmentation_pilot_100.jsonl",
        "brand_category": pilot_dir / "brand_category_pilot_100.jsonl",
    }
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    validate_compatibility(rows["compatibility"])
    validate_intent(rows["query_segmentation"])
    validate_brand_category(rows["brand_category"])
    return {
        "valid": True,
        "datasets": {name: summarize(data) for name, data in rows.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_all(args.pilot_dir)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
