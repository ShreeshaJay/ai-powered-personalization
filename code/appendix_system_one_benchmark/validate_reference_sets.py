"""Validate 2,000-item reference manifests and development-set separation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_pilots import ROOT, clean
from build_reference_sets import PILOT_DIR, read_jsonl


DEFAULT_REFERENCE_DIR = ROOT / "references"


def require_common(rows: list[dict[str, Any]], task: str) -> None:
    if len(rows) != 2000:
        raise ValueError(f"{task}: expected 2,000 rows, found {len(rows)}")
    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"{task}: duplicate item IDs")
    for row in rows:
        if row["task"] != task or row["split"] != "reference":
            raise ValueError(f"{row['item_id']}: invalid task or split")
        if row["adjudication_input"]["item_id"] != row["item_id"]:
            raise ValueError(f"{row['item_id']}: nested item ID mismatch")


def validate_compatibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require_common(rows, "product_accessory_compatibility")
    pilot_rows = read_jsonl(PILOT_DIR / "compatibility_pilot_100.jsonl")
    pilot_pairs = {
        (
            int(row["sampling_context"]["query_id"]),
            row["adjudication_input"]["anchor_product"]["product_id"],
            row["adjudication_input"]["candidate_product"]["product_id"],
        )
        for row in pilot_rows
    }
    pairs = []
    for row in rows:
        inputs = row["adjudication_input"]
        pair = (
            int(row["sampling_context"]["query_id"]),
            inputs["anchor_product"]["product_id"],
            inputs["candidate_product"]["product_id"],
        )
        if pair[1] == pair[2]:
            raise ValueError(f"{row['item_id']}: anchor equals candidate")
        if not inputs["anchor_product"]["title"] or not inputs["candidate_product"]["title"]:
            raise ValueError(f"{row['item_id']}: missing product title")
        pairs.append(pair)
    if len(pairs) != len(set(pairs)):
        raise ValueError("compatibility: duplicate query-anchor-candidate triples")
    overlap = set(pairs) & pilot_pairs
    if overlap:
        raise ValueError(f"compatibility: {len(overlap)} pilot pair overlaps")
    return {
        "unique_queries": len({pair[0] for pair in pairs}),
        "pilot_pair_overlap": 0,
    }


def validate_intent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require_common(rows, "commerce_query_segmentation")
    pilot_rows = read_jsonl(PILOT_DIR / "query_segmentation_pilot_100.jsonl")
    pilot_queries = {
        clean(row["adjudication_input"]["query"]).casefold()
        for row in pilot_rows
    }
    queries = [
        clean(row["adjudication_input"]["query"]).casefold() for row in rows
    ]
    if len(queries) != len(set(queries)):
        raise ValueError("query segmentation: duplicate normalized queries")
    overlap = set(queries) & pilot_queries
    if overlap:
        raise ValueError(f"query segmentation: {len(overlap)} pilot query overlaps")
    sources = Counter(row["sampling_context"]["source"] for row in rows)
    if sources != Counter({"esci": 1000, "orcas_i_gold": 1000}):
        raise ValueError(f"query segmentation: unexpected source counts {sources}")
    return {
        "unique_normalized_queries": len(set(queries)),
        "pilot_query_overlap": 0,
        "sources": dict(sorted(sources.items())),
    }


def validate_candidates(
    item_id: str, candidates: list[dict[str, str]], escape_id: str
) -> None:
    if len(candidates) != 9:
        raise ValueError(f"{item_id}: expected 9 candidates")
    ids = [candidate["id"] for candidate in candidates]
    values = [candidate["value"] for candidate in candidates]
    if len(ids) != len(set(ids)) or len(values) != len(set(values)):
        raise ValueError(f"{item_id}: duplicate candidates")
    if candidates[-1]["id"] != escape_id:
        raise ValueError(f"{item_id}: missing escape candidate")


def validate_brand_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require_common(rows, "query_to_brand_category")
    pilot_rows = read_jsonl(PILOT_DIR / "brand_category_pilot_100.jsonl")
    pilot_ids = {
        int(row["sampling_context"]["query_id"]) for row in pilot_rows
    }
    query_ids = []
    for row in rows:
        query_id = int(row["sampling_context"]["query_id"])
        query_ids.append(query_id)
        inputs = row["adjudication_input"]
        validate_candidates(row["item_id"], inputs["brand_candidates"], "none_or_other")
        validate_candidates(
            row["item_id"],
            inputs["category_candidates"],
            "other_or_ambiguous",
        )
        category_values = {
            candidate["value"] for candidate in inputs["category_candidates"]
        }
        weak_path = row["sampling_context"]["matched_category_path_weak"]
        if weak_path and weak_path not in category_values:
            raise ValueError(f"{row['item_id']}: weak category absent")
        brand_values = {
            candidate["value"] for candidate in inputs["brand_candidates"]
        }
        if any(
            brand not in brand_values
            for brand in row["sampling_context"]["explicit_brand_matches_weak"]
        ):
            raise ValueError(f"{row['item_id']}: explicit brand absent")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("brand/category: duplicate query IDs")
    overlap = set(query_ids) & pilot_ids
    if overlap:
        raise ValueError(f"brand/category: {len(overlap)} pilot query overlaps")
    return {"unique_queries": len(set(query_ids)), "pilot_query_overlap": 0}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [
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
            "min": min(sizes),
            "mean": round(sum(sizes) / len(sizes), 1),
            "max": max(sizes),
        },
    }


def validate_all(reference_dir: Path) -> dict[str, Any]:
    rows = {
        "compatibility": read_jsonl(reference_dir / "compatibility_reference_2000.jsonl"),
        "query_segmentation": read_jsonl(
            reference_dir / "query_segmentation_reference_2000.jsonl"
        ),
        "brand_category": read_jsonl(
            reference_dir / "brand_category_reference_2000.jsonl"
        ),
    }
    details = {
        "compatibility": validate_compatibility(rows["compatibility"]),
        "query_segmentation": validate_intent(rows["query_segmentation"]),
        "brand_category": validate_brand_category(rows["brand_category"]),
    }
    return {
        "valid": True,
        "datasets": {
            name: {**summarize(data), **details[name]}
            for name, data in rows.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_all(args.reference_dir)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
