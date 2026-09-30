"""Build separate 2,000-item reference manifests after rubric development."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from build_pilots import (
    ROOT,
    bounded_candidates,
    category_candidates,
    choose_anchor,
    clean,
    compact_product,
    exact_anchor_map,
    exact_brand_map,
    explicit_brand_matches,
    read_inputs,
)


DEFAULT_OUTPUT_DIR = ROOT / "references"
ORCAS_GOLD_PATH = ROOT / "data" / "orcas_i" / "ORCAS-I-gold.tsv"
PILOT_DIR = ROOT / "pilots"
SEED = 20260926


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def sample_rows(
    frame: pd.DataFrame,
    count: int,
    rng: random.Random,
    *,
    exclusion: set[Any],
    key,
) -> pd.DataFrame:
    indices = list(frame.index)
    rng.shuffle(indices)
    selected: list[int] = []
    seen: set[Any] = set()
    for index in indices:
        row = frame.loc[index]
        row_key = key(row)
        if row_key in exclusion or row_key in seen:
            continue
        selected.append(index)
        seen.add(row_key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} rows, selected {len(selected)} from {len(frame)}"
        )
    return frame.loc[selected].copy()


def pilot_compatibility_exclusions() -> set[tuple[int, str, str]]:
    rows = read_jsonl(PILOT_DIR / "compatibility_pilot_100.jsonl")
    return {
        (
            int(row["sampling_context"]["query_id"]),
            row["adjudication_input"]["anchor_product"]["product_id"],
            row["adjudication_input"]["candidate_product"]["product_id"],
        )
        for row in rows
    }


def build_compatibility_reference(
    examples: pd.DataFrame,
    products: pd.DataFrame,
    augmented: pd.DataFrame,
    rng: random.Random,
) -> list[dict[str, Any]]:
    anchors = exact_anchor_map(examples)
    pool = augmented[augmented["query_id"].isin(anchors)].copy()
    pool = pool[pool["product_id"].isin(products.index)].copy()
    pool["anchor_id"] = [
        choose_anchor(int(row.query_id), str(row.product_id), anchors)
        for row in pool.itertuples()
    ]
    pool = pool[pool["anchor_id"].notna()].copy()
    exclusions = pilot_compatibility_exclusions()
    pair_key = lambda row: (
        int(row["query_id"]),
        str(row["anchor_id"]),
        str(row["product_id"]),
    )
    strata = [
        (
            "generated_direct_accessory",
            pool[
                (pool["candidate_source"] == "complement_candidate")
                & (pool["relationship_type"] == "accessory")
            ],
            500,
        ),
        (
            "generated_cross_sell",
            pool[
                (pool["candidate_source"] == "complement_candidate")
                & (pool["relationship_type"] == "cross_sell")
            ],
            500,
        ),
        (
            "out_of_category_negative",
            pool[pool["candidate_source"] == "out_of_category_negative"],
            420,
        ),
        (
            "human_substitute_hard_negative",
            pool[
                (pool["candidate_source"] == "retrieval_top_k")
                & (pool["human_esci_label"] == "S")
            ],
            400,
        ),
        (
            "human_irrelevant_negative",
            pool[
                (pool["candidate_source"] == "retrieval_top_k")
                & (pool["human_esci_label"] == "I")
            ],
            180,
        ),
    ]
    selected: list[tuple[str, pd.Series]] = []
    for stratum, candidates, count in strata:
        sampled = sample_rows(
            candidates, count, rng, exclusion=exclusions, key=pair_key
        )
        selected.extend((stratum, row) for _, row in sampled.iterrows())
    rng.shuffle(selected)

    output: list[dict[str, Any]] = []
    for position, (stratum, row) in enumerate(selected, start=1):
        query_id = int(row["query_id"])
        candidate_id = str(row["product_id"])
        anchor_id = str(row["anchor_id"])
        item_id = f"compat_ref_{position:05d}"
        output.append(
            {
                "item_id": item_id,
                "task": "product_accessory_compatibility",
                "rubric_version": "v1",
                "split": "reference",
                "adjudication_input": {
                    "item_id": item_id,
                    "query_context": clean(row["query"], 300),
                    "anchor_product": compact_product(anchor_id, products),
                    "candidate_product": compact_product(candidate_id, products),
                },
                "sampling_context": {
                    "source_stratum": stratum,
                    "query_id": query_id,
                    "candidate_source": clean(row["candidate_source"]),
                    "relationship_type_weak": clean(row["relationship_type"]),
                    "complement_product_type_weak": clean(
                        row["complement_product_type"]
                    ),
                    "human_esci_label_if_available": clean(row["human_esci_label"]),
                },
            }
        )
    return output


def pilot_query_exclusions() -> set[str]:
    rows = read_jsonl(PILOT_DIR / "query_segmentation_pilot_100.jsonl")
    return {
        clean(row["adjudication_input"]["query"]).casefold()
        for row in rows
    }


def build_query_reference(
    taxonomy: pd.DataFrame, rng: random.Random
) -> list[dict[str, Any]]:
    exclusions = pilot_query_exclusions()
    taxonomy = taxonomy.copy()
    taxonomy["query_key"] = taxonomy["query"].map(lambda value: clean(value).casefold())
    taxonomy = taxonomy[~taxonomy["query_key"].isin(exclusions)].copy()
    strata = [
        (
            "esci_single_product",
            taxonomy[taxonomy["product_type_intent"] == "single_product"],
            550,
        ),
        (
            "esci_accessory_for_product",
            taxonomy[taxonomy["product_type_intent"] == "accessory_for_product"],
            120,
        ),
        (
            "esci_bundle_or_set",
            taxonomy[taxonomy["product_type_intent"] == "bundle_or_set"],
            100,
        ),
        (
            "esci_broad_gift_intent",
            taxonomy[taxonomy["product_type_intent"] == "broad_gift_intent"],
            80,
        ),
        (
            "esci_ambiguous",
            taxonomy[taxonomy["product_type_intent"] == "ambiguous"],
            75,
        ),
        (
            "esci_unknown",
            taxonomy[taxonomy["product_type_intent"] == "unknown"],
            75,
        ),
    ]
    selected: list[tuple[str, pd.Series]] = []
    used_queries: set[str] = set(exclusions)
    for stratum, candidates, count in strata:
        sampled = sample_rows(
            candidates,
            count,
            rng,
            exclusion=used_queries,
            key=lambda row: str(row["query_key"]),
        )
        used_queries.update(sampled["query_key"])
        selected.extend((stratum, row) for _, row in sampled.iterrows())

    orcas = pd.read_csv(ORCAS_GOLD_PATH, sep="\t")
    orcas["query_key"] = orcas["query"].map(lambda value: clean(value).casefold())
    orcas = orcas[~orcas["query_key"].isin(used_queries)].copy()
    orcas = sample_rows(
        orcas,
        1000,
        rng,
        exclusion=used_queries,
        key=lambda row: str(row["query_key"]),
    )

    output: list[dict[str, Any]] = []
    for stratum, row in selected:
        output.append(
            {
                "source": "esci",
                "source_stratum": stratum,
                "query": clean(row["query"], 400),
                "source_id": str(int(row["query_id"])),
                "weak_context": {
                    "product_type_intent_weak": clean(row["product_type_intent"]),
                    "anchor_product_type_weak": clean(row["anchor_product_type"]),
                },
            }
        )
    for _, row in orcas.iterrows():
        output.append(
            {
                "source": "orcas_i_gold",
                "source_stratum": f"orcas_{clean(row['label_manual']).lower()}",
                "query": clean(row["query"], 400),
                "source_id": str(row["qid"]),
                "weak_context": {
                    "orcas_manual_label": clean(row["label_manual"]),
                    "clicked_url": clean(row["url"], 500),
                },
            }
        )
    rng.shuffle(output)

    rows: list[dict[str, Any]] = []
    for position, item in enumerate(output, start=1):
        item_id = f"intent_ref_{position:05d}"
        rows.append(
            {
                "item_id": item_id,
                "task": "commerce_query_segmentation",
                "rubric_version": "v3",
                "split": "reference",
                "adjudication_input": {
                    "item_id": item_id,
                    "query": item["query"],
                },
                "sampling_context": {
                    "source": item["source"],
                    "source_stratum": item["source_stratum"],
                    "source_id": item["source_id"],
                    **item["weak_context"],
                },
            }
        )
    return rows


def pilot_brand_query_ids() -> set[int]:
    rows = read_jsonl(PILOT_DIR / "brand_category_pilot_100.jsonl")
    return {int(row["sampling_context"]["query_id"]) for row in rows}


def build_brand_category_reference(
    examples: pd.DataFrame,
    products: pd.DataFrame,
    taxonomy: pd.DataFrame,
    rng: random.Random,
) -> list[dict[str, Any]]:
    brand_map, brand_pool = exact_brand_map(examples, products)
    frame = taxonomy.copy()
    frame["query_id"] = frame["query_id"].astype(int)
    frame = frame[~frame["query_id"].isin(pilot_brand_query_ids())].copy()
    frame["exact_brands"] = frame["query_id"].map(
        lambda query_id: brand_map.get(int(query_id), [])
    )
    frame["explicit_brands"] = [
        explicit_brand_matches(str(query), brands)
        for query, brands in zip(frame["query"], frame["exact_brands"], strict=True)
    ]
    frame["has_category"] = (
        frame["matched_category_path"].fillna("").astype(str).str.strip() != ""
    )
    frame["has_explicit_brand"] = frame["explicit_brands"].map(bool)
    ambiguous_mask = (
        ~frame["has_category"]
        | frame["product_type_intent"].isin(["ambiguous", "unknown"])
    )
    strata = [
        (
            "explicit_brand_with_category",
            frame[frame["has_explicit_brand"] & frame["has_category"]],
            700,
        ),
        (
            "generic_query_with_category",
            frame[~frame["has_explicit_brand"] & frame["has_category"]],
            900,
        ),
        ("ambiguous_or_unmapped", frame[ambiguous_mask], 400),
    ]
    selected: list[tuple[str, pd.Series]] = []
    used_ids: set[int] = set()
    for stratum, candidates, count in strata:
        sampled = sample_rows(
            candidates,
            count,
            rng,
            exclusion=used_ids,
            key=lambda row: int(row["query_id"]),
        )
        used_ids.update(sampled["query_id"].astype(int))
        selected.extend((stratum, row) for _, row in sampled.iterrows())

    category_pool = sorted(
        {
            clean(path)
            for path in frame["matched_category_path"].dropna()
            if clean(path)
        }
    )
    rng.shuffle(selected)
    output: list[dict[str, Any]] = []
    for position, (stratum, row) in enumerate(selected, start=1):
        item_id = f"brand_category_ref_{position:05d}"
        exact_brands = list(row["exact_brands"])
        explicit_brands = list(row["explicit_brands"])
        target_path = clean(row["matched_category_path"])
        output.append(
            {
                "item_id": item_id,
                "task": "query_to_brand_category",
                "rubric_version": "v1",
                "split": "reference",
                "adjudication_input": {
                    "item_id": item_id,
                    "query": clean(row["query"], 400),
                    "brand_candidates": bounded_candidates(
                        explicit_brands + exact_brands[:3],
                        brand_pool,
                        8,
                        rng,
                        "brand",
                    )
                    + [{"id": "none_or_other", "value": "none or other brand"}],
                    "category_candidates": category_candidates(
                        target_path, category_pool, 8, rng
                    )
                    + [
                        {
                            "id": "other_or_ambiguous",
                            "value": "other, multiple, or ambiguous category",
                        }
                    ],
                },
                "sampling_context": {
                    "source_stratum": stratum,
                    "query_id": int(row["query_id"]),
                    "exact_product_brands_weak": exact_brands,
                    "explicit_brand_matches_weak": explicit_brands,
                    "matched_category_path_weak": target_path,
                    "product_type_intent_weak": clean(row["product_type_intent"]),
                },
            }
        )
    return output


def strata(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(row["sampling_context"]["source_stratum"] for row in rows).items())
    )


def build_all(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    print("Loading ESCI and enrichment inputs...", flush=True)
    examples, products, augmented, taxonomy = read_inputs()
    print("Building compatibility reference set...", flush=True)
    compatibility = build_compatibility_reference(
        examples, products, augmented, random.Random(seed + 1)
    )
    print("Building query segmentation reference set...", flush=True)
    query_segmentation = build_query_reference(taxonomy, random.Random(seed + 2))
    print("Building brand/category reference set...", flush=True)
    brand_category = build_brand_category_reference(
        examples, products, taxonomy, random.Random(seed + 3)
    )
    datasets = {
        "compatibility": compatibility,
        "query_segmentation": query_segmentation,
        "brand_category": brand_category,
    }
    summary: dict[str, Any] = {
        "seed": seed,
        "development_pilots_excluded": True,
        "datasets": {},
    }
    for name, rows in datasets.items():
        path = output_dir / f"{name}_reference_2000.jsonl"
        summary["datasets"][name] = {
            "path": str(path),
            "rows": len(rows),
            "source_strata": strata(rows),
            "sha256": write_jsonl(path, rows),
        }
    summary_path = output_dir / "reference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_all(args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
