"""Build deterministic 100-item pilot manifests for three adjudicated tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
APPENDIX_ROOT = ROOT.parent
WORKSPACE_ROOT = ROOT.parents[2]
ESCI_ROOT = WORKSPACE_ROOT / "Dataset" / "Amazon ESCI" / "shopping_queries_dataset"
ENRICHMENT_ROOT = APPENDIX_ROOT / "esci_llm_enrichment"
DEFAULT_OUTPUT_DIR = ROOT / "pilots"
SEED = 42

EXAMPLES_PATH = ESCI_ROOT / "shopping_queries_dataset_examples.parquet"
PRODUCTS_PATH = ESCI_ROOT / "shopping_queries_dataset_products.parquet"
AUGMENTED_PATH = ENRICHMENT_ROOT / "outputs" / "augmented_candidates.jsonl"
TAXONOMY_PATH = (
    ENRICHMENT_ROOT / "outputs" / "query_taxonomy_with_anchor_product_type.csv"
)

NAVIGATION_PATTERN = re.compile(
    r"(?i)(?:\b(?:store|website|official|login|customer service|near me)\b|\.com)"
)
EMPTY_VALUES = {"", "nan", "none", "null"}
GENERIC_BRANDS = {"generic", "unknown", "unbranded", "brand"}


def clean(value: Any, limit: int | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit] if limit else text


def normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def stable_sample(
    frame: pd.DataFrame,
    count: int,
    rng: random.Random,
    used_query_ids: set[int] | None = None,
) -> pd.DataFrame:
    if count == 0:
        return frame.iloc[0:0].copy()
    used_query_ids = used_query_ids if used_query_ids is not None else set()
    indices = list(frame.index)
    rng.shuffle(indices)
    selected: list[int] = []
    for index in indices:
        query_id = int(frame.at[index, "query_id"])
        if query_id in used_query_ids:
            continue
        selected.append(index)
        used_query_ids.add(query_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} unique queries but selected {len(selected)} "
            f"from {len(frame)} rows."
        )
    return frame.loc[selected].copy()


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    examples = pd.read_parquet(
        EXAMPLES_PATH,
        columns=[
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "split",
        ],
        filters=[("product_locale", "==", "us"), ("split", "==", "test")],
    )
    examples = examples[
        (examples["product_locale"] == "us") & (examples["split"] == "test")
    ].copy()

    products = pd.read_parquet(
        PRODUCTS_PATH,
        columns=[
            "product_id",
            "product_title",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        ],
        filters=[("product_locale", "==", "us")],
    )
    products = (
        products[products["product_locale"] == "us"]
        .drop_duplicates("product_id")
        .set_index("product_id", drop=False)
    )
    augmented = pd.read_json(AUGMENTED_PATH, lines=True)
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    return examples, products, augmented, taxonomy


def compact_product(product_id: str, products: pd.DataFrame) -> dict[str, str]:
    if product_id not in products.index:
        return {
            "product_id": product_id,
            "title": "",
            "brand": "",
            "color": "",
            "bullet_point": "",
        }
    row = products.loc[product_id]
    return {
        "product_id": product_id,
        "title": clean(row["product_title"], 500),
        "brand": clean(row["product_brand"], 120),
        "color": clean(row["product_color"], 120),
        "bullet_point": clean(row["product_bullet_point"], 700),
    }


def exact_anchor_map(examples: pd.DataFrame) -> dict[int, list[str]]:
    exact = examples[examples["esci_label"] == "E"].sort_values(
        ["query_id", "product_id"]
    )
    return {
        int(query_id): list(dict.fromkeys(group["product_id"].astype(str)))
        for query_id, group in exact.groupby("query_id", sort=False)
    }


def choose_anchor(
    query_id: int, candidate_id: str, anchors: dict[int, list[str]]
) -> str | None:
    for product_id in anchors.get(query_id, []):
        if product_id != candidate_id:
            return product_id
    return None


def build_compatibility(
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

    strata = [
        (
            "generated_direct_accessory",
            pool[
                (pool["candidate_source"] == "complement_candidate")
                & (pool["relationship_type"] == "accessory")
            ],
            40,
        ),
        (
            "generated_cross_sell",
            pool[
                (pool["candidate_source"] == "complement_candidate")
                & (pool["relationship_type"] == "cross_sell")
            ],
            20,
        ),
        (
            "out_of_category_negative",
            pool[pool["candidate_source"] == "out_of_category_negative"],
            20,
        ),
        (
            "human_substitute_hard_negative",
            pool[
                (pool["candidate_source"] == "retrieval_top_k")
                & (pool["human_esci_label"] == "S")
            ],
            10,
        ),
        (
            "human_irrelevant_negative",
            pool[
                (pool["candidate_source"] == "retrieval_top_k")
                & (pool["human_esci_label"] == "I")
            ],
            10,
        ),
    ]

    selected_rows: list[tuple[str, pd.Series]] = []
    used_query_ids: set[int] = set()
    for stratum, frame, count in strata:
        sampled = stable_sample(frame, count, rng, used_query_ids)
        selected_rows.extend((stratum, row) for _, row in sampled.iterrows())

    rng.shuffle(selected_rows)
    output: list[dict[str, Any]] = []
    for position, (stratum, row) in enumerate(selected_rows, start=1):
        query_id = int(row["query_id"])
        candidate_id = str(row["product_id"])
        anchor_id = str(row["anchor_id"])
        item_id = f"compat_{position:04d}"
        output.append(
            {
                "item_id": item_id,
                "task": "product_accessory_compatibility",
                "rubric_version": "v1",
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


def build_query_segmentation(
    taxonomy: pd.DataFrame, rng: random.Random
) -> list[dict[str, Any]]:
    taxonomy = taxonomy.copy()
    taxonomy["query_id"] = taxonomy["query_id"].astype(int)
    navigation = taxonomy[
        taxonomy["query"].fillna("").map(lambda value: bool(NAVIGATION_PATTERN.search(value)))
    ]
    strata = [
        ("navigation_probe", navigation, 10),
        (
            "single_product",
            taxonomy[taxonomy["product_type_intent"] == "single_product"],
            35,
        ),
        (
            "accessory_for_product",
            taxonomy[taxonomy["product_type_intent"] == "accessory_for_product"],
            15,
        ),
        (
            "bundle_or_set",
            taxonomy[taxonomy["product_type_intent"] == "bundle_or_set"],
            10,
        ),
        (
            "broad_gift_intent",
            taxonomy[taxonomy["product_type_intent"] == "broad_gift_intent"],
            10,
        ),
        (
            "ambiguous",
            taxonomy[taxonomy["product_type_intent"] == "ambiguous"],
            10,
        ),
        (
            "unknown",
            taxonomy[taxonomy["product_type_intent"] == "unknown"],
            10,
        ),
    ]
    used_query_ids: set[int] = set()
    selected_rows: list[tuple[str, pd.Series]] = []
    for stratum, frame, count in strata:
        sampled = stable_sample(frame, count, rng, used_query_ids)
        selected_rows.extend((stratum, row) for _, row in sampled.iterrows())

    rng.shuffle(selected_rows)
    output: list[dict[str, Any]] = []
    for position, (stratum, row) in enumerate(selected_rows, start=1):
        item_id = f"intent_{position:04d}"
        output.append(
            {
                "item_id": item_id,
                "task": "commerce_query_segmentation",
                "rubric_version": "v1",
                "adjudication_input": {
                    "item_id": item_id,
                    "query": clean(row["query"], 400),
                },
                "sampling_context": {
                    "source_stratum": stratum,
                    "query_id": int(row["query_id"]),
                    "product_type_intent_weak": clean(row["product_type_intent"]),
                    "anchor_product_type_weak": clean(row["anchor_product_type"]),
                    "taxonomy_confidence_weak": clean(
                        row["taxonomy_confidence_level"]
                    ),
                },
            }
        )
    return output


def exact_brand_map(
    examples: pd.DataFrame, products: pd.DataFrame
) -> tuple[dict[int, list[str]], list[str]]:
    exact = examples[examples["esci_label"] == "E"][
        ["query_id", "product_id"]
    ].copy()
    brand_lookup = products["product_brand"].to_dict()
    exact["brand"] = exact["product_id"].map(brand_lookup).map(clean)
    exact = exact[
        ~exact["brand"].str.lower().isin(GENERIC_BRANDS)
        & ~exact["brand"].str.lower().isin(EMPTY_VALUES)
    ]
    mapping = {
        int(query_id): list(dict.fromkeys(group["brand"]))[:20]
        for query_id, group in exact.groupby("query_id", sort=False)
    }
    counts = Counter(exact["brand"])
    pool = [brand for brand, _ in counts.most_common() if clean(brand)]
    return mapping, pool


def explicit_brand_matches(query: str, brands: Iterable[str]) -> list[str]:
    query_normalized = f" {normalized(query)} "
    matches = []
    for brand in brands:
        brand_normalized = normalized(brand)
        if len(brand_normalized) >= 3 and f" {brand_normalized} " in query_normalized:
            matches.append(brand)
    return matches


def bounded_candidates(
    targets: list[str],
    pool: list[str],
    count: int,
    rng: random.Random,
    prefix: str,
) -> list[dict[str, str]]:
    values = list(dict.fromkeys(clean(value) for value in targets if clean(value)))
    seen = set(values)
    needed = max(0, count - len(values))
    attempts = 0
    max_attempts = max(100, needed * 50)
    while len(values) < count and pool and attempts < max_attempts:
        candidate = clean(pool[rng.randrange(len(pool))])
        attempts += 1
        if candidate and candidate not in seen:
            values.append(candidate)
            seen.add(candidate)
    if len(values) < count:
        for candidate_raw in pool:
            candidate = clean(candidate_raw)
            if candidate and candidate not in seen:
                values.append(candidate)
                seen.add(candidate)
                if len(values) == count:
                    break
    if len(values) < count:
        raise ValueError(f"Insufficient unique candidates for {prefix}")
    values = values[:count]
    rng.shuffle(values)
    return [
        {"id": f"{prefix}_{index}", "value": value}
        for index, value in enumerate(values)
    ]


def category_candidates(
    target: str, category_pool: list[str], count: int, rng: random.Random
) -> list[dict[str, str]]:
    target = clean(target)
    same_root: list[str] = []
    if target:
        root = target.split(" > ", 1)[0]
        same_root = [
            path
            for path in category_pool
            if path != target and path.split(" > ", 1)[0] == root
        ]
        rng.shuffle(same_root)
    initial = ([target] if target else []) + same_root[:4]
    return bounded_candidates(initial, category_pool, count, rng, "category")


def build_brand_category(
    examples: pd.DataFrame,
    products: pd.DataFrame,
    taxonomy: pd.DataFrame,
    rng: random.Random,
) -> list[dict[str, Any]]:
    brand_map, brand_pool = exact_brand_map(examples, products)
    frame = taxonomy.copy()
    frame["query_id"] = frame["query_id"].astype(int)
    frame["exact_brands"] = frame["query_id"].map(
        lambda query_id: brand_map.get(int(query_id), [])
    )
    frame["explicit_brands"] = [
        explicit_brand_matches(str(query), brands)
        for query, brands in zip(
            frame["query"], frame["exact_brands"], strict=True
        )
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
            40,
        ),
        (
            "generic_query_with_category",
            frame[~frame["has_explicit_brand"] & frame["has_category"]],
            40,
        ),
        ("ambiguous_or_unmapped", frame[ambiguous_mask], 20),
    ]
    used_query_ids: set[int] = set()
    selected_rows: list[tuple[str, pd.Series]] = []
    for stratum, candidates, count in strata:
        sampled = stable_sample(candidates, count, rng, used_query_ids)
        selected_rows.extend((stratum, row) for _, row in sampled.iterrows())

    category_pool = sorted(
        {
            clean(path)
            for path in frame["matched_category_path"].dropna()
            if clean(path)
        }
    )
    rng.shuffle(selected_rows)
    output: list[dict[str, Any]] = []
    for position, (stratum, row) in enumerate(selected_rows, start=1):
        item_id = f"brand_category_{position:04d}"
        exact_brands = list(row["exact_brands"])
        explicit_brands = list(row["explicit_brands"])
        target_path = clean(row["matched_category_path"])
        output.append(
            {
                "item_id": item_id,
                "task": "query_to_brand_category",
                "rubric_version": "v1",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def source_strata(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(row["sampling_context"]["source_stratum"]) for row in rows
            ).items()
        )
    )


def build_all(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    examples, products, augmented, taxonomy = read_inputs()
    compatibility = build_compatibility(
        examples, products, augmented, random.Random(seed + 1)
    )
    intent = build_query_segmentation(taxonomy, random.Random(seed + 2))
    brand_category = build_brand_category(
        examples, products, taxonomy, random.Random(seed + 3)
    )
    datasets = {
        "compatibility": compatibility,
        "query_segmentation": intent,
        "brand_category": brand_category,
    }
    summary: dict[str, Any] = {
        "seed": seed,
        "rubric_version": "v1",
        "source_files": {
            "examples": str(EXAMPLES_PATH),
            "products": str(PRODUCTS_PATH),
            "augmented_candidates": str(AUGMENTED_PATH),
            "query_taxonomy": str(TAXONOMY_PATH),
        },
        "datasets": {},
    }
    for name, rows in datasets.items():
        path = output_dir / f"{name}_pilot_100.jsonl"
        summary["datasets"][name] = {
            "path": str(path),
            "rows": len(rows),
            "source_strata": source_strata(rows),
            "sha256": write_jsonl(path, rows),
        }
    summary_path = output_dir / "pilot_summary.json"
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
    summary = build_all(args.output_dir, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
