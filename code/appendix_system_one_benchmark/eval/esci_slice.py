"""Build a stratified ESCI evaluation slice from existing human and Gemini labels."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from adapters.schemas import compact_product
from build_pilots import EXAMPLES_PATH, PRODUCTS_PATH, clean


APPENDIX_ROOT = Path(__file__).resolve().parents[2]
LLM_LABELS_PATH = APPENDIX_ROOT / "outputs" / "metrics" / "llm_labels.json"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "references" / "esci_eval_slice.jsonl"
)
SEED = 42


def _read_human_pairs() -> pd.DataFrame:
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
    examples["product_title"] = examples["product_id"].map(
        lambda product_id: clean(products.at[product_id, "product_title"])
        if product_id in products.index
        else ""
    )
    examples["product_brand"] = examples["product_id"].map(
        lambda product_id: clean(products.at[product_id, "product_brand"])
        if product_id in products.index
        else ""
    )
    examples["product_color"] = examples["product_id"].map(
        lambda product_id: clean(products.at[product_id, "product_color"])
        if product_id in products.index
        else ""
    )
    examples["product_bullet_point"] = examples["product_id"].map(
        lambda product_id: clean(products.at[product_id, "product_bullet_point"])
        if product_id in products.index
        else ""
    )
    return examples


def _sample_frame(
    frame: pd.DataFrame, per_label: int, rng: random.Random
) -> pd.DataFrame:
    selected = []
    for label in ("E", "S", "C", "I"):
        pool = frame[frame["esci_label"] == label]
        if pool.empty:
            continue
        indices = list(pool.index)
        rng.shuffle(indices)
        take = indices[: min(per_label, len(indices))]
        selected.extend(take)
    sampled = frame.loc[selected].copy()
    sampled = sampled.sample(frac=1.0, random_state=SEED)
    return sampled


def _row_to_example(
    row: pd.Series | dict[str, Any],
    item_id: str,
    label_source: str,
) -> dict[str, Any]:
    data = row if isinstance(row, dict) else row.to_dict()
    product = compact_product(
        {
            "title": data.get("product_title", ""),
            "brand": data.get("product_brand", ""),
            "color": data.get("product_color", ""),
            "bullet_point": data.get("product_bullet_point", ""),
        }
    )
    return {
        "item_id": item_id,
        "task": "esci",
        "adjudication_input": {
            "query": clean(data.get("query", "")),
            "product": product,
        },
        "labels": {"esci_label": str(data["esci_label"])},
        "complete_labels": {"esci_label": str(data["esci_label"])},
        "complete_agreement": True,
        "sampling_context": {
            "query_id": int(data["query_id"]),
            "product_id": str(data["product_id"]),
            "source_stratum": label_source,
        },
        "label_source": label_source,
    }


def _iter_synthetic_pairs(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        store = json.load(handle)
    for query_id, products in store.get("labels", {}).items():
        for product_id, payload in products.items():
            label = str(payload.get("label", "")).upper()
            if label in {"E", "S", "C", "I"}:
                yield str(query_id), str(product_id), label


def build_esci_slice(
    human_per_label: int = 2000,
    synthetic_per_label: int = 0,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    rng = random.Random(SEED)
    human = _read_human_pairs()
    human_sample = _sample_frame(human, human_per_label, rng)
    examples = []
    for index, (_, row) in enumerate(human_sample.iterrows(), start=1):
        examples.append(
            _row_to_example(
                row,
                f"esci_human_{index:05d}",
                "human",
            )
        )

    synthetic_counts: Counter[str] = Counter()
    if synthetic_per_label > 0:
        human_keys = {
            (int(row["query_id"]), str(row["product_id"]))
            for _, row in human_sample.iterrows()
        }
        query_text = (
            human.drop_duplicates("query_id")
            .set_index("query_id")["query"]
            .to_dict()
        )
        products = (
            human.drop_duplicates("product_id")
            .set_index("product_id")[
                [
                    "product_title",
                    "product_brand",
                    "product_color",
                    "product_bullet_point",
                ]
            ]
            .to_dict("index")
        )
        buckets: dict[str, list[dict[str, Any]]] = {
            label: [] for label in ("E", "S", "C", "I")
        }
        for query_id, product_id, label in _iter_synthetic_pairs(LLM_LABELS_PATH):
            key = (int(query_id), product_id)
            if key in human_keys:
                continue
            if query_id not in query_text and int(query_id) not in query_text:
                continue
            meta = products.get(product_id)
            if not meta:
                continue
            buckets[label].append(
                {
                    "query_id": int(query_id),
                    "query": query_text.get(int(query_id), query_text.get(query_id, "")),
                    "product_id": product_id,
                    "esci_label": label,
                    **meta,
                }
            )
        for label, pool in buckets.items():
            rng.shuffle(pool)
            for row in pool[:synthetic_per_label]:
                synthetic_counts[label] += 1
                examples.append(
                    _row_to_example(
                        row,
                        f"esci_synthetic_{sum(synthetic_counts.values()):05d}",
                        "existing_synthetic",
                    )
                )

    rng.shuffle(examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    source_counts = Counter(row["label_source"] for row in examples)
    label_counts = Counter(row["labels"]["esci_label"] for row in examples)
    return {
        "output": str(output_path),
        "items": len(examples),
        "by_label_source": dict(source_counts),
        "by_esci_label": dict(label_counts),
        "human_available": {
            str(label): int(count)
            for label, count in human["esci_label"].value_counts().items()
        },
    }
