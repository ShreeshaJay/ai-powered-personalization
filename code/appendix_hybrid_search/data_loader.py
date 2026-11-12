"""US ESCI judgments and the shared product-text template.

The catalog is the set of product ids that appear in US rows with
``large_version == 1``. ESCI Task 1 used ``small_version == 1``; this
appendix does not. Empty template fields are omitted so a missing bullet
does not leave a dangling separator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EXAMPLES_FILE,
    LINEAR_GAINS,
    OFFICIAL_GAINS,
    PRODUCTS_FILE,
    TEXT_RULE,
    TEXT_TEMPLATE,
    sha256_text,
)

_LABEL_TO_SHORT = {
    "E": "E",
    "S": "S",
    "C": "C",
    "I": "I",
    "Exact": "E",
    "Substitute": "S",
    "Complement": "C",
    "Irrelevant": "I",
    "exact": "E",
    "substitute": "S",
    "complement": "C",
    "irrelevant": "I",
}


def normalize_ws(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split())


def normalize_query(value: object) -> str:
    return normalize_ws(value)


def render_product_text(title: object, bullet: object, brand: object) -> str:
    parts: list[str] = []
    title_text = normalize_ws(title)
    bullet_text = normalize_ws(bullet)
    brand_text = normalize_ws(brand)
    if title_text:
        parts.append(title_text)
    if bullet_text:
        parts.append(bullet_text)
    if brand_text:
        parts.append(f"Brand: {brand_text}")
    return ". ".join(parts)


def _field_token(value: object) -> str:
    """Missing catalog values are the token used in ``tasksource/esci`` product_text."""
    if value is None:
        return "None"
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if missing is True:
        return "None"
    return str(value)


def render_qdrant_text(
    title: object,
    brand: object,
    color: object,
    description: object,
    bullet_point: object,
) -> str:
    """Document text for the ESCI SPLADE checkpoint.

    This is the ``product_text`` column of ``tasksource/esci``: title, brand,
    color, description, then bullet point, joined by newlines. A missing
    field is the token ``None``. Characters inside a field are kept.
    """
    return "\n".join(
        _field_token(value) for value in (title, brand, color, description, bullet_point)
    )


def template_hash() -> str:
    spec = f"{TEXT_TEMPLATE}\n{TEXT_RULE}\n"
    return sha256_text(spec)


def short_label(value: object) -> str:
    text = str(value).strip()
    if text not in _LABEL_TO_SHORT:
        raise ValueError(f"Unrecognized ESCI label {value!r}.")
    return _LABEL_TO_SHORT[text]


@dataclass
class QueryJudgments:
    query_id: str
    query_text: str
    split: str
    labels: dict[str, str]
    product_order: list[str]

    def gains(self, table: dict[str, float]) -> dict[str, float]:
        return {product_id: float(table[label]) for product_id, label in self.labels.items()}

    def exact_ids(self) -> set[str]:
        return {product_id for product_id, label in self.labels.items() if label == "E"}

    def exact_or_substitute_ids(self) -> set[str]:
        return {product_id for product_id, label in self.labels.items() if label in {"E", "S"}}


@dataclass
class Catalog:
    product_ids: list[str]
    texts: dict[str, str]
    qdrant_texts: dict[str, str]
    frame: pd.DataFrame
    missing_product_rows: int
    template_sha256: str
    catalog_sha256: str
    text_sha256: str

    def texts_in_order(self, product_ids: list[str] | None = None) -> tuple[list[str], list[str]]:
        ids = self.product_ids if product_ids is None else product_ids
        return ids, [self.texts[product_id] for product_id in ids]


@dataclass
class ESCIData:
    data_dir: Path
    examples: pd.DataFrame
    catalog: Catalog
    queries: dict[str, QueryJudgments] = field(default_factory=dict)

    def by_split(self, split: str) -> dict[str, QueryJudgments]:
        return {
            query_id: item
            for query_id, item in self.queries.items()
            if item.split == split
        }


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _hash_lines(lines) -> str:
    """SHA-256 of the lines joined by newlines, with a trailing newline when non-empty."""
    digest = hashlib.sha256()
    count = 0
    for line in lines:
        if count:
            digest.update(b"\n")
        digest.update(str(line).encode("utf-8"))
        count += 1
    if count:
        digest.update(b"\n")
    return digest.hexdigest()


def load_esci(data_dir: Path) -> ESCIData:
    data_dir = Path(data_dir)
    examples_path = data_dir / EXAMPLES_FILE
    products_path = data_dir / PRODUCTS_FILE
    if not examples_path.exists() or not products_path.exists():
        raise FileNotFoundError(
            f"Expected {EXAMPLES_FILE} and {PRODUCTS_FILE} in {data_dir}."
        )
    examples = pd.read_parquet(examples_path)
    products = pd.read_parquet(products_path)
    _require_columns(
        examples,
        ["query_id", "query", "product_id", "product_locale", "esci_label", "large_version", "split"],
        examples_path.name,
    )
    _require_columns(
        products,
        [
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "product_locale",
        ],
        products_path.name,
    )
    us = examples[
        (examples["product_locale"].astype(str).str.lower() == "us")
        & (examples["large_version"] == 1)
    ].copy()
    us["query_id"] = us["query_id"].astype(str)
    us["product_id"] = us["product_id"].astype(str)
    us["esci_label"] = us["esci_label"].map(short_label)
    us["query"] = us["query"].map(normalize_query)
    us["split"] = us["split"].astype(str)
    test_query_ids = set(us.loc[us["split"] == "test", "query_id"])
    in_both_splits = test_query_ids & set(us.loc[us["split"] == "train", "query_id"])
    if in_both_splits:
        us = us[~((us["query_id"].isin(in_both_splits)) & (us["split"] == "train"))].copy()

    product_us = products[products["product_locale"].astype(str).str.lower() == "us"].copy()
    product_us["product_id"] = product_us["product_id"].astype(str)
    product_us = product_us.drop_duplicates("product_id", keep="first")
    product_lookup = product_us.set_index("product_id", drop=False)

    catalog_ids = sorted(us["product_id"].unique().tolist())
    texts: dict[str, str] = {}
    qdrant_texts: dict[str, str] = {}
    missing = 0
    for product_id in catalog_ids:
        if product_id not in product_lookup.index:
            missing += 1
            texts[product_id] = ""
            qdrant_texts[product_id] = ""
            continue
        row = product_lookup.loc[product_id]
        texts[product_id] = render_product_text(
            row["product_title"], row["product_bullet_point"], row["product_brand"]
        )
        qdrant_texts[product_id] = render_qdrant_text(
            row["product_title"],
            row["product_brand"],
            row["product_color"],
            row["product_description"],
            row["product_bullet_point"],
        )

    queries: dict[str, QueryJudgments] = {}
    for query_id, group in us.groupby("query_id", sort=False):
        splits = group["split"].unique().tolist()
        if len(splits) != 1:
            raise ValueError(f"Query {query_id} appears in multiple splits: {splits}.")
        texts_q = group["query"].unique().tolist()
        if len(texts_q) != 1:
            raise ValueError(f"Query {query_id} has more than one query string.")
        labels: dict[str, str] = {}
        order: list[str] = []
        for product_id, label in zip(group["product_id"].tolist(), group["esci_label"].tolist()):
            if product_id in labels and labels[product_id] != label:
                raise ValueError(f"Query {query_id} product {product_id} has conflicting labels.")
            if product_id not in labels:
                labels[product_id] = label
                order.append(product_id)
        queries[str(query_id)] = QueryJudgments(
            query_id=str(query_id),
            query_text=texts_q[0],
            split=splits[0],
            labels=labels,
            product_order=order,
        )

    train_ids = {qid for qid, item in queries.items() if item.split == "train"}
    test_ids = {qid for qid, item in queries.items() if item.split == "test"}
    if train_ids & test_ids:
        raise ValueError("Train and test query ids overlap.")

    catalog = Catalog(
        product_ids=catalog_ids,
        texts=texts,
        qdrant_texts=qdrant_texts,
        frame=product_us,
        missing_product_rows=missing,
        template_sha256=template_hash(),
        catalog_sha256=_hash_lines(catalog_ids),
        text_sha256=_hash_lines(f"{pid}\t{texts[pid]}" for pid in catalog_ids),
    )
    return ESCIData(data_dir=data_dir, examples=us.reset_index(drop=True), catalog=catalog, queries=queries)


def gain_table(name: str) -> dict[str, float]:
    if name == "official":
        return dict(OFFICIAL_GAINS)
    if name == "linear":
        return dict(LINEAR_GAINS)
    raise ValueError(f"Unknown gain table {name!r}.")
