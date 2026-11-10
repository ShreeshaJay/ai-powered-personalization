"""
Amazon KDD Cup 2023 sessions and product catalog for Chapter 12.

The raw sessions file has no user ID, no timestamp and no session ID. This module gives
every session a stable ``session_id``: its row number in ``sessions_train.csv`` before
any filtering. Splits, candidate slates and LLM session intents are all keyed by that ID,
so two scripts can never silently attach a feature to the wrong session.

Because the sessions carry no time axis, train, validation and test are random splits by
session. That makes test sessions exchangeable with training sessions, which is an
optimistic protocol compared with a temporal holdout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

SESSION_FILE = "sessions_train.csv"
PRODUCT_FILE = "products_train.csv"
SPLITS = ("train", "val", "test")


def parse_prev_items(column: str = "prev_items") -> pl.Expr:
    """Parse strings such as "['B09W9FND7K' 'B09JSPLN1M']" into a list of item IDs."""
    return (
        pl.col(column)
        .str.extract_all(r"'[^']+'")
        .list.eval(pl.element().str.strip_chars("'"))
    )


def load_raw_sessions(data_dir: Path, locale: str = "UK") -> pl.DataFrame:
    """Return session_id, history (list of item IDs) and target for one locale."""
    path = Path(data_dir) / SESSION_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README.md for the download steps.")
    return (
        pl.scan_csv(path, infer_schema_length=0)
        .with_row_index("session_id")
        .filter(pl.col("locale") == locale)
        .select(
            pl.col("session_id").cast(pl.Int64),
            parse_prev_items("prev_items").alias("history"),
            pl.col("next_item").alias("target"),
        )
        .collect()
    )


def load_products(data_dir: Path, locale: str = "UK") -> pl.DataFrame:
    """Return item_id, title, brand and desc for one locale (all columns as strings)."""
    path = Path(data_dir) / PRODUCT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README.md for the download steps.")
    return (
        pl.scan_csv(path, infer_schema_length=0)
        .filter(pl.col("locale") == locale)
        .select(
            pl.col("id").alias("item_id"),
            pl.col("title").fill_null(""),
            pl.col("brand").fill_null(""),
            pl.col("desc").fill_null(""),
        )
        .unique(subset="item_id", keep="first", maintain_order=True)
        .collect()
    )


def item_text(title: str, brand: str, desc: str, desc_max_chars: int = 400) -> str:
    """Chapter 10's item text format: '{title} [SEP] {brand} [SEP] {desc}', skipping empty parts."""
    parts = [p.strip() for p in (title, brand, desc[:desc_max_chars]) if p and p.strip()]
    return " [SEP] ".join(parts) if parts else "unknown product"


def assign_splits(session_ids: np.ndarray, fractions: Sequence[float], seed: int) -> np.ndarray:
    """Random split by session. Returns an array of 'train' / 'val' / 'test' labels."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must be three numbers summing to 1, got {fractions}")
    n = len(session_ids)
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    return labels


def prepare_sessions_and_items(
    sessions: pl.DataFrame,
    products: pl.DataFrame,
    *,
    min_prev_items: int,
    max_sessions: Optional[int],
    split_fractions: Sequence[float],
    catalog: str,
    extra_random_items: int,
    desc_max_chars: int,
    seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict]:
    """Filter, sample and split sessions; build the item table they need.

    Works on in-memory frames so that tests can pass synthetic data.
    """
    if catalog not in ("all", "referenced"):
        raise ValueError("catalog must be 'all' or 'referenced'")
    n_raw = sessions.height

    sessions = sessions.filter(pl.col("history").list.len() >= min_prev_items)
    n_long_enough = sessions.height
    catalog_ids = products.select("item_id")
    used = pl.concat([
        sessions.select("session_id", pl.col("history").alias("item_id")).explode("item_id"),
        sessions.select("session_id", pl.col("target").alias("item_id")),
    ])
    uncovered = used.join(catalog_ids, on="item_id", how="anti").select("session_id").unique()
    sessions = sessions.join(uncovered, on="session_id", how="anti")
    n_covered = sessions.height

    if max_sessions is not None and sessions.height > max_sessions:
        sessions = sessions.sample(n=max_sessions, seed=seed, shuffle=False)
    sessions = sessions.sort("session_id")
    splits = assign_splits(sessions["session_id"].to_numpy(), split_fractions, seed)
    sessions = sessions.with_columns(pl.Series("split", splits.tolist(), dtype=pl.Utf8))

    if catalog == "all":
        items = products
    else:
        referenced = pl.concat([
            sessions.select(pl.col("history").explode().alias("item_id")),
            sessions.select(pl.col("target").alias("item_id")),
        ]).unique()
        items = products.join(referenced, on="item_id", how="semi")
        if extra_random_items > 0:
            pool = products.join(referenced, on="item_id", how="anti")
            k = min(extra_random_items, pool.height)
            items = pl.concat([items, pool.sample(n=k, seed=seed)])
    items = items.sort("item_id").with_columns(
        pl.struct(["title", "brand", "desc"])
        .map_elements(lambda r: item_text(r["title"], r["brand"], r["desc"], desc_max_chars),
                      return_dtype=pl.Utf8)
        .alias("text")
    )

    split_counts = sessions.group_by("split").len().sort("split")
    meta = {
        "raw_sessions_in_locale": n_raw,
        "sessions_with_min_history": n_long_enough,
        "sessions_with_catalog_coverage": n_covered,
        "sessions_kept": sessions.height,
        "split_counts": dict(zip(split_counts["split"].to_list(), split_counts["len"].to_list())),
        "items_kept": items.height,
        "catalog": catalog,
    }
    return sessions, items, meta


class PreparedData:
    """Read-only view over the prepared sessions and item table of one work directory."""

    def __init__(self, sessions_path: Path, items_path: Path):
        if not sessions_path.exists() or not items_path.exists():
            raise FileNotFoundError(
                f"Prepared data not found under {sessions_path.parent}. Run prepare_data.py first."
            )
        self.sessions = pl.read_parquet(sessions_path)
        self.items = pl.read_parquet(items_path)
        self.item_ids: List[str] = self.items["item_id"].to_list()
        self.item_index: Dict[str, int] = {iid: i for i, iid in enumerate(self.item_ids)}

    @classmethod
    def from_paths(cls, paths) -> "PreparedData":
        return cls(paths.sessions, paths.items)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    def split(self, name: str) -> pl.DataFrame:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name}")
        return self.sessions.filter(pl.col("split") == name)

    def titles(self) -> List[str]:
        return [t if t else f"[untitled product {iid}]"
                for t, iid in zip(self.items["title"].to_list(), self.item_ids)]

    def texts(self) -> List[str]:
        return self.items["text"].to_list()

    def history_indices(self, frame: pl.DataFrame) -> List[List[int]]:
        return [[self.item_index[i] for i in h] for h in frame["history"].to_list()]

    def target_indices(self, frame: pl.DataFrame) -> np.ndarray:
        return np.array([self.item_index[t] for t in frame["target"].to_list()], dtype=np.int64)

    def train_item_counts(self) -> np.ndarray:
        """Number of training sessions in which each item appears (history or target)."""
        counts = np.zeros(self.num_items, dtype=np.int64)
        train = self.split("train")
        for history, target in zip(train["history"].to_list(), train["target"].to_list()):
            for iid in set(history) | {target}:
                counts[self.item_index[iid]] += 1
        return counts

    def train_target_counts(self) -> np.ndarray:
        """Number of training sessions whose target is each item (a popularity prior)."""
        counts = np.zeros(self.num_items, dtype=np.int64)
        for iid in self.split("train")["target"].to_list():
            counts[self.item_index[iid]] += 1
        return counts


def load_item_embeddings(path: Path, item_ids: Sequence[str]) -> Tuple[np.ndarray, Dict]:
    """Load embeddings aligned to ``item_ids`` and return (matrix, metadata).

    Raises if any item is missing, so a stale embedding file cannot silently misalign rows.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run encode_items.py first.")
    data = np.load(path, allow_pickle=False)
    stored_ids = data["item_ids"].tolist()
    index = {iid: i for i, iid in enumerate(stored_ids)}
    missing = [iid for iid in item_ids if iid not in index]
    if missing:
        raise ValueError(f"{len(missing)} items have no embedding in {path}; re-run encode_items.py")
    rows = np.array([index[iid] for iid in item_ids], dtype=np.int64)
    matrix = data["embeddings"][rows].astype(np.float32)
    meta = {"encoder": str(data["encoder"]) if "encoder" in data else "unknown"}
    return matrix, meta


def write_meta(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
