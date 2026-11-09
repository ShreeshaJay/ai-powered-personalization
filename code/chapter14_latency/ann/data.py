"""
Item vectors and session queries.

Two sources:

* **Coveo SIGIR eCom 2021** (Chapter 5's dataset). About half of the SKUs in
  ``sku_to_content.csv`` carry a 50-dimensional description vector; those are
  the vectors Chapter 5 searches. Session queries come from the browsing
  file: the mean of a session's recent viewed items, with the next viewed
  item as the target. This carries the quality comparison (recall against a
  downstream hit rate).
* **Synthetic clustered catalog**. Tens of thousands of vectors are too few
  to show realistic search latency, so a deterministic Gaussian-mixture
  catalog on the unit sphere (default one million items) carries scale.

Dataset files are read in place and never copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl


@dataclass
class Catalog:
    name: str
    vectors: np.ndarray                 # (n_items, dim), float32, unit norm
    popularity: np.ndarray              # (n_items,), non-negative counts or weights
    queries: np.ndarray                 # (n_queries, dim), float32, unit norm
    targets: Optional[np.ndarray] = None  # next-item index per query (Coveo only)
    histories: List[np.ndarray] = field(default_factory=list)  # item indices per query
    cluster_of: Optional[np.ndarray] = None  # synthetic only

    @property
    def n_items(self) -> int:
        return self.vectors.shape[0]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


# ------------------------------------------------------------------- Coveo
def _find(root: Path, name: str) -> Path:
    for sub in ("sampled", "train", "."):
        candidate = root / sub / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found under {root} (looked in sampled/, train/, and the folder itself)")


def _browsing_file(root: Path) -> Path:
    for name in ("browsing_train_sampled.csv", "browsing_train.csv"):
        try:
            return _find(root, name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no browsing_train file under {root}")


def load_coveo(root: Path, max_rows: int = 3_000_000, history: int = 5,
               max_queries: int = 20_000, seed: int = 0) -> Catalog:
    """Load Coveo description vectors and build next-item session queries.

    Only the first ``max_rows`` rows of the browsing file are read, which
    keeps memory and time bounded on the multi-gigabyte file.
    """
    sku_df = (
        pl.read_csv(_find(root, "sku_to_content.csv"), columns=["product_sku_hash", "description_vector"])
        .drop_nulls("description_vector")
    )
    vec_lists = (
        sku_df["description_vector"].str.strip_chars("[]").str.split(",")
        .list.eval(pl.element().str.strip_chars().cast(pl.Float32))
    )
    vectors = normalize(np.asarray(vec_lists.to_list(), dtype=np.float32))
    sku_index = pl.DataFrame({"product_sku_hash": sku_df["product_sku_hash"],
                              "item": np.arange(len(sku_df), dtype=np.int32)})

    events = (
        pl.scan_csv(_browsing_file(root))
        .select(["session_id_hash", "product_action", "product_sku_hash", "server_timestamp_epoch_ms"])
        .head(max_rows)
        .filter(pl.col("product_action") == "detail")
        .drop_nulls("product_sku_hash")
        .collect()
        .join(sku_index, on="product_sku_hash", how="inner")
        .sort(["session_id_hash", "server_timestamp_epoch_ms"])
    )
    popularity = np.bincount(events["item"].to_numpy(), minlength=len(vectors)).astype(np.float64)

    sessions = (
        events.group_by("session_id_hash", maintain_order=True)
        .agg(pl.col("item"))
        .get_column("item")
        .to_list()
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sessions))
    queries, targets, hists = [], [], []
    for idx in order:
        seq = np.asarray(sessions[idx], dtype=np.int64)
        if seq.size < 2:
            continue
        keep = np.concatenate([[True], seq[1:] != seq[:-1]])  # drop consecutive repeats
        seq = seq[keep]
        if seq.size < 2:
            continue
        hist, target = seq[:-1][-history:], seq[-1]
        if target in hist:
            continue
        queries.append(vectors[hist].mean(axis=0))
        targets.append(target)
        hists.append(hist)
        if len(queries) >= max_queries:
            break
    if not queries:
        raise ValueError("no usable sessions found; increase max_rows")
    return Catalog("coveo", vectors, popularity, normalize(np.asarray(queries)),
                   np.asarray(targets, dtype=np.int64), hists)


# --------------------------------------------------------------- synthetic
def synthetic_catalog(n_items: int = 1_000_000, dim: int = 64, n_clusters: int = 1_000,
                      n_queries: int = 20_000, history: int = 5, noise: float = 0.6,
                      zipf_exponent: float = 1.0, seed: int = 0, chunk: int = 200_000) -> Catalog:
    """Deterministic clustered catalog on the unit sphere with Zipf popularity.

    Queries are the normalized mean of ``history`` items drawn from one
    cluster, weighted by popularity, which mimics a user vector built from a
    short browsing history.
    """
    rng = np.random.default_rng(seed)
    centers = normalize(rng.standard_normal((n_clusters, dim)))
    cluster_of = rng.integers(0, n_clusters, n_items).astype(np.int32)
    vectors = np.empty((n_items, dim), dtype=np.float32)
    for lo in range(0, n_items, chunk):
        hi = min(lo + chunk, n_items)
        noise_block = rng.standard_normal((hi - lo, dim)).astype(np.float32) * (noise / np.sqrt(dim))
        vectors[lo:hi] = normalize(centers[cluster_of[lo:hi]] + noise_block)

    ranks = rng.permutation(n_items) + 1
    popularity = 1.0 / ranks.astype(np.float64) ** zipf_exponent

    order = np.argsort(cluster_of, kind="stable")
    starts = np.searchsorted(cluster_of[order], np.arange(n_clusters + 1))
    queries = np.empty((n_queries, dim), dtype=np.float32)
    hists = []
    for q in range(n_queries):
        c = int(rng.integers(0, n_clusters))
        members = order[starts[c]:starts[c + 1]]
        if members.size == 0:
            members = order[:1]
        w = popularity[members] / popularity[members].sum()
        hist = rng.choice(members, size=min(history, members.size), replace=False, p=w)
        hists.append(hist.astype(np.int64))
        queries[q] = vectors[hist].mean(axis=0)
    return Catalog("synthetic", vectors, popularity, normalize(queries), None, hists, cluster_of)


def load_catalog(kind: str, sigir_root: Optional[Path], quick: bool, seed: int,
                 n_items: Optional[int] = None) -> Catalog:
    """``kind`` is ``coveo``, ``synthetic``, or ``auto`` (Coveo when available)."""
    if kind == "auto":
        kind = "coveo" if sigir_root is not None and sigir_root.exists() else "synthetic"
    if kind == "coveo":
        if sigir_root is None or not sigir_root.exists():
            raise FileNotFoundError("Coveo data not found: set SIGIR_DATA_DIR or use the synthetic catalog")
        return load_coveo(sigir_root, max_rows=500_000 if quick else 3_000_000,
                          max_queries=3_000 if quick else 20_000, seed=seed)
    default_items = 50_000 if quick else 1_000_000
    return synthetic_catalog(n_items=n_items or default_items,
                             n_clusters=100 if quick else 1_000,
                             n_queries=3_000 if quick else 20_000, seed=seed)
