"""
Training data for the artifact build, from Yambda or from a synthetic generator.

Both sources return the same `TrainingData` with Yambda's column names and
types, so every later step runs unchanged on either:

    listens   uid, timestamp, item_id, is_organic, played_ratio_pct, track_length_seconds
    likes     uid, timestamp, item_id
    dislikes  uid, timestamp, item_id
    catalog   item IDs with a 128-dimensional content (audio) embedding each
    artist and album IDs per catalog item

Timestamps are Yambda's: seconds, recorded at 5-second resolution, the unit
the published Chapter 6 and 7 code reads.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ch13.build")

LISTEN_COLUMNS = ["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds"]


@dataclass
class TrainingData:
    name: str
    listens: pd.DataFrame
    likes: pd.DataFrame
    dislikes: pd.DataFrame
    item_ids: np.ndarray            # catalog, sorted int64
    content_embeddings: np.ndarray  # float32, row i belongs to item_ids[i]
    artist: Dict[int, int]
    album: Dict[int, int]
    source: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "users": int(self.listens["uid"].nunique()),
            "catalog_items": int(len(self.item_ids)),
            "listens": int(len(self.listens)),
            "likes": int(len(self.likes)),
            "dislikes": int(len(self.dislikes)),
            "embedding_dim": int(self.content_embeddings.shape[1]),
            "timestamp_min": int(self.listens["timestamp"].min()),
            "timestamp_max": int(self.listens["timestamp"].max()),
        }


# ---------------------------------------------------------------------------
# Synthetic, Yambda-shaped data
# ---------------------------------------------------------------------------

def make_synthetic(num_users: int = 300, num_items: int = 2000, num_clusters: int = 12,
                   listens_per_user=(60, 240), dim: int = 128, time_span: int = 2_000_000,
                   seed: int = 42) -> TrainingData:
    """
    Users prefer a few clusters of items ("genres"); items in a cluster have
    similar content embeddings; completion depends on that preference and on
    a per-item quality term. This gives every model a learnable signal while
    keeping the build to well under a minute on a laptop.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    item_cluster = rng.integers(0, num_clusters, num_items)
    embeddings = centers[item_cluster] + rng.normal(scale=0.8 / np.sqrt(dim), size=(num_items, dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    quality = rng.normal(scale=0.6, size=num_items)
    track_length = rng.integers(90, 420, num_items)

    artists_per_cluster = max(2, num_items // (num_clusters * 8))
    artist = {i: int(item_cluster[i] * 1000 + rng.integers(0, artists_per_cluster)) for i in range(num_items)}
    album = {i: int(artist[i] * 10 + rng.integers(0, 3)) for i in range(num_items)}

    members = [np.flatnonzero(item_cluster == c) for c in range(num_clusters)]
    popularity = []
    for idx in members:
        weights = 1.0 / np.arange(1, len(idx) + 1) ** 0.8
        rng.shuffle(weights)
        popularity.append(weights / weights.sum())

    rows, likes, dislikes = [], [], []
    for uid in range(num_users):
        preference = rng.dirichlet(np.full(num_clusters, 0.3))
        n = int(rng.integers(*listens_per_user))
        start = rng.integers(0, int(time_span * 0.3))
        timestamps = start + np.cumsum(rng.exponential((time_span - start) / (n + 1), size=n)).astype(np.int64)
        timestamps = np.minimum(timestamps, time_span - 1)
        cluster = rng.choice(num_clusters, p=preference)
        for ts in timestamps:
            if rng.random() > 0.75:
                cluster = rng.choice(num_clusters, p=preference)
            item = int(rng.choice(members[cluster], p=popularity[cluster]))
            organic = int(rng.random() < 0.45)
            affinity = np.clip(np.log(preference[cluster] * num_clusters + 1e-3), -3, 2)
            p_complete = 1.0 / (1.0 + np.exp(-(0.8 * affinity + quality[item] + 0.3 * organic - 0.2)))
            ratio = int(rng.integers(50, 101)) if rng.random() < p_complete else int(rng.integers(0, 50))
            rows.append((uid, int(ts), item, organic, ratio, int(track_length[item])))
            if ratio >= 80 and rng.random() < 0.06 + 0.1 * (quality[item] > 0.5):
                likes.append((uid, int(ts) + int(rng.integers(1, 2000)), item))
            elif ratio < 20 and rng.random() < 0.04:
                dislikes.append((uid, int(ts) + int(rng.integers(1, 500)), item))

    listens = pd.DataFrame(rows, columns=LISTEN_COLUMNS)
    listens = _cast_listens(listens).sort_values(["timestamp", "uid", "item_id"], kind="stable").reset_index(drop=True)
    events = ["uid", "timestamp", "item_id"]
    return TrainingData(
        name="synthetic",
        listens=listens,
        likes=_cast_events(pd.DataFrame(likes, columns=events)),
        dislikes=_cast_events(pd.DataFrame(dislikes, columns=events)),
        item_ids=np.arange(num_items, dtype=np.int64),
        content_embeddings=embeddings.astype(np.float32),
        artist=artist,
        album=album,
        source={"type": "synthetic", "seed": seed, "num_users": num_users, "num_items": num_items,
                "num_clusters": num_clusters},
    )


def _cast_listens(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({"uid": "uint32", "timestamp": "uint32", "item_id": "uint32", "is_organic": "uint8",
                      "played_ratio_pct": "uint16", "track_length_seconds": "uint32"})


def _cast_events(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({"uid": "uint32", "timestamp": "uint32", "item_id": "uint32"})


# ---------------------------------------------------------------------------
# Yambda sample
# ---------------------------------------------------------------------------

def yambda_flat_dir(data_dir: Path) -> Path:
    """The 50M interaction files: flat/50m/ in the Hugging Face layout, or flat/ in a flattened copy."""
    data_dir = Path(data_dir)
    for candidate in (data_dir / "flat" / "50m", data_dir / "flat"):
        if (candidate / "listens.parquet").is_file():
            return candidate
    return data_dir / "flat" / "50m"


def load_yambda(data_dir: Path, num_users: Optional[int] = 400, catalog_size: Optional[int] = 20000,
                seed: int = 42) -> TrainingData:
    """
    Sample `num_users` users at random, keep the `catalog_size` items those
    users played most that also have a content embedding, and restrict every
    table to that catalog. None for either limit keeps everything.
    """
    import polars as pl
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    flat = yambda_flat_dir(data_dir)
    for required in (flat / "listens.parquet", flat / "likes.parquet", flat / "dislikes.parquet",
                     data_dir / "embeddings.parquet", data_dir / "artist_item_mapping.parquet",
                     data_dir / "album_item_mapping.parquet"):
        if not required.is_file():
            raise FileNotFoundError(f"Yambda file not found: {required}. Pass --data-dir pointing at the folder "
                                    "that contains embeddings.parquet and flat/50m/listens.parquet "
                                    "(or flat/listens.parquet).")

    started = time.perf_counter()
    all_users = np.sort(pl.scan_parquet(flat / "listens.parquet").select("uid").unique().collect()["uid"].to_numpy())
    rng = np.random.default_rng(seed)
    users = all_users if num_users is None or num_users >= len(all_users) else \
        np.sort(rng.choice(all_users, size=num_users, replace=False))
    listens = (pl.scan_parquet(flat / "listens.parquet")
               .filter(pl.col("uid").is_in(users.tolist()))
               .collect())
    logger.info("Yambda: %d of %d users, %d listens (%.1fs)", len(users), len(all_users), listens.height,
                time.perf_counter() - started)

    plays = listens.group_by("item_id").agg(pl.len().alias("plays")).sort(["plays", "item_id"],
                                                                          descending=[True, False])
    embedding_file = pq.ParquetFile(data_dir / "embeddings.parquet")
    with_embedding = pq.read_table(data_dir / "embeddings.parquet", columns=["item_id"])["item_id"].to_numpy()
    ranked = plays["item_id"].to_numpy()
    ranked = ranked[np.isin(ranked, with_embedding)]
    catalog = np.sort(ranked if catalog_size is None else ranked[:catalog_size]).astype(np.int64)
    logger.info("Catalog: %d items with embeddings (of %d played)", len(catalog), plays.height)

    vectors = {}
    for group in range(embedding_file.num_row_groups):
        ids = embedding_file.read_row_group(group, columns=["item_id"])["item_id"].to_numpy()
        mask = np.isin(ids, catalog)
        if not mask.any():
            continue
        table = embedding_file.read_row_group(group, columns=["item_id", "normalized_embed"])
        rows = np.flatnonzero(mask)
        embeds = table["normalized_embed"].take(rows).to_pylist()
        for item_id, embed in zip(ids[rows], embeds):
            vectors[int(item_id)] = embed
    content = np.asarray([vectors[int(i)] for i in catalog], dtype=np.float32)
    logger.info("Read %d content embeddings (%.1fs)", len(content), time.perf_counter() - started)

    catalog_list = catalog.tolist()
    listens = listens.filter(pl.col("item_id").is_in(catalog_list)).select(LISTEN_COLUMNS).to_pandas()
    listens = _cast_listens(listens).sort_values(["timestamp", "uid", "item_id"], kind="stable").reset_index(drop=True)

    def events(name: str) -> pd.DataFrame:
        frame = (pl.scan_parquet(flat / f"{name}.parquet")
                 .filter(pl.col("uid").is_in(users.tolist()) & pl.col("item_id").is_in(catalog_list))
                 .select(["uid", "timestamp", "item_id"]).collect().to_pandas())
        return _cast_events(frame)

    def mapping(name: str, column: str) -> Dict[int, int]:
        frame = (pl.scan_parquet(data_dir / name).filter(pl.col("item_id").is_in(catalog_list))
                 .collect())
        # Same semantics as Chapter 8's load_artist_mapping: the last row for an item wins.
        return dict(zip(frame["item_id"].to_list(), frame[column].to_list()))

    data = TrainingData(
        name="yambda",
        listens=listens,
        likes=events("likes"),
        dislikes=events("dislikes"),
        item_ids=catalog,
        content_embeddings=content,
        artist=mapping("artist_item_mapping.parquet", "artist_id"),
        album=mapping("album_item_mapping.parquet", "album_id"),
        source={"type": "yambda", "variant": flat.relative_to(data_dir).as_posix(), "seed": seed,
                "num_users": int(len(users)), "catalog_size": int(len(catalog)),
                "embedding_column": "normalized_embed"},
    )
    logger.info("Yambda load complete (%.1fs): %s", time.perf_counter() - started, data.summary())
    return data
