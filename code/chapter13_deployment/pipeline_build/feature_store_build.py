"""
Write the online feature store from the fitted Chapter 6 HistoricalFeatureBuilder.

Tables (identical content in Parquet and SQLite):
    users       uid, 8 user aggregates, history (last 20 item IDs, oldest first)
    items       item_id, 9 item aggregates (Chapter 6's 7, including item_like_ratio,
                plus the like and dislike counts it is computed from; NULL when
                the item had no plays in the feature window), track_length_seconds,
                artist_id, album_id
    user_items  uid, item_id, previous_listen_count (pairs seen in the feature window)

The manifest records the builder's cold-start defaults, which the store
applies at read time in the same way HistoricalFeatureBuilder.transform does.

Users and items also carry `updated_at`, the event time at which the
snapshot computed them (the end of the feature window). feature_contract.json
declares every feature's group, type, encoding, default and maximum age
(Section 13.2.4); the store and both rankers read it.
"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from common.artifacts import sha256_file, write_json
from common.feature_store import CONTRACT_NAME, ITEM_ATTRIBUTES, MANIFEST_NAME, PARQUET_FILES, SQLITE_NAME, UPDATED_AT

logger = logging.getLogger("ch13.build")

# Maximum age of a value, in the dataset's event-time units (the units of
# context.timestamp). Illustrative choices, not tuned: with timestamps in
# seconds they are seven days for user and user-item aggregates and thirty
# days for item aggregates, which change more slowly.
MAX_AGE = {"user": 7 * 86400, "user_item": 7 * 86400, "item": 30 * 86400}

# Item columns HistoricalFeatureBuilder adds when it is fitted with likes and
# dislikes, beyond Chapter 6's published feature list. Chapter 6's XGBoost
# pipeline picks them up by name, so the contract must declare them.
LIKE_COUNT_FEATURES = ["item_likes", "item_dislikes"]


def _native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def build_feature_store(out_dir: Path, builder, data, *, feature_set_version: str, history_length: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    listens = data.listens
    ch6_columns = published_feature_columns()
    user_names = ch6_columns["user_features"]
    item_names = ch6_columns["item_features"] + [c for c in LIKE_COUNT_FEATURES if c in builder.item_features]
    user_item_names = ["previous_listen_count", "has_listened_before"]

    ordered = listens.sort_values(["uid", "timestamp"], kind="stable")
    history = ordered.groupby("uid", sort=True)["item_id"].apply(
        lambda s: [int(i) for i in s.to_numpy()[-history_length:]])
    users = pd.DataFrame({"uid": history.index.astype(np.int64), "history": history.values})
    users = users.merge(builder.user_features[["uid", *user_names]].astype({"uid": np.int64}), on="uid", how="outer")
    users["history"] = users["history"].apply(lambda h: h if isinstance(h, list) else [])
    users = users.sort_values("uid").reset_index(drop=True)

    track_length = listens.groupby("item_id")["track_length_seconds"].max()
    items = pd.DataFrame({"item_id": data.item_ids.astype(np.int64)})
    items = items.merge(builder.item_features[["item_id", *item_names]].astype({"item_id": np.int64}),
                        on="item_id", how="left")
    items["track_length_seconds"] = items["item_id"].map(track_length).astype("float64")
    items["artist_id"] = items["item_id"].map(data.artist).astype("float64")
    items["album_id"] = items["item_id"].map(data.album).astype("float64")

    pairs = builder.user_item_features[["uid", "item_id", "previous_listen_count"]].astype(
        {"uid": np.int64, "item_id": np.int64, "previous_listen_count": np.float64})
    pairs = pairs.sort_values(["uid", "item_id"]).reset_index(drop=True)

    reference_timestamp = int(listens["timestamp"].max())
    users[UPDATED_AT] = np.int64(reference_timestamp)
    items[UPDATED_AT] = np.int64(reference_timestamp)

    users.to_parquet(out_dir / PARQUET_FILES["users"], index=False)
    items.to_parquet(out_dir / PARQUET_FILES["items"], index=False)
    pairs.to_parquet(out_dir / PARQUET_FILES["user_items"], index=False)
    _write_sqlite(out_dir / SQLITE_NAME, users, items, pairs, user_names, item_names)

    track_lengths_per_item = listens.groupby("item_id")["track_length_seconds"].nunique()
    defaults = {
        "user": {k: _native(v) for k, v in builder.user_defaults.items()},
        "item": {k: _native(v) for k, v in builder.item_defaults.items()},
        "user_item": {k: _native(v) for k, v in builder.user_item_defaults.items()},
    }
    write_json(out_dir / CONTRACT_NAME, feature_contract(feature_set_version, defaults, user_names, item_names,
                                                         user_item_names))
    files = {name: sha256_file(out_dir / name) for name in (*PARQUET_FILES.values(), SQLITE_NAME, CONTRACT_NAME)}
    manifest = {
        "feature_set_version": feature_set_version,
        "source": "Chapter 6 HistoricalFeatureBuilder fitted on the feature window",
        "user_feature_names": user_names,
        "item_feature_names": item_names,
        "user_item_feature_names": user_item_names,
        "item_attributes": list(ITEM_ATTRIBUTES),
        "defaults": defaults,
        "history_length": history_length,
        "reference_timestamp": reference_timestamp,
        "counts": {"users": int(len(users)), "items": int(len(items)), "user_items": int(len(pairs))},
        "checks": {"items_with_varying_track_length": int((track_lengths_per_item > 1).sum())},
        "files": files,
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    logger.info("Feature store: %s", manifest["counts"])
    return manifest


def feature_contract(feature_set_version: str, defaults: dict, user_names, item_names, user_item_names) -> dict:
    features = {}
    for group, names in (("user", user_names), ("item", item_names), ("user_item", user_item_names)):
        for name in names:
            features[name] = {"group": group, "type": "float", "encoding": "raw", "default": defaults[group][name],
                              "max_age": MAX_AGE[group]}
    # An item attribute, not an aggregate: it does not go stale, and an item
    # without one is dropped from the candidates rather than given a default.
    features["track_length_seconds"] = {"group": "item", "type": "float", "encoding": "raw", "default": None,
                                        "max_age": None}
    return {
        "feature_set_version": feature_set_version,
        "age_unit": "dataset event-time units, the units of context.timestamp",
        "features": features,
    }


def published_feature_columns() -> dict:
    from common import published

    return published.chapter6("historical_features").get_historical_feature_columns()


def _write_sqlite(path: Path, users, items, pairs, user_names, item_names) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        user_cols = ", ".join(f"{name} REAL" for name in user_names)
        item_cols = ", ".join(f"{name} REAL" for name in item_names)
        conn.execute(f"CREATE TABLE users (uid INTEGER PRIMARY KEY, {user_cols}, history TEXT NOT NULL, "
                     f"{UPDATED_AT} INTEGER NOT NULL)")
        conn.execute(f"CREATE TABLE items (item_id INTEGER PRIMARY KEY, {item_cols}, "
                     f"track_length_seconds INTEGER, artist_id INTEGER, album_id INTEGER, {UPDATED_AT} INTEGER NOT NULL)")
        conn.execute("CREATE TABLE user_items (uid INTEGER NOT NULL, item_id INTEGER NOT NULL, "
                     "previous_listen_count REAL NOT NULL, PRIMARY KEY (uid, item_id)) WITHOUT ROWID")

        def clean(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return _native(value)

        conn.executemany(
            f"INSERT INTO users VALUES ({','.join('?' * (len(user_names) + 3))})",
            ((int(row.uid), *[clean(getattr(row, n)) for n in user_names], json.dumps(row.history),
              int(getattr(row, UPDATED_AT)))
             for row in users.itertuples(index=False)))
        conn.executemany(
            f"INSERT INTO items VALUES ({','.join('?' * (len(item_names) + 5))})",
            ((int(row.item_id), *[clean(getattr(row, n)) for n in item_names],
              *[None if clean(getattr(row, a)) is None else int(getattr(row, a)) for a in ITEM_ATTRIBUTES],
              int(getattr(row, UPDATED_AT)))
             for row in items.itertuples(index=False)))
        conn.executemany(
            "INSERT INTO user_items VALUES (?, ?, ?)",
            ((int(u), int(i), float(c)) for u, i, c in pairs.itertuples(index=False)))
        conn.commit()
    finally:
        conn.close()
