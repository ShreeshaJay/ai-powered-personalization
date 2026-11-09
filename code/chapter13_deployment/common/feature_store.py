"""
Online feature store behind one interface (Section 13.7).

Three backends hold the same tables, written by the artifact build from the
Chapter 6 HistoricalFeatureBuilder snapshot the rankers were trained on:

* sqlite  (default) one read-only file, indexed by key; no server to run.
* parquet           the same tables loaded into sorted NumPy arrays in memory.
* redis   (optional) a Redis server loaded by scripts/load_redis.py; start one
                     with `docker compose --profile redis up redis`.

Choose with FEATURE_STORE_BACKEND=sqlite|parquet|redis (REDIS_URL for redis).

Missing values are filled with the builder's cold-start defaults in exactly
the way HistoricalFeatureBuilder.transform fills them offline, so a user or
item the store has never seen gets the same features at serving time as it
would have had in training.

The feature contract (feature_contract.json, Section 13.2.4) declares each
feature's group, type, encoding, default and maximum age. Every user and item
row carries the event time at which its values were computed (`updated_at`);
user-item counts are written with the user's row and share its time. When a
lookup passes the request time (`as_of`), a value older than its maximum age
is treated as missing and replaced by the default. Every substitution, stale
or missing, is counted per feature, both for the process (`describe()`) and,
when the caller passes a FeatureSubstitutions, for the request.
"""

import json
import math
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .artifacts import ArtifactError, read_json, require_file, verify_files

MANIFEST_NAME = "feature_store_manifest.json"
CONTRACT_NAME = "feature_contract.json"
SQLITE_NAME = "features.sqlite"
PARQUET_FILES = {"users": "users.parquet", "items": "items.parquet", "user_items": "user_items.parquet"}
ITEM_ATTRIBUTES = ("track_length_seconds", "artist_id", "album_id")
UPDATED_AT = "updated_at"
_SQL_CHUNK = 900


@dataclass
class FeatureSubstitutions:
    """Feature name -> number of values replaced by the contract's default, by cause."""
    missing: Counter = field(default_factory=Counter)
    stale: Counter = field(default_factory=Counter)

    def add(self, other: "FeatureSubstitutions") -> None:
        self.missing.update(other.missing)
        self.stale.update(other.stale)


@dataclass
class UserRecord:
    user_id: int
    found: bool
    features: Dict[str, float]
    recent_item_ids: List[int] = field(default_factory=list)
    updated_at: Optional[int] = None


@dataclass
class ItemRecord:
    item_id: int
    found: bool
    features: Dict[str, float]
    track_length_seconds: Optional[int] = None
    artist_id: Optional[int] = None
    album_id: Optional[int] = None
    updated_at: Optional[int] = None


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _optional_int(value) -> Optional[int]:
    if _missing(value):
        return None
    value = int(value)
    return None if value < 0 else value


def check_contract(contract: dict, manifest: dict) -> None:
    """The contract must declare every stored feature with the store's default."""
    declared = contract.get("features", {})
    for group, names in (("user", manifest["user_feature_names"]), ("item", manifest["item_feature_names"]),
                         ("user_item", manifest["user_item_feature_names"])):
        for name in names:
            spec = declared.get(name)
            if spec is None:
                raise ArtifactError(f"Feature contract does not declare '{name}'.")
            if spec.get("group") != group or spec.get("default") != manifest["defaults"][group][name]:
                raise ArtifactError(f"Feature contract entry for '{name}' disagrees with the feature store "
                                    f"(group {spec.get('group')}, default {spec.get('default')}).")
    if contract.get("feature_set_version") != manifest["feature_set_version"]:
        raise ArtifactError("Feature contract and feature store come from different feature sets.")


def check_model_features(contract: dict, feature_names: Iterable[str], feature_set_version: str,
                         model: str) -> None:
    """
    Startup check for a ranker (Section 13.2.4): every feature the model reads
    is declared, as a raw float, by the contract of the feature set the model
    was trained on. A failure stops the service before it reports ready.
    """
    if contract.get("feature_set_version") != feature_set_version:
        raise ArtifactError(f"{model} was trained on {feature_set_version}, but the feature contract is for "
                            f"{contract.get('feature_set_version')}.")
    declared = contract.get("features", {})
    undeclared = [name for name in feature_names if name not in declared]
    if undeclared:
        raise ArtifactError(f"{model} reads features the contract does not declare: {undeclared}")
    mismatched = [name for name in feature_names
                  if (declared[name].get("type"), declared[name].get("encoding")) != ("float", "raw")]
    if mismatched:
        raise ArtifactError(f"{model} expects raw float features; the contract declares otherwise for {mismatched}")


class FeatureStore(ABC):
    backend = "abstract"
    # True when a lookup reads this process's memory or a local file and so
    # cannot hang on a network; False when it waits on another server.
    in_process = True

    def __init__(self, manifest: dict, contract: dict):
        check_contract(contract, manifest)
        self.manifest = manifest
        self.contract = contract
        self.feature_set_version: str = manifest["feature_set_version"]
        self.user_feature_names: List[str] = manifest["user_feature_names"]
        self.item_feature_names: List[str] = manifest["item_feature_names"]
        self.user_item_feature_names: List[str] = manifest["user_item_feature_names"]
        self.defaults: Dict[str, Dict[str, float]] = manifest["defaults"]
        self.reference_timestamp: int = int(manifest["reference_timestamp"])
        self.max_age: Dict[str, Optional[int]] = {name: spec.get("max_age")
                                                  for name, spec in contract["features"].items()}
        self.substitutions = FeatureSubstitutions()
        self._count_lock = threading.Lock()

    # -- backend hooks -------------------------------------------------------
    @abstractmethod
    def _fetch_user(self, user_id: int) -> Optional[Tuple[Dict[str, float], List[int], Optional[int]]]:
        """(features, history, updated_at) for a stored user."""

    @abstractmethod
    def _fetch_items(self, item_ids: List[int]) -> Dict[int, dict]:
        """item_id -> {feature, attribute or updated_at: raw value (may be None/NaN)} for stored items."""

    @abstractmethod
    def _fetch_user_items(self, user_id: int, item_ids: List[int]) -> Dict[int, float]:
        """item_id -> previous_listen_count for pairs present in the store."""

    def ping(self) -> bool:
        self._fetch_user(-1)
        return True

    def close(self) -> None:
        pass

    # -- public interface ----------------------------------------------------
    def is_stale(self, name: str, updated_at: Optional[int], as_of: Optional[int]) -> bool:
        max_age = self.max_age.get(name)
        if max_age is None or updated_at is None or as_of is None:
            return False
        return as_of - updated_at > max_age

    def _fill(self, values: Dict[str, float], names: Iterable[str], defaults: Dict[str, float],
              updated_at: Optional[int], as_of: Optional[int], seen: FeatureSubstitutions) -> Dict[str, float]:
        out = {}
        for name in names:
            if _missing(values.get(name)):
                seen.missing[name] += 1
                out[name] = float(defaults[name])
            elif self.is_stale(name, updated_at, as_of):
                seen.stale[name] += 1
                out[name] = float(defaults[name])
            else:
                out[name] = float(values[name])
        return out

    def _count(self, seen: FeatureSubstitutions, report: Optional[FeatureSubstitutions]) -> None:
        if not seen.missing and not seen.stale:
            return
        with self._count_lock:
            self.substitutions.add(seen)
        if report is not None:
            report.add(seen)

    def get_user(self, user_id: int, as_of: Optional[int] = None,
                 report: Optional[FeatureSubstitutions] = None) -> UserRecord:
        seen = FeatureSubstitutions()
        row = self._fetch_user(int(user_id))
        if row is None:
            features = self._fill({}, self.user_feature_names, self.defaults["user"], None, as_of, seen)
            self._count(seen, report)
            return UserRecord(user_id, False, features, [])
        values, history, updated_at = row
        features = self._fill(values, self.user_feature_names, self.defaults["user"], updated_at, as_of, seen)
        self._count(seen, report)
        return UserRecord(user_id, True, features, [int(i) for i in history], updated_at)

    def get_items(self, item_ids: List[int], as_of: Optional[int] = None,
                  report: Optional[FeatureSubstitutions] = None) -> Dict[int, ItemRecord]:
        """
        Substitutions are counted only for items that can be served: an unknown
        item, or one without a track length, is dropped from the candidates, so
        its features never reach a model.
        """
        ids = [int(i) for i in dict.fromkeys(item_ids)]
        rows = self._fetch_items(ids)
        records, seen = {}, FeatureSubstitutions()
        for item_id in ids:
            raw = rows.get(item_id)
            if raw is None:
                records[item_id] = ItemRecord(item_id, False,
                                              {name: float(self.defaults["item"][name])
                                               for name in self.item_feature_names})
                continue
            updated_at = _optional_int(raw.get(UPDATED_AT))
            item_seen = FeatureSubstitutions()
            features = self._fill(raw, self.item_feature_names, self.defaults["item"], updated_at, as_of, item_seen)
            track_length = _optional_int(raw.get("track_length_seconds"))
            if track_length is not None:
                features["track_length_seconds"] = float(track_length)
                seen.add(item_seen)
            records[item_id] = ItemRecord(item_id, True, features, track_length,
                                          _optional_int(raw.get("artist_id")), _optional_int(raw.get("album_id")),
                                          updated_at)
        self._count(seen, report)
        return records

    def get_user_items(self, user_id: int, item_ids: List[int], as_of: Optional[int] = None,
                       updated_at: Optional[int] = None,
                       report: Optional[FeatureSubstitutions] = None) -> Dict[int, Dict[str, float]]:
        """
        A pair absent from the store means no earlier listen, which is a value
        (zero), not a substitution. `updated_at` is the user's row time.
        """
        ids = [int(i) for i in dict.fromkeys(item_ids)]
        counts = self._fetch_user_items(int(user_id), ids)
        defaults = self.defaults["user_item"]
        stale = {name for name in self.user_item_feature_names if self.is_stale(name, updated_at, as_of)}
        out, seen = {}, FeatureSubstitutions()
        for item_id in ids:
            count = counts.get(item_id)
            if _missing(count):
                out[item_id] = {"previous_listen_count": 0.0, "has_listened_before": 0.0}
                continue
            values = {"previous_listen_count": float(count), "has_listened_before": 1.0}
            for name in stale:
                seen.stale[name] += 1
                values[name] = float(defaults[name])
            out[item_id] = values
        self._count(seen, report)
        return out

    def describe(self) -> dict:
        with self._count_lock:
            counted = {"missing": dict(self.substitutions.missing), "stale": dict(self.substitutions.stale)}
        return {"backend": self.backend, "feature_set_version": self.feature_set_version,
                "counts": self.manifest.get("counts", {}), "substitutions": counted}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

class SQLiteFeatureStore(FeatureStore):
    backend = "sqlite"

    def __init__(self, directory: Path, manifest: dict, contract: dict):
        super().__init__(manifest, contract)
        self.path = require_file(Path(directory) / SQLITE_NAME, "feature store database")
        self._local = threading.local()
        self._item_columns = list(self.item_feature_names) + list(ITEM_ATTRIBUTES) + [UPDATED_AT]

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._local.conn = conn
        return conn

    def _fetch_user(self, user_id):
        cols = ", ".join(self.user_feature_names)
        row = self._conn().execute(f"SELECT {cols}, history, {UPDATED_AT} FROM users WHERE uid = ?",
                                   (user_id,)).fetchone()
        if row is None:
            return None
        features = dict(zip(self.user_feature_names, row[:-2]))
        return features, json.loads(row[-2] or "[]"), _optional_int(row[-1])

    def _fetch_items(self, item_ids):
        out = {}
        cols = ", ".join(self._item_columns)
        for start in range(0, len(item_ids), _SQL_CHUNK):
            chunk = item_ids[start:start + _SQL_CHUNK]
            marks = ",".join("?" * len(chunk))
            for row in self._conn().execute(f"SELECT item_id, {cols} FROM items WHERE item_id IN ({marks})", chunk):
                out[int(row[0])] = dict(zip(self._item_columns, row[1:]))
        return out

    def _fetch_user_items(self, user_id, item_ids):
        out = {}
        for start in range(0, len(item_ids), _SQL_CHUNK):
            chunk = item_ids[start:start + _SQL_CHUNK]
            marks = ",".join("?" * len(chunk))
            query = f"SELECT item_id, previous_listen_count FROM user_items WHERE uid = ? AND item_id IN ({marks})"
            for item_id, count in self._conn().execute(query, [user_id, *chunk]):
                out[int(item_id)] = count
        return out


# ---------------------------------------------------------------------------
# Parquet (in memory)
# ---------------------------------------------------------------------------

class ParquetFeatureStore(FeatureStore):
    backend = "parquet"

    def __init__(self, directory: Path, manifest: dict, contract: dict):
        super().__init__(manifest, contract)
        import pandas as pd

        directory = Path(directory)
        users = pd.read_parquet(require_file(directory / PARQUET_FILES["users"], "users table")).sort_values("uid")
        self._uids = users["uid"].to_numpy(np.int64)
        self._user_values = users[self.user_feature_names].to_numpy(np.float64)
        self._histories = [list(h) if h is not None else [] for h in users["history"].tolist()]
        self._user_updated = users[UPDATED_AT].to_numpy(np.float64)

        items = pd.read_parquet(require_file(directory / PARQUET_FILES["items"], "items table")).sort_values("item_id")
        self._item_ids = items["item_id"].to_numpy(np.int64)
        self._item_columns = list(self.item_feature_names) + list(ITEM_ATTRIBUTES) + [UPDATED_AT]
        self._item_values = items[self._item_columns].to_numpy(np.float64)

        pairs = pd.read_parquet(require_file(directory / PARQUET_FILES["user_items"], "user-item table"))
        pairs = pairs.sort_values(["uid", "item_id"])
        self._pair_uids = pairs["uid"].to_numpy(np.int64)
        self._pair_items = pairs["item_id"].to_numpy(np.int64)
        self._pair_counts = pairs["previous_listen_count"].to_numpy(np.float64)

    @staticmethod
    def _find(sorted_keys: np.ndarray, key: int) -> Optional[int]:
        pos = int(np.searchsorted(sorted_keys, key))
        return pos if pos < len(sorted_keys) and sorted_keys[pos] == key else None

    def _fetch_user(self, user_id):
        pos = self._find(self._uids, user_id)
        if pos is None:
            return None
        return (dict(zip(self.user_feature_names, self._user_values[pos])), self._histories[pos],
                _optional_int(self._user_updated[pos]))

    def _fetch_items(self, item_ids):
        if not item_ids:
            return {}
        keys = np.asarray(item_ids, dtype=np.int64)
        pos = np.searchsorted(self._item_ids, keys).clip(0, max(len(self._item_ids) - 1, 0))
        hit = self._item_ids[pos] == keys if len(self._item_ids) else np.zeros(len(keys), bool)
        return {int(k): dict(zip(self._item_columns, self._item_values[p]))
                for k, p, h in zip(keys, pos, hit) if h}

    def _fetch_user_items(self, user_id, item_ids):
        lo = np.searchsorted(self._pair_uids, user_id, side="left")
        hi = np.searchsorted(self._pair_uids, user_id, side="right")
        if lo == hi or not item_ids:
            return {}
        items, counts = self._pair_items[lo:hi], self._pair_counts[lo:hi]
        keys = np.asarray(item_ids, dtype=np.int64)
        pos = np.searchsorted(items, keys).clip(0, len(items) - 1)
        hit = items[pos] == keys
        return {int(k): float(counts[p]) for k, p, h in zip(keys, pos, hit) if h}


# ---------------------------------------------------------------------------
# Redis (optional)
# ---------------------------------------------------------------------------

class RedisFeatureStore(FeatureStore):
    """Keys are namespaced by feature_set_version, so two versions can coexist during a switch."""

    backend = "redis"
    in_process = False

    def __init__(self, manifest: dict, contract: dict, client):
        super().__init__(manifest, contract)
        self.client = client
        self.prefix = self.feature_set_version

    def ping(self) -> bool:
        return bool(self.client.ping())

    def _fetch_user(self, user_id):
        raw = self.client.hgetall(f"{self.prefix}:user:{user_id}")
        if not raw:
            return None
        raw = {k.decode() if isinstance(k, bytes) else k: v for k, v in raw.items()}
        features = {name: float(raw[name]) if name in raw and raw[name] not in (b"", "") else None
                    for name in self.user_feature_names}
        updated = raw.get(UPDATED_AT)
        return features, json.loads(raw.get("history", "[]")), \
            _optional_int(float(updated)) if updated not in (None, b"", "") else None

    def _fetch_items(self, item_ids):
        pipe = self.client.pipeline(transaction=False)
        for item_id in item_ids:
            pipe.hgetall(f"{self.prefix}:item:{item_id}")
        out = {}
        for item_id, raw in zip(item_ids, pipe.execute()):
            if raw:
                raw = {k.decode() if isinstance(k, bytes) else k: v for k, v in raw.items()}
                out[item_id] = {k: (float(v) if v not in (b"", "") else None) for k, v in raw.items()}
        return out

    def _fetch_user_items(self, user_id, item_ids):
        if not item_ids:
            return {}
        values = self.client.hmget(f"{self.prefix}:ui:{user_id}", [str(i) for i in item_ids])
        return {i: float(v) for i, v in zip(item_ids, values) if v is not None}


def load_into_redis(directory: Path, client, batch_size: int = 2000) -> dict:
    """
    Copy the Parquet tables into Redis under the feature_set_version prefix.
    The `loaded` marker is written last, so a half-finished load is never
    served: RedisFeatureStore refuses to open without it.
    """
    import pandas as pd

    directory = Path(directory)
    manifest = load_manifest(directory)
    verify_files(directory, {name: manifest["files"][name] for name in PARQUET_FILES.values()}, "feature store")
    prefix = manifest["feature_set_version"]
    client.delete(f"{prefix}:loaded")
    pipe = client.pipeline(transaction=False)
    pending = 0

    def queued():
        nonlocal pending
        pending += 1
        if pending >= batch_size:
            pipe.execute()
            pending = 0

    def value(raw):
        return "" if _missing(raw) else float(raw)

    users = pd.read_parquet(directory / PARQUET_FILES["users"])
    for row in users.to_dict("records"):
        mapping = {name: value(row[name]) for name in (*manifest["user_feature_names"], UPDATED_AT)}
        mapping["history"] = json.dumps([int(i) for i in row["history"]])
        pipe.hset(f"{prefix}:user:{int(row['uid'])}", mapping=mapping)
        queued()
    items = pd.read_parquet(directory / PARQUET_FILES["items"])
    for row in items.to_dict("records"):
        mapping = {name: value(row[name])
                   for name in (*manifest["item_feature_names"], *ITEM_ATTRIBUTES, UPDATED_AT)}
        pipe.hset(f"{prefix}:item:{int(row['item_id'])}", mapping=mapping)
        queued()
    pairs = pd.read_parquet(directory / PARQUET_FILES["user_items"])
    for uid, group in pairs.groupby("uid", sort=False):
        pipe.hset(f"{prefix}:ui:{int(uid)}",
                  mapping={str(int(i)): float(c) for i, c in zip(group["item_id"], group["previous_listen_count"])})
        queued()
    pipe.execute()
    counts = {"users": len(users), "items": len(items), "user_items": len(pairs)}
    client.set(f"{prefix}:loaded", json.dumps(counts))
    return counts


# ---------------------------------------------------------------------------

def load_manifest(directory: Path) -> dict:
    return read_json(Path(directory) / MANIFEST_NAME, "feature store manifest")


def load_contract(directory: Path, verify: bool = True) -> dict:
    """The feature contract written next to the store; rankers read the same file at startup."""
    directory = Path(directory)
    if verify:
        files = load_manifest(directory).get("files", {})
        if CONTRACT_NAME not in files:
            raise ArtifactError(f"The feature store manifest does not list {CONTRACT_NAME}; rebuild the bundle.")
        verify_files(directory, {CONTRACT_NAME: files[CONTRACT_NAME]}, "feature contract")
    return read_json(directory / CONTRACT_NAME, "feature contract")


def open_feature_store(directory: Path, backend: Optional[str] = None, redis_client=None,
                       verify: bool = True) -> FeatureStore:
    directory = Path(directory)
    manifest = load_manifest(directory)
    contract = load_contract(directory, verify)
    backend = (backend or os.environ.get("FEATURE_STORE_BACKEND", "sqlite")).lower()
    files = manifest.get("files", {})
    if backend == "sqlite":
        if verify:
            verify_files(directory, {SQLITE_NAME: files[SQLITE_NAME]}, "feature store")
        return SQLiteFeatureStore(directory, manifest, contract)
    if backend == "parquet":
        if verify:
            verify_files(directory, {name: files[name] for name in PARQUET_FILES.values()}, "feature store")
        return ParquetFeatureStore(directory, manifest, contract)
    if backend == "redis":
        if redis_client is None:
            import redis

            redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        store = RedisFeatureStore(manifest, contract, redis_client)
        marker = redis_client.get(f"{store.prefix}:loaded")
        if not marker:
            raise ArtifactError(
                f"Redis has no features for {store.feature_set_version}. "
                "Load them with: python scripts/load_redis.py"
            )
        return store
    raise ArtifactError(f"Unknown FEATURE_STORE_BACKEND '{backend}' (use sqlite, parquet or redis)")
