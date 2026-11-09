"""
Ranking data for Exercise 2 Part B: a short Yambda window, or a synthetic stand-in.

Yambda uses the flat ``listens.parquet`` file, a global temporal split with
the last day as the test window and a 30-minute gap before it, and a listen
counting as positive when at least 50% of the track was played, as in
Chapter 7. Timestamps are seconds from an unknown origin, rounded to 5-second
bins: every value is a multiple of 5 and the 50M file spans about 301 days,
the collection period, so one unit is one second. Hour of day and day of week
are therefore phases relative to that origin, not clock time.

User and item history features are point-in-time: a row's features come
from the ``train_days`` window that ends where that row's window starts, so
a training row never sees its own label and training and test features are
built the same way. Aggregates taken over the training window itself would
include each training row's own outcome; that leak makes completion averages
far more predictive in training than at test time.

The synthetic generator plants feature crosses in the label so a DCN has
something to learn, and gives users and items skewed activity so the
"short versus long history" and "popular versus tail item" slices exist.

Pseudo-requests are user-day groups: all of one user's test rows on one day,
standing in for the candidates scored in one request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

SPARSE = ["uid", "item_id", "hour_of_day", "day_of_week"]
DENSE = ["is_organic", "track_length_seconds", "user_total_listens", "user_avg_completion",
         "user_unique_items", "user_organic_ratio", "item_total_plays", "item_avg_completion",
         "item_unique_listeners", "item_organic_ratio"]
SECONDS_PER_DAY = 86_400
GAP_SECONDS = 1_800


@dataclass
class Split:
    sparse: np.ndarray        # (n, 4) int64 encoded ids, 0 = unknown
    dense: np.ndarray         # (n, n_dense) float32, standardized
    label: np.ndarray         # (n,) float32
    group: np.ndarray         # (n,) int64 pseudo-request id
    user_history: np.ndarray  # (n,) history-window activity of the user
    item_plays: np.ndarray    # (n,) history-window plays of the item


@dataclass
class RankingData:
    name: str
    train: Split
    test: Split
    vocab: Dict[str, int]
    dense_names: List[str]


def _standardize(train: np.ndarray, test: np.ndarray):
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd == 0] = 1.0
    return ((train - mu) / sd).astype(np.float32), ((test - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------- Yambda
def history_features(history: pl.DataFrame):
    """Per-user and per-item activity aggregates over a window of listens."""
    user = history.group_by("uid").agg(
        pl.len().alias("user_total_listens"),
        pl.col("played_ratio_pct").mean().alias("user_avg_completion"),
        pl.col("item_id").n_unique().alias("user_unique_items"),
        pl.col("is_organic").mean().alias("user_organic_ratio"))
    item = history.group_by("item_id").agg(
        pl.len().alias("item_total_plays"),
        pl.col("played_ratio_pct").mean().alias("item_avg_completion"),
        pl.col("uid").n_unique().alias("item_unique_listeners"),
        pl.col("is_organic").mean().alias("item_organic_ratio"))
    return user, item


def load_yambda(data_dir: Path, train_days: int, max_train_rows: int, max_test_rows: int,
                seed: int) -> RankingData:
    path = Path(data_dir) / "listens.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set YAMBDA_DATA_DIR or use --data synthetic")
    lf = pl.scan_parquet(path)
    max_ts = int(lf.select(pl.col("timestamp").max()).collect().item())
    span = train_days * SECONDS_PER_DAY
    test_start = max_ts - SECONDS_PER_DAY
    train_end = test_start - GAP_SECONDS
    train_start = train_end - span
    cols = ["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct", "track_length_seconds"]
    base = (lf.select(cols).filter(pl.col("timestamp") >= max(train_start - span, 0)).collect()
            .with_columns(pl.col(c).cast(pl.Int64) for c in cols))

    def window(lo: int, hi: int) -> pl.DataFrame:
        return base.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))

    train = window(train_start, train_end)
    test = base.filter(pl.col("timestamp") >= test_start)
    if train.height > max_train_rows:
        train = train.sample(max_train_rows, seed=seed)
    if test.height > max_test_rows:
        test = test.sample(max_test_rows, seed=seed)

    users = train["uid"].unique().sort()
    items = train["item_id"].unique().sort()
    uid_map = pl.DataFrame({"uid": users, "uid_code": np.arange(1, len(users) + 1)})
    item_map = pl.DataFrame({"item_id": items, "item_code": np.arange(1, len(items) + 1)})

    def frame(df: pl.DataFrame, history: pl.DataFrame, start: int, n_days: int) -> pl.DataFrame:
        user, item = history_features(history)
        return (df.join(user, on="uid", how="left").join(item, on="item_id", how="left")
                .join(uid_map, on="uid", how="left").join(item_map, on="item_id", how="left")
                .with_columns(
                    ((pl.col("timestamp") // 3_600) % 24).alias("hour_of_day"),
                    ((pl.col("timestamp") // SECONDS_PER_DAY) % 7).alias("day_of_week"),
                    ((pl.col("timestamp") - start) // SECONDS_PER_DAY).clip(0, n_days - 1).alias("day"),
                    (pl.col("played_ratio_pct") >= 50).cast(pl.Float32).alias("label"))
                .fill_null(0))

    tr = frame(train, window(train_start - span, train_start), train_start, train_days)
    te = frame(test, window(train_end - span, train_end), test_start, 1)

    def arrays(df: pl.DataFrame):
        sparse = np.stack([df["uid_code"].to_numpy(), df["item_code"].to_numpy(),
                           df["hour_of_day"].to_numpy() + 1, df["day_of_week"].to_numpy() + 1], 1)
        dense = df.select(DENSE).to_numpy().astype(np.float64)
        for j in (1, 2, 4, 6, 8):  # counts and lengths on a log scale
            dense[:, j] = np.log1p(np.maximum(dense[:, j], 0))
        group = (df["uid"].to_numpy() * 1000 + df["day"].to_numpy() % 1000)
        return sparse.astype(np.int64), dense, df["label"].to_numpy().astype(np.float32), group, \
            df["user_total_listens"].to_numpy(), df["item_total_plays"].to_numpy()

    s_tr, d_tr, y_tr, g_tr, uh_tr, ip_tr = arrays(tr)
    s_te, d_te, y_te, g_te, uh_te, ip_te = arrays(te)
    d_tr, d_te = _standardize(d_tr, d_te)
    vocab = {"uid": len(users) + 1, "item_id": len(items) + 1, "hour_of_day": 25, "day_of_week": 8}
    return RankingData("yambda", Split(s_tr, d_tr, y_tr, g_tr, uh_tr, ip_tr),
                       Split(s_te, d_te, y_te, g_te, uh_te, ip_te), vocab, DENSE)


# ---------------------------------------------------------------- synthetic
def synthetic_ranking(n_users: int, n_items: int, n_train: int, n_test: int, seed: int) -> RankingData:
    """Deterministic interactions with planted crosses and skewed activity."""
    rng = np.random.default_rng(seed)
    user_w = 1.0 / np.arange(1, n_users + 1) ** 0.8
    item_w = 1.0 / np.arange(1, n_items + 1) ** 1.0
    user_bias = rng.normal(0, 0.5, n_users + 1)
    item_bias = rng.normal(0, 0.5, n_items + 1)
    user_vec = rng.normal(0, 1, (n_users + 1, 4))
    item_vec = rng.normal(0, 1, (n_items + 1, 4))
    n_dense = len(DENSE)

    def draw(n: int, day_offset: int):
        u = rng.choice(n_users, n, p=user_w / user_w.sum()) + 1
        it = rng.choice(n_items, n, p=item_w / item_w.sum()) + 1
        hour = rng.integers(0, 24, n)
        day = rng.integers(0, 7, n) + day_offset
        dense = rng.normal(0, 1, (n, n_dense)).astype(np.float32)
        logit = (user_bias[u] + item_bias[it] + (user_vec[u] * item_vec[it]).sum(1) * 0.5
                 + 0.8 * dense[:, 0] * dense[:, 1] - 0.6 * dense[:, 2] * dense[:, 3]
                 + 0.3 * np.cos(hour / 24 * 2 * np.pi) - 0.5)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
        sparse = np.stack([u, it, hour + 1, day % 7 + 1], 1).astype(np.int64)
        return sparse, dense, y, u * 1000 + day

    s_tr, d_tr, y_tr, g_tr = draw(n_train, 0)
    s_te, d_te, y_te, g_te = draw(n_test, 7)
    u_counts = np.bincount(s_tr[:, 0], minlength=n_users + 1)
    i_counts = np.bincount(s_tr[:, 1], minlength=n_items + 1)
    vocab = {"uid": n_users + 1, "item_id": n_items + 1, "hour_of_day": 25, "day_of_week": 8}
    return RankingData("synthetic",
                       Split(s_tr, d_tr, y_tr, g_tr, u_counts[s_tr[:, 0]], i_counts[s_tr[:, 1]]),
                       Split(s_te, d_te, y_te, g_te, u_counts[s_te[:, 0]], i_counts[s_te[:, 1]]),
                       vocab, DENSE)


def load_ranking_data(kind: str, yambda_root: Optional[Path], quick: bool, seed: int,
                      train_days: int) -> RankingData:
    if kind == "auto":
        kind = "yambda" if yambda_root is not None and (Path(yambda_root) / "listens.parquet").exists() \
            else "synthetic"
    if kind == "yambda":
        if yambda_root is None:
            raise FileNotFoundError("Yambda data not found: set YAMBDA_DATA_DIR or use --data synthetic")
        return load_yambda(yambda_root, train_days=1 if quick else train_days,
                           max_train_rows=200_000 if quick else 2_000_000,
                           max_test_rows=50_000 if quick else 300_000, seed=seed)
    return synthetic_ranking(n_users=5_000 if quick else 50_000, n_items=20_000 if quick else 200_000,
                             n_train=200_000 if quick else 2_000_000,
                             n_test=50_000 if quick else 300_000, seed=seed)
