"""
Chapter 15: Synthetic Yambda-Shaped Data
========================================
Writes a small listens.parquet with the Yambda flat schema so that the whole
exercise, including the Chapter 6 loader and feature pipeline, can run in
seconds without the dataset. Used by the smoke run and the end-to-end test.

The completion probability depends on a user propensity, an item quality,
the organic flag, track length, and whether the track is one of the user's
favourites, so the lookup-table and re-listen features carry real signal.
These numbers say nothing about Yambda; they only exercise the code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

TS_UNITS_PER_DAY = 86_400          # timestamps in seconds, as in Yambda


def write_synthetic_yambda(out_dir: Path, n_users: int = 400, n_items: int = 1500,
                           n_days: int = 40, events_per_user_day: float = 4.0,
                           seed: int = 0) -> Path:
    """Write out_dir/listens.parquet and return out_dir."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_propensity = rng.normal(0.0, 1.0, n_users)
    user_activity = rng.lognormal(0.0, 0.5, n_users)
    item_quality = rng.normal(0.0, 1.0, n_items)
    item_length = rng.integers(60, 420, n_items)
    popularity = 1.0 / np.arange(1, n_items + 1) ** 1.1
    popularity /= popularity.sum()

    rows = []
    for u in range(n_users):
        n = rng.poisson(events_per_user_day * n_days * user_activity[u])
        if n == 0:
            continue
        favourites = rng.choice(n_items, size=20, replace=False, p=popularity)
        is_fav = rng.random(n) < 0.35
        items = np.where(is_fav, rng.choice(favourites, size=n),
                         rng.choice(n_items, size=n, p=popularity))
        ts = np.sort(rng.integers(0, n_days * TS_UNITS_PER_DAY // 5, n)) * 5
        organic = (rng.random(n) < 0.35).astype(np.uint8)
        length_z = (item_length[items] - 240.0) / 100.0
        logit = (0.9 * user_propensity[u] + 0.7 * item_quality[items] + 0.5 * organic
                 + 0.6 * is_fav - 0.4 * length_z + rng.normal(0.0, 1.0, n))
        played = np.clip(100.0 / (1.0 + np.exp(-logit)) + rng.normal(0.0, 10.0, n), 0, 150)
        rows.append(pl.DataFrame({
            "uid": np.full(n, 100 + u, dtype=np.uint32),
            "timestamp": ts.astype(np.uint32),
            "item_id": (items + 1).astype(np.uint32),
            "is_organic": organic,
            "played_ratio_pct": played.round().astype(np.uint16),
            "track_length_seconds": item_length[items].astype(np.uint32),
        }))
    pl.concat(rows).sort(["uid", "timestamp"]).write_parquet(out_dir / "listens.parquet")
    return out_dir
