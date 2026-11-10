"""Noise band: five baseline seeds and a user-level bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def sample_std(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise ValueError("A spread needs at least two values")
    return float(array.std(ddof=1))


def user_bootstrap_std(frame: pl.DataFrame, *, replicates: int, seed: int) -> float:
    """Standard deviation of validation AUC across user resamples."""

    if replicates < 20:
        raise ValueError("Use at least 20 bootstrap replicates")
    users = frame["uid"].unique().to_list()
    groups = {
        user: frame.filter(pl.col("uid") == user)
        for user in users
    }
    rng = np.random.default_rng(int(seed))
    scores: list[float] = []
    for _ in range(int(replicates)):
        drawn = rng.choice(np.asarray(users), size=len(users), replace=True)
        sample = pl.concat([groups[user] for user in drawn.tolist()])
        labels = sample["label"].to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        scores.append(float(roc_auc_score(labels, sample["prediction"].to_numpy())))
    if len(scores) < 20:
        raise ValueError("Too few bootstrap resamples contained both classes")
    return sample_std(scores)


def build_noise_band(
    seed_aucs: list[float],
    bootstrap_std: float,
) -> dict[str, Any]:
    seed_spread = sample_std(seed_aucs)
    band = float(max(seed_spread, float(bootstrap_std)))
    return {
        "method": "baseline_seeds_plus_user_bootstrap",
        "baseline_seeds": len(seed_aucs),
        "baseline_seed_aucs": [float(value) for value in seed_aucs],
        "baseline_mean_auc": float(np.mean(np.asarray(seed_aucs, dtype=float))),
        "seed_std": seed_spread,
        "bootstrap_std": float(bootstrap_std),
        "noise_band": band,
        "rule": "winner_validation_auc_minus_baseline_mean_gt_noise_band",
    }
