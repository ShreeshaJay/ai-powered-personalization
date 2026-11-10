"""Metrics recomputed by the harness. Trials do not report these themselves."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Equal-width expected calibration error for scores in ``[0, 1]``."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1], right=True)
    total = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        confidence = float(np.mean(y_score[mask]))
        accuracy = float(np.mean(y_true[mask]))
        total += weight * abs(accuracy - confidence)
    return float(total)


def score_joined(frame: pl.DataFrame, *, split: str, n_bins: int = 15) -> dict[str, Any]:
    """Score a frame that already has ``label`` and ``prediction``."""

    y_true = frame["label"].to_numpy().astype(int)
    y_score = frame["prediction"].to_numpy().astype(float)
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty frame")
    if np.isnan(y_score).any() or (y_score < 0.0).any() or (y_score > 1.0).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    classes = set(np.unique(y_true).tolist())
    if not classes.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1, got {sorted(classes)}")
    if len(classes) < 2:
        raise ValueError("AUC needs both classes")
    clipped = np.clip(y_score, 1e-6, 1.0 - 1e-6)
    return {
        "split": split,
        "primary_metric": "auc_roc",
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "expected_calibration_error": expected_calibration_error(y_true, y_score, n_bins=n_bins),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(y_score)),
    }


def join_and_score(
    predictions: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    split: str,
) -> dict[str, Any]:
    required_pred = {"row_id", "prediction"}
    required_label = {"row_id", "uid", "label"}
    if not required_pred.issubset(predictions.columns):
        raise ValueError(f"Predictions need {sorted(required_pred)}")
    if not required_label.issubset(labels.columns):
        raise ValueError(f"Labels need {sorted(required_label)}")
    merged = labels.join(
        predictions.select(["row_id", "prediction"]),
        on="row_id",
        how="inner",
    )
    metrics = score_joined(merged, split=split)
    metrics["n_joined"] = int(merged.height)
    return metrics
