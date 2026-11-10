"""Leakage, coverage, and reproduction checks for one submission."""

from __future__ import annotations

from typing import Any, Iterable

import polars as pl


BANNED_FEATURES = {
    "timestamp",
    "played_ratio_pct",
    "label",
    "target",
    "split",
    "segment",
    "validation_label",
    "test_label",
    "row_id",
}


class LeakageError(ValueError):
    """The submission violates the sealed leakage policy."""


def check_feature_columns(columns: Iterable[str]) -> list[str]:
    violations = []
    for name in columns:
        lowered = str(name).strip().lower()
        if lowered in BANNED_FEATURES or "label" in lowered:
            violations.append(f"Feature `{name}` is outside the leakage policy")
    return violations


def check_metadata(metadata: dict[str, Any]) -> list[str]:
    violations = []
    if metadata.get("early_stopping_split") != "train_tail":
        violations.append("early_stopping_split must be train_tail")
    if metadata.get("model_selection_split") != "validation":
        violations.append("model_selection_split must be validation")
    if metadata.get("optimization_split") != "validation":
        violations.append("optimization_split must be validation")
    if metadata.get("preprocessor_fit_split") != "train":
        violations.append("preprocessor_fit_split must be train")
    if metadata.get("primary_metric") != "auc_roc":
        violations.append("primary_metric must be auc_roc")
    if metadata.get("primary_split") != "validation":
        violations.append("primary_split must be validation")
    return violations


def check_coverage(predictions: pl.DataFrame, labels: pl.DataFrame) -> list[str]:
    pred_ids = predictions["row_id"].to_list()
    label_ids = labels["row_id"].to_list()
    violations = []
    if len(pred_ids) != len(set(pred_ids)):
        violations.append("Predictions repeat row_id values")
    missing = set(label_ids) - set(pred_ids)
    extra = set(pred_ids) - set(label_ids)
    if missing:
        violations.append(f"Predictions drop {len(missing)} labelled rows")
    if extra:
        violations.append(f"Predictions include {len(extra)} rows that have no label")
    return violations


def check_reproduction(
    claimed: dict[str, Any] | None,
    recomputed: dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """A claimed metric that disagrees with the harness is a forged score."""

    if not claimed:
        return []
    violations = []
    for key, value in claimed.items():
        if key not in recomputed:
            continue
        if abs(float(value) - float(recomputed[key])) > tolerance:
            violations.append(f"Claimed {key} does not match the harness")
    return violations


def enforce(violations: list[str]) -> None:
    if violations:
        raise LeakageError("; ".join(violations))
