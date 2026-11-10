"""Frozen train, gap, validation, gap, test windows."""

from __future__ import annotations

from typing import Any

SECONDS_PER_DAY = 86400


def boundaries_from_end(
    test_end: int,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    gap_seconds: int,
) -> dict[str, int]:
    """Windows working backward from the last timestamp, in seconds.

    Layout: history, train, 30-minute gap, validation day, 30-minute gap,
    test day. History is as long as the train window and exists so the
    Chapter 6 pipeline can fit aggregates before the training rows.
    """

    test_end = int(test_end)
    test_start = test_end - int(test_days) * SECONDS_PER_DAY
    validation_end = test_start - int(gap_seconds)
    validation_start = validation_end - int(validation_days) * SECONDS_PER_DAY
    train_end = validation_start - int(gap_seconds)
    train_start = train_end - int(train_days) * SECONDS_PER_DAY
    history_start = train_start - int(train_days) * SECONDS_PER_DAY
    bounds = {
        "history_start": history_start,
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
        "gap_seconds": int(gap_seconds),
    }
    validate_boundaries(bounds, validation_days=validation_days, test_days=test_days)
    return bounds


def validate_boundaries(
    bounds: dict[str, int],
    *,
    validation_days: int,
    test_days: int,
) -> None:
    order = [
        bounds["history_start"],
        bounds["train_start"],
        bounds["train_end"],
        bounds["validation_start"],
        bounds["validation_end"],
        bounds["test_start"],
        bounds["test_end"],
    ]
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError(f"Split boundaries are not strictly ordered: {bounds}")
    gap_1 = bounds["validation_start"] - bounds["train_end"]
    gap_2 = bounds["test_start"] - bounds["validation_end"]
    if gap_1 != bounds["gap_seconds"] or gap_2 != bounds["gap_seconds"]:
        raise ValueError("Both gaps must equal the pre-registered gap")
    if bounds["validation_end"] - bounds["validation_start"] != validation_days * SECONDS_PER_DAY:
        raise ValueError("Validation window is not the pre-registered length")
    if bounds["test_end"] - bounds["test_start"] != test_days * SECONDS_PER_DAY:
        raise ValueError("Test window is not the pre-registered length")


def assign_segment(timestamp: int, bounds: dict[str, int]) -> str:
    timestamp = int(timestamp)
    if timestamp < bounds["train_start"]:
        return "history" if timestamp >= bounds["history_start"] else "out_of_range"
    if timestamp < bounds["train_end"]:
        return "train"
    if timestamp < bounds["validation_start"]:
        return "gap"
    if timestamp < bounds["validation_end"]:
        return "validation"
    if timestamp < bounds["test_start"]:
        return "gap"
    if timestamp <= bounds["test_end"]:
        return "test"
    return "out_of_range"


def bounds_document(bounds: dict[str, int], contract: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "manifest_id": contract["manifest_id"],
        "mode": mode,
        "label_definition": contract["label_definition"],
        "label_threshold": int(contract["label_threshold"]),
        "primary_metric": contract["primary_metric"],
        "primary_split": contract["primary_split"],
        "early_stopping_fraction": float(contract["split"]["early_stopping_fraction"]),
        "early_stopping_split": "train_tail",
        "boundaries": {key: int(value) for key, value in bounds.items()},
        "notes": [
            "Scoring uses the validation day only.",
            "Early stopping uses the last 10 percent of the training window.",
            "Test labels are written under the seal directory by the auditor.",
        ],
    }
