"""Small synthetic listens with a shift after the training window."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from harness.splits import SECONDS_PER_DAY, boundaries_from_end


def synthetic_frame(contract: dict[str, Any], seed: int) -> tuple[pl.DataFrame, dict[str, int]]:
    """Build history, train, validation, and test rows on the sealed clock.

    Inside the training window, longer tracks are more often completed.
    That relationship is absent on the validation day and the test day, and
    the base completion rate steps down after training. The shift is planted
    so a reader can see a gap; it is not a claim about Yambda.
    """

    split = contract["split"]
    spec = contract["synthetic"]
    test_end = (
        int(split["train_days"]) * SECONDS_PER_DAY * 2
        + int(split["gap_seconds"])
        + int(split["validation_days"]) * SECONDS_PER_DAY
        + int(split["gap_seconds"])
        + int(split["test_days"]) * SECONDS_PER_DAY
    )
    bounds = boundaries_from_end(
        test_end,
        train_days=int(split["train_days"]),
        validation_days=int(split["validation_days"]),
        test_days=int(split["test_days"]),
        gap_seconds=int(split["gap_seconds"]),
    )
    rng = np.random.default_rng(int(seed))
    users = int(spec["users"])
    items = int(spec["items"])
    pieces = [
        ("history", bounds["history_start"], bounds["train_start"], int(spec["rows_history"]), False),
        ("train", bounds["train_start"], bounds["train_end"], int(spec["rows_train"]), False),
        ("validation", bounds["validation_start"], bounds["validation_end"], int(spec["rows_validation"]), False),
        ("test", bounds["test_start"], bounds["test_end"], int(spec["rows_test"]), True),
    ]
    rates = {"history": 0.50, "train": 0.50, "validation": 0.38, "test": 0.30}
    rows: list[dict[str, Any]] = []
    row_id = 0
    for segment, start, end, count, include_end in pieces:
        timestamps = _timestamps(start, end, count, include_end=include_end)
        for index, timestamp in enumerate(timestamps):
            positive = bool(rng.random() < rates[segment])
            signal = 35.0 if positive and segment in ("history", "train") else 0.0
            length = 160.0 + signal + float(rng.normal(0.0, 18.0))
            rows.append(
                {
                    "row_id": row_id,
                    "segment": segment,
                    "uid": int(rng.integers(0, users)),
                    "item_id": int(rng.integers(0, items)),
                    "timestamp": int(timestamp),
                    "is_organic": int(rng.integers(0, 2)),
                    "played_ratio_pct": 78.0 if positive else 18.0,
                    "track_length_seconds": float(np.clip(length, 40.0, 400.0)),
                }
            )
            row_id += 1
    _repair_classes(rows)
    frame = _repair_with_chapter6_tail(pl.DataFrame(rows)).cast(
        {
            "row_id": pl.Int64,
            "uid": pl.Int32,
            "item_id": pl.Int32,
            "timestamp": pl.Int32,
            "is_organic": pl.Int8,
            "played_ratio_pct": pl.Float32,
            "track_length_seconds": pl.Float32,
        }
    )
    return frame, bounds


def _repair_classes(rows: list[dict[str, Any]]) -> None:
    """Keep both completion classes in every window and in the train tail."""

    by_segment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_segment.setdefault(row["segment"], []).append(row)
    for group in by_segment.values():
        _repair_group(group)
    train = by_segment["train"]
    tail_start = int(len(train) * 0.9)
    _repair_group(train[tail_start:])


def _repair_group(group: list[dict[str, Any]]) -> None:
    if len(group) < 2:
        raise ValueError("A window needs at least two rows to contain both classes")
    positives = [row for row in group if row["played_ratio_pct"] >= 50]
    if len(positives) == 0:
        group[0]["played_ratio_pct"] = 78.0
    elif len(positives) == len(group):
        group[0]["played_ratio_pct"] = 18.0


def _repair_with_chapter6_tail(frame: pl.DataFrame) -> pl.DataFrame:
    """Use Chapter 6's own quantile split so the real train tail has both classes."""

    from harness.chapter6 import polars_pipeline

    train = frame.filter(pl.col("segment") == "train")
    _, tail = polars_pipeline().split_validation_window(train, 0.10)
    positive = tail.filter(pl.col("played_ratio_pct") >= 50).height
    if 0 < positive < tail.height:
        return frame
    row_id = int(tail["row_id"][0])
    flipped = 18.0 if positive == tail.height else 78.0
    return frame.with_columns(
        pl.when(pl.col("row_id") == row_id)
        .then(pl.lit(flipped, dtype=pl.Float32))
        .otherwise(pl.col("played_ratio_pct"))
        .alias("played_ratio_pct")
    )


def _timestamps(start: int, end: int, count: int, *, include_end: bool) -> list[int]:
    if count <= 0:
        raise ValueError("Each window needs at least one row")
    span = end - start
    if span <= count:
        raise ValueError(f"Window [{start}, {end}) cannot hold {count} distinct timestamps")
    if include_end:
        return _timestamps(start, end, count - 1, include_end=False) + [end]
    return [start + (span * i) // count for i in range(count)]
