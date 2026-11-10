"""
Yandex Yambda -> HSTU (item, action, time, value) sequences with leave-one-out splits.

Per user, ordered by time:
    test        history = events[:-1],  target = last event
    validation  history = events[:-2],  target = penultimate event
    training    events[:-2], cut into windows of max_seq_len + 1 events that overlap by
                one event, so every next-event transition in the training range is
                predicted exactly once (the loss covers every position of a window)

The auxiliary value is played_ratio_pct / 100, clipped to [0, 1]. Events without a
played ratio (likes, dislikes and their undo events) carry NaN, which the rating loss
skips; earlier versions filled them with 0 and trained the head toward 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import polars as pl

# Yambda timestamps are seconds since the first event in the dataset, rounded down to a
# multiple of 5 (Yambda paper, Section 3.3). They are not Unix time and not counts of
# 5-second intervals: the 50M release spans about 301 days read as seconds.
SECONDS_PER_DAY = 86_400

# Native Yambda event types -> contiguous action ids (0 reserved for padding).
ACTION_MAP: Dict[str, int] = {
    "listen": 1,
    "like": 2,
    "dislike": 3,
    "unlike": 4,
    "undislike": 5,
}
NUM_ACTIONS = len(ACTION_MAP)

DATASET_ROOT = Path(os.environ.get("RECSYS_DATASET_ROOT", Path(__file__).resolve().parents[3] / "Dataset"))
DEFAULT_DATA_DIR = DATASET_ROOT / "Yandex" / "flat"


@dataclass
class SequenceSample:
    items: List[int]
    actions: List[int]
    times: List[int]
    values: List[float]
    target_item: int
    target_action: int
    target_time: int
    target_value: float
    uid: int


@dataclass
class HSTUDataBundle:
    num_items: int
    num_actions: int
    item_id_map: Dict[int, int]
    reverse_item_map: Dict[int, int]
    train_samples: List[SequenceSample]
    val_samples: List[SequenceSample]
    test_samples: List[SequenceSample]
    stats: Dict[str, Any] = field(default_factory=dict)


def seconds_for_days(days: float) -> int:
    return int(days * SECONDS_PER_DAY)


def load_yambda_events(
    data_dir: Path,
    max_days: Optional[float] = 30.0,
    use_last_window: bool = False,
) -> pl.DataFrame:
    """
    Load multi_event.parquet with an optional time window.

    Args:
        data_dir: Path to Yandex/flat.
        max_days: Keep this many days of data (timestamps are seconds). None keeps all rows.
        use_last_window: If True, keep the most recent max_days; else first max_days.

    The returned ``seconds`` column is the Yambda timestamp unchanged.
    """
    multi_path = Path(data_dir) / "multi_event.parquet"
    if not multi_path.exists():
        raise FileNotFoundError(f"Expected multi_event.parquet at {multi_path}")

    lf = pl.scan_parquet(str(multi_path)).select(
        [
            pl.col("uid").cast(pl.Int32),
            pl.col("item_id").cast(pl.Int32),
            pl.col("timestamp").cast(pl.Int64),
            pl.col("played_ratio_pct").cast(pl.Float32),
            pl.col("event_type").cast(pl.Utf8),
        ]
    )

    if max_days is not None:
        ts_stats = (
            pl.scan_parquet(str(multi_path))
            .select(
                pl.col("timestamp").min().alias("min_ts"),
                pl.col("timestamp").max().alias("max_ts"),
            )
            .collect()
        )
        min_ts = int(ts_stats["min_ts"][0])
        max_ts = int(ts_stats["max_ts"][0])
        window_seconds = seconds_for_days(max_days)
        if use_last_window:
            lo, hi = max_ts - window_seconds, max_ts
        else:
            lo, hi = min_ts, min_ts + window_seconds
        lf = lf.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") <= hi))

    df = (
        lf.sort(["uid", "timestamp"], maintain_order=True)
        .with_columns(
            pl.col("event_type")
            .replace_strict(ACTION_MAP, default=None)
            .alias("action_id"),
            pl.col("timestamp").alias("seconds"),
            (pl.col("played_ratio_pct") / 100.0).clip(0.0, 1.0).alias("value"),
        )
        .filter(pl.col("action_id").is_not_null())
        .collect(engine="streaming")
    )
    return df


def _build_item_maps(item_ids: pl.Series) -> Tuple[Dict[int, int], Dict[int, int], int]:
    unique_items = sorted(item_ids.unique().to_list())
    item_id_map = {orig: idx + 1 for idx, orig in enumerate(unique_items)}
    reverse_item_map = {v: k for k, v in item_id_map.items()}
    return item_id_map, reverse_item_map, len(unique_items)


def _truncate_tail(values: List[Any], max_len: int) -> List[Any]:
    if len(values) <= max_len:
        return values
    return values[-max_len:]


def _sample(items, actions, times, values, end: int, max_len: int, uid: int) -> SequenceSample:
    """History = events[:end] (most recent max_len), target = events[end]."""
    start = max(0, end - max_len)
    return SequenceSample(
        items=items[start:end], actions=actions[start:end], times=times[start:end], values=values[start:end],
        target_item=items[end], target_action=actions[end], target_time=times[end], target_value=values[end],
        uid=uid,
    )


def training_windows(n_events: int, max_seq_len: int) -> List[Tuple[int, int]]:
    """(start, end) windows over events[0:n_events] of at most max_seq_len + 1 events.

    Consecutive windows overlap by one event, so every transition t-1 -> t with
    1 <= t < n_events appears as a training target in exactly one window. Windows are
    laid out from the most recent event backwards.
    """
    windows: List[Tuple[int, int]] = []
    end = n_events
    span = max_seq_len + 1
    while end >= 2:
        start = max(0, end - span)
        windows.append((start, end))
        if start == 0:
            break
        end = start + 1
    return windows[::-1]


def split_user(uid: int, items: Sequence[int], actions: Sequence[int], times: Sequence[int],
               values: Sequence[float], max_seq_len: int):
    """Return (train_windows, val_sample, test_sample) for one user's time-ordered events."""
    items, actions, times, values = list(items), list(actions), list(times), list(values)
    n = len(items)
    test = _sample(items, actions, times, values, n - 1, max_seq_len, uid)
    val = _sample(items, actions, times, values, n - 2, max_seq_len, uid)
    train = []
    for start, end in training_windows(n - 2, max_seq_len):
        # stored as history = window[:-1], target = window[-1]; NextItemDataset rejoins them
        train.append(_sample(items[start:end], actions[start:end], times[start:end], values[start:end],
                             end - start - 1, max_seq_len, uid))
    return train, val, test


def repeat_rate(samples: Sequence[SequenceSample], window: Optional[int] = None) -> float:
    if not samples:
        return 0.0
    hits = 0
    for s in samples:
        hist = s.items if window is None else s.items[-window:]
        hits += s.target_item in set(hist)
    return hits / len(samples)


def build_leave_one_out_splits(
    df: pl.DataFrame,
    max_seq_len: int = 200,
    min_seq_len: int = 3,
    time_column: str = "seconds",
    value_column: str = "value",
    dataset_name: str = "yambda",
) -> HSTUDataBundle:
    """Leave-one-out split; ``df`` needs uid, item_id, action_id, a time column and a value column."""
    item_id_map, reverse_item_map, num_items = _build_item_maps(df["item_id"])
    num_actions = int(df["action_id"].max())

    train_samples: List[SequenceSample] = []
    val_samples: List[SequenceSample] = []
    test_samples: List[SequenceSample] = []

    grouped = df.group_by("uid", maintain_order=True).agg(
        pl.col("item_id").alias("items"),
        pl.col("action_id").alias("actions"),
        pl.col(time_column).alias("times"),
        pl.col(value_column).alias("values"),
    )
    users_kept = 0
    test_repeats_full_history = 0
    val_repeats_full_history = 0
    for row in grouped.iter_rows(named=True):
        if len(row["items"]) < min_seq_len:
            continue
        users_kept += 1
        mapped_items = [item_id_map[i] for i in row["items"]]
        test_repeats_full_history += mapped_items[-1] in set(mapped_items[:-1])
        val_repeats_full_history += mapped_items[-2] in set(mapped_items[:-2])
        train, val, test = split_user(
            int(row["uid"]),
            mapped_items,
            [int(a) for a in row["actions"]],
            [int(t) for t in row["times"]],
            [float(v) if v is not None and not math.isnan(v) else math.nan for v in row["values"]],
            max_seq_len,
        )
        train_samples.extend(train)
        val_samples.append(val)
        test_samples.append(test)

    stats = {
        "dataset": dataset_name,
        "num_rows": len(df),
        "num_users": df["uid"].n_unique(),
        "num_users_kept": users_kept,
        "num_items_mapped": num_items,
        "num_actions": num_actions,
        "train_windows": len(train_samples),
        "train_transitions": sum(len(s.items) for s in train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        # samples keep only the last max_seq_len events, so the full-history rates are counted above
        "test_target_in_history_rate": test_repeats_full_history / users_kept if users_kept else 0.0,
        "test_target_in_model_window_rate": repeat_rate(test_samples, max_seq_len),
        "val_target_in_history_rate": val_repeats_full_history / users_kept if users_kept else 0.0,
        "max_seq_len": max_seq_len,
    }
    if "event_type" in df.columns:
        stats["event_type_counts"] = dict(zip(*df["event_type"].value_counts(sort=True).to_dict(as_series=False).values()))
    return HSTUDataBundle(
        num_items=num_items,
        num_actions=num_actions,
        item_id_map=item_id_map,
        reverse_item_map=reverse_item_map,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        stats=stats,
    )


def load_hstu_data(
    data_dir: Path,
    max_days: Optional[float] = 30.0,
    max_seq_len: int = 200,
    use_last_window: bool = False,
) -> HSTUDataBundle:
    df = load_yambda_events(data_dir, max_days=max_days, use_last_window=use_last_window)
    bundle = build_leave_one_out_splits(df, max_seq_len=max_seq_len)
    bundle.num_actions = NUM_ACTIONS
    bundle.stats["time_unit"] = "seconds"
    bundle.stats["max_days"] = max_days
    bundle.stats["window"] = "last" if use_last_window else "first"
    bundle.stats["span_days"] = (
        (int(df["seconds"].max()) - int(df["seconds"].min())) / SECONDS_PER_DAY if len(df) else 0.0
    )
    return bundle


def save_data_bundle(bundle: HSTUDataBundle, path: Path) -> None:
    payload = {
        "num_items": bundle.num_items,
        "num_actions": bundle.num_actions,
        "stats": bundle.stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HSTU Yambda splits and print their statistics")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-days", type=float, default=30.0)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--use-last-window", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "data_stats.json",
    )
    args = parser.parse_args()

    bundle = load_hstu_data(
        args.data_dir,
        max_days=args.max_days,
        max_seq_len=args.max_seq_len,
        use_last_window=args.use_last_window,
    )
    save_data_bundle(bundle, args.output)
    print(json.dumps(bundle.stats, indent=2))


if __name__ == "__main__":
    main()
