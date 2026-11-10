"""Auditor step: freeze the split, write validation labels, seal the test labels.

Run from this folder:

    python build_harness.py --synthetic --run-dir runs/synthetic
    python build_harness.py --data-dir path/to/Yandex/flat --run-dir runs/yambda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from harness.contract import (
    artifacts_dir,
    contract_path,
    file_sha256,
    load_contract,
)
from harness.seal import seal_dir, write_seal
from harness.splits import assign_segment, boundaries_from_end, bounds_document
from harness.synthetic import synthetic_frame

RAW_COLUMNS = [
    "uid",
    "item_id",
    "timestamp",
    "is_organic",
    "played_ratio_pct",
    "track_length_seconds",
]


def build(run_dir: Path, *, synthetic: bool, data_dir: Path | None, force: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(f"{run_dir} already exists. Pass --force to rebuild it.")
    contract = load_contract()
    if synthetic:
        frame, bounds = synthetic_frame(contract, int(contract["seeds"]["master"]))
        mode = "synthetic"
    else:
        if data_dir is None:
            raise ValueError("A Yambda run needs --data-dir or YAMBDA_DATA_DIR")
        frame, bounds = _load_yambda(data_dir, contract)
        mode = "yambda"
    _write_outputs(run_dir, frame, bounds, contract, mode)


def _load_yambda(data_dir: Path, contract: dict) -> tuple[pl.DataFrame, dict]:
    path = Path(data_dir) / "listens.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            "listens.parquet was not found in the data directory. "
            "Chapter 6 documents the Yambda download."
        )
    frame = pl.scan_parquet(str(path))
    names = frame.collect_schema().names()
    missing = [column for column in RAW_COLUMNS if column not in names]
    if missing:
        raise ValueError(f"Listens file is missing columns: {missing}")
    max_ts = int(frame.select(pl.col("timestamp").max()).collect().item())
    split = contract["split"]
    bounds = boundaries_from_end(
        max_ts,
        train_days=int(split["train_days"]),
        validation_days=int(split["validation_days"]),
        test_days=int(split["test_days"]),
        gap_seconds=int(split["gap_seconds"]),
    )
    collected = (
        frame.select(RAW_COLUMNS)
        .filter(
            (pl.col("timestamp") >= bounds["history_start"])
            & (pl.col("timestamp") <= bounds["test_end"])
        )
        .collect()
        .sort("timestamp")
        .with_row_index("row_id")
    )
    collected = collected.with_columns(pl.col("row_id").cast(pl.Int64))
    segments = [assign_segment(int(ts), bounds) for ts in collected["timestamp"].to_list()]
    collected = collected.with_columns(pl.Series("segment", segments))
    return collected, bounds


def _write_outputs(run_dir: Path, frame: pl.DataFrame, bounds: dict, contract: dict, mode: str) -> None:
    threshold = int(contract["label_threshold"])
    frame = frame.with_columns(
        (pl.col("played_ratio_pct") >= threshold).cast(pl.Int8).alias("label")
    )
    artifacts = artifacts_dir(run_dir)
    sealed = seal_dir(run_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)

    def part(name: str) -> pl.DataFrame:
        return frame.filter(pl.col("segment") == name)

    history = part("history")
    train = part("train")
    validation = part("validation")
    test = part("test")
    for name, piece in (("train", train), ("validation", validation), ("test", test)):
        if piece.height == 0:
            raise RuntimeError(f"The {name} window has no rows")
        classes = piece["label"].n_unique()
        if classes < 2:
            raise RuntimeError(f"The {name} window has a single class")

    listens = pl.concat([history, train], how="vertical") if history.height else train
    listens.select(RAW_COLUMNS).write_parquet(artifacts / "listens.parquet")
    keep = ["row_id", *RAW_COLUMNS, "label"]
    train.select(keep).write_parquet(artifacts / "train_rows.parquet")
    validation.select(keep).write_parquet(artifacts / "validation_rows.parquet")
    validation.select(["row_id", "uid", "label"]).write_parquet(artifacts / "validation_labels.parquet")
    test.select(keep).write_parquet(sealed / "test_rows.parquet")
    test.select(["row_id", "uid", "label"]).write_parquet(sealed / "test_labels.parquet")

    manifest = bounds_document(bounds, contract, mode)
    manifest["row_counts"] = {
        "history": int(history.height),
        "train": int(train.height),
        "validation": int(validation.height),
        "test": int(test.height),
    }
    (artifacts / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_seal(
        run_dir,
        contract_hash=file_sha256(contract_path()),
        test_budget=int(contract["test_budget"]),
    )
    print(
        f"Harness built ({mode}): train {train.height}, "
        f"validation {validation.height}, test {test.height}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the sealed Appendix C harness")
    parser.add_argument("--synthetic", action="store_true", help="Build a tiny drifted dataset")
    parser.add_argument("--data-dir", type=Path, default=None, help="Yambda flat directory")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic and args.data_dir is None:
        env_dir = __import__("os").environ.get("YAMBDA_DATA_DIR")
        if env_dir:
            args.data_dir = Path(env_dir)
    run_dir = args.run_dir
    if run_dir is None:
        from harness.contract import ROOT
        run_dir = ROOT / "runs" / ("synthetic" if args.synthetic else "yambda")
    try:
        build(run_dir, synthetic=args.synthetic, data_dir=args.data_dir, force=args.force)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
