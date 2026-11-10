"""
MovieLens-1M -> HSTU (item, action, time, value) sequences with leave-one-out splits.

Follows Rahul Agarwal's MLWhiz tutorial for the action vocabulary:
  - Ratings mapped to POSITIVE (>=4), NEUTRAL (=3), NEGATIVE (<3)
  - Unix timestamps used directly for the relative time-bucket bias
The auxiliary value is the rating rescaled to [0, 1]: (rating - 1) / 4.
Splits use the same per-user protocol as data_loader.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import polars as pl

from data_loader import DATASET_ROOT, HSTUDataBundle, build_leave_one_out_splits, save_data_bundle

# Tutorial action vocabulary (0 = padding).
ML1M_ACTION_MAP: Dict[str, int] = {
    "POSITIVE": 1,
    "NEUTRAL": 2,
    "NEGATIVE": 3,
}
NUM_ML1M_ACTIONS = len(ML1M_ACTION_MAP)

DEFAULT_ML1M_DIR = DATASET_ROOT / "MovieLens" / "ml-1m"


def rating_to_action_id(rating: int) -> int:
    if rating >= 4:
        return ML1M_ACTION_MAP["POSITIVE"]
    if rating == 3:
        return ML1M_ACTION_MAP["NEUTRAL"]
    return ML1M_ACTION_MAP["NEGATIVE"]


def load_movielens_ratings(data_dir: Path) -> pl.DataFrame:
    ratings_path = Path(data_dir) / "ratings.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"MovieLens-1M not found at {ratings_path}. "
            "Download from https://grouplens.org/datasets/movielens/1m/"
        )
    lines = ratings_path.read_text(encoding="latin-1").splitlines()
    rows = [tuple(int(x) for x in line.split("::")) for line in lines if line.strip()]
    df = pl.DataFrame(rows, schema=["uid", "item_id", "rating", "timestamp"], orient="row")
    return (
        df.sort(["uid", "timestamp"], maintain_order=True)
        .with_columns(
            pl.when(pl.col("rating") >= 4)
            .then(pl.lit(ML1M_ACTION_MAP["POSITIVE"]))
            .when(pl.col("rating") == 3)
            .then(pl.lit(ML1M_ACTION_MAP["NEUTRAL"]))
            .otherwise(pl.lit(ML1M_ACTION_MAP["NEGATIVE"]))
            .alias("action_id"),
            ((pl.col("rating").cast(pl.Float32) - 1.0) / 4.0).alias("value"),
            pl.col("timestamp").cast(pl.Int64).alias("seconds"),
        )
    )


def load_movielens_data(data_dir: Path, max_seq_len: int = 200) -> HSTUDataBundle:
    df = load_movielens_ratings(data_dir)
    bundle = build_leave_one_out_splits(df, max_seq_len=max_seq_len, dataset_name="movielens-1m")
    bundle.num_actions = NUM_ML1M_ACTIONS
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MovieLens-1M HSTU splits and print their statistics")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ML1M_DIR)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "ml1m" / "data_stats.json",
    )
    args = parser.parse_args()

    bundle = load_movielens_data(args.data_dir, max_seq_len=args.max_seq_len)
    save_data_bundle(bundle, args.output)
    print(json.dumps(bundle.stats, indent=2))


if __name__ == "__main__":
    main()
