"""
Chapter 15: Serving-Side Corruption
===================================
Utilities that reproduce a silent, schema-valid upstream failure: a feature
arrives filled with a default value instead of its real value. Training is
never touched; only the frame the model scores is corrupted.

Two fill values are supported:

- zero-fill: the column arrives as 0, the classic "default when the join or
  upstream job silently failed" pattern
- mean-fill: the column arrives as its training-set mean, a "plausible"
  default that leaves the column mean unchanged

Two scopes are supported:

- all traffic
- a user segment chosen by a deterministic hash of the user ID, so the same
  users fall in the segment in every window and on every machine

Zeros and means keep each column's dtype and fall inside every type-and-range
contract, so schema validation still passes. That is the failure mode the
exercise is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import polars as pl

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL2 = np.uint64(0x94D049BB133111EB)


def splitmix64(x) -> np.ndarray:
    """SplitMix64 output for state x, with the constants of Vigna's reference splitmix64.c.

    splitmix64(0) is the first output of a SplitMix64 generator seeded with 0.
    Unsigned 64-bit arithmetic wraps, which the algorithm relies on.
    """
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _MUL1
        z = (z ^ (z >> np.uint64(27))) * _MUL2
        return z ^ (z >> np.uint64(31))


def hash_to_unit_interval(ids, salt: int = 0) -> np.ndarray:
    """Map integer IDs to [0, 1) with SplitMix64, using the top 53 bits for a float64."""
    ids = np.asarray(ids, dtype=np.uint64)
    salted = ids ^ splitmix64(np.array([salt], dtype=np.uint64))[0]
    return (splitmix64(salted) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def segment_mask(ids, fraction: float = 0.3, salt: int = 0) -> np.ndarray:
    """True for IDs whose hash falls below fraction: a stable pseudo-random segment."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return hash_to_unit_interval(ids, salt) < fraction


@dataclass(frozen=True)
class Scenario:
    """One serving-side corruption scenario."""
    name: str
    fill: str          # 'none', 'zero', or 'mean'
    scope: str         # 'none', 'all', or 'segment'
    description: str


SCENARIOS = (
    Scenario("clean", "none", "none",
             "Control: the test window scored as it arrived."),
    Scenario("zero_fill_all", "zero", "all",
             "Target features arrive as 0 for every request."),
    Scenario("zero_fill_segment", "zero", "segment",
             "Target features arrive as 0 for the hashed user segment only."),
    Scenario("mean_fill_all", "mean", "all",
             "Target features arrive as their training-set mean for every request."),
)


def _cast_scalar(value: float, dtype: pl.DataType):
    """Round to the nearest integer for integer columns so the dtype contract still holds."""
    if dtype.is_integer():
        return int(round(value))
    return float(value)


def compute_fill_values(train_df: pl.DataFrame, features: Sequence[str],
                        fill: str) -> Dict[str, float]:
    """The value each feature is overwritten with, cast to the feature's dtype."""
    values = {}
    for f in features:
        dtype = train_df.schema[f]
        if fill == "zero":
            raw = 0.0
        elif fill == "mean":
            raw = float(train_df[f].cast(pl.Float64).mean())
        else:
            raise ValueError(f"Unknown fill: {fill!r}")
        values[f] = _cast_scalar(raw, dtype)
    return values


def apply_corruption(df: pl.DataFrame, fill_values: Dict[str, float],
                     row_mask: Optional[Iterable[bool]] = None) -> pl.DataFrame:
    """Overwrite each feature with its fill value, on every row or only where row_mask is True."""
    mask = pl.lit(True) if row_mask is None else pl.lit(pl.Series(np.asarray(row_mask, dtype=bool)))
    exprs = []
    for f, v in fill_values.items():
        dtype = df.schema[f]
        exprs.append(pl.when(mask).then(pl.lit(v, dtype=dtype)).otherwise(pl.col(f)).alias(f))
    return df.with_columns(exprs)


def corrupt(df: pl.DataFrame, scenario: Scenario, fill_values: Dict[str, Dict[str, float]],
            segment: np.ndarray) -> pl.DataFrame:
    """Apply one scenario to a scoring frame.

    fill_values maps a fill name ('zero' or 'mean') to {feature: value};
    segment is the per-row boolean segment membership of df.
    """
    if scenario.fill == "none":
        return df
    mask = None if scenario.scope == "all" else segment
    return apply_corruption(df, fill_values[scenario.fill], mask)
