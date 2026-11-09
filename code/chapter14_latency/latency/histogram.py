"""
Log-bucketed latency histogram.

Percentiles cannot be averaged across hosts, time windows, or runs. The
correct way to combine latency measurements is to merge their histograms and
read percentiles from the merged counts. This module provides a small
histogram with bounded relative error, in the spirit of HdrHistogram:

* every bucket spans a constant ratio ``1 + precision``, so any reported
  percentile is within about ``precision / 2`` of a value that was recorded;
* histograms with the same layout merge by adding counts;
* ``record_corrected`` back-fills the samples a closed-loop load generator
  never sent while the server was stalled (coordinated omission correction,
  Tene, 2012).

Values are in milliseconds.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np


class LatencyHistogram:
    """Histogram of latencies in milliseconds with constant relative bucket width."""

    def __init__(self, lowest_ms: float = 1e-3, highest_ms: float = 1e6, precision: float = 0.01):
        if lowest_ms <= 0 or highest_ms <= lowest_ms:
            raise ValueError("need 0 < lowest_ms < highest_ms")
        if not 0 < precision < 1:
            raise ValueError("precision must be in (0, 1)")
        self.lowest_ms = float(lowest_ms)
        self.highest_ms = float(highest_ms)
        self.precision = float(precision)
        self._log_ratio = math.log1p(precision)
        self.n_buckets = int(math.ceil(math.log(highest_ms / lowest_ms) / self._log_ratio)) + 1
        self.counts = np.zeros(self.n_buckets, dtype=np.int64)
        self.total = 0
        self.min_ms = math.inf
        self.max_ms = -math.inf
        self._sum = 0.0

    # ------------------------------------------------------------------ layout
    def _index(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.lowest_ms, self.highest_ms)
        idx = np.floor(np.log(clipped / self.lowest_ms) / self._log_ratio).astype(np.int64)
        return np.clip(idx, 0, self.n_buckets - 1)

    def bucket_value(self, index: int) -> float:
        """Representative value of a bucket: geometric midpoint of its bounds."""
        lo = self.lowest_ms * math.exp(index * self._log_ratio)
        return lo * math.sqrt(1.0 + self.precision)

    def same_layout(self, other: "LatencyHistogram") -> bool:
        return (
            self.lowest_ms == other.lowest_ms
            and self.highest_ms == other.highest_ms
            and self.precision == other.precision
        )

    # --------------------------------------------------------------- recording
    def record(self, value_ms: float, count: int = 1) -> None:
        self.record_many(np.full(count, value_ms, dtype=np.float64))

    def record_many(self, values_ms: Iterable[float]) -> None:
        values = np.asarray(values_ms, dtype=np.float64).ravel()
        if values.size == 0:
            return
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("latencies must be finite and non-negative")
        np.add.at(self.counts, self._index(values), 1)
        self.total += int(values.size)
        self._sum += float(values.sum())
        self.min_ms = min(self.min_ms, float(values.min()))
        self.max_ms = max(self.max_ms, float(values.max()))

    def record_corrected(self, value_ms: float, expected_interval_ms: float) -> None:
        """Record a value and the samples coordinated omission hid.

        A closed-loop client that intended to send one request every
        ``expected_interval_ms`` but waited ``value_ms`` for a response skipped
        the requests it should have sent meanwhile. Those requests would have
        waited ``value_ms - k * expected_interval_ms`` for k = 1, 2, ...
        This mirrors HdrHistogram's ``recordValueWithExpectedInterval``.
        """
        if expected_interval_ms <= 0:
            raise ValueError("expected_interval_ms must be positive")
        values = [value_ms]
        missing = value_ms - expected_interval_ms
        while missing >= expected_interval_ms:
            values.append(missing)
            missing -= expected_interval_ms
        self.record_many(values)

    def merge(self, other: "LatencyHistogram") -> "LatencyHistogram":
        """Add another histogram's counts into this one (in place) and return self."""
        if not self.same_layout(other):
            raise ValueError("histograms must share lowest_ms, highest_ms, and precision")
        self.counts += other.counts
        self.total += other.total
        self._sum += other._sum
        self.min_ms = min(self.min_ms, other.min_ms)
        self.max_ms = max(self.max_ms, other.max_ms)
        return self

    def copy(self) -> "LatencyHistogram":
        clone = LatencyHistogram(self.lowest_ms, self.highest_ms, self.precision)
        return clone.merge(self)

    # ----------------------------------------------------------------- reading
    @property
    def mean_ms(self) -> float:
        return self._sum / self.total if self.total else math.nan

    def percentile(self, q: float) -> float:
        """Value at percentile ``q`` (0 to 100), using the nearest-rank definition."""
        if self.total == 0:
            return math.nan
        if not 0 <= q <= 100:
            raise ValueError("q must be between 0 and 100")
        if q == 0:
            return self.min_ms
        if q == 100:
            return self.max_ms
        rank = max(1, int(math.ceil(q / 100.0 * self.total)))
        index = int(np.searchsorted(np.cumsum(self.counts), rank))
        value = self.bucket_value(index)
        return float(min(max(value, self.min_ms), self.max_ms))

    def percentiles(self, qs: Iterable[float] = (50, 95, 99, 99.9)) -> Dict[str, float]:
        return {f"p{q:g}": self.percentile(q) for q in qs}

    def summary(self) -> Dict[str, float]:
        out = {"count": float(self.total), "mean_ms": self.mean_ms,
               "min_ms": self.min_ms if self.total else math.nan,
               "max_ms": self.max_ms if self.total else math.nan}
        out.update({k + "_ms": v for k, v in self.percentiles().items()})
        return out


def histogram_of(values_ms: Iterable[float], **kwargs) -> LatencyHistogram:
    """Convenience constructor: a histogram holding ``values_ms``."""
    hist = LatencyHistogram(**kwargs)
    hist.record_many(values_ms)
    return hist
