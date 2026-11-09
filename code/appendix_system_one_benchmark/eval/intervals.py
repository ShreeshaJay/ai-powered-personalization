"""Seeded paired bootstrap over items.

Every model scored on a field is resampled with the same item indices, so the
difference between two models is computed on identical resamples (a paired
bootstrap). Intervals are percentile intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np

from eval.calibration import DEFAULT_BINS, equal_width_bin_index

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 42
CHUNK = 200
METRICS = ("accuracy", "macro_f1_fixed", "ece_top_label_width")


@dataclass(frozen=True)
class ModelArrays:
    """One model's predictions on one field, encoded against a fixed label space.

    `reference` and `predicted` hold label-space indices; a predicted label outside
    the space gets index len(space). `confidence` is the top-label probability.
    """

    reference: np.ndarray
    predicted: np.ndarray
    confidence: np.ndarray

    @property
    def correct(self) -> np.ndarray:
        return self.reference == self.predicted


def encode(values: Sequence[str], space: Sequence[str]) -> np.ndarray:
    position = {label: index for index, label in enumerate(space)}
    return np.array([position.get(value, len(space)) for value in values], dtype=np.int64)


def resample_indices(
    n: int, resamples: int, seed: int, stream: int, chunk: int = CHUNK
) -> Iterator[np.ndarray]:
    """Yield (chunk, n) arrays of item indices drawn with replacement.

    The generator is keyed by (seed, stream), so each field gets its own fixed
    sequence of resamples regardless of which models are scored on it.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        yield rng.integers(0, n, size=(size, n))


def _macro_f1_batch(reference: np.ndarray, predicted: np.ndarray, labels: int) -> np.ndarray:
    width = labels + 1
    rows = reference.shape[0]
    codes = reference * width + predicted + (np.arange(rows)[:, None] * width * width)
    counts = np.bincount(codes.ravel(), minlength=rows * width * width).reshape(rows, width, width)
    tp = np.einsum("rii->ri", counts)[:, :labels].astype(float)
    predicted_total = counts.sum(axis=1)[:, :labels]
    reference_total = counts.sum(axis=2)[:, :labels]
    precision = np.divide(tp, predicted_total, out=np.zeros_like(tp), where=predicted_total > 0)
    recall = np.divide(tp, reference_total, out=np.zeros_like(tp), where=reference_total > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)
    return f1.mean(axis=1)


def _ece_batch(confidence: np.ndarray, correct: np.ndarray, bin_index: np.ndarray, bins: int) -> np.ndarray:
    rows, n = confidence.shape
    valid = bin_index >= 0
    codes = np.where(valid, bin_index + np.arange(rows)[:, None] * bins, rows * bins)
    size = rows * bins + 1
    conf_sum = np.bincount(codes.ravel(), weights=confidence.ravel(), minlength=size)[:-1]
    hit_sum = np.bincount(codes.ravel(), weights=correct.ravel().astype(float), minlength=size)[:-1]
    gap = np.abs(conf_sum - hit_sum).reshape(rows, bins)
    return gap.sum(axis=1) / n


def bootstrap_field(
    models: dict[str, ModelArrays],
    labels: int,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    stream: int = 0,
    bins: int = DEFAULT_BINS,
) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap distributions of accuracy, fixed-space macro-F1 and top-label ECE.

    Returns {model: {metric: array of shape (resamples,)}}. All models must hold
    the same items in the same order.
    """
    names = list(models)
    n = len(models[names[0]].reference)
    for name in names:
        if len(models[name].reference) != n:
            raise ValueError("every model must be scored on the same items")
    bin_index = {name: equal_width_bin_index(models[name].confidence, bins) for name in names}
    out = {name: {metric: [] for metric in METRICS} for name in names}
    for index in resample_indices(n, resamples, seed, stream):
        for name in names:
            arrays = models[name]
            reference = arrays.reference[index]
            predicted = arrays.predicted[index]
            correct = reference == predicted
            out[name]["accuracy"].append(correct.mean(axis=1))
            out[name]["macro_f1_fixed"].append(_macro_f1_batch(reference, predicted, labels))
            out[name]["ece_top_label_width"].append(
                _ece_batch(arrays.confidence[index], correct, bin_index[name][index], bins)
            )
    return {name: {metric: np.concatenate(values) for metric, values in metrics.items()} for name, metrics in out.items()}


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def paired_differences(
    distributions: dict[str, dict[str, np.ndarray]],
    estimates: dict[str, dict[str, float]],
    baseline: str | None,
    level: float = 0.95,
) -> list[dict[str, object]]:
    """Differences (a - b) with paired intervals: every model against the baseline,
    then every pair of non-baseline models, for each metric in METRICS."""
    names = list(distributions)
    others = [name for name in names if name != baseline]
    pairs = [(name, baseline) for name in others] if baseline in names else []
    pairs += list(combinations(others, 2))
    rows = []
    for a, b in pairs:
        for metric in METRICS:
            low, high = percentile_interval(distributions[a][metric] - distributions[b][metric], level)
            estimate = estimates[a][metric] - estimates[b][metric]
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "estimate": round(estimate, 4),
                    "low": round(low, 4),
                    "high": round(high, 4),
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return rows
