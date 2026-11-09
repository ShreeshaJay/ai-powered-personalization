"""Top-label calibration, proper scores and risk-coverage from probability vectors.

Unless a name says otherwise, "confidence" here is the top-label probability: the
probability a model assigned to the label it returned, read from its probability
vector. The adapters' own `confidence` field is a different statistic for most
adapters; functions that take it say so.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from eval.metrics import _key, top_label_ece

DEFAULT_BINS = 15
SEMANTICS_TOLERANCE = 0.01


def selected_probability(predicted: Any, probabilities: dict[str, float]) -> float:
    return float(probabilities.get(_key(predicted), 0.0))


def equal_width_bin_index(confidences: Sequence[float], bins: int = DEFAULT_BINS) -> np.ndarray:
    """Bin index per item with the `top_label_ece` convention; -1 outside [0, 1] and for NaN.

    Bin 0 is [0, 1/bins]; bin j > 0 is (j/bins, (j+1)/bins].
    """
    values = np.asarray(confidences, dtype=float)
    upper_edges = np.array([index / bins for index in range(1, bins + 1)])
    index = np.searchsorted(upper_edges, values, side="left")
    index[~((values >= 0.0) & (values <= 1.0))] = -1
    return index


def ece_equal_width(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
) -> float:
    """Top-label ECE over equal-width bins, using the committed estimator."""
    return top_label_ece([float(c) for c in confidences], [bool(y) for y in correct], bins=bins)


def ece_equal_mass(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
) -> float:
    """Top-label ECE over equal-mass bins.

    Items are stably sorted by confidence (ties keep pack order) and split into
    `bins` groups whose sizes differ by at most one (numpy.array_split).
    """
    values = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=float)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    total = 0.0
    for group in np.array_split(order, bins):
        if len(group) == 0:
            continue
        total += (len(group) / len(values)) * abs(values[group].mean() - hits[group].mean())
    return float(total)


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = DEFAULT_BINS,
    scheme: str = "width",
) -> list[dict[str, float]]:
    """Per-bin counts, mean confidence and accuracy for a reliability diagram."""
    values = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=float)
    rows = []
    if scheme == "width":
        index = equal_width_bin_index(values, bins)
        groups = [np.flatnonzero(index == b) for b in range(bins)]
        bounds = [(b / bins, (b + 1) / bins) for b in range(bins)]
    elif scheme == "mass":
        order = np.argsort(values, kind="stable")
        groups = list(np.array_split(order, bins))
        bounds = [
            (float(values[g].min()), float(values[g].max())) if len(g) else (math.nan, math.nan)
            for g in groups
        ]
    else:
        raise ValueError(f"Unknown binning scheme: {scheme}")
    for group, (low, high) in zip(groups, bounds):
        rows.append(
            {
                "lower": round(float(low), 6),
                "upper": round(float(high), 6),
                "items": int(len(group)),
                "mean_confidence": round(float(values[group].mean()), 6) if len(group) else None,
                "accuracy": round(float(hits[group].mean()), 6) if len(group) else None,
            }
        )
    return rows


def binned_brier_decomposition(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
) -> dict[str, float]:
    """Top-label Brier score and its binned reliability and resolution terms.

    The binary outcome is "the returned label is correct". With equal-width bins,
    brier ~= reliability - resolution + uncertainty; the identity is exact only
    when confidence is constant within each bin.
    """
    values = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=float)
    n = len(values)
    if n == 0:
        return {"brier": math.nan, "reliability": math.nan, "resolution": math.nan, "uncertainty": math.nan}
    base_rate = hits.mean()
    index = equal_width_bin_index(values, bins)
    reliability = resolution = 0.0
    for b in range(bins):
        group = index == b
        count = int(group.sum())
        if not count:
            continue
        weight = count / n
        reliability += weight * (values[group].mean() - hits[group].mean()) ** 2
        resolution += weight * (hits[group].mean() - base_rate) ** 2
    return {
        "brier": float(np.mean((values - hits) ** 2)),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(base_rate * (1.0 - base_rate)),
    }


def risk_coverage(
    confidences: Sequence[float],
    correct: Sequence[bool],
    coverages: tuple[float, ...] = (0.5, 0.8, 1.0),
) -> dict[str, Any]:
    """Area under the risk-coverage curve and accuracy at fixed coverage.

    Items are ranked by confidence, highest first, ties in pack order. Risk at
    coverage k/n is the error rate among the first k items; AURC is its mean over
    k = 1..n. Coverage cut-offs use the same rounding as `coverage_accuracy`.
    """
    values = np.asarray(confidences, dtype=float)
    hits = np.asarray(correct, dtype=float)
    n = len(values)
    if n == 0:
        return {"aurc": math.nan, "coverage_accuracy": {}}
    order = np.argsort(-values, kind="stable")
    errors = np.cumsum(1.0 - hits[order])
    risk = errors / np.arange(1, n + 1)
    accuracy_at = {}
    for coverage in coverages:
        keep = max(1, int(round(coverage * n)))
        accuracy_at[f"acc_at_{coverage:.2f}"] = round(float(hits[order[:keep]].mean()), 4)
    return {"aurc": float(risk.mean()), "coverage_accuracy": accuracy_at}


def macro_f1_fixed(y_true: Sequence[Any], y_pred: Sequence[Any], label_space: Sequence[Any]) -> float:
    """Macro-F1 averaged over a fixed label space.

    Every label in the space counts, including labels absent from both the
    reference and the predictions (F1 = 0 for those). `f1_report` instead averages
    over the labels present in either list.
    """
    labels = [_key(value) for value in label_space]
    reference = [_key(value) for value in y_true]
    predicted = [_key(value) for value in y_pred]
    scores = []
    for label in labels:
        tp = sum(r == label and p == label for r, p in zip(reference, predicted, strict=True))
        fp = sum(r != label and p == label for r, p in zip(reference, predicted, strict=True))
        fn = sum(r == label and p != label for r, p in zip(reference, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores)) if scores else math.nan


def macro_f1_union(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """`f1_report`'s macro-F1 (labels present in either list) before its 4-decimal rounding."""
    return macro_f1_fixed(y_true, y_pred, sorted({_key(value) for value in [*y_true, *y_pred]}))


def zero_probability_true_labels(y_true: Sequence[Any], probabilities: Sequence[dict[str, float]]) -> int:
    """Count of items whose reference label received probability exactly 0."""
    return int(sum(float(dist.get(_key(ref), 0.0)) == 0.0 for ref, dist in zip(y_true, probabilities, strict=True)))


def rescaled_margin(top_probability: float, options: int) -> float:
    """(p_max - 1/k) / (1 - 1/k): the margin over a uniform split, rescaled to [0, 1]."""
    if options < 2:
        return 1.0
    return (top_probability - 1.0 / options) / (1.0 - 1.0 / options)


def normalised_entropy_confidence(probabilities: Sequence[float]) -> float:
    """1 - H(p) / log k, clipped to [0, 1]."""
    k = len(probabilities)
    if k < 2:
        return 1.0
    p = np.asarray(probabilities, dtype=float)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(k), 0.0, 1.0))


def confidence_semantics(
    reported_confidence: Sequence[float],
    probabilities: Sequence[dict[str, float]],
    tolerance: float = SEMANTICS_TOLERANCE,
) -> dict[str, float]:
    """Share of items whose reported `confidence` matches each candidate formula."""
    n = len(reported_confidence)
    if n == 0:
        return {}
    matches = {"top_label_probability": 0, "rescaled_margin": 0, "normalised_entropy": 0}
    for reported, dist in zip(reported_confidence, probabilities, strict=True):
        values = list(dist.values())
        top = max(values)
        if abs(reported - top) <= tolerance:
            matches["top_label_probability"] += 1
        if abs(reported - rescaled_margin(top, len(values))) <= tolerance:
            matches["rescaled_margin"] += 1
        if abs(reported - normalised_entropy_confidence(values)) <= tolerance:
            matches["normalised_entropy"] += 1
    return {name: round(count / n, 4) for name, count in matches.items()}
