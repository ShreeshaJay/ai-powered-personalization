"""Classification, calibration, and latency metrics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def _key(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion(y_true: list[Any], y_pred: list[Any]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for reference, predicted in zip(y_true, y_pred, strict=True):
        matrix[_key(reference)][_key(predicted)] += 1
    return {
        label: dict(sorted(counter.items()))
        for label, counter in sorted(matrix.items())
    }


def f1_report(y_true: list[Any], y_pred: list[Any]) -> dict[str, float]:
    labels = sorted({_key(value) for value in y_true + y_pred})
    per_class = {}
    tp = fp = fn = 0
    f1_values = []
    for label in labels:
        true_pos = sum(
            1
            for reference, predicted in zip(y_true, y_pred, strict=True)
            if _key(reference) == label and _key(predicted) == label
        )
        false_pos = sum(
            1
            for reference, predicted in zip(y_true, y_pred, strict=True)
            if _key(reference) != label and _key(predicted) == label
        )
        false_neg = sum(
            1
            for reference, predicted in zip(y_true, y_pred, strict=True)
            if _key(reference) == label and _key(predicted) != label
        )
        precision = _safe_div(true_pos, true_pos + false_pos)
        recall = _safe_div(true_pos, true_pos + false_neg)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": true_pos + false_neg,
        }
        tp += true_pos
        fp += false_pos
        fn += false_neg
        f1_values.append(f1)
    micro_p = _safe_div(tp, tp + fp)
    micro_r = _safe_div(tp, tp + fn)
    return {
        "accuracy": round(
            _safe_div(
                sum(
                    _key(reference) == _key(predicted)
                    for reference, predicted in zip(y_true, y_pred, strict=True)
                ),
                len(y_true),
            ),
            4,
        ),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "micro_f1": round(_safe_div(2 * micro_p * micro_r, micro_p + micro_r), 4),
        "per_class": per_class,
    }


def top_label_ece(
    confidences: list[float],
    correct: list[bool],
    bins: int = 10,
) -> float:
    if not confidences:
        return float("nan")
    edges = [index / bins for index in range(bins + 1)]
    error = 0.0
    count = len(confidences)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [
            (confidence, is_correct)
            for confidence, is_correct in zip(confidences, correct, strict=True)
            if (confidence >= low if index == 0 else confidence > low)
            and confidence <= high
        ]
        if not selected:
            continue
        mean_conf = sum(item[0] for item in selected) / len(selected)
        mean_acc = sum(item[1] for item in selected) / len(selected)
        error += (len(selected) / count) * abs(mean_conf - mean_acc)
    return float(error)


def multiclass_brier(
    y_true: list[Any], probabilities: list[dict[str, float]]
) -> float:
    if not y_true:
        return float("nan")
    labels = sorted(
        {
            key
            for dist in probabilities
            for key in dist
        }
        | {_key(value) for value in y_true}
    )
    total = 0.0
    for reference, dist in zip(y_true, probabilities, strict=True):
        for label in labels:
            target = 1.0 if _key(reference) == label else 0.0
            total += (dist.get(label, 0.0) - target) ** 2
    return total / len(y_true)


def negative_log_likelihood(
    y_true: list[Any], probabilities: list[dict[str, float]]
) -> float:
    if not y_true:
        return float("nan")
    total = 0.0
    for reference, dist in zip(y_true, probabilities, strict=True):
        probability = max(float(dist.get(_key(reference), 0.0)), 1e-12)
        total += -math.log(probability)
    return total / len(y_true)


def coverage_accuracy(
    confidences: list[float],
    correct: list[bool],
    coverages: tuple[float, ...] = (0.5, 0.8, 1.0),
) -> dict[str, float]:
    if not confidences:
        return {f"acc_at_{coverage:.2f}": float("nan") for coverage in coverages}
    order = sorted(
        range(len(confidences)),
        key=lambda index: confidences[index],
        reverse=True,
    )
    result = {}
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(order))))
        chosen = order[:keep]
        accuracy = sum(correct[index] for index in chosen) / len(chosen)
        result[f"acc_at_{coverage:.2f}"] = round(accuracy, 4)
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def field_metrics(
    y_true: list[Any],
    y_pred: list[Any],
    probabilities: list[dict[str, float]],
    confidences: list[float],
) -> dict[str, Any]:
    correct = [
        _key(reference) == _key(predicted)
        for reference, predicted in zip(y_true, y_pred, strict=True)
    ]
    report = f1_report(y_true, y_pred)
    report.update(
        {
            "items": len(y_true),
            "ece": round(top_label_ece(confidences, correct), 4),
            "brier": round(multiclass_brier(y_true, probabilities), 4),
            "nll": round(negative_log_likelihood(y_true, probabilities), 4),
            "mean_confidence": round(
                sum(confidences) / len(confidences), 4
            )
            if confidences
            else float("nan"),
            "coverage_accuracy": coverage_accuracy(confidences, correct),
            "confusion": confusion(y_true, y_pred),
        }
    )
    return report


def exact_match(
    examples: list[dict[str, Any]],
    predictions: list[Any],
    fields: list[str],
) -> dict[str, Any]:
    pred_by_id = {prediction.item_id: prediction for prediction in predictions}
    eligible = [
        example
        for example in examples
        if example.get("complete_agreement") and example.get("complete_labels")
    ]
    matches = 0
    for example in eligible:
        prediction = pred_by_id[example["item_id"]]
        if all(
            _key(example["complete_labels"][field])
            == _key(prediction.fields[field].predicted)
            for field in fields
        ):
            matches += 1
    return {
        "items": len(eligible),
        "exact_match": round(_safe_div(matches, len(eligible)), 4),
    }


def latency_metrics(elapsed_ms: list[float]) -> dict[str, float]:
    if not elapsed_ms:
        return {
            "items": 0,
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "mean_ms": float("nan"),
            "items_per_second": float("nan"),
        }
    mean_ms = sum(elapsed_ms) / len(elapsed_ms)
    return {
        "items": len(elapsed_ms),
        "p50_ms": round(percentile(elapsed_ms, 0.50), 3),
        "p95_ms": round(percentile(elapsed_ms, 0.95), 3),
        "mean_ms": round(mean_ms, 3),
        "items_per_second": round(1000.0 / mean_ms, 3) if mean_ms else float("nan"),
    }
