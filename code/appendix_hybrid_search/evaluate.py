"""Protocols P1, P2 and P3, plus paired bootstrap and Holm correction.

Evaluation reads run files or in-memory rankings. It does not retrieve.
An empty run scores zero on a defined metric and is counted. A query whose
judgments are all Irrelevant is excluded from nDCG only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from config import OFFICIAL_GAINS, require_plan_hash
from data_loader import QueryJudgments
from fusion import HitList, Run, read_run

MetricMap = dict[str, float | None]


def dcg(gains: list[float] | np.ndarray, k: int | None = None) -> float:
    values = np.asarray(list(gains), dtype=np.float64)
    if k is not None:
        values = values[:k]
    if values.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, values.size + 2))
    return float(np.dot(values, discounts))


def ndcg_from_gains(ranked_gains: list[float], all_gains: list[float], k: int | None = None) -> float | None:
    ideal = dcg(sorted(all_gains, reverse=True), k)
    if ideal == 0.0:
        return None
    return dcg(ranked_gains, k) / ideal


def ndcg_for_ids(ranked_ids: list[str], gain_map: dict[str, float], k: int | None = None) -> float | None:
    ranked_gains = [float(gain_map.get(product_id, 0.0)) for product_id in ranked_ids]
    return ndcg_from_gains(ranked_gains, list(gain_map.values()), k)


def recall_at(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    found = sum(1 for product_id in retrieved[:k] if product_id in relevant)
    return found / len(relevant)


def judged_at(retrieved: list[str], judged: set[str], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for product_id in retrieved[:k] if product_id in judged) / float(k)


def condensed_ndcg(retrieved: list[str], gain_map: dict[str, float], k: int = 10) -> float | None:
    """Drop unjudged products, then compute nDCG at k. The ideal uses every judgment."""
    condensed = [product_id for product_id in retrieved if product_id in gain_map]
    return ndcg_for_ids(condensed, gain_map, k)


def _defined(gain_map: dict[str, float]) -> bool:
    return any(gain > 0.0 for gain in gain_map.values())


def p1_order(hits: HitList, judged_ids: list[str]) -> list[str]:
    """Place unscored judged products after every scored one."""
    judged = set(judged_ids)
    seen: set[str] = set()
    present: list[str] = []
    for product_id, _score in hits:
        if product_id in judged and product_id not in seen:
            present.append(product_id)
            seen.add(product_id)
    missing = sorted(product_id for product_id in judged_ids if product_id not in seen)
    return present + missing


def score_p1(run: Run, queries: dict[str, QueryJudgments], gains: dict[str, float]) -> dict[str, dict]:
    per_query: dict[str, dict] = {}
    for query_id, item in queries.items():
        gain_map = item.gains(gains)
        hits = run.get(query_id, [])
        defined = _defined(gain_map)
        if not hits:
            per_query[query_id] = {
                "ndcg": 0.0 if defined else None,
                "ndcg_at_10": 0.0 if defined else None,
                "empty": True,
            }
            continue
        order = p1_order(hits, item.product_order)
        per_query[query_id] = {
            "ndcg": ndcg_for_ids(order, gain_map, None),
            "ndcg_at_10": ndcg_for_ids(order, gain_map, 10),
            "empty": False,
        }
    return per_query


def score_p2(run: Run, queries: dict[str, QueryJudgments], gains: dict[str, float]) -> dict[str, dict]:
    per_query: dict[str, dict] = {}
    for query_id, item in queries.items():
        retrieved = [product_id for product_id, _score in run.get(query_id, [])]
        gain_map = item.gains(gains)
        judged = set(item.labels)
        per_query[query_id] = {
            "recall_at_100_exact": recall_at(retrieved, item.exact_ids(), 100),
            "recall_at_1000_exact": recall_at(retrieved, item.exact_ids(), 1000),
            "recall_at_100_exact_substitute": recall_at(retrieved, item.exact_or_substitute_ids(), 100),
            "recall_at_1000_exact_substitute": recall_at(retrieved, item.exact_or_substitute_ids(), 1000),
            "condensed_ndcg_at_10": condensed_ndcg(retrieved, gain_map, 10),
            "judged_at_10": judged_at(retrieved, judged, 10),
            "empty": len(retrieved) == 0,
        }
    return per_query


def macro(per_query: dict[str, dict], metric: str) -> dict:
    values = [row[metric] for row in per_query.values() if row[metric] is not None]
    return {
        "metric": metric,
        "mean": float(np.mean(values)) if values else None,
        "n_queries": len(per_query),
        "n_scored": len(values),
        "n_excluded": sum(1 for row in per_query.values() if row[metric] is None),
        "n_empty": sum(1 for row in per_query.values() if row.get("empty")),
    }


def aligned_diff(left: dict[str, dict], right: dict[str, dict], metric: str) -> tuple[list[str], np.ndarray]:
    ids = sorted(set(left) & set(right))
    kept: list[str] = []
    diffs: list[float] = []
    for query_id in ids:
        a = left[query_id][metric]
        b = right[query_id][metric]
        if a is None or b is None:
            continue
        kept.append(query_id)
        diffs.append(float(a) - float(b))
    return kept, np.asarray(diffs, dtype=np.float64)


def paired_bootstrap(diff: np.ndarray, n_resamples: int, seed: int) -> dict:
    if diff.size == 0:
        return {"mean": None, "low": None, "high": None, "p": None, "n": 0}
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p_le = float(np.mean(means <= 0.0))
    p_ge = float(np.mean(means >= 0.0))
    return {
        "mean": float(diff.mean()),
        "low": float(low),
        "high": float(high),
        "p": float(min(1.0, 2.0 * min(p_le, p_ge))),
        "n": int(diff.size),
    }


def holm(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Step-down Holm. A later hypothesis is not rejected after the first failure."""
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    rejected = [False] * len(p_values)
    thresholds = [0.0] * len(p_values)
    still_rejecting = True
    total = len(p_values)
    for rank, index in enumerate(order):
        threshold = alpha / (total - rank)
        thresholds[index] = threshold
        if still_rejecting and p_values[index] <= threshold:
            rejected[index] = True
        else:
            still_rejecting = False
    return [
        {"p": p_values[index], "threshold": thresholds[index], "reject": rejected[index]}
        for index in range(len(p_values))
    ]


def relative_difference(system_a: float, system_b: float) -> float:
    denominator = abs(system_b)
    if denominator == 0.0:
        return 0.0 if system_a == 0.0 else float("inf")
    return abs(system_a - system_b) / denominator


def interpretable(system_a: float, system_b: float, threshold: float = 0.01) -> bool:
    return relative_difference(system_a, system_b) >= threshold


def _mix_seed(seed: int, query_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def random_order_floor(
    queries: dict[str, QueryJudgments],
    gains: dict[str, float],
    shuffles: int,
    seed: int,
) -> dict:
    full_values: list[float] = []
    at10_values: list[float] = []
    excluded = 0
    for query_id, item in queries.items():
        gain_map = item.gains(gains)
        if not _defined(gain_map):
            excluded += 1
            continue
        generator = np.random.default_rng(_mix_seed(seed, query_id))
        ids = list(item.product_order)
        one_full: list[float] = []
        one_at10: list[float] = []
        for _ in range(shuffles):
            generator.shuffle(ids)
            one_full.append(float(ndcg_for_ids(ids, gain_map, None)))
            one_at10.append(float(ndcg_for_ids(ids, gain_map, 10)))
        full_values.append(float(np.mean(one_full)))
        at10_values.append(float(np.mean(one_at10)))
    return {
        "ndcg": float(np.mean(full_values)) if full_values else None,
        "ndcg_at_10": float(np.mean(at10_values)) if at10_values else None,
        "n_scored": len(full_values),
        "n_excluded": excluded,
        "shuffles": shuffles,
        "seed": seed,
    }


def hajek_mean(values: list[float], inclusion_probabilities: list[float]) -> float:
    if len(values) != len(inclusion_probabilities) or not values:
        raise ValueError("Hajek mean needs one probability per value.")
    weights = []
    for probability in inclusion_probabilities:
        if probability <= 0.0 or probability > 1.0:
            raise ValueError(f"Inclusion probability {probability} is outside (0, 1].")
        weights.append(1.0 / probability)
    weight_sum = float(sum(weights))
    return float(sum(weight * value for weight, value in zip(weights, values)) / weight_sum)


def quadratic_weighted_kappa(left: list[int], right: list[int], n_classes: int = 4) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa needs paired labels.")
    observed = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(left, right):
        observed[a, b] += 1.0
    observed /= observed.sum()
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)
    weights = np.zeros_like(observed)
    scale = float(n_classes - 1)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) / scale) ** 2
    observed_score = float(np.sum(weights * observed))
    expected_score = float(np.sum(weights * expected))
    if expected_score == 0.0:
        return 1.0 if observed_score == 0.0 else 0.0
    return 1.0 - observed_score / expected_score


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_result_size(path: Path, limit_bytes: int = 1_000_000) -> None:
    size = path.stat().st_size
    if size > limit_bytes:
        raise RuntimeError(f"{path.name} is {size} bytes, above the {limit_bytes} byte result limit.")


def record_test_scoring(output_dir: Path, key: str, plan_digest: str) -> None:
    """Refuse a second test scoring of the same configuration."""
    log_path = output_dir / "eval_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("key") == key and record.get("plan_sha256") == plan_digest:
                raise RuntimeError(
                    f"Test configuration {key} was already scored under this protocol digest."
                )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "plan_sha256": plan_digest, "split": "test"}) + "\n")


def guard_test_split(output_dir: Path, key: str, split: str) -> str | None:
    if split != "test":
        return None
    digest = require_plan_hash()
    record_test_scoring(output_dir, key, digest)
    return digest


def load_run_or_empty(path: Path, query_ids: Iterable[str]) -> Run:
    if not path.exists():
        return {query_id: [] for query_id in query_ids}
    run = read_run(path)
    for query_id in query_ids:
        run.setdefault(query_id, [])
    return run


def primary_comparisons(
    per_query_by_system: dict[str, dict[str, dict]],
    family: list[dict],
    n_resamples: int,
    seed: int,
    alpha: float,
    relative_threshold: float,
    selection: dict[str, str] | None = None,
) -> list[dict]:
    resolved = []
    p_values = []
    for item in family:
        system_a = item["system_a"]
        system_b = item["system_b"]
        if selection:
            system_a = selection.get(system_a, system_a)
            system_b = selection.get(system_b, system_b)
        if system_a.startswith("{") or system_b.startswith("{"):
            raise RuntimeError("Best-system placeholders are still unresolved.")
        if system_a not in per_query_by_system or system_b not in per_query_by_system:
            raise KeyError(f"Missing per-query metrics for {system_a} or {system_b}.")
        _ids, diff = aligned_diff(
            per_query_by_system[system_a],
            per_query_by_system[system_b],
            item["metric"],
        )
        estimate = paired_bootstrap(diff, n_resamples, seed)
        mean_a = macro(per_query_by_system[system_a], item["metric"])["mean"]
        mean_b = macro(per_query_by_system[system_b], item["metric"])["mean"]
        resolved.append(
            {
                "id": item["id"],
                "system_a": system_a,
                "system_b": system_b,
                "protocol": item["protocol"],
                "metric": item["metric"],
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference": estimate["mean"],
                "interval_low": estimate["low"],
                "interval_high": estimate["high"],
                "n": estimate["n"],
                "interpret": None
                if estimate["mean"] is None
                else interpretable(float(mean_a), float(mean_b), relative_threshold),
            }
        )
        p_values.append(1.0 if estimate["p"] is None else float(estimate["p"]))
    decisions = holm(p_values, alpha)
    for row, decision in zip(resolved, decisions):
        row["p"] = decision["p"]
        row["holm_threshold"] = decision["threshold"]
        row["holm_reject"] = decision["reject"]
    return resolved


# Official gains are the default used by callers that do not pass a table.
DEFAULT_GAINS = dict(OFFICIAL_GAINS)


def report_primary_family(
    per_p1: dict[str, dict],
    per_p2: dict[str, dict],
    family: list[dict],
    n_resamples: int,
    seed: int,
    alpha: float,
    relative_threshold: float,
    selection: dict[str, str] | None = None,
) -> list[dict]:
    """Holm correction is applied once across every resolved primary comparison."""
    resolved = []
    p_values = []
    for item in family:
        system_a = item["system_a"]
        system_b = item["system_b"]
        if selection:
            system_a = selection.get(system_a, system_a)
            system_b = selection.get(system_b, system_b)
        table = per_p1 if item["protocol"] == "p1" else per_p2
        if system_a not in table or system_b not in table:
            continue
        _ids, diff = aligned_diff(table[system_a], table[system_b], item["metric"])
        estimate = paired_bootstrap(diff, n_resamples, seed)
        mean_a = macro(table[system_a], item["metric"])["mean"]
        mean_b = macro(table[system_b], item["metric"])["mean"]
        resolved.append(
            {
                "id": item["id"],
                "system_a": system_a,
                "system_b": system_b,
                "protocol": item["protocol"],
                "metric": item["metric"],
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference": estimate["mean"],
                "interval_low": estimate["low"],
                "interval_high": estimate["high"],
                "n": estimate["n"],
                "interpret": None
                if mean_a is None or mean_b is None
                else interpretable(float(mean_a), float(mean_b), relative_threshold),
            }
        )
        p_values.append(1.0 if estimate["p"] is None else float(estimate["p"]))
    for row, decision in zip(resolved, holm(p_values, alpha)):
        row["p"] = decision["p"]
        row["holm_threshold"] = decision["threshold"]
        row["holm_reject"] = decision["reject"]
    return resolved
