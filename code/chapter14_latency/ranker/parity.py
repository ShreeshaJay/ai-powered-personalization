"""
Parity checks between a reference ranker and a faster variant.

The verification ladder of Section 14.11, offline part:

* score parity: absolute differences and correlations of predicted scores;
* rank parity per pseudo-request: top-K overlap, Kendall's tau, and
  rank-biased overlap (Webber, Moffat, and Zobel, 2010);
* metric parity: AUC of each model against the labels;
* calibration shift: change in mean prediction, reported beside the observed
  positive rate so the reference's own calibration is visible too;
* slice parity: the same checks on short- and long-history users and on
  popular and tail items.

Tolerances are inputs. The defaults in ``Tolerances`` exist so the script
runs; a team sets its own from what its downstream metrics can absorb.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class Tolerances:
    max_auc_drop: float = 0.002
    min_top10_overlap: float = 0.95
    max_abs_mean_shift: float = 0.005


def topk_overlap(ref: np.ndarray, var: np.ndarray, k: int) -> float:
    """Share of the reference top-k items that are also in the variant's top-k."""
    k = min(k, len(ref))
    if k == 0:
        return math.nan
    a = set(np.argsort(-ref, kind="stable")[:k].tolist())
    b = set(np.argsort(-var, kind="stable")[:k].tolist())
    return len(a & b) / k


def kendall_tau(ref: np.ndarray, var: np.ndarray) -> float:
    if len(ref) < 2:
        return math.nan
    return float(stats.kendalltau(ref, var).statistic)


def rbo(list_a: Sequence, list_b: Sequence, p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap of two rankings of equal length.

    Identical rankings score 1 and disjoint rankings score 0. Larger ``p``
    gives more weight to deeper ranks.
    """
    k = min(len(list_a), len(list_b))
    if k == 0:
        return math.nan
    seen_a, seen_b, overlap, total = set(), set(), 0, 0.0
    for d in range(1, k + 1):
        x, y = list_a[d - 1], list_b[d - 1]
        if x == y:
            overlap += 1
        else:
            overlap += (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        total += (overlap / d) * p ** d
    return float((overlap / k) * p ** k + (1 - p) / p * total)


def score_parity(ref: np.ndarray, var: np.ndarray) -> Dict[str, float]:
    diff = np.abs(var - ref)
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "pearson": float(np.corrcoef(ref, var)[0, 1]) if ref.std() > 0 and var.std() > 0 else math.nan,
            "spearman": float(stats.spearmanr(ref, var).statistic)}


def rank_parity(ref: np.ndarray, var: np.ndarray, groups: np.ndarray, min_size: int = 10,
                max_groups: int = 20_000, seed: int = 0) -> Dict[str, float]:
    """Mean rank agreement across pseudo-requests with at least ``min_size`` rows.

    Top-k overlap uses only groups with more than k rows: in a group of
    exactly k rows both top-k sets are the whole group and overlap is 1 by
    construction.
    """
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    big = [(s, e) for s, e in zip(starts, ends) if e - s >= min_size]
    if len(big) > max_groups:
        pick = np.random.default_rng(seed).choice(len(big), max_groups, replace=False)
        big = [big[i] for i in sorted(pick)]
    o10, o50, taus, rbos, n50 = [], [], [], [], 0
    for s, e in big:
        idx = order[s:e]
        r, v = ref[idx], var[idx]
        if e - s > 10:
            o10.append(topk_overlap(r, v, 10))
        if e - s > 50:
            o50.append(topk_overlap(r, v, 50))
            n50 += 1
        taus.append(kendall_tau(r, v))
        rbos.append(rbo(np.argsort(-r, kind="stable").tolist(), np.argsort(-v, kind="stable").tolist()))
    return {"groups": len(big), "groups_over_10": len(o10),
            "top10_overlap": float(np.nanmean(o10)) if o10 else math.nan,
            "groups_over_50": n50, "top50_overlap": float(np.nanmean(o50)) if o50 else math.nan,
            "kendall_tau": float(np.nanmean(taus)) if taus else math.nan,
            "rbo_p0.9": float(np.nanmean(rbos)) if rbos else math.nan}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else math.nan


def slices(user_history: np.ndarray, item_plays: np.ndarray) -> Dict[str, np.ndarray]:
    uh_cut = np.median(user_history)
    ip_cut = np.percentile(item_plays, 80)
    return {"all": np.ones(len(user_history), dtype=bool),
            "short_history_users": user_history <= uh_cut,
            "long_history_users": user_history > uh_cut,
            "popular_items": item_plays > ip_cut,
            "tail_items": item_plays <= ip_cut}


def parity_report(variant: str, ref: np.ndarray, var: np.ndarray, labels: np.ndarray,
                  groups: np.ndarray, user_history: np.ndarray, item_plays: np.ndarray,
                  tol: Tolerances, seed: int = 0) -> List[Dict[str, object]]:
    rows = []
    for name, mask in slices(user_history, item_plays).items():
        if mask.sum() < 2:
            continue
        r, v, y, g = ref[mask], var[mask], labels[mask], groups[mask]
        row = {"variant": variant, "slice": name, "rows": int(mask.sum())}
        row.update(score_parity(r, v))
        row.update(rank_parity(r, v, g, seed=seed))
        row["auc_ref"], row["auc_variant"] = _auc(y, r), _auc(y, v)
        row["auc_change"] = row["auc_variant"] - row["auc_ref"]
        row["label_rate"] = float(y.mean())
        row["mean_pred_ref"], row["mean_pred_variant"] = float(r.mean()), float(v.mean())
        row["mean_pred_shift"] = row["mean_pred_variant"] - row["mean_pred_ref"]
        failures = []
        if not math.isnan(row["auc_change"]) and row["auc_change"] < -tol.max_auc_drop:
            failures.append("auc")
        if not math.isnan(row["top10_overlap"]) and row["top10_overlap"] < tol.min_top10_overlap:
            failures.append("top10_overlap")
        if abs(row["mean_pred_shift"]) > tol.max_abs_mean_shift:
            failures.append("calibration")
        row["passes"] = not failures
        row["failed_checks"] = "+".join(failures)
        rows.append(row)
    return rows


def tolerances_dict(tol: Tolerances) -> Dict[str, float]:
    return asdict(tol)
