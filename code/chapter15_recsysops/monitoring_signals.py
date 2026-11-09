"""
Chapter 15: Monitoring Signals
==============================
Reusable estimators for the feature, score, quality, and pipeline monitors
used in the silent-degradation exercise. Each function takes plain arrays
(or a Polars DataFrame for the schema check) so it can be lifted into another
project without the rest of this folder.

Signal families
---------------
- Missingness and defaults: null rate, zero rate
- Location and spread: mean, standard deviation, quantiles
- Two-sample distribution tests: Kolmogorov-Smirnov statistic, population
  stability index (PSI)
- Score distribution: the same tests applied to model scores, plus the
  entropy of a fixed score histogram
- Quality, once labels exist: AUC, group AUC, NDCG@K, log loss, expected
  calibration error, calibration gap
- Pipeline health: schema pass rate, rows per day, join match rate

Every comparison returns a non-negative "badness" statistic so that one rule,
statistic > threshold, decides whether a monitor fires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score

DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


# =============================================================================
# Helpers
# =============================================================================

def _as_float(x) -> np.ndarray:
    """Return a 1-D float64 copy of x, with None and nulls mapped to NaN."""
    if isinstance(x, pl.Series):
        x = x.cast(pl.Float64).to_numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def _non_null(x) -> np.ndarray:
    arr = _as_float(x)
    return arr[~np.isnan(arr)]


# =============================================================================
# Missingness and defaults
# =============================================================================

def null_rate(x) -> float:
    """Fraction of rows that are null or NaN."""
    arr = _as_float(x)
    if arr.size == 0:
        return float("nan")
    return float(np.isnan(arr).mean())


def zero_rate(x) -> float:
    """Fraction of all rows whose value is exactly zero (nulls count as non-zero)."""
    arr = _as_float(x)
    if arr.size == 0:
        return float("nan")
    return float((arr == 0.0).mean())


# =============================================================================
# Location, spread, quantiles
# =============================================================================

def quantiles(x, qs: Sequence[float] = DEFAULT_QUANTILES) -> np.ndarray:
    """Quantiles of the non-null values, using linear interpolation (Hyndman-Fan type 7)."""
    arr = _non_null(x)
    if arr.size == 0:
        return np.full(len(qs), np.nan)
    return np.quantile(arr, qs, method="linear")


def summary_stats(x, qs: Sequence[float] = DEFAULT_QUANTILES) -> Dict[str, float]:
    """Row count, null rate, zero rate, mean, standard deviation, and quantiles."""
    arr = _non_null(x)
    out = {
        "n": int(_as_float(x).size),
        "null_rate": null_rate(x),
        "zero_rate": zero_rate(x),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
    }
    for q, v in zip(qs, quantiles(x, qs)):
        out[f"q{int(round(q * 100)):02d}"] = float(v)
    return out


def standardized_mean_shift(reference, current) -> float:
    """|mean(current) - mean(reference)| divided by the reference standard deviation.

    A constant reference has no spread to scale by, so any change in the mean
    returns infinity and no change returns zero.
    """
    ref, cur = _non_null(reference), _non_null(current)
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    diff = abs(cur.mean() - ref.mean())
    sd = ref.std()
    if sd == 0:
        return 0.0 if diff == 0 else float("inf")
    return float(diff / sd)


def standardized_quantile_shift(reference, current,
                                qs: Sequence[float] = DEFAULT_QUANTILES) -> float:
    """Largest |quantile shift| across qs, divided by the reference standard deviation."""
    ref = _non_null(reference)
    if ref.size == 0 or _non_null(current).size == 0:
        return float("nan")
    diff = float(np.max(np.abs(quantiles(current, qs) - quantiles(reference, qs))))
    sd = ref.std()
    if sd == 0:
        return 0.0 if diff == 0 else float("inf")
    return diff / sd


# =============================================================================
# Two-sample distribution tests
# =============================================================================

def ks_statistic(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two empirical CDFs."""
    a = np.sort(_non_null(reference))
    b = np.sort(_non_null(current))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_pvalue(reference, current) -> float:
    """Asymptotic p-value of the two-sample KS test.

    With tens of thousands of rows almost any real difference is
    "significant", so the exercise fires on the statistic, not the p-value.
    """
    a, b = _non_null(reference), _non_null(current)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b, method="asymp").pvalue)


def quantile_cut_points(reference, n_bins: int = 10) -> np.ndarray:
    """Interior bin edges at the reference quantiles, de-duplicated for discrete features."""
    ref = _non_null(reference)
    if ref.size == 0:
        return np.array([])
    probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(ref, probs, method="linear"))


def bin_proportions(x, cut_points: np.ndarray) -> np.ndarray:
    """Share of rows in each bin (-inf, c1], (c1, c2], ..., (ck, inf), plus a final null bin."""
    arr = _as_float(x)
    if arr.size == 0:
        return np.full(len(cut_points) + 2, np.nan)
    is_null = np.isnan(arr)
    idx = np.searchsorted(cut_points, arr[~is_null], side="left")
    counts = np.bincount(idx, minlength=len(cut_points) + 1).astype(np.float64)
    counts = np.append(counts, is_null.sum())
    return counts / arr.size


def psi_from_proportions(p_reference, p_current, eps: float = 1e-4) -> float:
    """PSI = sum over bins of (current - reference) * ln(current / reference).

    Empty bins are floored at eps so that the logarithm stays finite.
    """
    p = np.clip(np.asarray(p_reference, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(p_current, dtype=np.float64), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of current against reference, on reference-quantile bins."""
    cuts = quantile_cut_points(reference, n_bins)
    return psi_from_proportions(bin_proportions(reference, cuts),
                                bin_proportions(current, cuts), eps)


def histogram_entropy(x, bins: int = 20, value_range=(0.0, 1.0)) -> float:
    """Shannon entropy, in nats, of a fixed-width histogram (scores use 20 bins on [0, 1])."""
    arr = _non_null(x)
    if arr.size == 0:
        return float("nan")
    counts, _ = np.histogram(arr, bins=bins, range=value_range)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


# =============================================================================
# Quality and calibration (need labels)
# =============================================================================

def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve; NaN when only one class is present."""
    y = np.asarray(y_true)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_score)))


def group_auc(groups, y_true, y_score) -> float:
    """Impression-weighted mean of per-group AUC, over groups that contain both classes.

    Each group's AUC uses the rank-sum (Mann-Whitney) form with average ranks
    for ties: (R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg).
    """
    df = pl.DataFrame({"g": np.asarray(groups), "y": np.asarray(y_true, dtype=np.int64),
                       "s": np.asarray(y_score, dtype=np.float64)})
    df = df.with_columns(pl.col("s").rank("average").over("g").alias("r"))
    agg = (
        df.group_by("g")
        .agg(pl.len().alias("n"), pl.col("y").sum().alias("n_pos"),
             (pl.col("r") * pl.col("y")).sum().alias("r_pos"))
        .with_columns((pl.col("n") - pl.col("n_pos")).alias("n_neg"))
        .filter((pl.col("n_pos") > 0) & (pl.col("n_neg") > 0))
    )
    if agg.height == 0:
        return float("nan")
    auc = (agg["r_pos"] - agg["n_pos"] * (agg["n_pos"] + 1) / 2) / (agg["n_pos"] * agg["n_neg"])
    n = agg["n"].cast(pl.Float64)
    return float((auc * n).sum() / n.sum())


def ndcg_at_k(groups, y_true, y_score, k: int = 10, tiebreak=None) -> float:
    """Mean NDCG@k over groups with at least one relevant row.

    Rows are ranked by score within each group. Ties are broken by tiebreak
    (ascending) when given, so the result does not depend on row order.
    """
    n = len(np.asarray(groups))
    tb = np.arange(n) if tiebreak is None else np.asarray(tiebreak)
    df = pl.DataFrame({"g": np.asarray(groups), "y": np.asarray(y_true, dtype=np.float64),
                       "s": np.asarray(y_score, dtype=np.float64), "tb": tb})
    discount = 1.0 / (pl.col("rank") + 1).cast(pl.Float64).log(base=2)

    def _dcg(frame: pl.DataFrame, sort_cols, descending) -> pl.DataFrame:
        ranked = frame.sort(["g"] + sort_cols, descending=[False] + descending)
        ranked = ranked.with_columns(pl.col("y").cum_count().over("g").alias("rank"))
        return (ranked.filter(pl.col("rank") <= k)
                .group_by("g").agg((pl.col("y") * discount).sum().alias("dcg")))

    dcg = _dcg(df, ["s", "tb"], [True, False])
    idcg = _dcg(df, ["y", "tb"], [True, False]).rename({"dcg": "idcg"})
    joined = idcg.filter(pl.col("idcg") > 0).join(dcg, on="g", how="left")
    if joined.height == 0:
        return float("nan")
    return float((joined["dcg"] / joined["idcg"]).mean())


def log_loss(y_true, y_prob, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy, with probabilities clipped to [eps, 1 - eps]."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1 - eps)
    if y.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(y_true, y_prob, n_bins: int = 10) -> List[Dict[str, float]]:
    """Per-bin count, mean predicted probability, and observed positive rate (equal-width bins)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": b,
            "lower": b / n_bins,
            "upper": (b + 1) / n_bins,
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()) if m.any() else float("nan"),
            "observed": float(y[m].mean()) if m.any() else float("nan"),
        })
    return rows


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE = sum over equal-width bins of (bin share) * |observed rate - mean predicted|."""
    y = np.asarray(y_true)
    if y.size == 0:
        return float("nan")
    ece = 0.0
    for row in reliability_table(y_true, y_prob, n_bins):
        if row["n"]:
            ece += row["n"] / y.size * abs(row["observed"] - row["mean_pred"])
    return float(ece)


def calibration_gap(y_true, y_prob) -> float:
    """Mean predicted probability minus observed positive rate (calibration in the large)."""
    y = np.asarray(y_true, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.asarray(y_prob, dtype=np.float64)) - y.mean())


def quality_metrics(groups, y_true, y_prob, k: int = 10, tiebreak=None,
                    n_calibration_bins: int = 10) -> Dict[str, float]:
    """All label-dependent quality metrics for one slice of traffic."""
    return {
        "auc": roc_auc(y_true, y_prob),
        "gauc": group_auc(groups, y_true, y_prob),
        f"ndcg@{k}": ndcg_at_k(groups, y_true, y_prob, k=k, tiebreak=tiebreak),
        "log_loss": log_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_calibration_bins),
        "calibration_gap": calibration_gap(y_true, y_prob),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "mean_score": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
    }


# =============================================================================
# Pipeline health
# =============================================================================

def join_match_rate(keys, lookup_keys) -> float:
    """Fraction of rows whose key is present in the lookup table they are joined against."""
    keys = np.asarray(keys)
    if keys.size == 0:
        return float("nan")
    return float(np.isin(keys, np.asarray(lookup_keys)).mean())


@dataclass(frozen=True)
class ColumnContract:
    """A type-and-range data contract for one column, the kind a schema validator enforces."""
    name: str
    dtype: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    nullable: bool = False


def schema_pass_rate(df: pl.DataFrame, contract: Iterable[ColumnContract]) -> Dict[str, float]:
    """Check dtypes for every column, then nulls and bounds row by row.

    Returns the fraction of rows that pass every row-level check, and whether
    every column still has its contracted dtype.
    """
    contract = list(contract)
    dtype_ok = all(c.name in df.columns and str(df.schema[c.name]) == c.dtype for c in contract)
    checks = []
    for c in contract:
        if c.name not in df.columns:
            continue
        col = pl.col(c.name)
        ok = pl.lit(True)
        if not c.nullable:
            ok = ok & col.is_not_null() & ~col.cast(pl.Float64).is_nan()
        if c.lower is not None:
            ok = ok & (col >= c.lower)
        if c.upper is not None:
            ok = ok & (col <= c.upper)
        checks.append(ok.fill_null(False))
    if not checks or df.height == 0:
        return {"pass_rate": float("nan"), "dtype_ok": dtype_ok}
    row_ok = df.select(pl.all_horizontal(checks).alias("ok"))["ok"]
    return {"pass_rate": float(row_ok.mean()), "dtype_ok": dtype_ok}


# =============================================================================
# Monitor rules
# =============================================================================

@dataclass(frozen=True)
class Thresholds:
    """Alert thresholds, fixed before any run. Each fires when statistic > threshold.

    The PSI value follows the common credit-scoring rule of thumb for a
    significant shift; it carries no stated error rate. The rest are
    illustrative round numbers. The clean control in the exercise reports
    which of them are already exceeded without any corruption.
    """
    schema_pass_drop: float = 0.001
    rows_per_day_rel_change: float = 0.25
    join_rate_drop: float = 0.02
    null_rate_rise: float = 0.01
    zero_rate_rise: float = 0.02
    mean_shift_sd: float = 0.2
    quantile_shift_sd: float = 0.2
    ks: float = 0.1
    psi: float = 0.25
    score_entropy_change: float = 0.1
    auc_drop: float = 0.01
    gauc_drop: float = 0.01
    ndcg_drop: float = 0.01
    log_loss_rise: float = 0.01
    ece_rise: float = 0.01
    calibration_gap_rise: float = 0.01

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def distribution_signals(reference, current, th: Thresholds,
                         n_bins: int = 10) -> List[Dict[str, float]]:
    """Null rate, zero rate, mean, quantiles, KS, and PSI of current against reference."""
    ref_null, cur_null = null_rate(reference), null_rate(current)
    ref_zero, cur_zero = zero_rate(reference), zero_rate(current)
    ref_s, cur_s = summary_stats(reference), summary_stats(current)
    return [
        {"signal": "null_rate", "reference_value": ref_null, "current_value": cur_null,
         "statistic": cur_null - ref_null, "threshold": th.null_rate_rise},
        {"signal": "zero_rate", "reference_value": ref_zero, "current_value": cur_zero,
         "statistic": cur_zero - ref_zero, "threshold": th.zero_rate_rise},
        {"signal": "mean", "reference_value": ref_s["mean"], "current_value": cur_s["mean"],
         "statistic": standardized_mean_shift(reference, current), "threshold": th.mean_shift_sd},
        {"signal": "quantiles", "reference_value": ref_s["q50"], "current_value": cur_s["q50"],
         "statistic": standardized_quantile_shift(reference, current),
         "threshold": th.quantile_shift_sd},
        {"signal": "ks", "reference_value": float("nan"), "current_value": float("nan"),
         "statistic": ks_statistic(reference, current), "threshold": th.ks,
         "p_value": ks_pvalue(reference, current)},
        {"signal": "psi", "reference_value": float("nan"), "current_value": float("nan"),
         "statistic": psi(reference, current, n_bins=n_bins), "threshold": th.psi},
    ]


def score_signals(reference_scores, current_scores, th: Thresholds,
                  n_bins: int = 10) -> List[Dict[str, float]]:
    """Distribution signals on model scores, plus the change in score-histogram entropy."""
    rows = [r for r in distribution_signals(reference_scores, current_scores, th, n_bins)
            if r["signal"] in ("mean", "quantiles", "ks", "psi")]
    h_ref, h_cur = histogram_entropy(reference_scores), histogram_entropy(current_scores)
    rows.append({"signal": "entropy", "reference_value": h_ref, "current_value": h_cur,
                 "statistic": abs(h_cur - h_ref), "threshold": th.score_entropy_change})
    for r in rows:
        r["signal"] = f"score_{r['signal']}"
    return rows


def quality_signals(reference: Dict[str, float], current: Dict[str, float],
                    th: Thresholds, k: int = 10) -> List[Dict[str, float]]:
    """Quality-metric changes, signed so that a positive statistic means "worse"."""
    nd = f"ndcg@{k}"
    spec = [
        ("auc", reference["auc"] - current["auc"], th.auc_drop),
        ("gauc", reference["gauc"] - current["gauc"], th.gauc_drop),
        (nd, reference[nd] - current[nd], th.ndcg_drop),
        ("log_loss", current["log_loss"] - reference["log_loss"], th.log_loss_rise),
        ("ece", current["ece"] - reference["ece"], th.ece_rise),
        ("calibration_gap", abs(current["calibration_gap"]) - abs(reference["calibration_gap"]),
         th.calibration_gap_rise),
    ]
    return [{"signal": name, "reference_value": reference[name], "current_value": current[name],
             "statistic": stat, "threshold": thr} for name, stat, thr in spec]


def pipeline_signals(reference: Dict[str, float], current: Dict[str, float],
                     th: Thresholds) -> List[Dict[str, float]]:
    """Schema pass rate, rows per day, and join match rates."""
    def rel(a, b):
        return abs(b / a - 1.0) if a else float("inf")
    return [
        {"signal": "schema_pass", "reference_value": reference["schema_pass_rate"],
         "current_value": current["schema_pass_rate"],
         "statistic": reference["schema_pass_rate"] - current["schema_pass_rate"]
         + (0.0 if current["schema_dtype_ok"] else 1.0),
         "threshold": th.schema_pass_drop},
        {"signal": "rows_per_day", "reference_value": reference["rows_per_day"],
         "current_value": current["rows_per_day"],
         "statistic": rel(reference["rows_per_day"], current["rows_per_day"]),
         "threshold": th.rows_per_day_rel_change},
        {"signal": "join_rate_user", "reference_value": reference["join_rate_user"],
         "current_value": current["join_rate_user"],
         "statistic": reference["join_rate_user"] - current["join_rate_user"],
         "threshold": th.join_rate_drop},
        {"signal": "join_rate_item", "reference_value": reference["join_rate_item"],
         "current_value": current["join_rate_item"],
         "statistic": reference["join_rate_item"] - current["join_rate_item"],
         "threshold": th.join_rate_drop},
    ]
