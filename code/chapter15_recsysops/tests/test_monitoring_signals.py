"""Known-answer tests for the monitoring estimators."""

import math

import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import ndcg_score, roc_auc_score

import monitoring_signals as ms


# --- missingness -------------------------------------------------------------

def test_null_rate_counts_none_and_nan():
    assert ms.null_rate(pl.Series([1.0, None, float("nan"), 4.0])) == pytest.approx(0.5)
    assert ms.null_rate(np.array([1.0, 2.0])) == 0.0


def test_zero_rate_uses_all_rows():
    assert ms.zero_rate(pl.Series([0, 0, 1, None])) == pytest.approx(0.5)
    assert math.isnan(ms.zero_rate(np.array([])))


# --- quantiles and shifts ----------------------------------------------------

def test_quantiles_linear_interpolation():
    q = ms.quantiles([1, 2, 3, 4], qs=(0.25, 0.5, 0.75))
    np.testing.assert_allclose(q, [1.75, 2.5, 3.25])


def test_quantiles_ignore_nulls():
    q = ms.quantiles(pl.Series([1.0, None, 2.0, 3.0, float("nan"), 4.0]), qs=(0.5,))
    np.testing.assert_allclose(q, [2.5])


def test_summary_stats_keys():
    s = ms.summary_stats([0.0, 1.0, 2.0, 3.0])
    assert s["n"] == 4 and s["zero_rate"] == 0.25 and s["mean"] == 1.5
    assert s["q50"] == pytest.approx(1.5) and s["q01"] == pytest.approx(0.03)


def test_standardized_shifts():
    ref, cur = [0.0, 2.0], [3.0, 3.0]          # reference mean 1, population sd 1
    assert ms.standardized_mean_shift(ref, cur) == pytest.approx(2.0)
    assert ms.standardized_quantile_shift(ref, cur, qs=(0.5,)) == pytest.approx(2.0)


def test_shift_with_constant_reference():
    assert ms.standardized_mean_shift([5, 5], [5, 5]) == 0.0
    assert ms.standardized_mean_shift([5, 5], [6]) == math.inf


# --- KS ----------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ([1, 2, 3], [4, 5, 6], 1.0),
    ([1, 2, 3, 4], [3, 4, 5, 6], 0.5),
    ([1, 2, 3], [1, 2, 3], 0.0),
])
def test_ks_known_answers(a, b, expected):
    assert ms.ks_statistic(a, b) == pytest.approx(expected)


def test_ks_point_mass_at_median_is_one_half():
    # The empirical CDF of 0..99 is 0.5 just below 49.5 and 0.5 above it.
    assert ms.ks_statistic(np.arange(100), np.full(50, 49.5)) == pytest.approx(0.5)


def test_ks_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 20, 500), rng.integers(2, 22, 300)
    assert ms.ks_statistic(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic)
    assert 0.0 <= ms.ks_pvalue(a, b) <= 1.0


# --- PSI ---------------------------------------------------------------------

def test_psi_from_proportions_known_answer():
    # (0.25 - 0.5) ln(0.5) + (0.75 - 0.5) ln(1.5)
    assert ms.psi_from_proportions([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.2746530722, rel=1e-9)
    assert ms.psi_from_proportions([0.3, 0.7], [0.3, 0.7]) == 0.0


def test_psi_end_to_end_known_answer():
    # Reference 0..9 with 2 bins cuts at 4.5; current puts 1 of 4 rows at or below it.
    assert ms.quantile_cut_points(np.arange(10), n_bins=2).tolist() == [4.5]
    assert ms.psi(np.arange(10), [0, 5, 6, 7], n_bins=2) == pytest.approx(0.2746530722, rel=1e-9)


def test_bin_proportions_right_closed_with_null_bin():
    p = ms.bin_proportions(pl.Series([1.0, 4.5, 5.0, None]), np.array([4.5]))
    np.testing.assert_allclose(p, [0.5, 0.25, 0.25])


def test_cut_points_deduplicate_discrete_values():
    cuts = ms.quantile_cut_points([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], n_bins=10)
    assert len(cuts) == len(set(cuts.tolist())) and len(cuts) < 9


def test_psi_identical_is_zero_and_symmetric_terms_positive():
    rng = np.random.default_rng(2)
    x = rng.normal(size=2000)
    assert ms.psi(x, x) == pytest.approx(0.0)
    assert ms.psi(x, x + 1.0) > 0.25


# --- the mean-fill result the chapter relies on -------------------------------

def test_mean_fill_passes_null_zero_mean_but_not_ks_psi_quantiles():
    rng = np.random.default_rng(3)
    ref = rng.normal(10.0, 2.0, 5000)
    cur = np.full(5000, ref.mean())
    rows = {r["signal"]: r for r in ms.distribution_signals(ref, cur, ms.Thresholds())}
    for name in ("null_rate", "zero_rate", "mean"):
        assert rows[name]["statistic"] <= rows[name]["threshold"], name
    assert rows["ks"]["statistic"] >= 0.5
    assert rows["psi"]["statistic"] > 10 * rows["psi"]["threshold"]
    assert rows["quantiles"]["statistic"] > rows["quantiles"]["threshold"]


def test_mean_fill_ks_bound_with_reference_mass_at_the_fill_value():
    # KS of a point mass at m is max(F_ref(m-), 1 - F_ref(m)) >= (1 - P_ref(X = m)) / 2.
    rng = np.random.default_rng(7)
    ref = np.r_[np.zeros(300), rng.poisson(4.0, 700)].astype(float)
    m = float(round(ref.mean()))
    ks = ms.ks_statistic(ref, np.full(500, m))
    assert ks == pytest.approx(max(np.mean(ref < m), np.mean(ref > m)))
    assert ks >= (1 - np.mean(ref == m)) / 2


def test_mean_fill_lowers_zero_rate_instead_of_raising_it():
    ref = np.r_[np.zeros(50), np.linspace(1.0, 100.0, 950)]
    rows = {r["signal"]: r for r in ms.distribution_signals(ref, np.full(1000, ref.mean()), ms.Thresholds())}
    assert rows["zero_rate"]["statistic"] == pytest.approx(-0.05)


def test_zero_fill_fires_zero_rate_and_mean():
    rng = np.random.default_rng(4)
    ref = rng.normal(10.0, 2.0, 5000)
    rows = {r["signal"]: r for r in ms.distribution_signals(ref, np.zeros(5000), ms.Thresholds())}
    assert rows["zero_rate"]["statistic"] == pytest.approx(1.0)
    assert rows["mean"]["statistic"] > rows["mean"]["threshold"]
    assert rows["null_rate"]["statistic"] == 0.0


# --- entropy -----------------------------------------------------------------

def test_histogram_entropy():
    assert ms.histogram_entropy([0.01, 0.99]) == pytest.approx(math.log(2))
    assert ms.histogram_entropy([0.5, 0.5, 0.5]) == 0.0


# --- quality and calibration ---------------------------------------------------

def test_ece_and_calibration_gap_known_answer():
    p, y = [0.1, 0.1, 0.9, 0.9], [0, 1, 1, 1]
    # bin [0.1, 0.2): |0.5 - 0.1| * 0.5 ; bin [0.9, 1.0]: |1.0 - 0.9| * 0.5
    assert ms.expected_calibration_error(y, p) == pytest.approx(0.25)
    assert ms.calibration_gap(y, p) == pytest.approx(-0.25)


def test_reliability_table_puts_one_in_last_bin():
    rows = ms.reliability_table([1, 0], [1.0, 0.0], n_bins=10)
    assert rows[9]["n"] == 1 and rows[0]["n"] == 1


def test_log_loss_matches_sklearn():
    rng = np.random.default_rng(5)
    y, p = rng.integers(0, 2, 200), rng.uniform(0.01, 0.99, 200)
    assert ms.log_loss(y, p) == pytest.approx(sk_log_loss(y, p))


def test_group_auc_matches_weighted_sklearn():
    g = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.3, 0.3, 0.8, 0.2, 0.5, 0.2, 0.4, 0.6, 0.1])
    expected_num, expected_den = 0.0, 0
    for grp in (1, 2):                      # groups 3 and 4 have one class only
        m = g == grp
        expected_num += roc_auc_score(y[m], s[m]) * m.sum()
        expected_den += m.sum()
    assert ms.group_auc(g, y, s) == pytest.approx(expected_num / expected_den)


def test_ndcg_known_answer_and_sklearn():
    y, s = [1, 0, 1], [0.9, 0.8, 0.1]
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ms.ndcg_at_k([7, 7, 7], y, s, k=10) == pytest.approx(expected)
    assert ms.ndcg_at_k([7, 7, 7], y, s, k=10) == pytest.approx(ndcg_score([y], [s]))
    assert ms.ndcg_at_k([7, 7, 7], y, s, k=1) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_ndcg_matches_sklearn_over_many_groups(k):
    rng = np.random.default_rng(6)
    g = rng.integers(0, 40, 1500)
    y = (rng.random(1500) < 0.3).astype(int)
    s = rng.random(1500)
    expected = [ndcg_score([y[g == u]], [s[g == u]], k=k) if (g == u).sum() > 1 else 1.0
                for u in np.unique(g) if y[g == u].sum() > 0]
    assert ms.ndcg_at_k(g, y, s, k=k) == pytest.approx(np.mean(expected), rel=1e-12)


def test_ndcg_skips_groups_without_positives_and_uses_tiebreak():
    assert ms.ndcg_at_k([1, 1, 2], [1, 0, 0], [0.9, 0.1, 0.5]) == pytest.approx(1.0)
    assert ms.ndcg_at_k([1, 1], [0, 1], [0.5, 0.5], tiebreak=[1, 0]) == pytest.approx(1.0)
    assert ms.ndcg_at_k([1, 1], [0, 1], [0.5, 0.5], tiebreak=[0, 1]) == pytest.approx(1 / math.log2(3))


def test_quality_signals_positive_means_worse():
    ref = {"auc": 0.8, "gauc": 0.7, "ndcg@10": 0.9, "log_loss": 0.5, "ece": 0.02,
           "calibration_gap": -0.01}
    cur = {"auc": 0.7, "gauc": 0.75, "ndcg@10": 0.9, "log_loss": 0.6, "ece": 0.01,
           "calibration_gap": 0.05}
    rows = {r["signal"]: r["statistic"] for r in ms.quality_signals(ref, cur, ms.Thresholds())}
    assert rows["auc"] == pytest.approx(0.1) and rows["gauc"] == pytest.approx(-0.05)
    assert rows["log_loss"] == pytest.approx(0.1) and rows["ece"] == pytest.approx(-0.01)
    assert rows["calibration_gap"] == pytest.approx(0.04)


# --- pipeline health -----------------------------------------------------------

def test_join_match_rate():
    assert ms.join_match_rate([1, 2, 3, 4], [1, 2]) == 0.5


def test_schema_pass_rate_bounds_nulls_and_dtype():
    contract = [ms.ColumnContract("a", "Int32", lower=0.0), ms.ColumnContract("b", "Float32", 0.0, 1.0)]
    df = pl.DataFrame({"a": pl.Series([0, 5, -1, 2], dtype=pl.Int32),
                       "b": pl.Series([0.5, 1.0, 0.2, None], dtype=pl.Float32)})
    out = ms.schema_pass_rate(df, contract)
    assert out == {"pass_rate": 0.5, "dtype_ok": True}
    out = ms.schema_pass_rate(df.with_columns(pl.col("a").cast(pl.Float64)), contract)
    assert out["dtype_ok"] is False


def test_zero_fill_is_schema_valid():
    contract = [ms.ColumnContract("x", "Float32", lower=0.0)]
    df = pl.DataFrame({"x": pl.Series([0.0, 0.0, 0.0], dtype=pl.Float32)})
    assert ms.schema_pass_rate(df, contract)["pass_rate"] == 1.0


def test_pipeline_signals_dtype_break_fires():
    base = {"schema_pass_rate": 1.0, "schema_dtype_ok": True, "rows_per_day": 100.0,
            "join_rate_user": 1.0, "join_rate_item": 0.99}
    th = ms.Thresholds()
    rows = {r["signal"]: r["statistic"] for r in ms.pipeline_signals(base, dict(base), th)}
    assert all(v == 0.0 for v in rows.values())
    broken = {**base, "schema_dtype_ok": False, "rows_per_day": 50.0}
    rows = {r["signal"]: r["statistic"] for r in ms.pipeline_signals(base, broken, th)}
    assert rows["schema_pass"] == 1.0 and rows["rows_per_day"] == pytest.approx(0.5)
