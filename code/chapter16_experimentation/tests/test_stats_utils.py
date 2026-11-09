"""Known-answer tests for the A/B statistics in stats_utils.py."""

import numpy as np
import pytest
from scipy import integrate, stats

import stats_utils as su


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.exponential(2.0, 300), rng.exponential(2.3, 450)
    res = su.welch_test(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    ci = ref.confidence_interval(0.95)
    assert (res.ci_low, res.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-8)


def test_welch_from_moments_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    samples = [(rng.normal(0, 1, 50), rng.normal(0.2, 2, 70)) for _ in range(5)]
    m1 = [a.mean() for a, _ in samples]; v1 = [a.var(ddof=1) for a, _ in samples]
    m2 = [b.mean() for _, b in samples]; v2 = [b.var(ddof=1) for _, b in samples]
    _, _, _, p, _, _ = su.welch_from_moments(m1, v1, 50, m2, v2, 70)
    expected = [su.welch_test(a, b).p_value for a, b in samples]
    np.testing.assert_allclose(p, expected, rtol=1e-12)


def test_sample_size_rule_of_thumb():
    # 2 (1.959964 + 0.841621)^2 = 15.6978..., the source of "16 sigma^2 / delta^2".
    n = su.sample_size_per_arm(sigma=1.0, delta=1.0)
    assert n == pytest.approx(15.6978, abs=1e-3)
    assert su.sample_size_per_arm(sigma=3.0, delta=0.5) == pytest.approx(n * 36.0)
    assert su.power_two_sample(1.0, 1.0, n) == pytest.approx(0.80, abs=1e-3)


def test_cuped_theta_is_ols_slope():
    rng = np.random.default_rng(2)
    x = rng.gamma(2.0, 3.0, 2000)
    y = 1.5 + 0.7 * x + rng.normal(0, 2, 2000)
    slope = np.polyfit(x, y, 1)[0]
    assert su.cuped_theta(y, x) == pytest.approx(slope, rel=1e-10)


def test_cuped_preserves_mean_and_shrinks_variance_by_one_minus_rho_squared():
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 3.0, 5000)
    y = 0.8 * x + rng.normal(0, 3, 5000)
    adj = su.cuped_adjust(y, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert adj.mean() == pytest.approx(y.mean(), rel=1e-12)
    # In-sample identity: var(Y - theta X) = var(Y) (1 - rho^2) when theta is the OLS slope.
    assert adj.var() == pytest.approx(y.var() * (1 - rho ** 2), rel=1e-10)


def test_cuped_unbiased_across_aa_and_recovers_injected_effect():
    rng = np.random.default_rng(4)
    n, sims, effect = 2000, 1500, 0.5
    aa, ab = [], []
    for _ in range(sims):
        x = rng.gamma(2.0, 2.0, n)
        y0 = x + rng.normal(0, 1.5, n)
        z = rng.random(n) < 0.5
        aa.append(su.welch_test(su.cuped_adjust(y0, x)[z], su.cuped_adjust(y0, x)[~z]).diff)
        y1 = y0 + effect * z
        adj = su.cuped_adjust(y1, x)
        ab.append(adj[z].mean() - adj[~z].mean())
    aa, ab = np.array(aa), np.array(ab)
    se = aa.std() / np.sqrt(sims)
    assert abs(aa.mean()) < 4 * se
    assert abs(ab.mean() - effect) < 4 * se


def test_delta_method_exact_when_denominator_constant():
    rng = np.random.default_rng(5)
    num = rng.poisson(3.0, 400).astype(float)
    den = np.full(400, 5.0)
    assert su.delta_method_ratio_var(num, den) == pytest.approx(num.var(ddof=1) / (400 * 25.0), rel=1e-12)


def test_delta_method_matches_user_bootstrap():
    rng = np.random.default_rng(6)
    n = 3000
    den = rng.poisson(rng.gamma(0.8, 20, n)) + 1
    rate = rng.beta(3, 2, n)            # per-user completion propensity
    num = rng.binomial(den, rate)
    boot = su.poisson_bootstrap_ratio(num, den, 2000, rng)
    assert np.sqrt(su.delta_method_ratio_var(num, den)) == pytest.approx(boot.std(ddof=1), rel=0.08)
    # Per-event variance ignores within-user correlation and is far too small here.
    assert su.naive_ratio_var(num, den) < 0.5 * su.delta_method_ratio_var(num, den)


def test_srm_flags_small_imbalance_only_at_scale():
    assert su.srm_test([505_000, 495_000], [0.5, 0.5]).flagged
    assert not su.srm_test([505, 495], [0.5, 0.5]).flagged
    balanced = su.srm_test([500_000, 500_000], [0.5, 0.5])
    assert balanced.p_value == pytest.approx(1.0) and not balanced.flagged
    # chi^2 = sum (o - e)^2 / e = 2 * 5000^2 / 500000 = 100
    assert su.srm_test([505_000, 495_000], [0.5, 0.5]).chi2 == pytest.approx(100.0)


def test_msprt_matches_numerical_mixture_integral():
    diff, V, tau2 = 0.3, 0.04, 0.09
    lik = lambda th: stats.norm.pdf(diff, th, np.sqrt(V)) * stats.norm.pdf(th, 0, np.sqrt(tau2))
    numeric = integrate.quad(lik, -10, 10)[0] / stats.norm.pdf(diff, 0, np.sqrt(V))
    assert float(su.msprt_lambda(diff, V, tau2)) == pytest.approx(numeric, rel=1e-8)
    assert float(su.msprt_lambda(0.0, V, tau2)) == pytest.approx(np.sqrt(V / (V + tau2)))


def test_msprt_false_positive_rate_under_continuous_monitoring():
    rng = np.random.default_rng(7)
    sims, looks, per_look, alpha, sigma = 2000, 100, 20, 0.05, 1.0
    a = rng.normal(0, sigma, (sims, looks, per_look)).sum(axis=2).cumsum(axis=1)
    b = rng.normal(0, sigma, (sims, looks, per_look)).sum(axis=2).cumsum(axis=1)
    n = per_look * np.arange(1, looks + 1)
    diff = (a - b) / n
    V = 2 * sigma ** 2 / n
    lam = su.msprt_lambda(diff, V, tau2=0.05)
    reject = (lam >= 1 / alpha).any(axis=1)
    # Naive repeated z-tests on the same streams, for contrast.
    naive = (np.abs(diff / np.sqrt(V)) > 1.96).any(axis=1)
    assert reject.mean() <= alpha + 0.015
    assert naive.mean() > 2 * alpha


def test_always_valid_p_values_are_running_minimum():
    lam = np.array([0.5, 4.0, 2.0, 25.0, 10.0])
    np.testing.assert_allclose(su.always_valid_p_values(lam), [1.0, 0.25, 0.25, 0.04, 0.04])


def test_bonferroni_and_benjamini_hochberg():
    assert su.bonferroni_alpha(0.05, 28) == pytest.approx(0.05 / 28)
    np.testing.assert_array_equal(su.benjamini_hochberg([0.01, 0.04, 0.03, 0.005]), [True, True, True, True])
    np.testing.assert_array_equal(su.benjamini_hochberg([0.01, 0.02, 0.03, 0.5]), [True, True, True, False])
    np.testing.assert_array_equal(su.benjamini_hochberg([0.2, 0.3, 0.9]), [False, False, False])


def test_benjamini_hochberg_is_step_up_and_matches_scipy():
    # Sorted thresholds at q = 0.05, m = 4: 0.0125, 0.025, 0.0375, 0.05. The second p-value
    # misses its own threshold, but the third meets its threshold, so step-up rejects all three.
    np.testing.assert_array_equal(su.benjamini_hochberg([0.036, 0.01, 0.03, 0.9]), [True, True, True, False])
    rng = np.random.default_rng(8)
    for _ in range(200):
        p = np.r_[rng.uniform(0, 0.02, rng.integers(0, 6)), rng.uniform(0, 1, rng.integers(1, 30))]
        rng.shuffle(p)
        np.testing.assert_array_equal(su.benjamini_hochberg(p, q=0.1), stats.false_discovery_control(p) <= 0.1)


def test_effective_sample_size():
    assert su.effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert su.effective_sample_size(np.r_[np.zeros(9), 7.0]) == pytest.approx(1.0)
    assert su.effective_sample_size([1.0, 2.0, 3.0]) == pytest.approx(36.0 / 14.0)


def test_winsorize_caps_only_above():
    np.testing.assert_array_equal(su.winsorize_upper([1, 5, 10, 100], 10), [1, 5, 10, 10])
