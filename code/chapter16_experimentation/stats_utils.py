"""
Chapter 16: Experimentation and Bandits - statistics for online experiments
===========================================================================
Small, dependency-light implementations of the estimators used in
Sections 16.3 and 16.4:

- Welch two-sample test (from raw data or from per-arm moments, vectorized)
- Sample size and power for a difference in means
- Winsorizing (capping) with a pre-specified cap
- CUPED adjustment with theta = cov(X, Y) / var(X)          (Deng et al., 2013)
- Sample ratio mismatch (SRM) chi-square check            (Fabijan et al., 2019)
- Delta-method variance for ratio metrics          (Deng, Knoblich, and Lu, 2018)
- User-level (Poisson) bootstrap for ratio metrics
- Mixture sequential probability ratio test (mSPRT)       (Johari et al., 2017)
- Bonferroni and Benjamini-Hochberg corrections
- Effective sample size of importance weights

Every function works on NumPy arrays and has a unit test in tests/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats


# =============================================================================
# Two-sample tests
# =============================================================================

@dataclass
class TestResult:
    diff: float
    se: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float


def welch_from_moments(mean1, var1, n1, mean2, var2, n2, alpha: float = 0.05):
    """Welch's unequal-variance t-test from per-arm moments (vectorized).

    `var1`, `var2` are unbiased sample variances (ddof=1). Returns
    (diff, se, t, p, ci_low, ci_high) as arrays, with diff = mean1 - mean2.
    """
    mean1, var1, n1, mean2, var2, n2 = (np.asarray(a, dtype=float) for a in (mean1, var1, n1, mean2, var2, n2))
    a = var1 / n1
    b = var2 / n2
    se = np.sqrt(a + b)
    diff = mean1 - mean2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = diff / se
        df = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
    p = 2.0 * stats.t.sf(np.abs(t), df)
    q = stats.t.ppf(1.0 - alpha / 2.0, df)
    return diff, se, t, p, diff - q * se, diff + q * se


def welch_test(y_treat: np.ndarray, y_ctrl: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Welch two-sample t-test of mean(y_treat) - mean(y_ctrl).

    Matches `scipy.stats.ttest_ind(..., equal_var=False)` and adds the
    Welch-Satterthwaite confidence interval.
    """
    y_treat = np.asarray(y_treat, dtype=float)
    y_ctrl = np.asarray(y_ctrl, dtype=float)
    out = welch_from_moments(
        y_treat.mean(), y_treat.var(ddof=1), y_treat.size,
        y_ctrl.mean(), y_ctrl.var(ddof=1), y_ctrl.size, alpha,
    )
    return TestResult(*(float(v) for v in out))


def z_test_from_moments(diff, var_diff, alpha: float = 0.05):
    """Normal-approximation test of a difference with known variance (vectorized)."""
    diff = np.asarray(diff, dtype=float)
    se = np.sqrt(np.asarray(var_diff, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = diff / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    q = stats.norm.ppf(1.0 - alpha / 2.0)
    return diff, se, z, p, diff - q * se, diff + q * se


# =============================================================================
# Power and sample size (Section 16.4.1)
# =============================================================================

def sample_size_per_arm(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """n per arm = 2 (z_{1-alpha/2} + z_{1-beta})^2 sigma^2 / delta^2.

    At alpha = 0.05 and 80% power the constant is about 15.7, which is the
    source of the 16 sigma^2 / delta^2 rule of thumb (Kohavi, Tang, and Xu, 2020).
    """
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    return 2.0 * (z_a + z_b) ** 2 * sigma ** 2 / delta ** 2


def power_two_sample(sigma: float, delta: float, n_per_arm: float, alpha: float = 0.05) -> float:
    """Power of a two-sided z-test for a difference in means, both tails included."""
    se = np.sqrt(2.0 * sigma ** 2 / n_per_arm)
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    shift = delta / se
    return float(stats.norm.sf(z_a - shift) + stats.norm.cdf(-z_a - shift))


# =============================================================================
# Capping (Section 16.4.2)
# =============================================================================

def winsorize_upper(y: np.ndarray, cap: float) -> np.ndarray:
    """Cap values above `cap`. The cap must be fixed before looking at results."""
    return np.minimum(np.asarray(y, dtype=float), cap)


# =============================================================================
# CUPED (Section 16.4.4)
# =============================================================================

def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """theta = cov(X, Y) / var(X), the OLS slope of Y on X.

    Estimate theta on the pooled data of both arms. X must be measured before
    the experiment so treatment cannot affect it.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        return 0.0
    return float(np.dot(xc, y - y.mean()) / denom)


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: Optional[float] = None, x_mean: Optional[float] = None) -> np.ndarray:
    """Y' = Y - theta (X - mean(X)). Leaves the mean unchanged, shrinks variance by 1 - rho^2."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta is None:
        theta = cuped_theta(y, x)
    if x_mean is None:
        x_mean = float(x.mean())
    return y - theta * (x - x_mean)


# =============================================================================
# Sample ratio mismatch (Section 16.3.5)
# =============================================================================

@dataclass
class SRMResult:
    observed: tuple
    expected: tuple
    chi2: float
    p_value: float
    flagged: bool


def srm_test(counts: Sequence[int], ratios: Sequence[float], threshold: float = 0.001) -> SRMResult:
    """Chi-square goodness-of-fit test of observed arm sizes against the designed split.

    `scipy.stats.chisquare` compares observed counts with expected counts.
    A very small p-value (a common platform threshold is 0.001) means the
    assignment or logging is broken and the experiment result should not be read.
    """
    counts = np.asarray(counts, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    expected = counts.sum() * ratios / ratios.sum()
    chi2, p = stats.chisquare(counts, expected)
    return SRMResult(tuple(counts.astype(int).tolist()), tuple(expected.tolist()), float(chi2), float(p), bool(p < threshold))


# =============================================================================
# Ratio metrics: delta method and user-level bootstrap (Section 16.4.3)
# =============================================================================

def ratio_of_means(num: np.ndarray, den: np.ndarray) -> float:
    return float(np.sum(num) / np.sum(den))


def delta_method_ratio_var(num: np.ndarray, den: np.ndarray) -> float:
    """Variance of R = mean(num) / mean(den) when the unit of randomization is the user.

    `num` and `den` hold one value per user, for example completed listens and
    listens. First-order Taylor expansion (Deng, Knoblich, and Lu, 2018):

        Var(R) ~ (1/n) [ var(N)/mu_D^2 - 2 mu_N cov(N, D)/mu_D^3 + mu_N^2 var(D)/mu_D^4 ]
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = num.size
    mu_n = num.mean()
    mu_d = den.mean()
    var_n = num.var(ddof=1)
    var_d = den.var(ddof=1)
    cov_nd = np.cov(num, den, ddof=1)[0, 1]
    return float((var_n / mu_d ** 2 - 2.0 * mu_n * cov_nd / mu_d ** 3 + mu_n ** 2 * var_d / mu_d ** 4) / n)


def naive_ratio_var(num: np.ndarray, den: np.ndarray) -> float:
    """Per-event binomial variance p(1-p)/N_events. Wrong when events cluster within users."""
    p = ratio_of_means(num, den)
    return float(p * (1.0 - p) / np.sum(den))


def poisson_bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """User-level Poisson bootstrap replicates of sum(num)/sum(den).

    Each user receives a Poisson(1) weight per replicate, which approximates
    resampling users with replacement and keeps all of a user's events together.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    w = rng.poisson(1.0, size=(n_boot, num.size)).astype(float)
    return (w @ num) / (w @ den)


# =============================================================================
# Sequential testing (Section 16.4.5)
# =============================================================================

def msprt_lambda(diff, var_diff, tau2: float):
    """Mixture SPRT likelihood ratio for a normal difference with N(0, tau^2) mixing.

    For an estimate diff ~ N(theta, V) and theta ~ N(0, tau^2) under the
    alternative, the mixture likelihood ratio against theta = 0 is

        Lambda = sqrt(V / (V + tau^2)) * exp( diff^2 tau^2 / (2 V (V + tau^2)) )

    (Johari et al., 2017). Rejecting the first time Lambda >= 1/alpha keeps the
    false positive rate at most alpha no matter how often the test is checked.
    """
    diff = np.asarray(diff, dtype=float)
    v = np.asarray(var_diff, dtype=float)
    log_lam = 0.5 * np.log(v / (v + tau2)) + diff ** 2 * tau2 / (2.0 * v * (v + tau2))
    return np.exp(log_lam)


def always_valid_p_values(lambdas: np.ndarray, axis: int = -1) -> np.ndarray:
    """p_n = min(1, min_{m <= n} 1 / Lambda_m), the always-valid p-value sequence."""
    lambdas = np.asarray(lambdas, dtype=float)
    with np.errstate(divide="ignore"):
        inv = 1.0 / lambdas
    return np.minimum(1.0, np.minimum.accumulate(inv, axis=axis))


# =============================================================================
# Multiple comparisons (Section 16.4.6)
# =============================================================================

def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Per-test level that bounds the family-wise error rate at alpha."""
    return alpha / n_tests


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Boolean rejections controlling the false discovery rate at level q."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresholds = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresholds
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.nonzero(passed)[0])
        reject[order[: k + 1]] = True
    return reject


# =============================================================================
# Importance weights (Section 16.9.3)
# =============================================================================

def effective_sample_size(weights: np.ndarray) -> float:
    """(sum w)^2 / sum w^2: how many equally weighted samples the weights are worth."""
    w = np.asarray(weights, dtype=float)
    s2 = np.sum(w ** 2)
    if s2 == 0:
        return 0.0
    return float(np.sum(w) ** 2 / s2)
