"""
Chapter 16: Experimentation and Bandits - A/B test simulation
=============================================================
Semi-synthetic A/B tests on real Yambda listening behaviour (Sections 16.3.5,
16.4, and 16.6.1; Figures 16.2, 16.3, and 16.5).

Semi-synthetic means: per-user variance comes from real users, arms are
assigned by hashing, and treatment effects are injected with a known size, so
the true answer is known. Nothing here estimates a real effect of any real
change; the injected effects are illustrative.

Modes
-----
aa       many A/A splits; p-values should be uniform, false positives near alpha
srm      a logging bug drops heavy users from one arm; the SRM check flags it
power    power against users for raw, winsorized, and CUPED analysis
cuped    Figure 16.2: pre/post scatter and raw-versus-CUPED effect estimates
ratio    completion rate per listen; CI coverage of naive, delta-method, bootstrap
peeking  Figure 16.3: daily looks at A/A tests under four stopping rules
novelty  Figure 16.5: novelty and primacy effects by days since first exposure
all      every mode above

Usage
-----
    # Yambda (directory containing listens.parquet)
    $env:YAMBDA_DATA_DIR = "C:\\path\\to\\Dataset\\Yandex\\flat"
    python ab_test_simulation.py --mode all

    # No dataset needed
    python ab_test_simulation.py --mode all --synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

import stats_utils as su
from common import (
    PALETTE, FIG_WIDTH_IN, add_common_args, log, mode_rng, prepare_out_dir, resolve_yambda_dir,
    run_metadata, save_figure, setup_matplotlib, write_csv, write_json,
)

SECONDS_PER_DAY = 86400
WINDOW_DAYS = 28
ALPHA = 0.05


# =============================================================================
# Data: one row per user, one column per experiment day
# =============================================================================

@dataclass
class UserPanel:
    """Per-user daily metrics over the experiment window plus a pre-period total.

    Only users active at least once in the experiment window are kept: a user
    who never arrives is never exposed to either variant.
    """
    pre: np.ndarray               # (n,) metric total over the pre-period
    daily: np.ndarray             # (n, D) metric per experiment day
    daily_listens: np.ndarray     # (n, D) all listens per day (ratio denominator)
    daily_completed: np.ndarray   # (n, D) completed listens per day (ratio numerator)
    source: str
    metric: str

    @property
    def n_users(self) -> int:
        return self.daily.shape[0]

    @property
    def total(self) -> np.ndarray:
        return self.daily.sum(axis=1)

    @property
    def first_day(self) -> np.ndarray:
        """Day of first activity in the window; the day the user enters the test."""
        return np.argmax(self.daily_listens > 0, axis=1)

    def resample(self, n: int, rng: np.random.Generator) -> "UserPanel":
        """Draw n users with replacement to emulate a larger population."""
        idx = rng.integers(0, self.n_users, size=n)
        return UserPanel(self.pre[idx], self.daily[idx], self.daily_listens[idx],
                         self.daily_completed[idx], self.source, self.metric)


def load_yambda_panel(data_dir: Path, completion_threshold: int, metric: str) -> UserPanel:
    """Aggregate Yambda listens into a 28-day pre-period and a 28-day experiment window.

    Reads only uid, timestamp, is_organic, and played_ratio_pct. Timestamps
    are seconds from the start of the log, so day = timestamp // 86400. The
    experiment window is the last 28 days; the pre-period is the 28 before it.
    """
    import polars as pl

    path = Path(data_dir) / "listens.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Set YAMBDA_DATA_DIR or pass --data-dir, or use --synthetic.")
    lf = pl.scan_parquet(path).select(["uid", "timestamp", "is_organic", "played_ratio_pct"])
    last_day = int(lf.select((pl.col("timestamp").max() // SECONDS_PER_DAY)).collect().item())
    exp_start = last_day - WINDOW_DAYS + 1
    pre_start = exp_start - WINDOW_DAYS
    daily = (
        lf.with_columns((pl.col("timestamp") // SECONDS_PER_DAY).cast(pl.Int32).alias("day"))
        .filter(pl.col("day") >= pre_start)
        .group_by(["uid", "day"])
        .agg(
            pl.len().alias("listens"),
            (pl.col("is_organic") == 0).sum().alias("rec_listens"),
            (pl.col("played_ratio_pct") >= completion_threshold).sum().alias("completed"),
        )
        .collect()
    )
    metric_col = "rec_listens" if metric == "rec_listens" else "listens"
    exp = daily.filter(pl.col("day") >= exp_start)
    uids = np.sort(exp["uid"].unique().to_numpy())
    n = uids.size
    rows = np.searchsorted(uids, exp["uid"].to_numpy())
    cols = exp["day"].to_numpy() - exp_start

    def to_matrix(col: str) -> np.ndarray:
        m = np.zeros((n, WINDOW_DAYS), dtype=np.int64)
        m[rows, cols] = exp[col].to_numpy()
        return m

    pre_df = daily.filter(pl.col("day") < exp_start).group_by("uid").agg(pl.col(metric_col).sum())
    pre_uid = pre_df["uid"].to_numpy()
    pos = np.minimum(np.searchsorted(uids, pre_uid), n - 1)
    hit = uids[pos] == pre_uid
    pre = np.zeros(n, dtype=np.int64)
    pre[pos[hit]] = pre_df[metric_col].to_numpy()[hit]
    return UserPanel(pre, to_matrix(metric_col), to_matrix("listens"), to_matrix("completed"),
                     source="yambda", metric=metric_col)


def synthetic_panel(n_users: int, rng: np.random.Generator) -> UserPanel:
    """Heavy-tailed synthetic listening panel with a correlated pre-period.

    Generative story (all parameters illustrative, documented in the README):
      rate_u     ~ Gamma(shape=0.7, mean=8 listens per active day)  heavy tail
      active_u   ~ Beta(2, 3)        probability of being active on a day
      drift_u    ~ LogNormal(0, 0.5) pre-period rate = rate_u * drift_u
      listens    ~ Bernoulli(active_u) * Poisson(rate_u) per day: negative-binomial-like
      complete_u ~ Beta(4, 3);  completed ~ Binomial(listens, complete_u)
    """
    shape, mean_rate = 0.7, 8.0
    rate = rng.gamma(shape, mean_rate / shape, size=n_users)
    active = rng.beta(2.0, 3.0, size=n_users)
    drift = rng.lognormal(0.0, 0.5, size=n_users)
    complete = rng.beta(4.0, 3.0, size=n_users)
    on = rng.random((n_users, WINDOW_DAYS)) < active[:, None]
    listens = rng.poisson(rate[:, None] * on)
    completed = rng.binomial(listens, complete[:, None])
    pre_days_active = rng.binomial(WINDOW_DAYS, active)
    pre = rng.poisson(rate * drift * pre_days_active)
    keep = listens.sum(axis=1) > 0
    return UserPanel(pre[keep], listens[keep], listens[keep], completed[keep],
                     source="synthetic", metric="listens")


# =============================================================================
# Assignment and effect injection
# =============================================================================

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def hash_assign(unit_ids: np.ndarray, salt: str, treat_fraction: float = 0.5) -> np.ndarray:
    """Deterministic arm assignment: hash(unit id, experiment salt) -> uniform -> arm.

    The same unit always lands in the same arm of a given experiment, and a
    different salt gives an independent split, which is how concurrent
    experiments in different layers stay uncorrelated (Section 16.3.2).
    Returns True for treatment.
    """
    salt_int = np.uint64(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little"))
    with np.errstate(over="ignore"):
        h = _splitmix64(np.asarray(unit_ids).astype(np.uint64) ^ salt_int)
    u = (h >> np.uint64(11)).astype(np.float64) / float(2 ** 53)
    return u < treat_fraction


def inject_count_effect(y: np.ndarray, rel_effect, rng: np.random.Generator) -> np.ndarray:
    """Change counts by a known relative effect while keeping them integer counts.

    Positive effects add Poisson(rel_effect * y) extra events; negative effects
    keep each event with probability 1 + rel_effect. Either way E[y'] = (1 + rel_effect) y.
    """
    y = np.asarray(y)
    eff = np.broadcast_to(np.asarray(rel_effect, dtype=float), y.shape)
    up = rng.poisson(np.where(eff > 0, eff, 0.0) * y)
    down = rng.binomial(y, np.clip(1.0 + np.where(eff < 0, eff, 0.0), 0.0, 1.0))
    return np.where(eff >= 0, y + up, down)


def inject_ratio_effect(completed: np.ndarray, listens: np.ndarray, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Turn each non-completed listen into a completed one with probability flip_prob.

    The completion rate c becomes c + flip_prob (1 - c) in expectation and
    always stays within [0, 1].
    """
    return completed + rng.binomial(listens - completed, flip_prob)


def _arm_test(y: np.ndarray, z: np.ndarray) -> su.TestResult:
    return su.welch_test(y[z], y[~z])


# =============================================================================
# Modes
# =============================================================================

def mode_aa(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    n_sims = 200 if args.quick else 1000
    ids = np.arange(panel.n_users)
    y, x = panel.total.astype(float), panel.pre.astype(float)
    p_raw, p_cuped = np.empty(n_sims), np.empty(n_sims)
    est_cuped = np.empty(n_sims)
    for s in range(n_sims):
        z = hash_assign(ids, f"aa-{args.seed}-{s}")
        p_raw[s] = _arm_test(y, z).p_value
        r = _arm_test(su.cuped_adjust(y, x), z)
        p_cuped[s], est_cuped[s] = r.p_value, r.diff
    summary = {
        "n_users": panel.n_users, "n_sims": n_sims,
        "false_positive_rate_raw": float(np.mean(p_raw < ALPHA)),
        "false_positive_rate_cuped": float(np.mean(p_cuped < ALPHA)),
        "ks_uniform_p_raw": float(stats.kstest(p_raw, "uniform").pvalue),
        "mean_cuped_estimate_aa": float(est_cuped.mean()),
        "share_treatment_mean": float(np.mean([hash_assign(ids, f"aa-{args.seed}-{s}").mean() for s in range(20)])),
    }
    write_csv(out / "aa_pvalues.csv", ({"sim": i, "p_raw": a, "p_cuped": b} for i, (a, b) in enumerate(zip(p_raw, p_cuped))))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.6))
    ax.hist(p_raw, bins=20, range=(0, 1), color=PALETTE["primary"], edgecolor="white")
    ax.axhline(n_sims / 20, color=PALETTE["baseline"], ls="--", lw=1, label="uniform expectation")
    ax.set_xlabel("p-value"); ax.set_ylabel("A/A experiments"); ax.legend(frameon=False)
    save_figure(fig, out / "aa_pvalue_histogram.png"); plt.close(fig)
    return summary


def mode_srm(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    # Keep the population at full size even in quick mode: with too few users
    # the chi-square check cannot see a bug of this size.
    n_users = args.srm_users
    n_sims = 10 if args.quick else 200
    heavy_q, drop = 0.9, args.srm_drop
    flagged = {"clean": 0, "buggy": 0}
    naive_sig = {"clean": 0, "buggy": 0}
    example = None
    for s in range(n_sims):
        pop = panel.resample(n_users, rng)
        y = pop.total.astype(float)
        z = hash_assign(np.arange(n_users), f"srm-{args.seed}-{s}")
        heavy = y >= np.quantile(y, heavy_q)
        # The bug: a treatment code path loses the logs of some heavy users,
        # for example because it times out more often for long histories.
        lost = z & heavy & (rng.random(n_users) < drop)
        for label, keep in (("clean", np.ones(n_users, bool)), ("buggy", ~lost)):
            zk, yk = z[keep], y[keep]
            srm = su.srm_test([int(zk.sum()), int((~zk).sum())], [0.5, 0.5])
            res = su.welch_test(yk[zk], yk[~zk])
            flagged[label] += srm.flagged
            naive_sig[label] += res.p_value < ALPHA
            if s == 0 and label == "buggy":
                example = {"arm_sizes": srm.observed, "expected": srm.expected, "srm_chi2": srm.chi2,
                           "srm_p": srm.p_value, "srm_flagged": srm.flagged,
                           "naive_relative_diff": res.diff / yk[~zk].mean(), "naive_p": res.p_value}
    return {
        "n_users": n_users, "n_sims": n_sims, "heavy_quantile": heavy_q, "drop_prob": drop,
        "srm_flag_rate_clean": flagged["clean"] / n_sims, "srm_flag_rate_buggy": flagged["buggy"] / n_sims,
        "naive_significant_rate_clean": naive_sig["clean"] / n_sims,
        "naive_significant_rate_buggy": naive_sig["buggy"] / n_sims,
        "example_buggy_run": example,
    }


def mode_power(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    sizes = [2_000, 10_000] if args.quick else [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    n_sims = 40 if args.quick else 300
    rows = []
    for n in sizes:
        hits = {"raw": 0, "winsorized": 0, "cuped": 0}
        for s in range(n_sims):
            pop = panel.resample(n, rng)
            z = hash_assign(np.arange(n), f"power-{args.seed}-{n}-{s}")
            y = pop.total.copy()
            y[z] = inject_count_effect(y[z], args.effect, rng)
            y = y.astype(float)
            cap = np.quantile(y, 0.99)  # pre-specified rule: pooled 99th percentile
            variants = {"raw": y, "winsorized": su.winsorize_upper(y, cap),
                        "cuped": su.cuped_adjust(y, pop.pre.astype(float))}
            for k, v in variants.items():
                r = _arm_test(v, z)
                hits[k] += (r.p_value < ALPHA) and (r.diff > 0)
        rows.append({"n_users": n, **{f"power_{k}": v / n_sims for k, v in hits.items()}})
        log(f"  power n={n}: " + ", ".join(f"{k}={v / n_sims:.2f}" for k, v in hits.items()))
    write_csv(out / "power_curves.csv", rows)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
    for k, c in (("raw", PALETTE["baseline"]), ("winsorized", PALETTE["secondary"]), ("cuped", PALETTE["primary"])):
        ax.plot([r["n_users"] for r in rows], [r[f"power_{k}"] for r in rows], "o-", color=c, label=k)
    ax.axhline(0.8, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("users in experiment (both arms)"); ax.set_ylabel("power"); ax.legend(frameon=False)
    save_figure(fig, out / "power_curves.png"); plt.close(fig)
    return {"effect": args.effect, "n_sims": n_sims, "rows": rows}


def mode_cuped(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    n_sims = 200 if args.quick else 1000
    y0, x = panel.total, panel.pre.astype(float)
    ids = np.arange(panel.n_users)
    true_effect = args.effect * y0.mean()
    raw, adj = np.empty(n_sims), np.empty(n_sims)
    for s in range(n_sims):
        z = hash_assign(ids, f"cuped-{args.seed}-{s}")
        y = y0.copy()
        y[z] = inject_count_effect(y[z], args.effect, rng)
        y = y.astype(float)
        raw[s] = _arm_test(y, z).diff
        adj[s] = _arm_test(su.cuped_adjust(y, x), z).diff
    rho = float(np.corrcoef(x, y0)[0, 1])
    summary = {
        "n_users": panel.n_users, "n_sims": n_sims, "injected_relative_effect": args.effect,
        "true_effect_abs": true_effect, "theta": su.cuped_theta(y0.astype(float), x), "rho_pre_post": rho,
        "predicted_variance_ratio_1_minus_rho2": 1 - rho ** 2,
        "observed_variance_ratio": float(adj.var() / raw.var()),
        "mean_raw": float(raw.mean()), "mean_cuped": float(adj.mean()),
        "sd_raw": float(raw.std()), "sd_cuped": float(adj.std()),
    }
    write_csv(out / "cuped_estimates.csv", ({"sim": i, "raw": a, "cuped": b} for i, (a, b) in enumerate(zip(raw, adj))))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.9))
    sub = rng.choice(panel.n_users, size=min(3000, panel.n_users), replace=False)
    lx, ly = np.log1p(x[sub]), np.log1p(y0[sub].astype(float))
    ax = axes[0]
    ax.scatter(lx, ly, s=4, alpha=0.3, color=PALETTE["primary"], linewidths=0)
    has_pre = lx > 0
    slope, intercept = np.polyfit(lx[has_pre], ly[has_pre], 1)
    grid = np.linspace(lx[has_pre].min(), lx.max(), 50)
    ax.plot(grid, intercept + slope * grid, color=PALETTE["highlight"], lw=1.5,
            label="least-squares fit,\nusers with pre-period data")
    ax.set_xlabel("log(1 + pre-period listens)"); ax.set_ylabel("log(1 + experiment-period listens)")
    ax.set_title("(a)", loc="left"); ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    lo, hi = np.percentile(np.r_[raw, adj], [0.5, 99.5])
    grid = np.linspace(lo, hi, 300)
    dens_raw, dens_adj = stats.gaussian_kde(raw)(grid), stats.gaussian_kde(adj)(grid)
    ax.plot(grid, dens_raw, color=PALETTE["baseline"], lw=1.6, label="difference in means")
    ax.plot(grid, dens_adj, color=PALETTE["primary"], lw=1.6, label="CUPED-adjusted")
    ax.axvline(true_effect, color="k", ls="--", lw=1, label="injected effect")
    ax.set_xlabel("estimated effect (listens per user)"); ax.set_ylabel("density"); ax.set_yticks([])
    # Headroom above the peak, so the key never covers a curve.
    ax.set_ylim(0, 1.5 * max(dens_raw.max(), dens_adj.max()))
    ax.set_title("(b)", loc="left"); ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save_figure(fig, out / "fig16_2_cuped.png"); plt.close(fig)
    return summary


def mode_ratio(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    n_sims = 200 if args.quick else 1000
    n_boot_sims = 30 if args.quick else 200
    n_boot = 200
    num = panel.daily_completed.sum(axis=1)
    den = panel.daily_listens.sum(axis=1)
    ids = np.arange(panel.n_users)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    cover = {"naive": 0, "delta": 0, "bootstrap": 0}
    reject_effect = {"naive": 0, "delta": 0}
    se_ratio = []
    for s in range(n_sims):
        z = hash_assign(ids, f"ratio-{args.seed}-{s}")
        diff = su.ratio_of_means(num[z], den[z]) - su.ratio_of_means(num[~z], den[~z])
        se_naive = np.sqrt(su.naive_ratio_var(num[z], den[z]) + su.naive_ratio_var(num[~z], den[~z]))
        se_delta = np.sqrt(su.delta_method_ratio_var(num[z], den[z]) + su.delta_method_ratio_var(num[~z], den[~z]))
        cover["naive"] += abs(diff) <= z_crit * se_naive
        cover["delta"] += abs(diff) <= z_crit * se_delta
        se_ratio.append(se_delta / se_naive)
        if s < n_boot_sims:
            bt = su.poisson_bootstrap_ratio(num[z], den[z], n_boot, rng)
            bc = su.poisson_bootstrap_ratio(num[~z], den[~z], n_boot, rng)
            cover["bootstrap"] += abs(diff) <= z_crit * np.std(bt - bc, ddof=1)
        # Same split with a known injected improvement in completion.
        num_t = inject_ratio_effect(num[z], den[z], args.ratio_flip, rng)
        d1 = su.ratio_of_means(num_t, den[z]) - su.ratio_of_means(num[~z], den[~z])
        reject_effect["naive"] += abs(d1) > z_crit * np.sqrt(su.naive_ratio_var(num_t, den[z]) + su.naive_ratio_var(num[~z], den[~z]))
        reject_effect["delta"] += abs(d1) > z_crit * np.sqrt(su.delta_method_ratio_var(num_t, den[z]) + su.delta_method_ratio_var(num[~z], den[~z]))
    summary = {
        "n_users": panel.n_users, "n_listens": int(den.sum()), "completion_rate": float(num.sum() / den.sum()),
        "n_sims": n_sims, "n_bootstrap_sims": n_boot_sims, "n_bootstrap_reps": n_boot,
        "aa_coverage_naive": cover["naive"] / n_sims, "aa_coverage_delta": cover["delta"] / n_sims,
        "aa_coverage_bootstrap": cover["bootstrap"] / n_boot_sims,
        "median_se_ratio_delta_over_naive": float(np.median(se_ratio)),
        "injected_flip_prob": args.ratio_flip,
        "power_naive": reject_effect["naive"] / n_sims, "power_delta": reject_effect["delta"] / n_sims,
    }
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.4))
    labels = ["per-listen (naive)", "delta method", "user bootstrap"]
    vals = [summary["aa_coverage_naive"], summary["aa_coverage_delta"], summary["aa_coverage_bootstrap"]]
    ax.barh(labels, vals, color=[PALETTE["highlight"], PALETTE["primary"], PALETTE["tertiary"]])
    ax.axvline(1 - ALPHA, color="k", ls="--", lw=1, label="nominal 95%")
    ax.set_xlim(0, 1); ax.set_xlabel("A/A confidence-interval coverage"); ax.legend(frameon=False, loc="lower center")
    save_figure(fig, out / "ratio_ci_coverage.png"); plt.close(fig)
    return summary


def mode_peeking(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    n_sims = 300 if args.quick else 2000
    D = WINDOW_DAYS
    C = np.cumsum(panel.daily, axis=1).astype(float)          # cumulative metric since entry
    M = (panel.first_day[:, None] <= np.arange(D)[None, :]).astype(float)  # entered by day d
    C2 = C ** 2
    ids = np.arange(panel.n_users)
    # Mixing scale for mSPRT: a relative effect of args.tau_rel on the final-day mean.
    tau2 = (args.tau_rel * C[:, -1].mean()) ** 2
    p_all = np.empty((n_sims, D))
    lam_all = np.empty((n_sims, D))
    chunk = 100
    for start in range(0, n_sims, chunk):
        S = range(start, min(start + chunk, n_sims))
        Z = np.stack([hash_assign(ids, f"peek-{args.seed}-{s}") for s in S]).astype(float)
        stats_arm = []
        for W in (Z, 1.0 - Z):
            n = W @ M
            s1 = W @ C
            s2 = W @ C2
            mean = s1 / n
            var = (s2 - n * mean ** 2) / (n - 1)
            stats_arm.append((mean, var, n))
        (m1, v1, n1), (m0, v0, n0) = stats_arm
        diff, se, _, p, _, _ = su.welch_from_moments(m1, v1, n1, m0, v0, n0)
        p_all[start:start + len(S)] = p
        lam_all[start:start + len(S)] = su.msprt_lambda(diff, se ** 2, tau2)
    sig_naive = np.maximum.accumulate(p_all < ALPHA, axis=1)
    sig_bonf = np.maximum.accumulate(p_all < su.bonferroni_alpha(ALPHA, D), axis=1)
    sig_msprt = np.maximum.accumulate(lam_all >= 1.0 / ALPHA, axis=1)
    fixed = float(np.mean(p_all[:, -1] < ALPHA))
    rows = [{"day": d + 1, "naive_daily_looks": sig_naive[:, d].mean(), "bonferroni": sig_bonf[:, d].mean(),
             "msprt": sig_msprt[:, d].mean()} for d in range(D)]
    write_csv(out / "peeking_curves.csv", rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.0))
    days = np.arange(1, D + 1)
    ax.plot(days, sig_naive.mean(0), color=PALETTE["highlight"], lw=1.8, label="Stop at first p < 0.05, daily looks")
    # The two corrected curves nearly coincide; dashing Bonferroni and drawing it last keeps both visible.
    ax.plot(days, sig_msprt.mean(0), color=PALETTE["primary"], lw=1.8, label="Always-valid mSPRT")
    ax.plot(days, sig_bonf.mean(0), color=PALETTE["baseline"], lw=1.8, ls="--", label="Bonferroni across looks")
    ax.plot([D], [fixed], "D", color="k", ms=5, label="Fixed horizon, one analysis")
    ax.axhline(ALPHA, color="k", ls="--", lw=0.8)
    ax.text(1, ALPHA, " alpha = 0.05", va="bottom", fontsize=7)
    ax.set_xlim(0.5, D + 0.5); ax.set_ylim(0, max(sig_naive.mean(0).max(), ALPHA) * 1.15)
    ax.set_xlabel("analysis day"); ax.set_ylabel("share of A/A tests declared significant")
    # Below the axes, so the key never covers the rising curve.
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save_figure(fig, out / "fig16_3_peeking.png"); plt.close(fig)
    return {"n_users": panel.n_users, "n_sims": n_sims, "looks": D, "tau_rel": args.tau_rel,
            "fpr_naive_by_day28": float(sig_naive[:, -1].mean()), "fpr_bonferroni": float(sig_bonf[:, -1].mean()),
            "fpr_msprt": float(sig_msprt[:, -1].mean()), "fpr_fixed_horizon": fixed}


def novelty_curve(d: np.ndarray, delta0: float, delta_inf: float, tau: float) -> np.ndarray:
    """delta(d) = delta_inf + (delta0 - delta_inf) exp(-d / tau)."""
    return delta_inf + (delta0 - delta_inf) * np.exp(-np.asarray(d, dtype=float) / tau)


def mode_novelty(panel: UserPanel, args, rng, out: Path, plt) -> Dict:
    n_users = 20_000 if args.quick else args.novelty_users
    D, window = WINDOW_DAYS, 14
    scenarios = {
        "novelty": dict(delta0=0.15, delta_inf=0.03, tau=4.0),
        "primacy": dict(delta0=-0.08, delta_inf=0.04, tau=5.0),
    }
    pop = panel.resample(n_users, rng)
    entry = pop.first_day
    since = np.arange(D)[None, :] - entry[:, None]            # days since first exposure
    z = hash_assign(np.arange(n_users), f"novelty-{args.seed}")
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    results, curves = {}, []
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.9), sharey=True)
    for ax, (name, sc) in zip(axes, scenarios.items()):
        y = pop.daily.copy()
        eff = np.where(since >= 0, novelty_curve(np.maximum(since, 0), **sc), 0.0)
        y[z] = inject_count_effect(y[z], eff[z], rng)
        # Pre-specified, arm-blind winsorizing of user-days: without it a single extreme
        # user-day, repeated by the resampling, can dominate that day's estimate.
        cap = float(np.quantile(y, 0.999))
        y = np.minimum(y, cap)
        est, lo, hi = [], [], []
        for d in range(D):
            ok = entry + d < D
            vals = y[ok, np.minimum(entry[ok] + d, D - 1)].astype(float)
            zt = z[ok]
            mt, mc = vals[zt].mean(), vals[~zt].mean()
            r = mt / mc
            se = r * np.sqrt(vals[zt].var(ddof=1) / (zt.sum() * mt ** 2) + vals[~zt].var(ddof=1) / ((~zt).sum() * mc ** 2))
            est.append(r - 1); lo.append(r - 1 - z_crit * se); hi.append(r - 1 + z_crit * se)
        entered = entry < window
        tot = y[entered, :window].sum(axis=1).astype(float)
        zt = z[entered]
        pooled = tot[zt].mean() / tot[~zt].mean() - 1
        truth = novelty_curve(np.arange(D), **sc)
        results[name] = {**sc, "user_day_cap": cap, "pooled_two_week_estimate": pooled, "daily_estimates": est}
        curves += [{"scenario": name, "day_since_exposure": d, "estimate": e, "ci_low": l, "ci_high": h, "true": t}
                   for d, (e, l, h, t) in enumerate(zip(est, lo, hi, truth))]
        days = np.arange(D)
        ax.axvspan(-0.5, window - 0.5, color="#D9D9D9", alpha=0.5, lw=0, label="two-week test window")
        ax.fill_between(days, lo, hi, color=PALETTE["primary"], alpha=0.25, lw=0)
        ax.plot(days, est, color=PALETTE["primary"], lw=1.5, label="estimated effect, 95% interval")
        ax.plot(days, truth, "k--", lw=1, label="injected true effect")
        ax.axhline(sc["delta_inf"], color="k", ls=":", lw=0.8, label="long-run effect")
        ax.axhline(0, color=PALETTE["baseline"], lw=0.5)
        ax.plot([window - 1], [pooled], "o", color=PALETTE["highlight"], ms=6, label="pooled two-week estimate")
        ax.set_title(f"({'a' if name == 'novelty' else 'b'}) {name}", loc="left")
        ax.set_xlabel("days since first exposure")
    axes[0].set_ylabel("relative treatment effect")
    from matplotlib.ticker import PercentFormatter
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, out / "fig16_5_novelty_primacy.png"); plt.close(fig)
    write_csv(out / "novelty_curves.csv", curves)
    return {"n_users": n_users, "scenarios": results}


MODES = {"aa": mode_aa, "srm": mode_srm, "power": mode_power, "cuped": mode_cuped,
         "ratio": mode_ratio, "peeking": mode_peeking, "novelty": mode_novelty}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    p.add_argument("--synthetic", action="store_true", help="Use the synthetic panel instead of Yambda")
    p.add_argument("--data-dir", type=str, default=None, help="Yambda flat directory (default: $YAMBDA_DATA_DIR)")
    p.add_argument("--synthetic-users", type=int, default=10_000)
    p.add_argument("--metric", choices=["listens", "rec_listens"], default="listens",
                   help="Primary count metric: all listens or recommendation-driven listens")
    p.add_argument("--completion-threshold", type=int, default=80,
                   help="played_ratio_pct at or above which a listen counts as completed (illustrative)")
    p.add_argument("--effect", type=float, default=0.02, help="Injected relative effect on the count metric")
    p.add_argument("--ratio-flip", type=float, default=0.01, help="Injected completion flip probability (ratio mode)")
    p.add_argument("--srm-users", type=int, default=200_000)
    p.add_argument("--srm-drop", type=float, default=0.2, help="Share of heavy treatment users whose logs are lost")
    p.add_argument("--novelty-users", type=int, default=100_000)
    p.add_argument("--tau-rel", type=float, default=0.05, help="mSPRT mixing scale as a relative effect")
    add_common_args(p, "ab_test")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    out = prepare_out_dir(args.out_dir)
    plt = setup_matplotlib()
    if args.synthetic:
        panel = synthetic_panel(args.synthetic_users if not args.quick else 4_000, mode_rng(args.seed, "synthetic-panel"))
    else:
        data_dir = resolve_yambda_dir(args.data_dir)
        if data_dir is None:
            log("No Yambda directory. Set YAMBDA_DATA_DIR, pass --data-dir, or run with --synthetic.")
            return 2
        panel = load_yambda_panel(data_dir, args.completion_threshold, args.metric)
    total = panel.total
    data_summary = {
        "source": panel.source, "metric": panel.metric, "n_users": panel.n_users,
        "mean_per_user": float(total.mean()), "median_per_user": float(np.median(total)),
        "p99_per_user": float(np.quantile(total, 0.99)),
        "corr_pre_post": float(np.corrcoef(panel.pre, total)[0, 1]),
        "share_users_without_pre_period": float(np.mean(panel.pre == 0)),
    }
    log(f"Panel: {data_summary}")
    modes = list(MODES) if args.mode == "all" else [args.mode]
    results = {"data": data_summary}
    for m in modes:
        t0 = time.time()
        log(f"[{m}] running")
        results[m] = MODES[m](panel, args, mode_rng(args.seed, m), out, plt)
        results[m]["runtime_seconds"] = round(time.time() - t0, 2)
        log(f"[{m}] done in {results[m]['runtime_seconds']}s")
    results["meta"] = run_metadata(args, started)
    name = "summary.json" if args.mode == "all" else f"summary_{args.mode}.json"
    write_json(out / name, results)
    log(f"Wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
