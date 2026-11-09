"""
Chapter 16: Experimentation and Bandits - bandits and off-policy evaluation
===========================================================================
A fully synthetic "shelf selection" problem (Sections 16.8 and 16.9;
Figures 16.8 and 16.9, and the qualitative content of Table 16.3).

K = 10 arms stand for carousels, artwork variants, or value-function
configurations. Contexts are x = [1, z] with z in R^8. Reward is
Bernoulli(sigmoid(theta_a . z + b_a)), tuned to low, click-like rates. The
true value of any policy is computed by Monte Carlo from the known reward
probabilities, so every estimate can be scored against the truth.

Modes
-----
mab         Figure 16.8a: context-free Bernoulli bandit; uniform, epsilon-greedy, UCB1, Thompson
contextual  Figure 16.8b: the same policies plus LinUCB, regret against the context-aware oracle
ope         Figure 16.9a: IPS, clipped IPS, SNIPS, DM, DR on epsilon-greedy logs
support     Figure 16.9b: estimator RMSE as the logging epsilon and the log size vary
all         every mode above

Usage
-----
    python bandit_ope_simulation.py --mode all
    python bandit_ope_simulation.py --mode mab contextual    # Figure 16.8 only
    python bandit_ope_simulation.py --mode all --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import ope
from bandits import (
    UCB1, BernoulliBandit, EpsilonGreedy, LinUCB, LogisticContextualBandit, ThompsonBernoulli,
    UniformRandom, run_policy,
)
from common import (
    PALETTE, FIG_WIDTH_IN, add_common_args, log, prepare_out_dir, run_metadata,
    save_figure, setup_matplotlib, write_csv, write_json,
)

N_ARMS = 10
DIM = 8
POLICY_COLOURS = {
    "Uniform random": PALETTE["baseline"],
    "Epsilon-greedy": PALETTE["secondary"],
    "UCB1": PALETTE["tertiary"],
    "Thompson sampling": PALETTE["primary"],
    "LinUCB": PALETTE["highlight"],
}
ESTIMATOR_COLOURS = {"IPS": PALETTE["highlight"], "SNIPS": PALETTE["secondary"], "DM": PALETTE["baseline"],
                     "DR": PALETTE["primary"], "Replay": PALETTE["tertiary"]}


def make_env(seed: int) -> LogisticContextualBandit:
    # Weight scale large enough that the best arm depends on the context.
    return LogisticContextualBandit(N_ARMS, DIM, np.random.default_rng(seed), base_rate=0.03, weight_scale=1.0)


def _policies(rng: np.random.Generator, contextual: bool, dim: int, linucb_alpha: float):
    pols = [UniformRandom(N_ARMS, rng), EpsilonGreedy(N_ARMS, 0.1, rng), UCB1(N_ARMS), ThompsonBernoulli(N_ARMS, rng)]
    if contextual:
        pols.append(LinUCB(N_ARMS, dim, alpha=linucb_alpha))
    return pols


def _regret_runs(env, T: int, n_seeds: int, seed: int, contextual: bool, linucb_alpha: float):
    curves: Dict[str, List[np.ndarray]] = {}
    for s in range(n_seeds):
        stream = env.stream(T, np.random.default_rng([seed, s, 1]))
        for pol in _policies(np.random.default_rng([seed, s, 2]), contextual, env.dim, linucb_alpha):
            curves.setdefault(pol.name, []).append(np.cumsum(run_policy(pol, stream)))
    return {k: np.vstack(v) for k, v in curves.items()}


def _regret_summary(curves, T: int, step: int):
    rows = []
    idx = np.unique(np.r_[np.arange(step - 1, T, step), T - 1])
    for name, c in curves.items():
        q25, q50, q75 = np.percentile(c[:, idx], [25, 50, 75], axis=0)
        rows += [{"policy": name, "round": int(t + 1), "median": a, "q25": b, "q75": d}
                 for t, a, b, d in zip(idx, q50, q25, q75)]
    final = {name: {"median_final_regret": float(np.median(c[:, -1])),
                    "regret_second_half_over_first_half": float(np.median((c[:, -1] - c[:, T // 2 - 1]) / np.maximum(c[:, T // 2 - 1], 1e-12)))}
             for name, c in curves.items()}
    return rows, final


def _plot_regret(ax, rows, title):
    for name, colour in POLICY_COLOURS.items():
        r = [x for x in rows if x["policy"] == name]
        if not r:
            continue
        t = [x["round"] for x in r]
        ax.plot(t, [x["median"] for x in r], color=colour, lw=1.6, label=name)
        ax.fill_between(t, [x["q25"] for x in r], [x["q75"] for x in r], color=colour, alpha=0.2, lw=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative regret")


def mode_mab(args, rng, out: Path, plt, store: Dict) -> Dict:
    T = 3_000 if args.quick else 20_000
    n_seeds = 4 if args.quick else 20
    env = BernoulliBandit.random(N_ARMS, np.random.default_rng(args.seed + 1))
    curves = _regret_runs(env, T, n_seeds, args.seed, contextual=False, linucb_alpha=args.linucb_alpha)
    rows, final = _regret_summary(curves, T, step=max(T // 200, 1))
    write_csv(out / "regret_mab.csv", rows)
    store["mab"] = rows
    return {"T": T, "n_seeds": n_seeds, "arm_means": env.means, "policies": final}


def mode_contextual(args, rng, out: Path, plt, store: Dict) -> Dict:
    T = 3_000 if args.quick else 20_000
    n_seeds = 4 if args.quick else 20
    env = make_env(args.seed + 2)
    curves = _regret_runs(env, T, n_seeds, args.seed, contextual=True, linucb_alpha=args.linucb_alpha)
    rows, final = _regret_summary(curves, T, step=max(T // 200, 1))
    write_csv(out / "regret_contextual.csv", rows)
    store["contextual"] = rows
    X = env.sample_contexts(100_000, np.random.default_rng(args.seed + 3))
    p = env.expected_rewards(X)
    return {"T": T, "n_seeds": n_seeds, "linucb_alpha": args.linucb_alpha,
            "oracle_value": float(p.max(1).mean()), "best_single_arm_value": float(p.mean(0).max()),
            "share_contexts_where_best_single_arm_is_best": float(np.mean(p.argmax(1) == p.mean(0).argmax())),
            "policies": final}


# =============================================================================
# Off-policy evaluation
# =============================================================================

class OPESetup:
    """Environment, a production (logging) policy, and a target policy with known value."""

    def __init__(self, seed: int, quick: bool, linucb_alpha: float):
        self.env = make_env(seed + 2)
        rng = np.random.default_rng(seed + 10)
        # Production policy: greedy on a ridge model fitted to uniform warm-up logs.
        Xw = self.env.sample_contexts(2_000, rng)
        aw = rng.integers(0, N_ARMS, Xw.shape[0])
        rw = (rng.random(Xw.shape[0]) < self.env.expected_rewards(Xw)[np.arange(Xw.shape[0]), aw]).astype(float)
        self.theta_prod = ope.fit_ridge_per_arm(Xw, aw, rw, N_ARMS)
        # Target policy: greedy policy of a LinUCB model trained online on separate traffic.
        lin = LinUCB(N_ARMS, self.env.dim, alpha=linucb_alpha)
        run_policy(lin, self.env.stream(5_000, rng))
        self.theta_target = lin.theta
        # True values by Monte Carlo on fresh contexts with the known reward probabilities.
        n_mc = 100_000 if quick else 1_000_000
        Xm = self.env.sample_contexts(n_mc, rng)
        pm = self.env.expected_rewards(Xm)
        rows = np.arange(n_mc)
        self.true_value = float(pm[rows, self.target_actions(Xm)].mean())
        self.production_value = float(pm[rows, self.production_actions(Xm)].mean())
        self.agreement = float(np.mean(self.target_actions(Xm) == self.production_actions(Xm)))
        self.n_mc = n_mc

    def production_actions(self, X):
        return np.argmax(X @ self.theta_prod.T, axis=1)

    def target_actions(self, X):
        return np.argmax(X @ self.theta_target.T, axis=1)

    def log(self, n: int, epsilon: float, rng: np.random.Generator):
        """Epsilon-greedy logs around production, with propensities recorded at decision time."""
        X = self.env.sample_contexts(n, rng)
        probs = ope.epsilon_greedy_probs(self.production_actions(X), N_ARMS, epsilon)
        a = ope.sample_actions(probs, rng)
        rows = np.arange(n)
        r = (rng.random(n) < self.env.expected_rewards(X)[rows, a]).astype(float)
        return X, a, r, probs[rows, a]

    def estimate(self, n: int, epsilon: float, rng: np.random.Generator, with_replay: bool = False):
        X, a, r, ps = self.log(n, epsilon, rng)
        target = ope.one_hot(self.target_actions(X), N_ARMS)
        q_hat = ope.predict_rewards(X, ope.fit_ridge_per_arm(X, a, r, N_ARMS))
        est = ope.evaluate_all(r, a, ps, target, q_hat)
        if with_replay:
            est["Replay"] = ope.replay(r, a, ps, target)
        return est


def mode_ope(args, rng, out: Path, plt, store: Dict) -> Dict:
    reps = 30 if args.quick else 200
    n_logs, eps = 10_000, 0.1
    setup = OPESetup(args.seed, args.quick, args.linucb_alpha)
    runs = [setup.estimate(n_logs, eps, np.random.default_rng([args.seed, 99, i])) for i in range(reps)]
    names = [k for k in runs[0] if k != "ESS"]
    summary = {}
    for k in names:
        v = np.array([r[k] for r in runs])
        summary[k] = {"bias": float(v.mean() - setup.true_value), "sd": float(v.std(ddof=1)),
                      "rmse": float(np.sqrt(np.mean((v - setup.true_value) ** 2)))}
    write_csv(out / "ope_estimates.csv", ({"rep": i, **r} for i, r in enumerate(runs)))
    store["ope"] = {"runs": runs, "true": setup.true_value, "names": names}
    return {"reps": reps, "n_logs": n_logs, "logging_epsilon": eps, "true_target_value": setup.true_value,
            "logging_production_value": setup.production_value, "target_production_agreement": setup.agreement,
            "monte_carlo_contexts": setup.n_mc, "mean_ess": float(np.mean([r["ESS"] for r in runs])),
            "estimators": summary}


def mode_support(args, rng, out: Path, plt, store: Dict) -> Dict:
    reps = 20 if args.quick else 100
    epsilons = [0.01, 0.03, 0.1, 0.3, 1.0]
    sizes = [1_000, 10_000] if args.quick else [1_000, 10_000, 100_000]
    setup = OPESetup(args.seed, args.quick, args.linucb_alpha)
    rows = []
    for n in sizes:
        for eps in epsilons:
            runs = [setup.estimate(n, eps, np.random.default_rng([args.seed, 7, n, int(eps * 1000), i]), with_replay=eps == 1.0)
                    for i in range(reps)]
            row = {"n_logs": n, "epsilon": eps, "mean_ess": float(np.mean([r["ESS"] for r in runs]))}
            for k in ("IPS", "SNIPS", "DM", "DR", "Replay"):
                if k in runs[0]:
                    v = np.array([r[k] for r in runs], dtype=float)
                    row[f"rmse_{k}"] = float(np.sqrt(np.nanmean((v - setup.true_value) ** 2)))
                    row[f"bias_{k}"] = float(np.nanmean(v) - setup.true_value)
            rows.append(row)
        log(f"  support n_logs={n} done")
    write_csv(out / "support_sweep.csv", rows)
    store["support"] = rows
    return {"reps": reps, "true_target_value": setup.true_value, "rows": rows}


# =============================================================================
# Figures
# =============================================================================

def _plot_ope_box(ax, data):
    names = [n for n in data["names"] if n != "Clipped IPS (M=100)"]
    vals = [[r[n] for r in data["runs"]] for n in names]
    labels = [n.replace("Clipped IPS (M=10)", "Clipped IPS") for n in names]
    bp = ax.boxplot(vals, tick_labels=labels, patch_artist=True, widths=0.6, flierprops={"markersize": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["primary"]); patch.set_alpha(0.5)
    ax.axhline(data["true"], color="k", ls="--", lw=1, label="true value")
    ax.set_ylabel("estimated policy value"); ax.set_title("(a)", loc="left")
    ax.tick_params(axis="x", labelrotation=30)
    # Above the plotting area, so the key never covers a box.
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)


def _plot_support(ax, rows, n_logs: int):
    r = [x for x in rows if x["n_logs"] == n_logs]
    eps = [x["epsilon"] for x in r]
    for k in ("IPS", "SNIPS", "DM", "DR"):
        ax.plot(eps, [x[f"rmse_{k}"] for x in r], "o-", color=ESTIMATOR_COLOURS[k], lw=1.5, ms=3, label=k)
    rep = [x for x in r if "rmse_Replay" in x]
    if rep:
        ax.plot([x["epsilon"] for x in rep], [x["rmse_Replay"] for x in rep], "s", color=ESTIMATOR_COLOURS["Replay"], label="Replay (uniform logs)")
    ax.axvspan(min(eps) * 0.8, 0.035, color="#D9D9D9", alpha=0.6, lw=0)
    ax.text(min(eps) * 0.85, 0.03, "target actions\nrarely logged", transform=ax.get_xaxis_transform(),
            fontsize=7, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    # Room below the lowest curve, so the band's label never sits under a line.
    lowest = min(x[f"rmse_{k}"] for x in r for k in ("IPS", "SNIPS", "DM", "DR"))
    ax.set_ylim(bottom=lowest / 2.5)
    ax.set_xlabel("logging epsilon"); ax.set_ylabel("RMSE"); ax.set_title("(b)", loc="left")
    # Below the axes, so the key never covers a curve.
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)


def make_figures(store: Dict, out: Path, plt) -> List[str]:
    """Figure 16.8 from mab/contextual and Figure 16.9 from ope/support.

    With both modes of a pair, the full two-panel figure is written; with one,
    only that panel is written, under a name that says which panel it is.
    """
    made = []
    regret = [(k, t) for k, t in (("mab", "(a) context-free"), ("contextual", "(b) contextual")) if k in store]
    if regret:
        fig, axes = plt.subplots(1, len(regret), figsize=(FIG_WIDTH_IN if len(regret) == 2 else FIG_WIDTH_IN / 1.6, 2.9),
                                 squeeze=False)
        for ax, (k, title) in zip(axes[0], regret):
            _plot_regret(ax, store[k], title)
        handles, labels = axes[0][-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5 if len(regret) == 2 else 2, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=(0, 0.08 if len(regret) == 2 else 0.16, 1, 1))
        name = "fig16_8_bandit_regret.png" if len(regret) == 2 else f"fig16_8_panel_{regret[0][0]}.png"
        save_figure(fig, out / name); plt.close(fig)
        made.append(name)
    panels = [k for k in ("ope", "support") if k in store]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH_IN if len(panels) == 2 else FIG_WIDTH_IN / 1.6, 3.0),
                                 squeeze=False)
        for ax, k in zip(axes[0], panels):
            if k == "ope":
                _plot_ope_box(ax, store["ope"])
            else:
                _plot_support(ax, store["support"], 10_000)
        fig.tight_layout()
        name = "fig16_9_ope.png" if len(panels) == 2 else f"fig16_9_panel_{panels[0]}.png"
        save_figure(fig, out / name); plt.close(fig)
        made.append(name)
    return made


MODES = {"mab": mode_mab, "contextual": mode_contextual, "ope": mode_ope, "support": mode_support}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", nargs="+", choices=list(MODES) + ["all"], default=["all"],
                   help="One or more modes; 'mab contextual' gives Figure 16.8, 'ope support' Figure 16.9")
    p.add_argument("--linucb-alpha", type=float, default=0.5, help="LinUCB exploration width")
    add_common_args(p, "bandit_ope")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    out = prepare_out_dir(args.out_dir)
    plt = setup_matplotlib()
    modes = list(MODES) if "all" in args.mode else [m for m in MODES if m in args.mode]
    results, store = {}, {}
    for m in modes:
        t0 = time.time()
        log(f"[{m}] running")
        results[m] = MODES[m](args, rng, out, plt, store)
        results[m]["runtime_seconds"] = round(time.time() - t0, 2)
        log(f"[{m}] done in {results[m]['runtime_seconds']}s")
    results["figures"] = make_figures(store, out, plt)
    results["meta"] = run_metadata(args, started)
    name = "summary.json" if modes == list(MODES) else f"summary_{'_'.join(modes)}.json"
    write_json(out / name, results)
    log(f"Wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
