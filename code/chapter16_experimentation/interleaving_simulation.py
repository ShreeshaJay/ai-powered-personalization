"""
Chapter 16: Experimentation and Bandits - interleaving versus A/B testing
=========================================================================
A fully synthetic simulation of two rankers compared by team-draft
interleaving and by a user-split A/B test (Section 16.7; Figures 16.6 and 16.7).

Simulator
---------
- Users and items have latent vectors (d = 16); true relevance of item i to
  user u is sigmoid(alpha * u.v + item_bias + offset), so most items are rarely clicked.
- Each session draws a user and 100 candidate items.
- Each ranker scores candidates as the true logit plus Gaussian noise. The
  ranker with less noise is truly better; expected NDCG@10 under the true
  relevance confirms which one.
- Clicks follow a position-based model: P(click at rank k) = relevance / k^eta.

Modes
-----
sanity       identical rankers: about 50% interleaving wins, false positives near alpha
sensitivity  Figure 16.7: share of experiments reaching a correct significant decision
example      one seeded impression, saved as JSON for Figure 16.6
all          every mode above

Usage
-----
    python interleaving_simulation.py --mode all
    python interleaving_simulation.py --mode all --quick
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import stats_utils as su
from common import (
    PALETTE, FIG_WIDTH_IN, add_common_args, log, mode_rng, prepare_out_dir, run_metadata,
    save_figure, setup_matplotlib, write_csv, write_json,
)
from interleaving import (
    TEAM_A, interleaving_sign_test, team_draft_batch, team_draft_interleave, team_draft_outcome,
)

ALPHA = 0.05
K = 10                 # list length shown to the user
N_CANDIDATES = 100


@dataclass
class World:
    users: np.ndarray        # (n_users, d)
    items: np.ndarray        # (n_items, d)
    item_bias: np.ndarray    # (n_items,)
    activity: np.ndarray     # (n_users,) multiplier on click probability; heavy-tailed
    alpha: float
    offset: float
    eta: float

    @classmethod
    def create(cls, rng: np.random.Generator, n_users: int = 5000, n_items: int = 1000, dim: int = 16,
               alpha: float = 1.5, offset: float = -3.0, eta: float = 1.0, activity_sigma: float = 1.0) -> "World":
        scale = dim ** -0.25    # makes u.v roughly standard normal
        # Some users click far more than others whatever they are shown. This
        # between-user variance inflates A/B noise, while interleaving compares
        # both rankers within the same session and cancels it.
        activity = rng.lognormal(-activity_sigma ** 2 / 2, activity_sigma, n_users)
        return cls(rng.normal(0, scale, (n_users, dim)), rng.normal(0, scale, (n_items, dim)),
                   rng.normal(0, 0.5, n_items), activity, alpha, offset, eta)

    def exam(self, k: int) -> np.ndarray:
        return 1.0 / np.arange(1, k + 1) ** self.eta


@dataclass
class SessionPool:
    """Pre-simulated sessions; experiments subsample from these."""
    clicks_a: np.ndarray      # clicks when the session sees ranker A only
    clicks_b: np.ndarray      # clicks when the session sees ranker B only
    outcome: np.ndarray       # team-draft result: +1 A wins, -1 B wins, 0 tie
    ndcg_a: float
    ndcg_b: float


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def simulate_pool(world: World, n_sessions: int, noise_a: float, noise_b: float, rng: np.random.Generator,
                  identical: bool = False, chunk: int = 5000) -> SessionPool:
    exam = world.exam(K)
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    out_a, out_b, out_i, ndcg_a, ndcg_b = [], [], [], [], []
    n_items = world.items.shape[0]
    for start in range(0, n_sessions, chunk):
        m = min(chunk, n_sessions - start)
        u = rng.integers(0, world.users.shape[0], m)
        cand = np.argpartition(rng.random((m, n_items)), N_CANDIDATES, axis=1)[:, :N_CANDIDATES]
        logit = world.alpha * np.einsum("md,mcd->mc", world.users[u], world.items[cand]) + world.item_bias[cand] + world.offset
        rel = 1.0 / (1.0 + np.exp(-logit))
        score_a = logit + rng.normal(0, noise_a, logit.shape)
        score_b = score_a.copy() if identical else logit + rng.normal(0, noise_b, logit.shape)
        top_a, top_b = _top_k(score_a, K), _top_k(score_b, K)
        rel_a = np.take_along_axis(rel, top_a, axis=1)
        rel_b = np.take_along_axis(rel, top_b, axis=1)
        ideal = np.sort(rel, axis=1)[:, ::-1][:, :K] @ disc
        ndcg_a.append((rel_a @ disc) / ideal)
        ndcg_b.append((rel_b @ disc) / ideal)
        exam_u = np.minimum(world.activity[u][:, None] * exam[None, :], 1.0)
        out_a.append((rng.random((m, K)) < rel_a * exam_u).sum(axis=1))
        out_b.append((rng.random((m, K)) < rel_b * exam_u).sum(axis=1))
        coins = rng.integers(0, 2, (m, (K + 1) // 2))
        items, teams = team_draft_batch(top_a, top_b, K, coins)
        rel_i = np.take_along_axis(rel, items, axis=1)
        clicked = rng.random((m, K)) < rel_i * exam_u
        wins_a = (clicked & (teams == TEAM_A)).sum(axis=1)
        wins_b = (clicked & (teams != TEAM_A)).sum(axis=1)
        out_i.append(np.sign(wins_a - wins_b))
    return SessionPool(np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_i),
                       float(np.concatenate(ndcg_a).mean()), float(np.concatenate(ndcg_b).mean()))


def ab_decisions(pool: SessionPool, n_sessions: int, reps: int, rng: np.random.Generator):
    """Per replication: A/B test with n_sessions split evenly. Returns (correct, any_significant)."""
    half = max(n_sessions // 2, 2)
    correct = np.zeros(reps, bool)
    signif = np.zeros(reps, bool)
    step = max(1, 4_000_000 // half)
    for s in range(0, reps, step):
        r = min(step, reps - s)
        ya = pool.clicks_a[rng.integers(0, pool.clicks_a.size, (r, half))].astype(float)
        yb = pool.clicks_b[rng.integers(0, pool.clicks_b.size, (r, half))].astype(float)
        diff, _, _, p, _, _ = su.welch_from_moments(ya.mean(1), ya.var(1, ddof=1), half, yb.mean(1), yb.var(1, ddof=1), half)
        p = np.nan_to_num(p, nan=1.0)
        signif[s:s + r] = p < ALPHA
        correct[s:s + r] = (p < ALPHA) & (diff > 0)
    return correct, signif


def interleaving_decisions(pool: SessionPool, n_sessions: int, reps: int, rng: np.random.Generator):
    correct = np.zeros(reps, bool)
    signif = np.zeros(reps, bool)
    for s in range(reps):
        o = pool.outcome[rng.integers(0, pool.outcome.size, n_sessions)]
        wa, wb, p, _ = interleaving_sign_test(o)
        signif[s] = p < ALPHA
        correct[s] = (p < ALPHA) and (wa > wb)
    return correct, signif


# =============================================================================
# Modes
# =============================================================================

def mode_sanity(world: World, args, rng, out: Path, plt) -> Dict:
    # Every test resamples the same pool, so the pool's own chance imbalance acts as a
    # small true effect shared by all tests. A pool hundreds of times larger than one test
    # keeps that imbalance negligible next to a single test's noise.
    n_pool = 200_000 if args.quick else 1_000_000
    n_sessions = 1_000 if args.quick else 2_000
    reps = 100 if args.quick else 500
    res = {}
    for label, identical in (("identical_rankers", True), ("equal_quality_independent_noise", False)):
        pool = simulate_pool(world, n_pool, args.noise_a, args.noise_a, rng, identical=identical)
        nontie = pool.outcome != 0
        _, ab_sig = ab_decisions(pool, n_sessions, reps, rng)
        _, il_sig = interleaving_decisions(pool, n_sessions, reps, rng)
        res[label] = {
            "win_rate_a_among_nonties": float(np.mean(pool.outcome[nontie] > 0)),
            "tie_rate": float(1 - nontie.mean()),
            "false_positive_rate_ab": float(ab_sig.mean()),
            "false_positive_rate_interleaving": float(il_sig.mean()),
        }
    return {"n_pool": n_pool, "n_sessions_per_test": n_sessions, "reps": reps, **res}


def mode_sensitivity(world: World, args, rng, out: Path, plt) -> Dict:
    n_pool = 50_000 if args.quick else 200_000
    reps = 100 if args.quick else 500
    sizes = np.unique(np.logspace(2, 4, 5).astype(int)) if args.quick else np.unique(np.logspace(2, 5, 13).astype(int))
    gaps = {"large gap": args.noise_b_large, "small gap": args.noise_b_small}
    rows, ground_truth = [], {}
    for gap, noise_b in gaps.items():
        pool = simulate_pool(world, n_pool, args.noise_a, noise_b, rng)
        ground_truth[gap] = {"noise_a": args.noise_a, "noise_b": noise_b, "ndcg10_a": pool.ndcg_a, "ndcg10_b": pool.ndcg_b,
                             "clicks_per_session_a": float(pool.clicks_a.mean()), "clicks_per_session_b": float(pool.clicks_b.mean()),
                             "interleaving_win_rate_a": float(np.mean(pool.outcome[pool.outcome != 0] > 0))}
        for n in sizes:
            ab, _ = ab_decisions(pool, int(n), reps, rng)
            il, _ = interleaving_decisions(pool, int(n), reps, rng)
            rows.append({"gap": gap, "n_sessions": int(n), "ab_correct": float(ab.mean()), "interleaving_correct": float(il.mean())})
        log(f"  {gap}: NDCG@10 A={pool.ndcg_a:.4f} B={pool.ndcg_b:.4f}")
    write_csv(out / "sensitivity.csv", rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.1))
    for gap, ls in (("large gap", "-"), ("small gap", "--")):
        r = [x for x in rows if x["gap"] == gap]
        n = [x["n_sessions"] for x in r]
        ax.plot(n, [x["interleaving_correct"] for x in r], ls, color=PALETTE["primary"], lw=1.8, label=f"Team-draft interleaving, {gap}")
        ax.plot(n, [x["ab_correct"] for x in r], ls, color=PALETTE["baseline"], lw=1.8, label=f"A/B test, {gap}")
    ax.axhline(0.8, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("sessions in the experiment"); ax.set_ylabel("share of experiments with a\ncorrect significant decision")
    # Below the axes, so the key never covers a curve or the 0.8 reference line.
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save_figure(fig, out / "fig16_7_interleaving_sensitivity.png"); plt.close(fig)
    return {"n_pool": n_pool, "reps": reps, "ground_truth": ground_truth, "rows": rows}


def mode_example(world: World, args, rng, out: Path, plt) -> Dict:
    """One impression for Figure 16.6, chosen deterministically from the seed.

    Sessions are drawn in sequence from a dedicated generator until one meets
    the display criteria: the two top-6 lists share at least three items in a
    different order, each ranker wins at least one coin flip, and the clicks
    produce a win for one team.
    """
    ex_rng = np.random.default_rng(args.seed + 1606)
    n_show, n_cand = 6, 8
    letters = list(string.ascii_uppercase[:n_cand])
    for attempt in range(1, 10_000):
        rel = ex_rng.uniform(0.1, 0.8, n_cand)
        logit = np.log(rel / (1 - rel))
        sa = logit + ex_rng.normal(0, args.noise_a, n_cand)
        sb = logit + ex_rng.normal(0, args.noise_b_large, n_cand)
        ra = [letters[i] for i in np.argsort(-sa)[:n_show]]
        rb = [letters[i] for i in np.argsort(-sb)[:n_show]]
        shared = [x for x in ra if x in rb]
        if len(shared) < 3 or ra == rb:
            continue
        coins = ex_rng.integers(0, 2, n_show)
        td = team_draft_interleave(ra, rb, n_show, coins=coins)
        if len(set(td.first_picker)) < 2:
            continue
        exam = world.exam(n_show)
        rel_by = dict(zip(letters, rel))
        clicked = [bool(ex_rng.random() < rel_by[it] * exam[k]) for k, it in enumerate(td.items)]
        outcome = team_draft_outcome(td.teams, clicked)
        if outcome == 0 or sum(clicked) < 2:
            continue
        rounds, pos = [], 0
        for r, first in enumerate(td.first_picker):
            picks = []
            for _ in range(2):
                if pos < len(td.items):
                    picks.append({"team": "A" if td.teams[pos] == TEAM_A else "B", "item": td.items[pos]})
                    pos += 1
            rounds.append({"round": r + 1, "first_picker": "A" if first == TEAM_A else "B", "picks": picks})
        tally_a = sum(c for c, t in zip(clicked, td.teams) if t == TEAM_A)
        tally_b = sum(clicked) - tally_a
        example = {
            "attempt": attempt, "ranker_a": ra, "ranker_b": rb, "rounds": rounds,
            "interleaved": [{"position": k + 1, "item": it, "team": "A" if t == TEAM_A else "B", "clicked": c}
                            for k, (it, t, c) in enumerate(zip(td.items, td.teams, clicked))],
            "team_a_clicks": tally_a, "team_b_clicks": tally_b,
            "outcome": "A wins" if outcome > 0 else "B wins",
        }
        write_json(out / "interleaving_example.json", example)
        return example
    raise RuntimeError("No example session met the display criteria")


MODES = {"sanity": mode_sanity, "sensitivity": mode_sensitivity, "example": mode_example}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    p.add_argument("--noise-a", type=float, default=1.0, help="Score noise of the better ranker A")
    p.add_argument("--noise-b-small", type=float, default=1.1, help="Ranker B noise for the small quality gap")
    p.add_argument("--noise-b-large", type=float, default=1.4, help="Ranker B noise for the large quality gap")
    add_common_args(p, "interleaving")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    out = prepare_out_dir(args.out_dir)
    plt = setup_matplotlib()
    world = World.create(np.random.default_rng(args.seed + 7))
    modes = list(MODES) if args.mode == "all" else [args.mode]
    results = {}
    for m in modes:
        t0 = time.time()
        log(f"[{m}] running")
        results[m] = MODES[m](world, args, mode_rng(args.seed, m), out, plt)
        results[m]["runtime_seconds"] = round(time.time() - t0, 2)
        log(f"[{m}] done in {results[m]['runtime_seconds']}s")
    results["meta"] = run_metadata(args, started)
    name = "summary.json" if args.mode == "all" else f"summary_{args.mode}.json"
    write_json(out / name, results)
    log(f"Wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
