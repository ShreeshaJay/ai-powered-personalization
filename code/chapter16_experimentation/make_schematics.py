"""
Chapter 16: Experimentation and Bandits - schematic figures
===========================================================
Draws the chapter's code-generated diagrams:

    Figure 16.1  The evaluation ladder                         (Section 16.2.2)
    Figure 16.4  How experiment designs contain interference   (Section 16.5.2)
    Figure 16.6  Team-draft interleaving, one impression       (Section 16.7.2)

Figure 16.6 is rendered from `interleaving_example.json`, written by
`interleaving_simulation.py --mode example`. If that file is missing, the
example is generated here with the same seed, so the figure always matches
the algorithm's output.

Usage
-----
    python make_schematics.py
    python make_schematics.py --example-json outputs/interleaving/interleaving_example.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from common import (
    DEFAULT_OUTPUT_ROOT, PALETTE, FIG_WIDTH_IN, add_common_args, log, prepare_out_dir,
    run_metadata, save_figure, setup_matplotlib, write_json,
)

TEAM_COLOURS = {"A": PALETTE["primary"], "B": PALETTE["secondary"]}


def _tint(hex_colour: str, amount: float):
    """Blend a colour with white; amount 0 gives white, 1 the colour itself."""
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(1 - amount * (1 - rgb))


# =============================================================================
# Figure 16.1
# =============================================================================

def figure_ladder(out: Path, plt) -> str:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("Offline\nmetrics", "Does it rank\nheld-out\ninteractions\nbetter?"),
        ("Off-policy\nevaluation", "What would it\nhave earned on\nlogged traffic?"),
        ("Interleaving", "Which ranker\ndo users\nprefer?"),
        ("A/B test", "Does it change\nuser and business\noutcomes, and\nby how much?"),
        ("Long-term\nholdback", "Does the\neffect last?"),
    ]
    dots = [8, 5, 3, 2, 1]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4.2))
    ax.set_xlim(-0.9, 10.3); ax.set_ylim(-1.3, 7.9); ax.axis("off")
    w, h, gap, rise = 1.9, 2.4, 0.12, 1.0
    for i, ((name, question), n_dots) in enumerate(zip(steps, dots)):
        x0, y0 = i * (w + gap), i * rise
        shade = 0.22 + 0.78 * i / (len(steps) - 1)
        face = _tint(PALETTE["primary"], shade)
        text_colour = "white" if shade > 0.6 else "#1F1F1F"
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=face, edgecolor=PALETTE["primary"], lw=1.0))
        ax.text(x0 + w / 2, y0 + h - 0.15, name, ha="center", va="top", fontsize=7.5, fontweight="bold", color=text_colour)
        ax.text(x0 + w / 2, y0 + 0.15, question, ha="center", va="bottom", fontsize=6.3, color=text_colour, linespacing=1.15)
        xs = x0 + w / 2 + (np.arange(n_dots) - (n_dots - 1) / 2) * 0.2
        ax.scatter(xs, np.full(n_dots, y0 + h + 0.3), s=14, color=PALETTE["highlight"], zorder=3)
    ax.scatter([0.1], [7.45], s=14, color=PALETTE["highlight"])
    ax.text(0.3, 7.45, "candidates remaining at each rung", fontsize=7, color=PALETTE["highlight"], ha="left", va="center")
    ax.add_patch(FancyArrowPatch((0.0, -0.45), (10.0, -0.45), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#404040"))
    ax.text(5.0, -0.65, "Cost, time, and risk to users per decision", ha="center", va="top", fontsize=8)
    ax.add_patch(FancyArrowPatch((-0.45, 0.0), (-0.45, 6.6), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#404040"))
    ax.text(-0.68, 3.3, "Fidelity to long-term outcomes", rotation=90, ha="center", va="center", fontsize=8)
    name = "fig16_1_evaluation_ladder.png"
    save_figure(fig, out / name); plt.close(fig)
    return name


# =============================================================================
# Figure 16.4
# =============================================================================

def _box(ax, xy, w, h, label, face, edge="#404040", fontsize=7, colour="black"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=face, edgecolor=edge, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center", fontsize=fontsize, color=colour)


def _arrow(ax, a, b, colour="#404040", style="-|>", lw=0.9, rad=0.0, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, lw=lw, color=colour,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def _user_column(ax, arms: np.ndarray, x: float = 0.0, cell: float = 0.5):
    from matplotlib.patches import Rectangle
    for i, t in enumerate(arms):
        ax.add_patch(Rectangle((x, (len(arms) - 1 - i) * cell), cell * 0.9, cell * 0.9,
                               facecolor=PALETTE["primary"] if t else PALETTE["baseline"], edgecolor="white"))
    ax.text(x + cell * 0.45, len(arms) * cell + 0.15, "users", ha="center", fontsize=7)


def figure_interference(out: Path, plt, seed: int) -> str:
    from matplotlib.patches import Patch, Rectangle

    rng = np.random.default_rng(seed + 1604)
    arms = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=bool)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 5.6))
    ys_t = [(11 - i) * 0.5 + 0.22 for i in np.flatnonzero(arms)]
    ys_c = [(11 - i) * 0.5 + 0.22 for i in np.flatnonzero(~arms)]

    # (a) user-randomized, one model trained on both arms' logs
    ax = axes[0, 0]
    _user_column(ax, arms)
    _box(ax, (2.6, 2.5), 1.9, 1.0, "one model", _tint(PALETTE["highlight"], 0.25))
    for y in ys_t + ys_c:
        _arrow(ax, (0.5, y), (2.55, 3.0), colour=PALETTE["primary"] if y in ys_t else PALETTE["baseline"], lw=0.5, style="-")
    _arrow(ax, (3.55, 3.55), (0.85, 5.9), colour=PALETTE["highlight"], rad=0.35, lw=1.3, ls="--")
    ax.text(3.0, 5.35, "trained on both arms,\nserves both arms:\ndata leaks between arms", fontsize=6.5,
            color=PALETTE["highlight"], ha="left")
    ax.set_title("(a) User-randomized, shared model:\nno protection against shared training data", loc="left", fontsize=8)

    # (b) data-diverted: one model per arm, each trained only on its own arm's logs
    ax = axes[0, 1]
    _user_column(ax, arms)
    _box(ax, (2.6, 4.0), 2.5, 0.9, "treatment model", _tint(PALETTE["primary"], 0.35), fontsize=6.5)
    _box(ax, (2.6, 1.0), 2.5, 0.9, "control model", _tint(PALETTE["baseline"], 0.35), fontsize=6.5)
    for y in ys_t:
        _arrow(ax, (0.5, y), (2.55, 4.45), colour=PALETTE["primary"], lw=0.5, style="-")
    for y in ys_c:
        _arrow(ax, (0.5, y), (2.55, 1.45), colour=PALETTE["baseline"], lw=0.5, style="-")
    ax.text(5.25, 3.0, "each model learns\nonly from its\nown arm's logs", fontsize=6.5, ha="left", va="center")
    ax.set_title("(b) Data-diverted:\nshared training data", loc="left", fontsize=8)

    # (c) user-corpus co-diverted: treatment users meet treatment items only
    ax = axes[1, 0]
    n = 12
    users_t = np.arange(n) < n // 2
    for i in range(n):
        for j in range(n):
            if users_t[i] == users_t[j]:
                face = PALETTE["primary"] if users_t[i] else PALETTE["baseline"]
                ax.add_patch(Rectangle((j * 0.5, (n - 1 - i) * 0.5), 0.46, 0.46, facecolor=face, edgecolor="white", lw=0.3))
            else:
                ax.add_patch(Rectangle((j * 0.5, (n - 1 - i) * 0.5), 0.46, 0.46, facecolor="#EFEFEF",
                                       edgecolor="#A0A0A0", hatch="////", lw=0.3))
    ax.text(3.0, 6.25, "items", ha="center", fontsize=7)
    ax.text(-0.25, 3.0, "users", ha="right", va="center", fontsize=7, rotation=90)
    ax.text(4.5, 4.5, "not\neligible", ha="center", va="center", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_title("(c) User-corpus co-diverted:\nshared training data and corpus effects", loc="left", fontsize=8)

    # (d) switchback: regions alternate between arms over time blocks
    ax = axes[1, 1]
    regions, blocks = 4, 12
    seq = rng.integers(0, 2, (regions, blocks)).astype(bool)
    for r in range(regions):
        for b in range(blocks):
            ax.add_patch(Rectangle((b * 0.5, (regions - 1 - r) * 0.9), 0.5, 0.8,
                                   facecolor=PALETTE["primary"] if seq[r, b] else PALETTE["baseline"], edgecolor="white", lw=0.5))
            ax.text(b * 0.5 + 0.25, (regions - 1 - r) * 0.9 + 0.4, "T" if seq[r, b] else "C", ha="center", va="center",
                    fontsize=6, color="white")
            if b > 0 and seq[r, b] != seq[r, b - 1]:
                ax.add_patch(Rectangle((b * 0.5, (regions - 1 - r) * 0.9), 0.09, 0.8, facecolor="white",
                                       edgecolor=PALETTE["highlight"], hatch="xxxx", lw=0.0, alpha=0.9))
        ax.text(-0.15, (regions - 1 - r) * 0.9 + 0.4, f"region {r + 1}", ha="right", va="center", fontsize=6.5)
    ax.text(3.0, -0.35, "time blocks", ha="center", va="top", fontsize=7)
    ax.text(6.15, 1.8, "hatched strips:\ncarryover after\neach switch", fontsize=6.5, color=PALETTE["highlight"], va="center")
    ax.set_title("(d) Switchback:\nmarket-wide and temporal spillover", loc="left", fontsize=8)

    for ax in axes.flat:
        ax.set_aspect("equal"); ax.axis("off"); ax.autoscale_view()
    axes[0, 0].set_xlim(-0.3, 6.6); axes[0, 1].set_xlim(-0.3, 7.2)
    axes[0, 0].set_ylim(-0.3, 6.6); axes[0, 1].set_ylim(-0.3, 6.6)
    axes[1, 0].set_xlim(-0.6, 6.4); axes[1, 0].set_ylim(-0.3, 6.6)
    axes[1, 1].set_xlim(-1.2, 8.0); axes[1, 1].set_ylim(-0.8, 3.8)
    fig.legend(handles=[Patch(facecolor=PALETTE["primary"], label="treatment"),
                        Patch(facecolor=PALETTE["baseline"], label="control"),
                        Patch(facecolor="#EFEFEF", edgecolor="#A0A0A0", hatch="////", label="excluded from both arms")],
               loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    name = "fig16_4_interference_designs.png"
    save_figure(fig, out / name); plt.close(fig)
    return name


# =============================================================================
# Figure 16.6
# =============================================================================

def load_or_make_example(path: Optional[Path], seed: int, out: Path) -> dict:
    if path is not None and path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    import interleaving_simulation as sim
    args = sim.build_parser().parse_args(["--seed", str(seed)])
    world = sim.World.create(np.random.default_rng(seed + 7))
    return sim.mode_example(world, args, None, out, None)


def figure_team_draft(example: dict, out: Path, plt) -> str:
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.5))
    ax.set_xlim(0, 13.4); ax.set_ylim(-0.7, 7.6); ax.axis("off")
    row_h, top = 0.9, 6.3

    def item_cell(x, y, label, face, text_colour="black", w=1.1):
        ax.add_patch(FancyBboxPatch((x, y - 0.33), w, 0.66, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=face, edgecolor="#606060", lw=0.6))
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=8, color=text_colour, fontweight="bold")

    for col, (key, title) in enumerate((("ranker_a", "Ranker A"), ("ranker_b", "Ranker B"))):
        x = 0.4 + col * 1.75
        ax.text(x + 0.55, top + 0.8, title, ha="center", fontsize=7.5, fontweight="bold", color=TEAM_COLOURS["A" if col == 0 else "B"])
        for k, it in enumerate(example[key]):
            item_cell(x, top - k * row_h, it, "white")
    for k in range(len(example["ranker_a"])):
        ax.text(0.25, top - k * row_h, str(k + 1), ha="right", va="center", fontsize=7, color="#606060")

    # Round-by-round table
    x0 = 4.0
    ax.text(x0 + 1.75, top + 0.8, "Draft rounds", ha="center", fontsize=8, fontweight="bold")
    for xpos, hdr in ((0.2, "round"), (1.15, "first pick\n(coin flip)"), (2.75, "picks")):
        ax.text(x0 + xpos, top + 0.15, hdr, ha="center", va="center", fontsize=6.3, color="#404040")
    for r, rnd in enumerate(example["rounds"]):
        y = top - 0.6 - r * 1.3
        ax.text(x0 + 0.2, y, str(rnd["round"]), ha="center", va="center", fontsize=8)
        ax.text(x0 + 1.15, y, rnd["first_picker"], ha="center", va="center", fontsize=8, fontweight="bold",
                color=TEAM_COLOURS[rnd["first_picker"]])
        for p, pick in enumerate(rnd["picks"]):
            item_cell(x0 + 2.0 + p * 0.8, y, f"{pick['team']}:{pick['item']}", _tint(TEAM_COLOURS[pick["team"]], 0.35), w=0.7)

    # Interleaved list
    x1 = 8.4
    ax.text(x1 + 0.7, top + 0.8, "Interleaved list shown", ha="center", fontsize=8, fontweight="bold")
    for row in example["interleaved"]:
        y = top - (row["position"] - 1) * row_h
        item_cell(x1, y, row["item"], TEAM_COLOURS[row["team"]], text_colour="white", w=1.4)
        ax.text(x1 - 0.15, y, str(row["position"]), ha="right", va="center", fontsize=7, color="#606060")
        if row["clicked"]:
            ax.scatter([x1 + 1.65], [y], marker="o", s=18, color=PALETTE["highlight"])
            ax.text(x1 + 1.85, y, "click", ha="left", va="center", fontsize=7, color=PALETTE["highlight"], fontweight="bold")

    # Tally
    tx, ty = 10.25, -0.65
    ax.add_patch(Rectangle((tx - 0.15, ty), 3.1, 1.35, facecolor="#F7F7F7", edgecolor="#808080", lw=0.6))
    ax.text(tx, ty + 1.1, f"Team A clicks: {example['team_a_clicks']}", fontsize=7.5, va="top", color=TEAM_COLOURS["A"])
    ax.text(tx, ty + 0.75, f"Team B clicks: {example['team_b_clicks']}", fontsize=7.5, va="top", color=TEAM_COLOURS["B"])
    ax.text(tx, ty + 0.38, f"Outcome: {example['outcome']}", fontsize=7.5, va="top", fontweight="bold")
    name = "fig16_6_team_draft_example.png"
    save_figure(fig, out / name); plt.close(fig)
    return name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example-json", type=str, default=str(DEFAULT_OUTPUT_ROOT / "interleaving" / "interleaving_example.json"))
    add_common_args(p, "figures")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    out = prepare_out_dir(args.out_dir)
    plt = setup_matplotlib()
    made = [figure_ladder(out, plt), figure_interference(out, plt, args.seed)]
    example = load_or_make_example(Path(args.example_json) if args.example_json else None, args.seed, out)
    made.append(figure_team_draft(example, out, plt))
    for m in made:
        log(f"Wrote {out / m}")
    write_json(out / "summary_schematics.json", {"figures": made, "meta": run_metadata(args, started)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
