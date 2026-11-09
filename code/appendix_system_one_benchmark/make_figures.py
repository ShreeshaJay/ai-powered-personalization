"""Draw the appendix figures from the replay pack (CPU, needs matplotlib).

    python make_figures.py [--pack-dir replay] [--output-dir outputs/replay/figures]

Writes three untitled PNGs (captions live in the book, not in the images):
  confidence_meanings.png        compatibility: reported `confidence` against top-label probability
  calibration_two_numbers.png    ECE on reported `confidence` against top-label ECE, four fields
  judge_agreement_by_stratum.png query segmentation: share of items where both judges agree on all fields
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.calibration import DEFAULT_BINS, ece_equal_width, selected_probability
from eval.replay_pack import DEFAULT_PACK_DIR, ORCAS_STRATUM, PROVENANCE_NAME, ReplayPack, field_view, load_pack

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "replay" / "figures"
NAVY = "#1E2761"
ACCENT = "#3B82F6"
ACCENT2 = "#10B981"
ACCENT3 = "#F59E0B"
MUTED = "#64748B"
CARD_BG = "#F1F5F9"
WIDTH_IN = 6.0
DPI = 300
BASELINE = "majority"
ECE_FIELDS = (
    ("compatibility", "label", "Compatibility"),
    ("query_segmentation", "goal", "Query goal"),
    ("brand_category", "brand_choice", "Brand choice"),
    ("esci", "esci_label", "ESCI"),
)
STRATUM_NAMES = {
    "esci_single_product": "ESCI: single product",
    "esci_accessory_for_product": "ESCI: accessory for a product",
    "esci_bundle_or_set": "ESCI: bundle or set",
    "esci_broad_gift_intent": "ESCI: broad or gift intent",
    "esci_ambiguous": "ESCI: ambiguous",
    "esci_unknown": "ESCI: unknown",
    ORCAS_STRATUM: "ORCAS-I (all categories)",
}
ALL_ITEMS = "all"


def style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata={"Software": None}, facecolor="white")


def role(pack: ReplayPack, slug: str) -> str:
    return pack.manifest["models"][slug]["role_label"]


def confidence_meanings(pack: ReplayPack, plt, path: Path) -> None:
    task = "compatibility"
    slugs = [slug for slug in pack.slugs if slug != BASELINE]
    options = len(next(iter(pack.predictions[(slugs[0], task)])).fields["label"].probabilities)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, 4.4), sharex=True, sharey=True)
    grid = [0.0, 1.0]
    for ax, slug in zip(axes.flat, slugs):
        rows = pack.predictions[(slug, task)]
        top = [selected_probability(row.fields["label"].predicted, row.fields["label"].probabilities) for row in rows]
        reported = [row.fields["label"].confidence for row in rows]
        ax.plot(grid, grid, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
        low = 1.0 / options
        ax.plot([low, 1.0], [0.0, 1.0], color=ACCENT3, linestyle=":", linewidth=1.3, zorder=1)
        ax.scatter(top, reported, s=4, color=NAVY, alpha=0.35, linewidths=0, zorder=2)
        ax.set_title(role(pack, slug), loc="left", fontsize=8.5, color=NAVY, pad=4)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_facecolor(CARD_BG)
    legend_ax = axes.flat[len(slugs)] if len(slugs) < axes.size else None
    for ax in list(axes.flat)[len(slugs):]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=NAVY, marker="o", linestyle="", markersize=4, alpha=0.6, label="One item"),
        plt.Line2D([], [], color=MUTED, linestyle="--", label="Confidence = probability"),
        plt.Line2D([], [], color=ACCENT3, linestyle=":", linewidth=1.3, label=f"Rescaled margin, {options} options"),
    ]
    if legend_ax is not None:
        legend_ax.legend(handles=handles, loc="center", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Top-label probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reported confidence")
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def calibration_two_numbers(pack: ReplayPack, plt, path: Path, bins: int) -> None:
    slugs = [slug for slug in pack.slugs if slug != BASELINE]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.6), sharex=True, sharey=True)
    values = {}
    for task, field, _title in ECE_FIELDS:
        for slug in slugs:
            view = field_view(pack, slug, task, field)
            values[(task, field, slug)] = (
                ece_equal_width(view.reported_confidence, view.correct, bins),
                ece_equal_width(view.top_label_probability, view.correct, bins),
            )
    limit = max(max(pair) for pair in values.values()) * 1.08
    positions = list(range(len(slugs)))[::-1]
    for ax, (task, field, title) in zip(axes.flat, ECE_FIELDS):
        for y, slug in zip(positions, slugs):
            reported, top = values[(task, field, slug)]
            ax.plot([reported, top], [y, y], color=MUTED, linewidth=1.2, zorder=1)
            ax.scatter([reported], [y], s=28, facecolors="white", edgecolors=MUTED, linewidths=1.2, zorder=2)
            ax.scatter([top], [y], s=28, color=NAVY, zorder=3)
        ax.text(0.97, 0.04, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=NAVY)
        ax.set_yticks(positions)
        ax.set_yticklabels([role(pack, slug) for slug in slugs])
        ax.set_xlim(0, limit)
        ax.set_ylim(-0.6, len(slugs) - 0.4)
        ax.grid(axis="x", color=CARD_BG, linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel(f"ECE ({bins} equal-width bins)")
    handles = [
        plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="", label="On reported confidence"),
        plt.Line2D([], [], color=NAVY, marker="o", linestyle="", label="On top-label probability"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)
    plt.close(fig)


def judge_agreement_by_stratum(provenance: dict, plt, path: Path) -> None:
    task = "query_segmentation"
    strata = {name: entry for name, entry in provenance["agreement_by_stratum"][task].items() if name != ALL_ITEMS}
    order = [name for name in STRATUM_NAMES if name in strata] + sorted(set(strata) - set(STRATUM_NAMES))
    shares = [strata[name]["complete_agreement"] for name in order]
    items = [strata[name]["items"] for name in order]
    overall = sum(share * count for share, count in zip(shares, items)) / sum(items)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    positions = list(range(len(order)))[::-1]
    colours = [ACCENT3 if name == ORCAS_STRATUM else ACCENT for name in order]
    ax.barh(positions, shares, color=colours, height=0.62, zorder=2)
    ax.vlines(overall, -0.6, len(order) - 0.35, color=NAVY, linestyle="--", linewidth=1.0, zorder=1)
    for y, share, count in zip(positions, shares, items):
        ax.text(
            share + 0.01,
            y,
            f"{100 * share:.0f}%  (n = {count:,})",
            va="center",
            fontsize=8.5,
            color=NAVY,
            zorder=3,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
    ax.text(
        overall,
        len(order) - 0.35,
        f"All {sum(items):,} items: {100 * overall:.0f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=NAVY,
        zorder=3,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([STRATUM_NAMES.get(name, name) for name in order])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) + 0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{100 * value:.0f}%"))
    ax.set_xlabel("Items where both judges agree on all four fields")
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("figures: matplotlib is not installed; install the optional figure requirement")
        return 1
    style(plt)
    pack = load_pack(args.pack_dir)
    provenance = json.loads((args.pack_dir / PROVENANCE_NAME).read_text(encoding="utf-8"))
    outputs = {
        "confidence_meanings.png": lambda path: confidence_meanings(pack, plt, path),
        "calibration_two_numbers.png": lambda path: calibration_two_numbers(pack, plt, path, args.bins),
        "judge_agreement_by_stratum.png": lambda path: judge_agreement_by_stratum(provenance, plt, path),
    }
    for name, draw in outputs.items():
        draw(args.output_dir / name)
        print(f"figures: wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
