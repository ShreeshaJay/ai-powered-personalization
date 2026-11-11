"""Figures A.3 and A.4 from saved metric JSON.

Charts have axis labels and legends and no in-image title. Colors are the
book palette: navy, blue, green, amber, and muted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1E2761"
BLUE = "#3B82F6"
GREEN = "#10B981"
AMBER = "#F59E0B"
MUTED = "#64748B"

A3_SERIES = (
    ("rexbert_micro_zero_shot", "RexBERT-micro zero-shot", NAVY),
    ("rexbert_micro_fine_tuned_seed0", "RexBERT-micro fine-tuned", BLUE),
    ("minilm_zero_shot", "MiniLM zero-shot", GREEN),
)

A4_GROUPS = {
    "baseline": MUTED,
    "zero_shot": BLUE,
    "fine_tuned": GREEN,
}


def apply_font() -> str:
    from matplotlib import font_manager

    names = {font.name for font in font_manager.fontManager.ttflist}
    chosen = "Calibri" if "Calibri" in names else "DejaVu Sans"
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def _density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(grid)
    spread = float(np.std(values))
    bandwidth = 1.06 * spread * (values.size ** (-0.2)) if spread > 0 else 0.05
    bandwidth = max(bandwidth, 0.02)
    diff = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diff * diff) / (bandwidth * np.sqrt(2.0 * np.pi))
    return kernel.mean(axis=1)


def figure_a3(payload: dict, out_path: Path) -> None:
    apply_font()
    grid = np.linspace(-0.2, 1.0, 256)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    panels = (
        (axes[0], "random_pairs", "Random catalog pairs"),
        (axes[1], "positive_pairs", "Held-out next-item pairs"),
    )
    for axis, key, xlabel in panels:
        block = payload.get(key, {})
        for series_key, label, color in A3_SERIES:
            if series_key not in block:
                continue
            values = np.asarray(block[series_key], dtype=np.float64)
            axis.plot(grid, _density(values, grid), color=color, label=label, linewidth=1.8)
        axis.set_xlabel(xlabel)
        axis.set_xlim(-0.2, 1.0)
        axis.set_title("")
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def figure_a4(payload: dict, out_path: Path) -> None:
    apply_font()
    rows = list(payload.get("rows", []))
    labels = [row["label"] for row in rows] or ["(none)"]
    y = np.arange(len(rows)) if rows else np.array([0.0])
    panels = (
        ("recall_at_50", "Recall@50"),
        ("mrr", "MRR"),
        ("cold_recall_at_50", "Cold-target Recall@50"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, max(3.5, 0.38 * max(len(rows), 1) + 1.4)), sharey=True)
    for axis, (key, xlabel) in zip(axes, panels):
        if not rows:
            axis.set_xlabel(xlabel)
            axis.set_title("")
            continue
        for position, row in enumerate(rows):
            center = row.get(key)
            if center is None:
                continue
            low = row.get(f"{key}_low", center)
            high = row.get(f"{key}_high", center)
            color = A4_GROUPS.get(row.get("group", ""), NAVY)
            axis.errorbar(
                center,
                position,
                xerr=[[max(center - low, 0.0)], [max(high - center, 0.0)]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=2,
                markersize=5,
            )
        axis.set_xlabel(xlabel)
        axis.set_title("")
        axis.set_ylim(-0.5, len(rows) - 0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def refuse_demo_overwrite(metrics_dir: Path) -> None:
    existing = metrics_dir / "diagnostics.json"
    if not existing.is_file():
        return
    current = json.loads(existing.read_text(encoding="utf-8"))
    if not current.get("synthetic"):
        raise SystemExit("Refusing to replace a measured diagnostics file with the demo.")


def write_figures(metrics_dir: Path, figure_dir: Path) -> list[str]:
    written = []
    a3 = metrics_dir / "diagnostics.json"
    a4 = metrics_dir / "figure_a4.json"
    if a3.is_file():
        figure_a3(json.loads(a3.read_text(encoding="utf-8")), figure_dir / "figure_A_3.png")
        written.append("figure_A_3.png")
    if a4.is_file():
        figure_a4(json.loads(a4.read_text(encoding="utf-8")), figure_dir / "figure_A_4.png")
        written.append("figure_A_4.png")
    return written


def _demo_payloads() -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    a3 = {
        "synthetic": True,
        "random_pairs": {
            "rexbert_micro_zero_shot": rng.normal(0.55, 0.08, size=400).tolist(),
            "rexbert_micro_fine_tuned_seed0": rng.normal(0.15, 0.12, size=400).tolist(),
            "minilm_zero_shot": rng.normal(0.2, 0.1, size=400).tolist(),
        },
        "positive_pairs": {
            "rexbert_micro_zero_shot": rng.normal(0.6, 0.08, size=400).tolist(),
            "rexbert_micro_fine_tuned_seed0": rng.normal(0.55, 0.1, size=400).tolist(),
            "minilm_zero_shot": rng.normal(0.45, 0.1, size=400).tolist(),
        },
    }
    a4 = {
        "synthetic": True,
        "rows": [
            {
                "label": "Popularity",
                "group": "baseline",
                "recall_at_50": 0.12,
                "recall_at_50_low": 0.10,
                "recall_at_50_high": 0.14,
                "mrr": 0.04,
                "mrr_low": 0.03,
                "mrr_high": 0.05,
                "cold_recall_at_50": 0.02,
                "cold_recall_at_50_low": 0.01,
                "cold_recall_at_50_high": 0.03,
            },
            {
                "label": "RexBERT-micro zero-shot",
                "group": "zero_shot",
                "recall_at_50": 0.22,
                "recall_at_50_low": 0.19,
                "recall_at_50_high": 0.25,
                "mrr": 0.08,
                "mrr_low": 0.07,
                "mrr_high": 0.09,
                "cold_recall_at_50": 0.05,
                "cold_recall_at_50_low": 0.03,
                "cold_recall_at_50_high": 0.07,
            },
        ],
    }
    return a3, a4


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figures A.3 and A.4 from metric JSON.")
    parser.add_argument("--metrics_dir", default="outputs/metrics/cpu_small")
    parser.add_argument("--figure_dir", default="outputs/figures/cpu_small")
    parser.add_argument("--demo", action="store_true", help="Draw from built-in synthetic JSON.")
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir)
    figure_dir = Path(args.figure_dir)
    if args.demo:
        if args.metrics_dir == "outputs/metrics/cpu_small":
            metrics_dir = Path("outputs/metrics/demo")
        if args.figure_dir == "outputs/figures/cpu_small":
            figure_dir = Path("outputs/figures/demo")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        refuse_demo_overwrite(metrics_dir)
        a3, a4 = _demo_payloads()
        (metrics_dir / "diagnostics.json").write_text(json.dumps(a3), encoding="utf-8")
        (metrics_dir / "figure_a4.json").write_text(json.dumps(a4), encoding="utf-8")
    written = write_figures(metrics_dir, figure_dir)
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
