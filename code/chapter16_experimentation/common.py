"""
Chapter 16: Experimentation and Bandits - shared helpers
========================================================
Command-line conventions, output handling, plotting style, and the Yambda
data-path convention shared by every entry point in this folder.

Every entry script accepts:
    --seed      integer seed; every random draw flows from it
    --out-dir   where PNG, CSV, and JSON outputs are written
    --quick     a reduced configuration for smoke runs (well under a minute)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BASE_DIR / "outputs"

# Office palette, consistent with the book's #4F81BD heading colour.
PALETTE = {
    "primary": "#4F81BD",    # treatment or primary series
    "baseline": "#7F7F7F",   # control or baseline
    "highlight": "#C0504D",
    "secondary": "#F79646",
    "tertiary": "#9BBB59",
}

FIG_WIDTH_IN = 6.5
FIG_DPI = 300


def add_common_args(parser: argparse.ArgumentParser, default_subdir: str) -> None:
    parser.add_argument("--seed", type=int, default=16, help="Random seed (default: 16)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT / default_subdir),
        help=f"Output directory (default: outputs/{default_subdir})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced replications and sizes for a fast smoke run",
    )


def prepare_out_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def mode_rng(seed: int, name: str) -> np.random.Generator:
    """An independent random stream for one mode, derived from the seed and the mode name.

    Each mode draws only from its own stream, so `--mode cuped` on its own gives
    exactly the numbers and figure that `--mode all` gives for that mode.
    """
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return np.random.default_rng([seed, tag])


def resolve_yambda_dir(cli_value: Optional[str]) -> Optional[Path]:
    """Return the Yambda `flat` directory from --data-dir or YAMBDA_DATA_DIR.

    Same convention as Chapters 6 and 7: the directory that contains
    `listens.parquet`, for example `.../Dataset/Yandex/flat`.
    """
    value = cli_value or os.environ.get("YAMBDA_DATA_DIR", "")
    if not value:
        return None
    return Path(value)


def run_metadata(args: argparse.Namespace, started: float, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    libraries = {"numpy": np.__version__}
    for name in ("scipy", "polars", "matplotlib"):
        module = sys.modules.get(name)
        if module is not None:
            libraries[name] = getattr(module, "__version__", None)
    meta = {
        "args": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "runtime_seconds": round(time.time() - started, 2),
        "python": platform.python_version(),
        "libraries": libraries,
    }
    if extra:
        meta.update(extra)
    return meta


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_builtin(payload), fh, indent=2)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(_to_builtin(r)) for r in rows]
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib():
    """Use a non-interactive backend and the book's plotting defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": FIG_DPI,
        }
    )
    return plt


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")


def log(msg: str) -> None:
    print(msg, flush=True)
