"""
Chapter 14: Latency Optimization - shared configuration
=======================================================
Paths, seeds, and output locations used by every entry point.

Nothing here contains an absolute path. Data locations come from environment
variables (the same ones Chapters 5 to 7 use), with a fallback to a sibling
``Dataset/`` folder when the code sits in the author's working tree.

Large artifacts (indexes, traces, ONNX models) go to a per-user cache folder
outside the code tree; small summary files go to ``summary/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).resolve().parent

# Fixed seed for every generator, sampler, and model initialization.
SEED = 14

# Summary files (small CSV and Markdown) live inside the code folder.
SUMMARY_DIR = CODE_DIR / "summary"
QUICK_SUMMARY_DIR = SUMMARY_DIR / "quick"


def default_output_root() -> Path:
    """Return the platform-appropriate folder for large outputs.

    Order of precedence:
      1. ``RECSYS_CH14_OUT`` if set.
      2. Windows: ``%LOCALAPPDATA%/recsys_book/ch14``.
      3. macOS:   ``~/Library/Caches/recsys_book/ch14``.
      4. Linux and others: ``$XDG_CACHE_HOME/recsys_book/ch14`` or
         ``~/.cache/recsys_book/ch14``.
    """
    override = os.environ.get("RECSYS_CH14_OUT")
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "recsys_book" / "ch14"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "recsys_book" / "ch14"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "recsys_book" / "ch14"


def output_dir(*parts: str, quick: bool = False) -> Path:
    """Create and return a subfolder of the large-output root."""
    root = default_output_root() / ("quick" if quick else "full")
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def summary_dir(quick: bool = False, override: Optional[str] = None) -> Path:
    """Create and return the folder for small summary files."""
    path = Path(override) if override else (QUICK_SUMMARY_DIR if quick else SUMMARY_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sibling_dataset(*parts: str) -> Optional[Path]:
    candidate = CODE_DIR.parents[1].joinpath("Dataset", *parts)
    return candidate if candidate.exists() else None


def sigir_dir() -> Optional[Path]:
    """Coveo SIGIR eCom 2021 data folder (contains ``sampled/`` or ``train/``)."""
    env = os.environ.get("SIGIR_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return _sibling_dataset("SIGIR-ecom-data-challenge")


def yambda_dir() -> Optional[Path]:
    """Yambda flat-format folder (contains ``listens.parquet``)."""
    env = os.environ.get("YAMBDA_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return _sibling_dataset("Yandex", "flat")
