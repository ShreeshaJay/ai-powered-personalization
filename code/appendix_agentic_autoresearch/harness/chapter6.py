"""Import Chapter 6's Polars feature pipeline without copying it."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType

_MODULE: ModuleType | None = None


def pipeline_path() -> Path:
    """Sibling ``chapter6_ranking`` next to this appendix folder."""

    return (
        Path(__file__).resolve().parents[2]
        / "chapter6_ranking"
        / "utils"
        / "polars_pipeline.py"
    )


def polars_pipeline() -> ModuleType:
    """Load ``utils/polars_pipeline.py`` from the published Chapter 6 code."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = pipeline_path()
    if not path.is_file():
        raise FileNotFoundError(
            "Chapter 6 feature pipeline was not found at "
            "code/chapter6_ranking/utils/polars_pipeline.py. "
            "Run this appendix from the book repository, beside chapter6_ranking."
        )
    spec = importlib.util.spec_from_file_location("chapter6_polars_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logging.getLogger("chapter6_polars_pipeline").setLevel(logging.WARNING)
    _MODULE = module
    return module
