"""Seeding, device selection, logging and result writing shared by every script."""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def run_record(script: str, args: Any, **extra: Any) -> Dict[str, Any]:
    """Standard header for every metrics file: what ran, with which arguments, on what."""
    return {
        "script": script,
        "command": " ".join(sys.argv),
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "environment": environment_info(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
