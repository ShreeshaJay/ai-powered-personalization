"""Shared utilities: reproducibility, metrics, logging."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


@dataclass
class RunMetadata:
    model: str
    seed: int
    data_dir: str
    max_days: int
    max_seq_len: int
    dim: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    num_epochs: int
    num_negatives: int
    temperature: float
    hardware: Dict[str, Any]
    dataset_stats: Dict[str, Any]
    wall_clock_seconds: float
    metrics: Dict[str, float]


def save_run_metadata(path: Path, metadata: RunMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)


def hit_rate_at_k(ranks: Iterable[int], k: int = 10) -> float:
    ranks_list = list(ranks)
    return sum(1 for r in ranks_list if r < k) / max(len(ranks_list), 1)


def ndcg_at_k(ranks: Iterable[int], k: int = 10) -> float:
    ranks_list = list(ranks)
    if not ranks_list:
        return 0.0
    return float(np.mean([1.0 / np.log2(r + 2) if r < k else 0.0 for r in ranks_list]))


def compute_rank(scores: torch.Tensor, target: int) -> int:
    """0-based rank of ``target`` in descending score order; tied items count as ranked above it."""
    t = scores[target]
    return int((scores > t).sum() + (scores == t).sum() - 1)


@torch.no_grad()
def full_catalog_ranks(scores: torch.Tensor, targets: torch.Tensor, history: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Ranks of each target among items 1..N.

    scores (B, N + 1) with column 0 the padding item; history (B, T) padded with 0.
    Returns ``masked`` (items in the history window removed; a target inside the window
    gets rank N, a miss), ``unmasked`` (nothing removed) and ``target_in_window``.
    Ties count against the target.
    """
    scores = scores.float().clone()
    scores[:, 0] = float("-inf")
    t = scores.gather(1, targets[:, None])
    unmasked = (scores > t).sum(1) + (scores == t).sum(1) - 1

    seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, history, True)
    seen[:, 0] = True
    in_window = seen.gather(1, targets[:, None]).squeeze(1)
    masked_scores = scores.masked_fill(seen, float("-inf"))
    tm = masked_scores.gather(1, targets[:, None])
    masked = (masked_scores > tm).sum(1) + (masked_scores == tm).sum(1) - 1
    masked = torch.where(in_window, torch.full_like(masked, scores.size(1) - 1), masked)
    return {"masked": masked, "unmasked": unmasked, "target_in_window": in_window}


def summarize_ranks(masked: List[int], unmasked: List[int], in_window: List[bool], k: int = 10) -> Dict[str, float]:
    """HR@k / NDCG@k under three views of the same evaluation samples.

    ``hr@k`` / ``ndcg@k``: seen-item masking, all samples (the protocol of the original
    exploration; repeat targets count as misses). ``unmasked_*``: no masking, so repeats
    can be hit. ``novel_*``: masked ranks restricted to samples whose target is not in
    the history window.
    """
    novel = [r for r, seen in zip(masked, in_window) if not seen]
    return {
        f"hr@{k}": hit_rate_at_k(masked, k),
        f"ndcg@{k}": ndcg_at_k(masked, k),
        f"unmasked_hr@{k}": hit_rate_at_k(unmasked, k),
        f"unmasked_ndcg@{k}": ndcg_at_k(unmasked, k),
        f"novel_hr@{k}": hit_rate_at_k(novel, k),
        f"novel_ndcg@{k}": ndcg_at_k(novel, k),
        "num_samples": len(masked),
        "num_novel_samples": len(novel),
        "target_in_window_rate": sum(in_window) / max(len(in_window), 1),
    }


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
