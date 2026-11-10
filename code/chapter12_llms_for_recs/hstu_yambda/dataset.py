"""PyTorch Dataset helpers for HSTU / SASRec training."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from data_loader import HSTUDataBundle, SequenceSample


class NextItemDataset(Dataset):
    """A training window: history + target joined, so the loss can use every position."""

    def __init__(self, samples: List[SequenceSample], max_seq_len: int) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        items = (sample.items + [sample.target_item])[-(self.max_seq_len + 1):]
        actions = (sample.actions + [sample.target_action])[-(self.max_seq_len + 1):]
        times = (sample.times + [sample.target_time])[-(self.max_seq_len + 1):]
        values = (sample.values + [sample.target_value])[-(self.max_seq_len + 1):]
        return {
            "items": torch.tensor(items, dtype=torch.long),
            "actions": torch.tensor(actions, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
            "uid": torch.tensor(sample.uid, dtype=torch.long),
        }


class EvalDataset(Dataset):
    """Single-target evaluation sample (history only + one target item)."""

    def __init__(self, samples: List[SequenceSample], max_seq_len: int) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "items": torch.tensor(sample.items[-self.max_seq_len:], dtype=torch.long),
            "actions": torch.tensor(sample.actions[-self.max_seq_len:], dtype=torch.long),
            "times": torch.tensor(sample.times[-self.max_seq_len:], dtype=torch.long),
            "target_item": torch.tensor(sample.target_item, dtype=torch.long),
            "uid": torch.tensor(sample.uid, dtype=torch.long),
        }


def pad_1d(values: List[torch.Tensor], pad_value: float = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence; mask is True on real positions."""
    max_len = max(v.size(0) for v in values)
    batch_size = len(values)
    out = values[0].new_full((batch_size, max_len), pad_value)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, v in enumerate(values):
        out[i, : v.size(0)] = v
        mask[i, : v.size(0)] = True
    return out, mask


def collate_train(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    items, mask = pad_1d([b["items"] for b in batch], pad_value=0)
    actions, _ = pad_1d([b["actions"] for b in batch], pad_value=0)
    times, _ = pad_1d([b["times"] for b in batch], pad_value=0)
    values, _ = pad_1d([b["values"] for b in batch], pad_value=math.nan)
    return {
        "items": items,
        "actions": actions,
        "times": times,
        "values": values,
        "mask": mask,
        "uids": torch.stack([b["uid"] for b in batch]),
    }


def collate_eval(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    items, mask = pad_1d([b["items"] for b in batch], pad_value=0)
    actions, _ = pad_1d([b["actions"] for b in batch], pad_value=0)
    times, _ = pad_1d([b["times"] for b in batch], pad_value=0)
    return {
        "items": items,
        "actions": actions,
        "times": times,
        "mask": mask,
        "targets": torch.stack([b["target_item"] for b in batch]),
        "uids": torch.stack([b["uid"] for b in batch]),
    }


def sample_negatives(target_items: torch.Tensor, num_items: int, num_negatives: int,
                     max_rounds: int = 5) -> torch.Tensor:
    """Uniform negatives in [1, num_items] for each target, re-drawn where they equal the target.

    Returns (N, num_negatives). A collision that survives ``max_rounds`` re-draws is left in
    place; the losses mask any negative equal to its positive, so it never counts.
    """
    device = target_items.device
    n = target_items.numel()
    negatives = torch.randint(1, num_items + 1, (n, num_negatives), device=device)
    for _ in range(max_rounds):
        clash = negatives.eq(target_items.reshape(-1, 1))
        if not clash.any():
            break
        resample = torch.randint(1, num_items + 1, (n, num_negatives), device=device)
        negatives = torch.where(clash, resample, negatives)
    return negatives


def build_datasets(bundle: HSTUDataBundle, max_seq_len: int) -> Tuple[NextItemDataset, EvalDataset, EvalDataset]:
    return (
        NextItemDataset(bundle.train_samples, max_seq_len=max_seq_len),
        EvalDataset(bundle.val_samples, max_seq_len=max_seq_len),
        EvalDataset(bundle.test_samples, max_seq_len=max_seq_len),
    )
