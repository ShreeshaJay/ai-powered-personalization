"""
Session sequences of Semantic ID tokens for TIGER (Section 12.2).

Every item has a fixed-length Semantic ID of L tokens (the RQ-VAE levels plus the
collision suffix), so no end-of-sequence token is needed: decoding always stops after L
steps. Token IDs are offset per level so that level l's code c and level m's code c are
different tokens:

    PAD = 0, BOS = 1
    level l, code c  ->  NUM_SPECIAL + sum(level_sizes[:l]) + c

The encoder input is the flattened Semantic IDs of the most recent history items. History
is truncated by whole items, so the encoder never sees half an ID.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_TOKEN = 0
BOS_TOKEN = 1
NUM_SPECIAL_TOKENS = 2


def level_offsets(level_sizes: Sequence[int]) -> np.ndarray:
    return NUM_SPECIAL_TOKENS + np.concatenate([[0], np.cumsum(level_sizes)[:-1]]).astype(np.int64)


def vocab_size(level_sizes: Sequence[int]) -> int:
    return NUM_SPECIAL_TOKENS + int(sum(level_sizes))


def sids_to_tokens(semantic_ids: np.ndarray, level_sizes: Sequence[int]) -> np.ndarray:
    """(N, L) codes -> (N, L) token IDs. Raises on a code outside its level's range."""
    semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
    sizes = np.asarray(level_sizes, dtype=np.int64)
    if semantic_ids.shape[1] != len(sizes):
        raise ValueError(f"expected {len(sizes)} levels, got {semantic_ids.shape[1]}")
    if (semantic_ids < 0).any() or (semantic_ids >= sizes).any():
        raise ValueError("a Semantic ID code is outside its level's codebook")
    return semantic_ids + level_offsets(level_sizes)


def tokens_to_sids(tokens: np.ndarray, level_sizes: Sequence[int]) -> np.ndarray:
    """Inverse of ``sids_to_tokens``. Raises instead of clamping invalid tokens."""
    tokens = np.asarray(tokens, dtype=np.int64)
    codes = tokens - level_offsets(level_sizes)
    if (codes < 0).any() or (codes >= np.asarray(level_sizes)).any():
        raise ValueError("token outside its level's range")
    return codes


class SemanticIDTrie:
    """Prefix tree over the catalog's token sequences, for constrained decoding."""

    def __init__(self, item_tokens: np.ndarray):
        self.num_levels = item_tokens.shape[1]
        self.children: Dict[Tuple[int, ...], List[int]] = {}
        self.item_of: Dict[Tuple[int, ...], int] = {}
        for item, row in enumerate(map(tuple, item_tokens.tolist())):
            if row in self.item_of:
                raise ValueError(f"two items share the token sequence {row}; add the collision suffix")
            self.item_of[row] = item
            for depth in range(self.num_levels):
                self.children.setdefault(row[:depth], [])
        for row in self.item_of:
            for depth in range(self.num_levels):
                self.children[row[:depth]].append(row[depth])
        self.children = {k: sorted(set(v)) for k, v in self.children.items()}

    def allowed(self, prefix: Tuple[int, ...]) -> List[int]:
        return self.children.get(prefix, [])

    def item(self, tokens: Tuple[int, ...]) -> Optional[int]:
        return self.item_of.get(tokens)


class SemanticIDSessions(Dataset):
    """(history token sequence, next-item token sequence) pairs."""

    def __init__(self, histories: Sequence[Sequence[int]], targets: Sequence[int],
                 item_tokens: np.ndarray, max_history_items: int):
        self.histories = [list(h)[-max_history_items:] for h in histories]
        self.targets = np.asarray(targets, dtype=np.int64)
        self.item_tokens = torch.as_tensor(item_tokens, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.item_tokens[self.histories[idx]].reshape(-1)
        target = self.item_tokens[self.targets[idx]]
        return {
            "encoder_input": enc,
            "decoder_input": torch.cat([torch.tensor([BOS_TOKEN]), target[:-1]]),
            "decoder_target": target,
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Left-pad encoder inputs so the most recent item is always at the end."""
    max_len = max(len(b["encoder_input"]) for b in batch)
    enc = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["encoder_input"]
        enc[i, max_len - len(seq):] = seq
    return {
        "encoder_input": enc,
        "encoder_mask": enc != PAD_TOKEN,
        "decoder_input": torch.stack([b["decoder_input"] for b in batch]),
        "decoder_target": torch.stack([b["decoder_target"] for b in batch]),
    }


def load_semantic_ids(path, item_ids: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """Return (semantic_ids aligned to item_ids, level_sizes)."""
    data = np.load(path, allow_pickle=False)
    stored = {iid: i for i, iid in enumerate(data["item_ids"].tolist())}
    missing = [iid for iid in item_ids if iid not in stored]
    if missing:
        raise ValueError(f"{len(missing)} items have no Semantic ID; re-run train_rqvae.py")
    rows = np.array([stored[iid] for iid in item_ids])
    return data["semantic_ids"][rows], [int(s) for s in data["level_sizes"]]
