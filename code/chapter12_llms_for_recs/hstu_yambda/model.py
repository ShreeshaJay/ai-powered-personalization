"""
HSTU (Hierarchical Sequential Transduction Unit) from scratch in PyTorch.

Architecture follows Rahul Agarwal's MLWhiz tutorial and cross-checks Meta's
generative-recommenders reference implementation:
  https://github.com/meta-recsys/generative-recommenders
  generative_recommenders/research/modeling/sequential/hstu.py
Licensed under Apache 2.0.

Each block, for input X (batch, n, d):
    U, V, Q, K = split(SiLU(Linear(LayerNorm(X))))
    A = SiLU(Q K^T + rab) * causal * key_padding / max_seq_len     (pointwise, no softmax)
    Y = X + Dropout(Linear(LayerNorm(A V) * U))

Simplifications against the reference: five hand-set time buckets instead of 128 log
buckets, a single attention head, and no jagged tensors. The attention normaliser is the
fixed max_seq_len, so a sequence's output does not depend on how long the other sequences
in its batch are.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedInputEmbedding(nn.Module):
    """Item + action embeddings fused through a small MLP."""

    def __init__(self, num_items: int, num_actions: int, dim: int) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.action_emb = nn.Embedding(num_actions + 1, dim, padding_idx=0)
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, item_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        item_vecs = self.item_emb(item_ids)
        action_vecs = self.action_emb(action_ids)
        return self.fuse(torch.cat([item_vecs, action_vecs], dim=-1))


class RelativeAttentionBias(nn.Module):
    """Learnable relative position and log-spaced time biases."""

    TIME_BUCKETS = [
        (0, 3600),
        (3600, 86_400),
        (86_400, 86_400 * 7),
        (86_400 * 7, 86_400 * 30),
        (86_400 * 30, float("inf")),
    ]

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_bias = nn.Embedding(2 * max_seq_len - 1, 1)
        self.time_bias = nn.Embedding(len(self.TIME_BUCKETS), 1)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.time_bias.weight)

    def _bucket(self, time_deltas: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(time_deltas, dtype=torch.long)
        for bucket_id, (lo, hi) in enumerate(self.TIME_BUCKETS):
            mask = (time_deltas >= lo) & (time_deltas < hi)
            out = torch.where(mask, torch.full_like(out, bucket_id), out)
        return out

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = times.shape
        del batch_size
        idx = torch.arange(seq_len, device=times.device)
        rel_pos = (idx.view(seq_len, 1) - idx.view(1, seq_len)) + (self.max_seq_len - 1)
        pos_b = self.pos_bias(rel_pos).squeeze(-1)
        time_deltas = (times.unsqueeze(2) - times.unsqueeze(1)).abs()
        time_b = self.time_bias(self._bucket(time_deltas)).squeeze(-1)
        return pos_b.unsqueeze(0) + time_b


class HSTUBlock(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.norm_in = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, 4 * dim)
        self.rab = RelativeAttentionBias(max_seq_len)
        self.norm = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        proj = F.silu(self.linear_in(self.norm_in(x)))
        k, q, v, u = proj.chunk(4, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) + self.rab(times)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))
        pad_mask = attn_mask.unsqueeze(1).to(dtype=x.dtype)
        activated = F.silu(scores) * causal * pad_mask
        attn_out = torch.matmul(activated, v) / self.max_seq_len

        gated = self.norm(attn_out) * u
        output = self.dropout(self.linear_out(gated))
        return output + x


class HSTUEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_actions: int,
        dim: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = FusedInputEmbedding(num_items, num_actions, dim)
        self.blocks = nn.ModuleList(
            [HSTUBlock(dim, max_seq_len, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        action_ids: torch.Tensor,
        times: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(item_ids, action_ids)
        for block in self.blocks:
            x = block(x, times, attn_mask)
        return x


class HSTURecommender(nn.Module):
    """
    HSTU encoder + sampled-softmax retrieval head (cosine similarity / temperature).
    Optional auxiliary head for played_ratio_pct regression.
    """

    def __init__(
        self,
        num_items: int,
        num_actions: int,
        dim: int = 64,
        num_layers: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.2,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.temperature = temperature
        self.encoder = HSTUEncoder(
            num_items=num_items,
            num_actions=num_actions,
            dim=dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.rating_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def encode_sequence(
        self,
        item_ids: torch.Tensor,
        action_ids: torch.Tensor,
        times: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(item_ids, action_ids, times, attn_mask)
        lengths = attn_mask.sum(dim=1).long().clamp(min=1)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_idx = lengths - 1
        return hidden[batch_idx, last_idx]

    def item_embeddings(self) -> torch.Tensor:
        return self.encoder.embed.item_emb.weight

    def score_items(self, seq_repr: torch.Tensor, item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        item_emb = self.item_embeddings()
        if item_ids is None:
            candidates = item_emb
        else:
            candidates = item_emb[item_ids]
        seq_norm = F.normalize(seq_repr, dim=-1, eps=1e-8)
        item_norm = F.normalize(candidates, dim=-1, eps=1e-8)
        if item_ids is None:
            return torch.matmul(seq_norm, item_norm.T) / self.temperature
        return (seq_norm * item_norm).sum(dim=-1) / self.temperature

    def retrieval_logits(self, user_state: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embeddings()
        us_n = F.normalize(user_state, dim=-1, eps=1e-8)
        emb_n = F.normalize(item_emb, dim=-1, eps=1e-8)
        return us_n @ emb_n.T

    @torch.no_grad()
    def predict_next(
        self,
        item_ids: torch.Tensor,
        action_ids: torch.Tensor,
        times: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_repr = self.encode_sequence(item_ids, action_ids, times, attn_mask)
        return self.score_items(seq_repr)
