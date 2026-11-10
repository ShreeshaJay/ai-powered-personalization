"""
MLP re-ranker over frozen embedding tables (Section 12.4).

Features are computed on the fly from index tensors, so no feature matrix is
materialised. All tables share one text-encoder space (checked when they are loaded).

Feature sets, each a superset of the previous one:

    content                    candidate content vector; cosine to the last history item
    content_history            + mean of the history content vectors, its elementwise
                                 product with the candidate, and their cosine
    content_history_llm_item   + the candidate's LLM item-knowledge vector, its cosine to
                                 the history's knowledge vectors and to the history mean,
                                 and a flag for whether the knowledge vector exists
    full                       + the LLM session-intent vector, its cosines to the
                                 candidate's content and knowledge vectors, and a flag

The history representation is a plain mean of item vectors. It is not a sequential
model (SASRec or similar); Chapter 11 covers those.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_SETS = ("content", "content_history", "content_history_llm_item", "full")


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=1e-8).unsqueeze(-1)


def _masked_mean(table: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vecs = table[idx] * mask.unsqueeze(-1)
    return vecs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


class AugmentedRanker(nn.Module):
    def __init__(self, feature_set: str, content: torch.Tensor,
                 llm_item: Optional[torch.Tensor] = None, llm_item_present: Optional[torch.Tensor] = None,
                 intent: Optional[torch.Tensor] = None, intent_present: Optional[torch.Tensor] = None,
                 hidden_dims: Sequence[int] = (256, 64), dropout: float = 0.2):
        super().__init__()
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.level = FEATURE_SETS.index(feature_set)
        d = content.shape[1]
        self.register_buffer("content", content.float())
        if self.level >= 2:
            if llm_item is None or llm_item_present is None:
                raise ValueError(f"{feature_set} needs LLM item-knowledge vectors")
            self.register_buffer("llm_item", llm_item.float())
            self.register_buffer("llm_item_present", llm_item_present.float())
        if self.level >= 3:
            if intent is None or intent_present is None:
                raise ValueError("full needs LLM session-intent vectors")
            self.register_buffer("intent", intent.float())
            self.register_buffer("intent_present", intent_present.float())
        in_dim = [d + 1, 3 * d + 2, 4 * d + 5, 5 * d + 8][self.level]
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(nn.LayerNorm(in_dim), *layers)
        self.in_dim = in_dim

    def features(self, cand: torch.Tensor, hist: torch.Tensor, hist_mask: torch.Tensor,
                 session_row: torch.Tensor) -> torch.Tensor:
        """cand (B,), hist (B, H) padded item indices, hist_mask (B, H) float, session_row (B,)."""
        e_c = self.content[cand]
        last_pos = (hist_mask.sum(dim=1).long() - 1).clamp(min=0)
        e_last = self.content[hist.gather(1, last_pos.unsqueeze(1)).squeeze(1)]
        feats = [e_c, _cos(e_c, e_last)]
        if self.level >= 1:
            h = _masked_mean(self.content, hist, hist_mask)
            feats += [h, e_c * h, _cos(e_c, h)]
        if self.level >= 2:
            k_c = self.llm_item[cand]
            k_mask = hist_mask * self.llm_item_present[hist]
            k_h = _masked_mean(self.llm_item, hist, k_mask)
            feats += [k_c, _cos(k_c, k_h), _cos(k_c, h), self.llm_item_present[cand].unsqueeze(-1)]
        if self.level >= 3:
            s = self.intent[session_row]
            feats += [s, _cos(s, e_c), _cos(s, k_c), self.intent_present[session_row].unsqueeze(-1)]
        return torch.cat(feats, dim=-1)

    def forward(self, cand, hist, hist_mask, session_row) -> torch.Tensor:
        return self.mlp(self.features(cand, hist, hist_mask, session_row)).squeeze(-1)
