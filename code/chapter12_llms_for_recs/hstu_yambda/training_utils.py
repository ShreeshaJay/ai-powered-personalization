"""Training helpers aligned with the MLWhiz HSTU notebook."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HSTURecommender

# Defaults from hstu-from-scratch.ipynb / rectools reference.
W_RETRIEVAL = 1.0
W_RATING = 0.2
ADAM_BETAS = (0.9, 0.98)
GRAD_CLIP = 1.0
NORM_EPS = 1e-8


def init_hstu_weights(model: HSTURecommender) -> None:
    """Xavier normal init for 2D+ params; zero padding rows (matches notebook)."""
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param.data)
    with torch.no_grad():
        model.encoder.embed.item_emb.weight[0].zero_()
        model.encoder.embed.action_emb.weight[0].zero_()


def init_sasrec_weights(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param.data)
        if name in ("item_emb.weight", "item_embedding.weight"):
            with torch.no_grad():
                param[0].zero_()


def sampled_softmax_loss(h: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor,
                         pos_items: torch.Tensor, neg_items: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per-row cross-entropy of the positive against sampled negatives, with cosine logits.

    h (N, D), pos_emb (N, D), neg_emb (N, K, D). Negatives equal to the positive are masked.
    """
    h_n = F.normalize(h, dim=-1, eps=NORM_EPS)
    pos_logit = (h_n * F.normalize(pos_emb, dim=-1, eps=NORM_EPS)).sum(-1, keepdim=True) / temperature
    neg_logit = torch.einsum("nd,nkd->nk", h_n, F.normalize(neg_emb, dim=-1, eps=NORM_EPS)) / temperature
    neg_logit = neg_logit.masked_fill(neg_items.eq(pos_items.unsqueeze(-1)), float("-inf"))
    logits = torch.cat([pos_logit, neg_logit], dim=-1).float()
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device), reduction="none")


def compute_hstu_loss(
    model: HSTURecommender,
    input_items: torch.Tensor,
    input_actions: torch.Tensor,
    input_times: torch.Tensor,
    input_mask: torch.Tensor,
    target_items: torch.Tensor,
    target_values: torch.Tensor,
    target_mask: torch.Tensor,
    neg_items: torch.Tensor,
    w_retrieval: float = W_RETRIEVAL,
    w_rating: float = W_RATING,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-task loss: sampled-softmax retrieval + rating MSE (MLWhiz notebook).

    Only positions with a real next event count (input_mask & target_mask). The rating
    term also skips targets without a value (NaN), such as likes and dislikes on Yambda.
    """
    h = model.encoder(input_items, input_actions, input_times, input_mask)
    valid = input_mask & target_mask
    h_v = h[valid]
    pos = target_items[valid]
    neg = neg_items[valid]
    item_emb = model.item_embeddings()
    pos_emb = item_emb[pos]
    l_retrieval = sampled_softmax_loss(h_v, pos_emb, item_emb[neg], pos, neg, model.temperature).mean()

    values = target_values[valid]
    has_value = ~torch.isnan(values)
    if has_value.any():
        pred = model.rating_head(h_v[has_value] + pos_emb[has_value]).squeeze(-1).float()
        l_rating = F.mse_loss(pred, values[has_value])
    else:
        l_rating = h_v.new_zeros(())

    total = w_retrieval * l_retrieval + w_rating * l_rating
    return total, {
        "loss": float(total.detach()),
        "retrieval": float(l_retrieval.detach()),
        "rating": float(l_rating.detach()),
        "rated_fraction": float(has_value.float().mean()) if len(values) else 0.0,
    }
