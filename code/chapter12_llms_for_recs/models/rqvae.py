"""
Residual-Quantised autoencoder (RQ-VAE) for Semantic IDs, after TIGER (Rajput et al., 2023).

Each item embedding x is encoded to z_e, then quantised level by level: level l picks the
nearest codeword to the residual the earlier levels left behind.

    Encoder:  Linear(d_in, h) -> ReLU -> Linear(h, d_z)                     z_e
    Level l:  c_l = argmin_k || r_{l-1} - e_{l,k} ||,   r_l = r_{l-1} - e_{l,c_l},  r_0 = z_e
    Decoder:  z_q = sum_l e_{l,c_l}  ->  Linear(d_z, h) -> ReLU -> Linear(h, d_in)  x_hat

    loss = MSE(x, x_hat) + beta * sum_l MSE(r_{l-1}, e_{l,c_l})

Codebooks are not trained by gradient descent. Each codeword is an exponential moving
average of the residuals assigned to it. Two details keep codebooks from collapsing onto a
handful of entries:

* k-means initialisation: codewords start at k-means centroids of real residuals, so
  every entry begins near data.
* dead-code reset: an entry whose moving-average usage falls below a fraction of the
  average entry's usage is moved to a randomly chosen residual from the current batch.

The decoder input uses one straight-through estimator, z_e + (z_q - z_e).detach(), so the
reconstruction gradient reaches the encoder once. The commitment terms reach the encoder
through the residuals r_{l-1}, which depend on z_e.

Items whose codes are identical after quantisation (collisions) are separated by an extra
final token, 0, 1, 2, ..., as TIGER does; see ``add_collision_suffix``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def kmeans(x: torch.Tensor, k: int, iters: int, generator: torch.Generator) -> torch.Tensor:
    """Lloyd's k-means with random initial centroids; empty clusters are re-seeded."""
    n = x.shape[0]
    if n < k:
        extra = x[torch.randint(n, (k - n,), generator=generator)]
        x_init = torch.cat([x, extra + 1e-3 * torch.randn(extra.shape, generator=generator)], dim=0)
        return x_init
    centroids = x[torch.randperm(n, generator=generator)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        counts = torch.bincount(assign, minlength=k).float()
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        empty = counts == 0
        centroids = sums / counts.clamp(min=1).unsqueeze(1)
        if empty.any():
            centroids[empty] = x[torch.randint(n, (int(empty.sum()),), generator=generator)]
    return centroids


class VectorQuantizer(nn.Module):
    """One quantisation level with an EMA-updated codebook and dead-code reset."""

    def __init__(self, codebook_size: int, dim: int, ema_decay: float = 0.99,
                 commitment_weight: float = 0.25, dead_code_fraction: float = 0.1,
                 eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight
        self.dead_code_fraction = dead_code_fraction
        self.eps = eps
        self.register_buffer("codebook", torch.randn(codebook_size, dim) * 0.01)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.clone())
        self.register_buffer("initialized", torch.tensor(False))
        self.resets = 0

    @torch.no_grad()
    def init_codebook(self, residuals: torch.Tensor, iters: int, generator: torch.Generator) -> None:
        centroids = kmeans(residuals.float(), self.codebook_size, iters, generator)
        self.codebook.copy_(centroids)
        self.ema_embed_sum.copy_(centroids)
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def assign(self, z: torch.Tensor) -> torch.Tensor:
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        return distances.argmin(dim=1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (codewords without gradient, codes, commitment loss)."""
        codes = self.assign(z.detach())
        if self.training:
            self._ema_update(z.detach(), codes)
        z_q = self.codebook[codes]
        commitment = self.commitment_weight * F.mse_loss(z, z_q)
        return z_q, codes, commitment

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, codes: torch.Tensor) -> None:
        one_hot = F.one_hot(codes, self.codebook_size).type(z.dtype)
        self.ema_cluster_size.mul_(self.ema_decay).add_(one_hot.sum(0), alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(one_hot.t() @ z, alpha=1 - self.ema_decay)
        # Additive smoothing only keeps the division finite for rarely used entries; it does
        # not revive dead codes. The reset below does that.
        n = self.ema_cluster_size.sum()
        size = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        self.codebook.copy_(self.ema_embed_sum / size.unsqueeze(1))

        dead = self.ema_cluster_size < self.dead_code_fraction * self.ema_cluster_size.mean()
        num_dead = int(dead.sum())
        if num_dead:
            pick = torch.randint(z.shape[0], (num_dead,), device=z.device)
            fresh = z[pick] + 1e-4 * torch.randn(num_dead, self.dim, device=z.device, dtype=z.dtype)
            self.codebook[dead] = fresh
            self.ema_embed_sum[dead] = fresh
            self.ema_cluster_size[dead] = 1.0
            self.resets += num_dead


class RQVAE(nn.Module):
    def __init__(self, input_dim: int, encoder_hidden_dim: int = 256, latent_dim: int = 64,
                 num_levels: int = 3, codebook_size: int = 256, ema_decay: float = 0.99,
                 commitment_weight: float = 0.25, dead_code_fraction: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim), nn.ReLU(), nn.Linear(encoder_hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim), nn.ReLU(), nn.Linear(encoder_hidden_dim, input_dim)
        )
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, latent_dim, ema_decay, commitment_weight, dead_code_fraction)
            for _ in range(num_levels)
        ])

    @torch.no_grad()
    def initialize_codebooks(self, x: torch.Tensor, iters: int = 20, seed: int = 42) -> None:
        """k-means initialisation, level by level, on the residuals of a data sample."""
        generator = torch.Generator().manual_seed(seed)
        was_training = self.training
        self.eval()
        residual = self.encoder(x).cpu()
        for quantizer in self.quantizers:
            quantizer.init_codebook(residual, iters, generator)
            residual = residual - quantizer.codebook.cpu()[quantizer.assign(residual.to(quantizer.codebook.device)).cpu()]
        self.train(was_training)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_e, z_q, codes, commitment) where z_q is the sum of chosen codewords."""
        z_e = self.encoder(x)
        residual = z_e
        z_q = torch.zeros_like(z_e)
        codes: List[torch.Tensor] = []
        commitment = z_e.new_zeros(())
        for quantizer in self.quantizers:
            codeword, level_codes, level_loss = quantizer(residual)
            z_q = z_q + codeword
            residual = residual - codeword
            codes.append(level_codes)
            commitment = commitment + level_loss
        return z_e, z_q, torch.stack(codes, dim=1), commitment

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e, z_q, codes, commitment = self.encode(x)
        z_st = z_e + (z_q - z_e).detach()
        x_hat = self.decoder(z_st)
        reconstruction = F.mse_loss(x_hat, x)
        return {"x_hat": x_hat, "codes": codes, "reconstruction_loss": reconstruction,
                "commitment_loss": commitment, "total_loss": reconstruction + commitment}

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
        was_training = self.training
        self.eval()
        out = [self.encode(x[i:i + batch_size])[2].cpu() for i in range(0, len(x), batch_size)]
        self.train(was_training)
        return torch.cat(out).numpy()

    def count_parameters(self) -> Dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        books = sum(q.codebook.numel() for q in self.quantizers)
        return {"encoder": enc, "decoder": dec, "codebook_entries": books, "trainable": enc + dec}


def add_collision_suffix(codes: np.ndarray) -> Tuple[np.ndarray, int]:
    """Append a final level that numbers items sharing the same codes: 0, 1, 2, ...

    Returns (semantic_ids of shape (N, L + 1), suffix vocabulary size). Items keep their
    input order within a collision group, so the result is deterministic.
    """
    seen: Dict[Tuple[int, ...], int] = {}
    suffix = np.zeros(len(codes), dtype=np.int64)
    for i, row in enumerate(map(tuple, codes.tolist())):
        suffix[i] = seen.get(row, 0)
        seen[row] = suffix[i] + 1
    return np.concatenate([codes, suffix[:, None]], axis=1), int(suffix.max()) + 1


def code_statistics(codes: np.ndarray, codebook_size: int) -> Dict[str, object]:
    """Usage computed from actual assignments over the whole catalog."""
    n, levels = codes.shape
    stats: Dict[str, object] = {"num_items": int(n), "levels": []}
    for level in range(levels):
        counts = np.bincount(codes[:, level], minlength=codebook_size)
        p = counts[counts > 0] / n
        prefix = {tuple(r) for r in codes[:, :level + 1].tolist()}
        stats["levels"].append({
            "level": level + 1,
            "codes_used": int((counts > 0).sum()),
            "codebook_size": int(codebook_size),
            "perplexity": float(np.exp(-(p * np.log(p)).sum())),
            "distinct_prefixes": len(prefix),
        })
    groups: Dict[Tuple[int, ...], int] = {}
    for row in map(tuple, codes.tolist()):
        groups[row] = groups.get(row, 0) + 1
    in_collision = sum(c for c in groups.values() if c > 1)
    stats["distinct_full_codes"] = len(groups)
    stats["items_sharing_codes_with_another_item"] = float(in_collision / n)
    stats["largest_collision_group"] = int(max(groups.values()))
    return stats
