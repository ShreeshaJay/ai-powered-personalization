"""
TIGER-style generative retrieval (Rajput et al., 2023), Section 12.2.

An encoder reads the flattened Semantic IDs of the session history; a decoder emits the
next item's Semantic ID one level at a time. Each decoding step is a softmax over that
level's codebook only.

Retrieval uses beam search constrained by a prefix tree of the catalog's Semantic IDs:
at every step a beam may only extend to a token that leads to at least one real item.
Every finished beam is therefore a valid ID of exactly one item, and no lookup fallback
is needed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.semantic_id_dataset import BOS_TOKEN, PAD_TOKEN, SemanticIDTrie, level_offsets, vocab_size


class TIGER(nn.Module):
    def __init__(self, level_sizes: Sequence[int], hidden_dim: int = 128, num_heads: int = 4,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4, ffn_dim: int = 1024,
                 dropout: float = 0.1, max_history_items: int = 15):
        super().__init__()
        self.level_sizes = [int(s) for s in level_sizes]
        self.num_levels = len(self.level_sizes)
        self.offsets = level_offsets(self.level_sizes).tolist()
        self.vocab_size = vocab_size(self.level_sizes)
        self.max_history_items = max_history_items

        self.token_embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.encoder_position = nn.Embedding(max_history_items * self.num_levels, hidden_dim)
        self.decoder_position = nn.Embedding(self.num_levels, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, ffn_dim, dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers, norm=nn.LayerNorm(hidden_dim),
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers, norm=nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.token_embedding.weight[PAD_TOKEN].zero_()

    def encode(self, encoder_input: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        length = encoder_input.shape[1]
        if length > self.encoder_position.num_embeddings:
            raise ValueError("history longer than max_history_items * num_levels tokens")
        # Positions count back from the most recent token, so left padding does not shift them.
        positions = torch.arange(length - 1, -1, -1, device=encoder_input.device)
        x = self.token_embedding(encoder_input) + self.encoder_position(positions)
        return self.encoder(self.dropout(x), src_key_padding_mask=~encoder_mask)

    def decode(self, memory: torch.Tensor, encoder_mask: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        steps = decoder_input.shape[1]
        positions = torch.arange(steps, device=decoder_input.device)
        y = self.token_embedding(decoder_input) + self.decoder_position(positions)
        causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=y.device), diagonal=1)
        h = self.decoder(self.dropout(y), memory, tgt_mask=causal, memory_key_padding_mask=~encoder_mask)
        return self.output(h)

    def level_logits(self, logits: torch.Tensor, level: int) -> torch.Tensor:
        start = self.offsets[level]
        return logits[..., start:start + self.level_sizes[level]]

    def forward(self, encoder_input, encoder_mask, decoder_input) -> torch.Tensor:
        return self.decode(self.encode(encoder_input, encoder_mask), encoder_mask, decoder_input)

    def loss(self, batch: Dict[str, torch.Tensor], level_weights: Sequence[float],
             label_smoothing: float = 0.0) -> Tuple[torch.Tensor, List[float]]:
        logits = self(batch["encoder_input"], batch["encoder_mask"], batch["decoder_input"])
        total = logits.new_zeros(())
        per_level = []
        for level in range(self.num_levels):
            target = batch["decoder_target"][:, level] - self.offsets[level]
            ce = F.cross_entropy(self.level_logits(logits[:, level], level), target,
                                 label_smoothing=label_smoothing)
            total = total + level_weights[level] * ce
            per_level.append(float(ce.detach()))
        return total / sum(level_weights), per_level

    @torch.no_grad()
    def constrained_beam_search(self, encoder_input: torch.Tensor, encoder_mask: torch.Tensor,
                                trie: SemanticIDTrie, beam_width: int) -> List[List[Tuple[int, float]]]:
        """Return, per session, up to ``beam_width`` (item index, log-probability) pairs, best first."""
        self.eval()
        batch, device = encoder_input.shape[0], encoder_input.device
        memory = self.encode(encoder_input, encoder_mask)
        memory = memory.repeat_interleave(beam_width, dim=0)
        mem_mask = encoder_mask.repeat_interleave(beam_width, dim=0)

        tokens = torch.full((batch * beam_width, 1), BOS_TOKEN, dtype=torch.long, device=device)
        scores = torch.full((batch, beam_width), float("-inf"), device=device)
        scores[:, 0] = 0.0  # one live beam per session at the start
        prefixes: List[Tuple[int, ...]] = [()] * (batch * beam_width)

        for level in range(self.num_levels):
            logits = self.decode(memory, mem_mask, tokens)[:, -1]
            log_probs = F.log_softmax(self.level_logits(logits, level).float(), dim=-1)
            size = self.level_sizes[level]
            mask = torch.full_like(log_probs, float("-inf"))
            offset = self.offsets[level]
            for row, prefix in enumerate(prefixes):
                allowed = trie.allowed(prefix)
                if allowed:
                    mask[row, torch.as_tensor(allowed, device=device) - offset] = 0.0
            cand = (scores.reshape(-1, 1) + log_probs + mask).reshape(batch, beam_width * size)
            k = min(beam_width, cand.shape[1])
            top_scores, top_idx = cand.topk(k, dim=1)
            if k < beam_width:
                pad = torch.full((batch, beam_width - k), float("-inf"), device=device)
                top_scores = torch.cat([top_scores, pad], dim=1)
                top_idx = torch.cat([top_idx, torch.zeros_like(pad, dtype=torch.long)], dim=1)
            parent = top_idx // size
            new_token = top_idx % size + offset
            flat_parent = (parent + torch.arange(batch, device=device).unsqueeze(1) * beam_width).reshape(-1)
            tokens = torch.cat([tokens[flat_parent], new_token.reshape(-1, 1)], dim=1)
            prefixes = [prefixes[p] + (int(t),) for p, t in zip(flat_parent.tolist(), new_token.reshape(-1).tolist())]
            scores = top_scores

        results: List[List[Tuple[int, float]]] = []
        for b in range(batch):
            row: List[Tuple[int, float]] = []
            for j in range(beam_width):
                score = float(scores[b, j])
                if np.isfinite(score):
                    item = trie.item(prefixes[b * beam_width + j])
                    if item is not None:
                        row.append((item, score))
            results.append(row)
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
