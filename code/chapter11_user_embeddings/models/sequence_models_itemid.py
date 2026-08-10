"""
Sequence Models with Learnable Item ID Embeddings (Section 11.2 Variant)

This module provides SASRec and GRU4Rec variants that use LEARNABLE item ID
embeddings instead of frozen SBERT embeddings from Chapter 10.

Key architectural difference from sequence_models.py:
  - Original:  Input = frozen 384-dim SBERT → Linear(384, 128) projection
  - This file: Input = nn.Embedding(num_items, 128)  ← LEARNABLE

Everything else is IDENTICAL: same Transformer/GRU architecture, same
128 → 384 output projection, same L2-normalization, same forward interface.

The output still maps to 384-dim and is trained via MNR loss against frozen
SBERT target embeddings.  This means:
  - The model learns item representations OPTIMIZED for next-item prediction
  - But the output space is still aligned to SBERT (for FAISS retrieval)
  - We can directly compare against the frozen-embedding variants

Why this matters (pedagogical):
  - Frozen SBERT embeddings capture CONTENT similarity ("blue blanket" ≈ "grey throw")
  - Learnable IDs capture BEHAVIORAL co-occurrence ("phone case" → "screen protector")
  - On MIND, frozen SBERT puts many news articles in dense topic clusters,
    causing mode collapse with MNR loss.  Learnable IDs can differentiate
    articles within the same topic based on behavioral patterns.

Trade-off:
  - Learnable IDs cannot generalize to unseen items (cold-start problem)
  - Frozen embeddings work for any item with text metadata
  - This is the classic expressiveness vs. generalization trade-off

Parameter counts (approximate, Amazon with 471K items):
  SASRec-ID:  ~60.7M params (dominated by nn.Embedding: 471K × 128 = 60.3M)
  GRU4Rec-ID: ~60.5M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .sequence_models import SASRecBlock


# ============================================================================
# SASRec with Learnable Item ID Embeddings
# ============================================================================

class SASRecItemID(nn.Module):
    """SASRec variant with learnable item ID embeddings.

    Replaces the Linear(384, 128) input projection with nn.Embedding(num_items, 128).
    All other components (Transformer blocks, output projection, positional
    embeddings) are identical to the original SASRec.

    Architecture:
        Item IDs (B, L) integer indices
          -> nn.Embedding(num_items, 128)   ← NEW: learnable
          -> + Learnable positional embeddings (max_len, 128)
          -> LayerNorm + Dropout
          -> N x TransformerBlock (same as original SASRec)
          -> Final LayerNorm
          -> Linear(128, output_dim) output projection
          -> L2-normalize
          -> User embedding at last valid position
    """

    def __init__(
        self,
        num_items: int,
        output_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        ffn_dim: int = 512,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # --- Learnable item embedding: replaces Linear(384, 128) ---
        # padding_idx=0 ensures padded positions get zero embeddings
        self.item_embedding = nn.Embedding(
            num_items, hidden_dim, padding_idx=padding_idx
        )

        # --- Learnable positional embeddings (same as original) ---
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # --- Pre-sequence LayerNorm + Dropout (same as original) ---
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # --- Transformer encoder blocks (SAME as original SASRec) ---
        self.transformer_blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Final LayerNorm (same as original) ---
        self.final_layernorm = nn.LayerNorm(hidden_dim)

        # --- Output projection: 128 → 384 (same as original) ---
        # Maps to SBERT space for FAISS retrieval
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights (same as original SASRec)."""
        # Item embedding: normal init with small std (standard for lookup tables)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        # Zero out padding index
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].fill_(0)

        # Position embedding
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Other layers: Xavier uniform
        for name, param in self.named_parameters():
            if 'item_embedding' in name or 'position_embedding' in name:
                continue  # Already initialized
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (same as original)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _generate_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create padding mask (same as original)."""
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def forward(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SASRec-ItemID.

        Args:
            item_ids: (B, L) integer item indices (0-padded).
                     Index 0 is reserved for padding.
            lengths: (B,) actual sequence lengths before padding.

        Returns:
            user_embeddings: (B, output_dim) L2-normalized user embeddings.
            all_hidden: (B, L, hidden_dim) hidden states at all positions.
        """
        B, L = item_ids.shape
        device = item_ids.device

        # Look up item embeddings: (B, L) -> (B, L, 128)
        hidden = self.item_embedding(item_ids)

        # Add positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        positions = positions.clamp(max=self.max_seq_len - 1)
        hidden = hidden + self.position_embedding(positions)

        # Pre-sequence LayerNorm + Dropout
        hidden = self.input_layernorm(hidden)
        hidden = self.input_dropout(hidden)

        # Generate masks
        causal_mask = self._generate_causal_mask(L, device)
        padding_mask = self._generate_padding_mask(lengths, L)

        # Transformer blocks (SAME as original)
        for block in self.transformer_blocks:
            hidden = block(hidden, causal_mask, padding_mask)

        # Final LayerNorm
        hidden = self.final_layernorm(hidden)
        all_hidden = hidden  # (B, L, 128)

        # Output projection: (B, L, 128) -> (B, L, 384)
        output = self.output_projection(hidden)

        # Extract at last valid position
        last_indices = (lengths - 1).long()
        batch_indices = torch.arange(B, device=device)
        user_embeddings = output[batch_indices, last_indices]

        # L2-normalize
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        return user_embeddings, all_hidden


# ============================================================================
# GRU4Rec with Learnable Item ID Embeddings
# ============================================================================

class GRU4RecItemID(nn.Module):
    """GRU4Rec variant with learnable item ID embeddings.

    Replaces the Linear(384, 128) input projection with nn.Embedding(num_items, 128).
    All other components (GRU layers, output projection) are identical.

    Architecture:
        Item IDs (B, L) integer indices
          -> nn.Embedding(num_items, 128)   ← NEW: learnable
          -> LayerNorm + Dropout
          -> GRU(input_size=128, hidden_size=128, num_layers=2, dropout=0.2)
          -> Linear(128, output_dim) output projection
          -> L2-normalize
          -> User embedding at last valid timestep
    """

    def __init__(
        self,
        num_items: int,
        output_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.num_items = num_items
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # --- Learnable item embedding ---
        self.item_embedding = nn.Embedding(
            num_items, hidden_dim, padding_idx=padding_idx
        )

        # --- Pre-GRU LayerNorm + Dropout (same as original) ---
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # --- Stacked GRU (SAME as original) ---
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Output projection: 128 → 384 (same as original) ---
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Item embedding
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].fill_(0)

        # GRU weights (same as original)
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GRU4Rec-ItemID.

        Args:
            item_ids: (B, L) integer item indices (0-padded).
            lengths: (B,) actual sequence lengths before padding.

        Returns:
            user_embeddings: (B, output_dim) L2-normalized user embeddings.
            all_hidden: (B, L, hidden_dim) hidden states at all timesteps.
        """
        B, L = item_ids.shape
        device = item_ids.device

        # Look up item embeddings: (B, L) -> (B, L, 128)
        hidden = self.item_embedding(item_ids)
        hidden = self.input_layernorm(hidden)
        hidden = self.input_dropout(hidden)

        # Pack sequences (same as original)
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.gru(packed)

        all_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=L
        )

        # Output projection: (B, L, 128) -> (B, L, 384)
        output = self.output_projection(all_hidden)

        # Extract at last valid timestep
        last_indices = (lengths - 1).long()
        batch_indices = torch.arange(B, device=device)
        user_embeddings = output[batch_indices, last_indices]

        # L2-normalize
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        return user_embeddings, all_hidden


# ============================================================================
# Model Factory
# ============================================================================

def create_itemid_model(
    model_type: str,
    num_items: int,
    output_dim: int = 384,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
    ffn_dim: int = 512,
    max_seq_len: int = 50,
    dropout: float = 0.2,
    padding_idx: int = 0,
) -> nn.Module:
    """Create an Item-ID sequence model by type name.

    Args:
        model_type: "sasrec" or "gru4rec"
        num_items: Size of item vocabulary (including padding index 0)
        Other args: architecture hyperparameters

    Returns:
        Instantiated nn.Module (SASRecItemID or GRU4RecItemID)
    """
    if model_type.lower() == "sasrec":
        return SASRecItemID(
            num_items=num_items,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )
    elif model_type.lower() == "gru4rec":
        return GRU4RecItemID(
            num_items=num_items,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'sasrec' or 'gru4rec'."
        )
