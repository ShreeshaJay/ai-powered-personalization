"""
Sequence Models for User Embeddings (Section 11.2)

Implements SASRec (Self-Attentive Sequential Recommendation) and GRU4Rec
for learning user embeddings from sequences of frozen item embeddings.

Both models are representative architectures of their paradigms:
  - SASRec  -> Transformer / self-attention family
  - GRU4Rec -> RNN / recurrence family

Students familiar with these two can explore variants such as BERT4Rec
(bidirectional masking), NARM (GRU + attention), or swap the loss function
for BPR or sampled softmax.

Architecture (shared by both models):
  Input (B, L, 384) frozen SBERT from Ch10
    -> Linear projection: 384 -> 128 (learnable)
    -> Sequence model (SASRec or GRU4Rec)
    -> Output projection: 128 -> 384 (learnable)
    -> L2-normalize
    -> User/Session embedding for FAISS retrieval

The output projection maps from the model's internal 128-dim space back to
the 384-dim item embedding space.  The MNR training loss explicitly forces
this projection to produce vectors that are cosine-similar to the frozen
SBERT item embeddings, so the trained user embeddings live in the same
semantic space as the Chapter 10 item embeddings.  This allows direct
retrieval using the existing FAISS index without rebuilding it.

Parameter counts (approximate):
  SASRec  (2 layers, 128 hidden, 2 heads): ~500K params (~2 MB)
  GRU4Rec (2 layers, 128 hidden):          ~300K params (~1.2 MB)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# ============================================================================
# SASRec: Self-Attentive Sequential Recommendation
# ============================================================================

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018).

    Uses a stack of Transformer encoder blocks with CAUSAL masking —
    each position can only attend to itself and previous positions.

    This is the key architectural difference from BERT4Rec: SASRec uses
    left-to-right (autoregressive) attention, which matches the production
    scenario where the system processes items as the user browses, without
    knowledge of future interactions.

    Architecture:
        Input SBERT (B, L, 384)
          -> Linear(384, 128)
          -> + Learnable positional embeddings (max_len, 128)
          -> LayerNorm + Dropout
          -> N x TransformerBlock:
               - Pre-LN Multi-Head Self-Attention (causal mask)
               - Pre-LN Feed-Forward Network (GELU activation)
               - Residual connections + Dropout
          -> Final LayerNorm
          -> Linear(128, 384) output projection
          -> L2-normalize
          -> User embedding at last valid position
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        ffn_dim: int = 512,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # --- Input projection: 384 -> 128 ---
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # --- Learnable positional embeddings ---
        # Short sequences (max 20-50) make learnable embeddings more
        # expressive than sinusoidal, with negligible parameter cost.
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # --- Pre-sequence LayerNorm + Dropout ---
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # --- Transformer encoder blocks ---
        # We use Pre-LN (LayerNorm before attention/FFN), following the
        # original SASRec implementation.  Pre-LN is more stable for
        # training without extensive learning rate tuning.
        self.transformer_blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Final LayerNorm ---
        self.final_layernorm = nn.LayerNorm(hidden_dim)

        # --- Output projection: 128 -> 384 ---
        # Maps back to the item embedding space so we can use the existing
        # FAISS index built from Chapter 10 embeddings.
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization (standard for Transformers)."""
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal (lower-triangular) attention mask.

        Returns a (seq_len, seq_len) boolean mask where True means
        "this position should NOT attend here" (following PyTorch convention
        for nn.MultiheadAttention's attn_mask parameter).

        Position i can attend to positions 0..i (inclusive).

           j=0  j=1  j=2  j=3
        i=0 [ F    T    T    T ]    i=0 can only attend to j=0
        i=1 [ F    F    T    T ]    i=1 can attend to j=0,1
        i=2 [ F    F    F    T ]    i=2 can attend to j=0,1,2
        i=3 [ F    F    F    F ]    i=3 can attend to j=0,1,2,3

        (F = attend, T = mask out)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def _generate_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Create a padding mask where True means "this position is padding".

        Args:
            lengths: (B,) actual sequence lengths
            max_len: maximum sequence length in the batch

        Returns:
            (B, max_len) boolean mask, True for padded positions
        """
        batch_size = lengths.size(0)
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
        padding_mask = positions >= lengths.unsqueeze(1)  # (B, max_len)
        return padding_mask

    def forward(
        self,
        item_embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SASRec.

        Args:
            item_embeddings: (B, L, 384) padded sequences of frozen SBERT
                            embeddings, chronologically ordered (left = oldest,
                            right = most recent).
            lengths: (B,) actual sequence lengths before padding.

        Returns:
            user_embeddings: (B, 384) L2-normalized user embeddings at the
                            last valid position.  Ready for FAISS retrieval.
            all_hidden: (B, L, 128) hidden states at all positions.
                       Used for multi-position training loss.
        """
        B, L, _ = item_embeddings.shape
        device = item_embeddings.device

        # Project input: (B, L, 384) -> (B, L, 128)
        hidden = self.input_projection(item_embeddings)

        # Add positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        # Clamp positions to max_seq_len - 1 for safety
        positions = positions.clamp(max=self.max_seq_len - 1)
        hidden = hidden + self.position_embedding(positions)

        # Pre-sequence LayerNorm + Dropout
        hidden = self.input_layernorm(hidden)
        hidden = self.input_dropout(hidden)

        # Generate masks
        causal_mask = self._generate_causal_mask(L, device)         # (L, L)
        padding_mask = self._generate_padding_mask(lengths, L)      # (B, L)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden = block(hidden, causal_mask, padding_mask)

        # Final LayerNorm
        hidden = self.final_layernorm(hidden)

        # Save all hidden states (128-dim) for multi-position loss
        all_hidden = hidden  # (B, L, 128)

        # Output projection: (B, L, 128) -> (B, L, 384)
        output_384 = self.output_projection(hidden)

        # Extract embedding at the last valid position for each sequence
        # lengths is 1-indexed, so the last valid index is lengths - 1
        last_indices = (lengths - 1).long()  # (B,)
        batch_indices = torch.arange(B, device=device)
        user_embeddings = output_384[batch_indices, last_indices]  # (B, 384)

        # L2-normalize for cosine similarity / FAISS IndexFlatIP
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        return user_embeddings, all_hidden


class SASRecBlock(nn.Module):
    """Single Transformer encoder block with Pre-LayerNorm.

    Pre-LN means LayerNorm is applied BEFORE the attention/FFN sublayers,
    not after.  This is more training-stable and is used in the original
    SASRec implementation.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention (causal) -> Dropout -> + residual
          -> LayerNorm -> FFN (Linear -> GELU -> Linear) -> Dropout -> + residual
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()

        # Pre-LN for attention
        self.attn_layernorm = nn.LayerNorm(hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (B, L, D) format
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-LN for FFN
        self.ffn_layernorm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (B, L, D) input hidden states
            causal_mask: (L, L) causal attention mask
            padding_mask: (B, L) padding mask

        Returns:
            (B, L, D) output hidden states
        """
        # Self-attention with residual
        residual = hidden
        hidden = self.attn_layernorm(hidden)
        attn_output, _ = self.multihead_attn(
            query=hidden,
            key=hidden,
            value=hidden,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
        )
        hidden = residual + self.attn_dropout(attn_output)

        # FFN with residual
        residual = hidden
        hidden = self.ffn_layernorm(hidden)
        hidden = residual + self.ffn(hidden)

        return hidden


# ============================================================================
# GRU4Rec: GRU-Based Sequential Recommendation
# ============================================================================

class GRU4Rec(nn.Module):
    """GRU-based Sequential Recommendation (Hidasi et al., ICLR 2016).

    Uses a stack of GRU layers to process sequences recurrently.  The GRU
    naturally handles variable-length sequences via packed representations,
    without needing explicit positional encoding or causal masks.

    Architecture:
        Input SBERT (B, L, 384)
          -> Linear(384, 128)
          -> LayerNorm + Dropout
          -> GRU(input_size=128, hidden_size=128, num_layers=2, dropout=0.2)
          -> Linear(128, 384) output projection
          -> L2-normalize
          -> User embedding at last valid timestep
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # --- Input projection: 384 -> 128 ---
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # --- Pre-GRU LayerNorm + Dropout ---
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # --- Stacked GRU ---
        # PyTorch's nn.GRU handles multi-layer stacking and inter-layer
        # dropout internally.  batch_first=True for (B, L, D) format.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Output projection: 128 -> 384 ---
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for GRU weights (standard practice)."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Linear layers
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GRU4Rec.

        Args:
            item_embeddings: (B, L, 384) padded sequences of frozen SBERT
                            embeddings, chronologically ordered.
            lengths: (B,) actual sequence lengths before padding.

        Returns:
            user_embeddings: (B, 384) L2-normalized user embeddings at the
                            last valid timestep.
            all_hidden: (B, L, 128) hidden states at all timesteps.
                       Used for multi-position training loss.
        """
        B, L, _ = item_embeddings.shape
        device = item_embeddings.device

        # Project input: (B, L, 384) -> (B, L, 128)
        hidden = self.input_projection(item_embeddings)
        hidden = self.input_layernorm(hidden)
        hidden = self.input_dropout(hidden)

        # Pack sequences for efficient variable-length processing
        # This avoids wasting computation on padding tokens.
        lengths_cpu = lengths.cpu().clamp(min=1)  # pack_padded_sequence needs CPU lengths
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        # GRU forward pass
        packed_output, _ = self.gru(packed)

        # Unpack back to padded format
        all_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=L,
        )  # (B, L, 128)

        # Output projection: (B, L, 128) -> (B, L, 384)
        output_384 = self.output_projection(all_hidden)

        # Extract embedding at the last valid timestep for each sequence
        last_indices = (lengths - 1).long()  # (B,)
        batch_indices = torch.arange(B, device=device)
        user_embeddings = output_384[batch_indices, last_indices]  # (B, 384)

        # L2-normalize for cosine similarity / FAISS IndexFlatIP
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        return user_embeddings, all_hidden


# ============================================================================
# Model Factory
# ============================================================================

def create_sequence_model(
    model_type: str,
    input_dim: int = 384,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
    ffn_dim: int = 512,
    max_seq_len: int = 50,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a sequence model by type name.

    Args:
        model_type: "sasrec" or "gru4rec"
        Other args: architecture hyperparameters

    Returns:
        Instantiated nn.Module (SASRec or GRU4Rec)
    """
    if model_type.lower() == "sasrec":
        return SASRec(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    elif model_type.lower() == "gru4rec":
        return GRU4Rec(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose 'sasrec' or 'gru4rec'."
        )


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
