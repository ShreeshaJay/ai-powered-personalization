"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
(He et al., SIGIR 2020)

Section 11.3 — Graph Neural Network for User Embeddings (MIND-only)

LightGCN is the simplest Graph Neural Network for recommendation:
it removes ALL feature transformations and nonlinearities from standard GCN,
relying solely on neighborhood aggregation via sparse matrix multiplication.

Architecture:
    E^{(0)}  = [user_emb; item_emb]           ← learnable nn.Embedding
    E^{(k+1)} = D^{-1/2} A D^{-1/2} E^{(k)}  ← parameter-free propagation
    E_final  = mean(E^{(0)}, ..., E^{(K)})    ← layer combination

The ONLY learnable parameters are the initial embedding tables:
    - user_embedding: nn.Embedding(num_users, hidden_dim)
    - item_embedding: nn.Embedding(num_items, hidden_dim)

No weight matrices, no activation functions, no dropout by default.
This simplicity makes LightGCN ideal for a textbook introduction to
graph-based recommendation.

Why this works for MIND (where sequence models failed):
    - Section 11.2 sequence models failed on MIND due to MNR loss causing
      mode collapse in dense SBERT topic clusters
    - LightGCN uses BPR loss (pairwise), which avoids dense similarity matrices
    - Learnable embeddings differentiate articles by behavioral co-occurrence,
      not content similarity
    - Graph propagation captures multi-hop collaborative patterns:
      "users who read A also read B"

Implementation: Pure PyTorch with torch.sparse.mm() — no torch-geometric.
The graph is small enough (~80K nodes) for full-batch GCN.

Parameter count:
    users: ~40K × 64 = 2.6M
    items: ~51K × 64 = 3.3M
    Total: ~5.9M (all in embeddings — no other parameters)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """LightGCN for bipartite user-item recommendation.

    The forward pass performs K rounds of parameter-free graph propagation
    via sparse matrix multiplication, then averages all K+1 layer embeddings
    to produce the final user and item representations.

    Args:
        num_users: Number of user nodes in the graph.
        num_items: Number of item nodes in the graph.
        hidden_dim: Embedding dimension (default: 64).
        num_layers: Number of GCN propagation layers (default: 3).
        dropout: Dropout on propagated embeddings (default: 0.0).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # The ONLY learnable parameters in LightGCN
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with normal distribution (LightGCN convention)."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

    def init_item_embeddings_from_pretrained(
        self,
        pretrained_matrix: np.ndarray,
    ) -> None:
        """Override item embeddings with pre-trained values (e.g., PCA-projected SBERT).

        This replaces the random N(0, 0.1) initialization with content-based
        embeddings from Chapter 10.  The embeddings remain TRAINABLE — this
        tests whether content-based initialization improves convergence or
        final metrics compared to random init.

        The pretrained_matrix must already be projected to hidden_dim
        (e.g., 384 → 64 via PCA).  The row ordering must match the
        item_id_to_idx mapping from MINDUserDataset, which is guaranteed
        when loading from the same .npz file used by mind_graph_builder.

        Args:
            pretrained_matrix: (num_items, hidden_dim) numpy array of
                pre-trained item embeddings, already projected to hidden_dim.

        Raises:
            ValueError: If shape doesn't match (num_items, hidden_dim).
        """
        expected_shape = (self.num_items, self.hidden_dim)
        if pretrained_matrix.shape != expected_shape:
            raise ValueError(
                f"Pretrained matrix shape {pretrained_matrix.shape} "
                f"doesn't match expected {expected_shape}"
            )

        with torch.no_grad():
            self.item_embedding.weight.copy_(
                torch.from_numpy(pretrained_matrix).float()
            )
        logger.info(
            f"  Initialized item_embedding with pre-trained vectors "
            f"(shape={pretrained_matrix.shape}, trainable=True)"
        )

    def forward(
        self,
        adj_norm: torch.sparse.FloatTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-batch GCN propagation over the bipartite graph.

        Computes K rounds of neighborhood aggregation and returns the
        layer-averaged embeddings for all users and items.

        Args:
            adj_norm: (N, N) symmetrically normalized adjacency matrix
                      where N = num_users + num_items.  Sparse FloatTensor.

        Returns:
            user_embeddings: (num_users, hidden_dim) final user embeddings.
            item_embeddings: (num_items, hidden_dim) final item embeddings.
        """
        # -----------------------------------------------------------------
        # Step 1: Build initial embedding matrix E^{(0)}
        #
        # Concatenate user and item embeddings into one big matrix.
        # Rows 0..num_users-1 = user embeddings
        # Rows num_users..N-1 = item embeddings
        # This matches the node ordering in the adjacency matrix.
        # -----------------------------------------------------------------
        all_embeddings = torch.cat([
            self.user_embedding.weight,   # (num_users, hidden_dim)
            self.item_embedding.weight,   # (num_items, hidden_dim)
        ], dim=0)  # (N, hidden_dim)

        # Save each layer's output for final averaging
        layer_embeddings = [all_embeddings]

        # -----------------------------------------------------------------
        # Step 2: K rounds of graph propagation
        #
        # Each round: every node's embedding becomes the weighted average
        # of its neighbors' embeddings (via sparse matrix multiplication).
        #
        # What this does intuitively:
        #   Layer 1: Users absorb info from articles they read.
        #            Articles absorb info from users who read them.
        #   Layer 2: Users get info from other users who read the same articles
        #            (2-hop neighbors: user → article → other users).
        #   Layer 3: 3-hop collaborative signal.
        #
        # The adjacency matrix is normalized, so popular articles and
        # heavy readers don't dominate the signal.
        # -----------------------------------------------------------------
        for k in range(self.num_layers):
            # E^{(k+1)} = A_norm @ E^{(k)}
            # This single sparse matrix multiply does ALL the neighbor
            # averaging for ALL nodes simultaneously.
            all_embeddings = torch.sparse.mm(adj_norm, all_embeddings)

            # Optional dropout (typically 0.0 for LightGCN — the original
            # paper uses no dropout because the model has no weight matrices
            # to overfit with; the only parameters are the embeddings)
            if self.training and self.dropout > 0:
                all_embeddings = F.dropout(
                    all_embeddings, p=self.dropout, training=True
                )

            layer_embeddings.append(all_embeddings)

        # -----------------------------------------------------------------
        # Step 3: Layer combination — average all layers
        #
        # E_final = (1 / (K+1)) * (E^{(0)} + E^{(1)} + ... + E^{(K)})
        #
        # Why average instead of just using the last layer?
        #   - E^{(0)} = the node's own identity (no neighbor info)
        #   - E^{(1)} = direct neighbors (1-hop)
        #   - E^{(2)} = 2-hop neighborhood
        #   - E^{(3)} = 3-hop neighborhood
        # Averaging combines local identity with increasingly broad
        # collaborative signal — each layer captures a different scale.
        # -----------------------------------------------------------------
        stacked = torch.stack(layer_embeddings, dim=0)  # (K+1, N, hidden_dim)
        final_embeddings = stacked.mean(dim=0)            # (N, hidden_dim)

        # -----------------------------------------------------------------
        # Step 4: Split back into user and item embeddings
        # (reverse the concatenation from Step 1)
        # -----------------------------------------------------------------
        user_embeddings = final_embeddings[:self.num_users]
        item_embeddings = final_embeddings[self.num_users:]

        return user_embeddings, item_embeddings

    # ========================================================================
    # Loss Functions
    # ========================================================================

    @staticmethod
    def compute_bpr_loss(
        user_embs: torch.Tensor,
        pos_item_embs: torch.Tensor,
        neg_item_embs: torch.Tensor,
    ) -> torch.Tensor:
        """BPR (Bayesian Personalized Ranking) loss.

        loss = -log(sigmoid(score_pos - score_neg))

        This pairwise loss avoids the dense similarity matrix that causes
        mode collapse with MNR loss on MIND's dense topic clusters.

        Args:
            user_embs: (B, hidden_dim) user embeddings.
            pos_item_embs: (B, hidden_dim) positive item embeddings.
            neg_item_embs: (B, hidden_dim) negative item embeddings.

        Returns:
            Scalar loss tensor.
        """
        pos_scores = (user_embs * pos_item_embs).sum(dim=1)  # (B,)
        neg_scores = (user_embs * neg_item_embs).sum(dim=1)  # (B,)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    @staticmethod
    def compute_l2_reg(
        user_embs_0: torch.Tensor,
        pos_item_embs_0: torch.Tensor,
        neg_item_embs_0: torch.Tensor,
    ) -> torch.Tensor:
        """L2 regularization on INITIAL (pre-GCN) embeddings.

        LightGCN regularizes E^{(0)}, not the propagated embeddings,
        because graph propagation is parameter-free — all learnable
        parameters live in the initial embedding tables.

        Args:
            user_embs_0: (B, hidden_dim) raw user embeddings from E^{(0)}.
            pos_item_embs_0: (B, hidden_dim) raw positive item embeddings.
            neg_item_embs_0: (B, hidden_dim) raw negative item embeddings.

        Returns:
            Scalar regularization loss (mean over batch).
        """
        reg = (
            user_embs_0.norm(2).pow(2)
            + pos_item_embs_0.norm(2).pow(2)
            + neg_item_embs_0.norm(2).pow(2)
        ) / user_embs_0.shape[0]
        return reg

    @staticmethod
    def compute_mnr_loss(
        user_embs: torch.Tensor,
        pos_item_embs: torch.Tensor,
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """MNR (Multiple Negatives Ranking) loss for comparison.

        Uses in-batch negatives, same as Section 11.2 sequence models.
        Expected to FAIL on MIND (mode collapse in dense topic clusters).
        Included for pedagogical comparison: confirms that the loss function
        is the root cause, not the model architecture.

        Args:
            user_embs: (B, hidden_dim) user embeddings.
            pos_item_embs: (B, hidden_dim) positive item embeddings.
            temperature: Softmax temperature (default: 0.05).

        Returns:
            Scalar loss tensor.
        """
        user_norm = F.normalize(user_embs, p=2, dim=1)
        item_norm = F.normalize(pos_item_embs, p=2, dim=1)
        similarity = torch.mm(user_norm, item_norm.t()) / temperature  # (B, B)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        return F.cross_entropy(similarity, labels)


# ============================================================================
# Factory Function
# ============================================================================

def create_lightgcn(
    num_users: int,
    num_items: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.0,
) -> LightGCN:
    """Create a LightGCN model.

    Args:
        num_users: Number of user nodes.
        num_items: Number of item nodes.
        hidden_dim: Embedding dimension (default: 64).
        num_layers: GCN propagation layers (default: 3).
        dropout: Dropout rate (default: 0.0).

    Returns:
        Instantiated LightGCN module.
    """
    return LightGCN(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
