"""
Section 11.3: LightGCN Training for MIND User Embeddings

Trains LightGCN on the bipartite user-item graph constructed from MIND
user click histories.  Uses BPR (Bayesian Personalized Ranking) loss
with per-edge negative sampling.

MIND-only — Amazon KDD has no persistent user IDs, so a bipartite
user-item graph cannot be constructed.

Why BPR instead of MNR?
    Section 11.2 demonstrated that MNR loss causes mode collapse on MIND
    when combined with dense SBERT topic clusters.  BPR is pairwise (not
    in-batch), so it avoids this problem entirely.

    An MNR variant is included for pedagogical comparison — it is expected
    to underperform, confirming the diagnosis.

Training procedure:
    1. Build bipartite graph from MIND behaviors (excluding eval users)
    2. Split edges 90/10 into train/val
    3. For each batch of edges:
       a. Full graph forward pass → all user & item embeddings
       b. Sample negative items for each positive edge
       c. BPR loss + L2 regularization on initial embeddings
       d. Backward pass + optimizer step
    4. Early stopping on validation BPR loss

Usage:
    python train_lightgcn.py --loss bpr           # Default: BPR loss
    python train_lightgcn.py --loss mnr           # MNR for comparison
    python train_lightgcn.py --loss both          # Train both variants
    python train_lightgcn.py --loss bpr --quick   # Smoke test (3 epochs)
"""

import argparse
import json
import time
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEFAULT_MIND_USER_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_LIGHTGCN_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
)
from data.mind_graph_builder import build_bipartite_graph, build_normalized_adjacency
from models.lightgcn import LightGCN, create_lightgcn, count_parameters


# ============================================================================
# Dataset: BPR Edge Sampling
# ============================================================================

class BPREdgeDataset(Dataset):
    """Dataset that yields (user, pos_item, neg_item) triples.

    Each training example starts from a known edge (user, pos_item)
    and samples a random negative item that this user has NOT interacted with.

    Negative sampling is done per-access for diversity across epochs.
    Rejection sampling terminates quickly: with ~51K items and median
    user history of ~11 articles, P(sampling a positive) < 0.02%.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        num_items: int,
        user_positive_items: Dict[int, Set[int]],
    ):
        self.edges = edges
        self.num_items = num_items
        self.user_positive_items = user_positive_items

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        user_idx, pos_item_idx = self.edges[idx]

        # Sample negative: random item NOT in user's positive set
        neg_item_idx = np.random.randint(0, self.num_items)
        positive_set = self.user_positive_items.get(user_idx, set())
        while neg_item_idx in positive_set:
            neg_item_idx = np.random.randint(0, self.num_items)

        return user_idx, pos_item_idx, neg_item_idx


def bpr_collate_fn(
    batch: List[Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate BPR triples into tensors."""
    users, pos_items, neg_items = zip(*batch)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long),
    )


# ============================================================================
# Training Epoch
# ============================================================================

def train_epoch_bpr(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l2_reg_weight: float = 1e-4,
) -> float:
    """One BPR training epoch.

    LightGCN does a FULL GRAPH forward pass per batch (the graph is small
    enough that this is efficient), then looks up embeddings for the
    batch's (user, pos_item, neg_item) triples.

    Args:
        model: LightGCN model.
        adj_norm: Normalized adjacency matrix (on device).
        dataloader: BPR edge DataLoader.
        optimizer: Optimizer.
        device: torch device.
        l2_reg_weight: Weight for L2 regularization on initial embeddings.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="  Training", leave=False)
    except ImportError:
        iterator = dataloader

    for user_ids, pos_ids, neg_ids in iterator:
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)

        # ---------------------------------------------------------------
        # Full graph forward pass: propagate ALL user+item embeddings
        # through K GCN layers.  This is repeated per batch because
        # the embedding weights change after each optimizer.step().
        # With ~80K nodes × 64-dim × 3 layers, each forward takes ~6ms.
        # ---------------------------------------------------------------
        all_user_embs, all_item_embs = model(adj_norm)

        # ---------------------------------------------------------------
        # Index into the full embedding matrices to get this batch's
        # (user, positive_item, negative_item) embeddings.
        # These are the POST-propagation embeddings (after GCN layers).
        # ---------------------------------------------------------------
        user_embs = all_user_embs[user_ids]       # (B, hidden_dim)
        pos_item_embs = all_item_embs[pos_ids]    # (B, hidden_dim)
        neg_item_embs = all_item_embs[neg_ids]    # (B, hidden_dim)

        # ---------------------------------------------------------------
        # BPR loss: encourage user·positive > user·negative
        # loss = -log(sigmoid(score_pos - score_neg))
        # ---------------------------------------------------------------
        bpr_loss = LightGCN.compute_bpr_loss(user_embs, pos_item_embs, neg_item_embs)

        # ---------------------------------------------------------------
        # L2 regularization on the INITIAL (pre-GCN, layer-0) embeddings.
        # We look up the RAW embedding vectors (not propagated) because
        # those are the actual learnable parameters.  This prevents them
        # from growing unbounded during training.
        # ---------------------------------------------------------------
        user_embs_0 = model.user_embedding(user_ids)
        pos_embs_0 = model.item_embedding(pos_ids)
        neg_embs_0 = model.item_embedding(neg_ids)
        l2_loss = LightGCN.compute_l2_reg(user_embs_0, pos_embs_0, neg_embs_0)

        # Total loss = BPR + lambda * L2
        loss = bpr_loss + l2_reg_weight * l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_epoch_mnr(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.05,
    l2_reg_weight: float = 1e-4,
) -> float:
    """One MNR training epoch (for pedagogical comparison).

    Uses in-batch negatives instead of explicit negative sampling.
    Expected to FAIL on MIND due to mode collapse.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(dataloader, desc="  Training (MNR)", leave=False)
    except ImportError:
        iterator = dataloader

    for user_ids, pos_ids, neg_ids in iterator:
        # Note: neg_ids are ignored for MNR (uses in-batch negatives)
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)

        # Full graph forward pass
        all_user_embs, all_item_embs = model(adj_norm)

        # Look up batch embeddings
        user_embs = all_user_embs[user_ids]       # (B, hidden_dim)
        pos_item_embs = all_item_embs[pos_ids]    # (B, hidden_dim)

        # MNR loss (in-batch negatives)
        mnr_loss = LightGCN.compute_mnr_loss(user_embs, pos_item_embs, temperature)

        # L2 regularization
        user_embs_0 = model.user_embedding(user_ids)
        pos_embs_0 = model.item_embedding(pos_ids)
        l2_loss = (user_embs_0.norm(2).pow(2) + pos_embs_0.norm(2).pow(2)) / user_ids.shape[0]

        loss = mnr_loss + l2_reg_weight * l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# Validation
# ============================================================================

def validate_bpr(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    dataloader: DataLoader,
    device: torch.device,
    l2_reg_weight: float = 1e-4,
) -> float:
    """Validate on held-out edges using BPR loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for user_ids, pos_ids, neg_ids in dataloader:
            user_ids = user_ids.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)

            all_user_embs, all_item_embs = model(adj_norm)

            user_embs = all_user_embs[user_ids]
            pos_item_embs = all_item_embs[pos_ids]
            neg_item_embs = all_item_embs[neg_ids]

            bpr_loss = LightGCN.compute_bpr_loss(user_embs, pos_item_embs, neg_item_embs)

            user_embs_0 = model.user_embedding(user_ids)
            pos_embs_0 = model.item_embedding(pos_ids)
            neg_embs_0 = model.item_embedding(neg_ids)
            l2_loss = LightGCN.compute_l2_reg(user_embs_0, pos_embs_0, neg_embs_0)

            loss = bpr_loss + l2_reg_weight * l2_loss
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_mnr(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.05,
    l2_reg_weight: float = 1e-4,
) -> float:
    """Validate using MNR loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for user_ids, pos_ids, neg_ids in dataloader:
            user_ids = user_ids.to(device)
            pos_ids = pos_ids.to(device)

            all_user_embs, all_item_embs = model(adj_norm)

            user_embs = all_user_embs[user_ids]
            pos_item_embs = all_item_embs[pos_ids]

            mnr_loss = LightGCN.compute_mnr_loss(user_embs, pos_item_embs, temperature)

            user_embs_0 = model.user_embedding(user_ids)
            pos_embs_0 = model.item_embedding(pos_ids)
            l2_loss = (user_embs_0.norm(2).pow(2) + pos_embs_0.norm(2).pow(2)) / user_ids.shape[0]

            loss = mnr_loss + l2_reg_weight * l2_loss
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# Data Preparation
# ============================================================================

def build_training_data(
    graph_data: Dict,
    config,
    quick: bool = False,
) -> Tuple[BPREdgeDataset, BPREdgeDataset]:
    """Split edges into train/val and create datasets.

    Args:
        graph_data: Output from build_bipartite_graph().
        config: LightGCNConfig.
        quick: If True, subset edges for smoke testing.

    Returns:
        (train_dataset, val_dataset)
    """
    edges = graph_data["user_item_edges"]
    num_items = graph_data["num_items"]
    user_positive_items = graph_data["user_positive_items"]

    # Shuffle edges
    rng = np.random.RandomState(config.random_seed)
    perm = rng.permutation(len(edges))
    shuffled_edges = [edges[i] for i in perm]

    # Quick mode: subset edges
    if quick:
        max_edges = min(10_000, len(shuffled_edges))
        shuffled_edges = shuffled_edges[:max_edges]
        logger.info(f"  Quick mode: using {max_edges:,} edges")

    # Split into train/val
    split_idx = int(len(shuffled_edges) * config.train_fraction)
    train_edges = shuffled_edges[:split_idx]
    val_edges = shuffled_edges[split_idx:]

    logger.info(f"  Train edges: {len(train_edges):,}")
    logger.info(f"  Val edges:   {len(val_edges):,}")

    train_dataset = BPREdgeDataset(train_edges, num_items, user_positive_items)
    val_dataset = BPREdgeDataset(val_edges, num_items, user_positive_items)

    return train_dataset, val_dataset


# ============================================================================
# Main Training Function
# ============================================================================

def train(
    loss_type: str = "bpr",
    quick: bool = False,
    sbert_init: bool = False,
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    hidden_dim_override: Optional[int] = None,
    num_layers_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
) -> Path:
    """Train LightGCN on MIND.

    Args:
        loss_type: "bpr" or "mnr".
        quick: Smoke test mode (3 epochs, subset edges).
        sbert_init: If True, initialize item embeddings with PCA-projected
            SBERT vectors from Chapter 10 instead of random N(0, 0.1).
        Various overrides for hyperparameters.

    Returns:
        Path to the output model directory.
    """
    config = DEFAULT_LIGHTGCN_CONFIG
    mind_config = DEFAULT_MIND_USER_CONFIG
    eval_config = DEFAULT_EVALUATION_CONFIG

    # Apply overrides
    epochs = epochs_override or (3 if quick else config.epochs)
    lr = lr_override or config.learning_rate
    hidden_dim = hidden_dim_override or config.hidden_dim
    num_layers = num_layers_override or config.num_layers
    batch_size = batch_size_override or config.batch_size

    init_label = "SBERT-PCA" if sbert_init else "Random"
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  LightGCN Training — MIND — Loss: {loss_type.upper()} — Init: {init_label}")
    logger.info(f"{'=' * 70}")

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # ----- Build graph -----
    logger.info("\n[Step 1] Building bipartite graph...")
    
    graph_data = build_bipartite_graph(mind_config, eval_config)

    adj_norm = build_normalized_adjacency(
        graph_data["user_item_edges"],
        graph_data["num_users"],
        graph_data["num_items"],
    )
    adj_norm = adj_norm.to(device)

    num_users = graph_data["num_users"]
    num_items = graph_data["num_items"]

    # ----- Build datasets -----
    logger.info("\n[Step 2] Preparing train/val edge splits...")
    train_dataset, val_dataset = build_training_data(graph_data, config, quick=quick)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bpr_collate_fn,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bpr_collate_fn,
        num_workers=0,
        drop_last=False,
    )

    # ----- Create model -----
    logger.info("\n[Step 3] Creating LightGCN model...")
    model = create_lightgcn(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=config.dropout,
    )
    model = model.to(device)

    total_params = count_parameters(model)
    user_params = model.user_embedding.weight.numel()
    item_params = model.item_embedding.weight.numel()

    logger.info(f"  Parameters: {total_params:,}")
    logger.info(f"    User embeddings: {user_params:,} ({user_params * 4 / 1e6:.1f} MB)")
    logger.info(f"    Item embeddings: {item_params:,} ({item_params * 4 / 1e6:.1f} MB)")
    logger.info(f"  Architecture: {num_layers} layers, {hidden_dim}-dim, dropout={config.dropout}")

    # ----- SBERT initialization (optional) -----
    if sbert_init:
        logger.info("\n[Step 3b] Initializing item embeddings with PCA-projected SBERT...")

        # Load Chapter 10's cached SBERT embeddings
        emb_path = Path(mind_config.item_embedding_file)
        if not emb_path.exists():
            raise FileNotFoundError(
                f"SBERT embeddings not found at {emb_path}. "
                f"Run Chapter 10 first to generate them."
            )

        sbert_data = np.load(emb_path, allow_pickle=True)
        sbert_embeddings = sbert_data["embeddings"]  # (num_items, 384)
        sbert_dim = sbert_embeddings.shape[1]

        logger.info(
            f"  Loaded SBERT embeddings: shape={sbert_embeddings.shape} "
            f"(384-dim → {hidden_dim}-dim via PCA)"
        )

        # Verify item count matches
        if sbert_embeddings.shape[0] != num_items:
            raise ValueError(
                f"SBERT item count ({sbert_embeddings.shape[0]}) doesn't match "
                f"graph item count ({num_items}). Index alignment requires same items."
            )

        # PCA projection: 384 → hidden_dim via truncated SVD
        # Center the data (subtract mean per dimension)
        mean_vec = sbert_embeddings.mean(axis=0)
        centered = sbert_embeddings - mean_vec

        # SVD: centered = U @ diag(S) @ Vt
        # Keep top hidden_dim components: projected = U[:, :d] * S[:d]
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = U[:, :hidden_dim] * S[:hidden_dim]  # (num_items, hidden_dim)

        # Variance retained
        total_var = np.sum(S ** 2)
        retained_var = np.sum(S[:hidden_dim] ** 2)
        logger.info(
            f"  PCA: {sbert_dim}-dim → {hidden_dim}-dim, "
            f"variance retained: {retained_var / total_var:.1%}"
        )

        # Initialize item embeddings with projected SBERT vectors
        model.init_item_embeddings_from_pretrained(projected)

    # ----- Optimizer -----
    # Note: weight_decay=0 because we handle L2 reg manually (on initial embeddings only)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    # ----- Output directory -----
    dir_suffix = f"lightgcn_{loss_type}_sbert" if sbert_init else f"lightgcn_{loss_type}"
    output_dir = MODELS_DIR / config.output_subdir / dir_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output dir: {output_dir}")

    # ----- Training loop -----
    logger.info(f"\n[Step 4] Training ({epochs} epochs, batch_size={batch_size})...")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_history = []
    start_time = time.time()

    # Select training/validation functions based on loss type
    if loss_type == "bpr":
        train_fn = lambda dl: train_epoch_bpr(
            model, adj_norm, dl, optimizer, device, config.l2_reg_weight
        )
        val_fn = lambda dl: validate_bpr(
            model, adj_norm, dl, device, config.l2_reg_weight
        )
    else:  # mnr
        train_fn = lambda dl: train_epoch_mnr(
            model, adj_norm, dl, optimizer, device,
            config.mnr_temperature, config.l2_reg_weight
        )
        val_fn = lambda dl: validate_mnr(
            model, adj_norm, dl, device,
            config.mnr_temperature, config.l2_reg_weight
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_fn(train_loader)

        # Validate
        val_loss = val_fn(val_loader)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        logger.info(
            f"  Epoch {epoch:>3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": current_lr,
            "time_seconds": round(epoch_time, 1),
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": "lightgcn",
                "loss_type": loss_type,
                "sbert_init": sbert_init,
                "config": {
                    "num_users": num_users,
                    "num_items": num_items,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": config.dropout,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "best_model.pt")
            logger.info(f"    ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info(
                    f"  Early stopping: no improvement for {config.patience} epochs"
                )
                break

    total_time = time.time() - start_time
    logger.info(f"\n  Training complete in {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")

    # ----- Save graph mappings (needed at evaluation time) -----
    np.savez(
        output_dir / "graph_mappings.npz",
        train_user_ids=np.array(list(graph_data["train_user_id_to_idx"].keys())),
        train_user_indices=np.array(list(graph_data["train_user_id_to_idx"].values())),
        item_ids=np.array(graph_data["item_ids"]),
        # item_id_to_idx is the same as dataset's mapping (0-based)
        eval_user_ids=np.array(list(graph_data["eval_user_ids"])),
    )

    # ----- Save training metadata -----
    metadata = {
        "model_type": "lightgcn",
        "loss_type": loss_type,
        "sbert_init": sbert_init,
        "dataset": "mind",
        "num_users": num_users,
        "num_items": num_items,
        "num_edges": graph_data["num_edges"],
        "parameters": total_params,
        "user_embedding_params": user_params,
        "item_embedding_params": item_params,
        "epochs_completed": len(training_history),
        "epochs_planned": epochs,
        "best_val_loss": round(best_val_loss, 6),
        "training_time_seconds": round(total_time, 1),
        "device": str(device),
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": config.dropout,
            "learning_rate": lr,
            "l2_reg_weight": config.l2_reg_weight,
            "batch_size": batch_size,
            "train_fraction": config.train_fraction,
            "patience": config.patience,
        },
        "training_history": training_history,
        "timestamp": datetime.now().isoformat(),
    }

    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved to {meta_path.name}")

    return output_dir


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train LightGCN on MIND (Section 11.3)"
    )
    parser.add_argument(
        "--loss", type=str, default="bpr",
        choices=["bpr", "mnr", "both"],
        help="Loss function: bpr (default), mnr (comparison), or both"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 3 epochs, subset edges"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Override hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=None,
        help="Override number of GCN layers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--sbert_init", action="store_true",
        help="Initialize item embeddings with PCA-projected SBERT from Ch10 "
             "(instead of random N(0,0.1))"
    )

    args = parser.parse_args()

    loss_types = ["bpr", "mnr"] if args.loss == "both" else [args.loss]

    for loss_type in loss_types:
        output_dir = train(
            loss_type=loss_type,
            quick=args.quick,
            sbert_init=args.sbert_init,
            epochs_override=args.epochs,
            lr_override=args.lr,
            hidden_dim_override=args.hidden_dim,
            num_layers_override=args.num_layers,
            batch_size_override=args.batch_size,
        )
        logger.info(f"\n  Model saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
