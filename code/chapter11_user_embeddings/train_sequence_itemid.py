"""
Training Script for Section 11.2 Variant: Sequence Models with Learnable Item IDs

This is a variant of train_sequence.py that uses LEARNABLE item ID embeddings
(nn.Embedding) instead of frozen SBERT embeddings from Chapter 10.

The key difference:
  - Original:  Model input = frozen 384-dim SBERT → Linear(384, 128)
  - This file: Model input = nn.Embedding(num_items, 128)  ← LEARNABLE

Everything else is identical: same loss (MNR), same evaluation protocol,
same data splits (same seed=42), same hyperparameters.

This allows a clean A/B comparison:
  - Frozen SBERT: captures content similarity (text-based)
  - Learnable IDs: captures behavioral co-occurrence (interaction-based)

Usage:
    # Train SASRec-ID on Amazon KDD
    python train_sequence_itemid.py --model sasrec --dataset amazon

    # Train both models on both datasets
    python train_sequence_itemid.py --model both --dataset both

    # Quick mode for smoke testing
    python train_sequence_itemid.py --model sasrec --dataset amazon --quick
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from config import (
    DEFAULT_AMAZON_SESSION_CONFIG,
    DEFAULT_MIND_USER_CONFIG,
    DEFAULT_SEQUENCE_MODEL_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
)
from data.amazon_session_loader import AmazonSessionDataset
from data.mind_user_loader import MINDUserDataset
from models.sequence_models_itemid import create_itemid_model
from models.sequence_models import count_parameters

# Reuse loss functions from the original training script
from train_sequence import sequence_mnr_loss


# ============================================================================
# Item-ID Training Dataset
# ============================================================================

class ItemIDTrainingDataset(Dataset):
    """PyTorch Dataset for Item-ID sequence model training.

    Unlike SequenceTrainingDataset (which stores embedding indices and looks
    up 384-dim vectors), this dataset stores raw item indices and returns
    them directly.  The model's nn.Embedding layer handles the lookup.

    We still need the frozen SBERT embeddings for the TARGET — because the
    MNR loss compares the model's output against frozen SBERT embeddings.
    Only the INPUT changes (from frozen embeddings to learnable IDs).

    For each training example:
      input_ids:  [item_idx_1, item_idx_2, ..., item_idx_L]  (integers)
      target_embs: [sbert_emb_2, sbert_emb_3, ..., sbert_emb_{L+1}]  (384-dim)
    """

    def __init__(
        self,
        sequences: List[List[int]],
        item_embeddings: np.ndarray,
        max_seq_len: int,
    ):
        """
        Args:
            sequences: List of item index sequences.  Indices are 1-based
                      (0 reserved for padding in nn.Embedding).
            item_embeddings: (N_items, 384) frozen SBERT embeddings (for targets).
            max_seq_len: Maximum sequence length.
        """
        self.sequences = sequences
        self.item_embeddings = item_embeddings
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return a single training example.

        Returns:
            input_ids: (L,) int64 array — item indices for the model input.
            target_embeddings: (L, 384) float32 array — frozen SBERT for targets.
            length: int — actual sequence length.
        """
        seq = self.sequences[idx]

        input_ids = seq[:-1]
        target_ids = seq[1:]

        # Right-truncate if needed (keep most recent)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            target_ids = target_ids[-self.max_seq_len:]

        length = len(input_ids)

        # Input: integer indices (for nn.Embedding lookup)
        input_array = np.array(input_ids, dtype=np.int64)

        # Target: frozen SBERT embeddings (for MNR loss)
        target_embs = self.item_embeddings[target_ids]  # (L, 384)

        return input_array, target_embs, length


def itemid_collate_fn(
    batch: List[Tuple[np.ndarray, np.ndarray, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate with dynamic padding for Item-ID sequences.

    Pads input_ids with 0 (the padding index for nn.Embedding).
    Pads target_embeddings with zeros.

    Returns:
        input_ids_padded:  (B, L_max) int64 tensor
        target_padded:     (B, L_max, 384) float32 tensor
        lengths:           (B,) int64 tensor
    """
    input_ids_list, target_embs_list, lengths_list = zip(*batch)

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    emb_dim = target_embs_list[0].shape[1]

    # Pad input IDs with 0 (padding_idx)
    input_ids_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_padded = torch.zeros(batch_size, max_len, emb_dim, dtype=torch.float32)

    for i, (ids, tgt, ln) in enumerate(zip(input_ids_list, target_embs_list, lengths_list)):
        input_ids_padded[i, :ln] = torch.from_numpy(ids)
        target_padded[i, :ln] = torch.from_numpy(tgt)

    return input_ids_padded, target_padded, lengths


# ============================================================================
# Multi-Position Loss for Item-ID Models
# ============================================================================

def compute_multi_position_loss_itemid(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_embeddings: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute MNR loss at all valid positions (same logic as original).

    The only difference: input is item IDs (integers), not embeddings.
    """
    device = input_ids.device

    user_embeddings, all_hidden = model(input_ids, lengths)
    all_output = model.output_projection(all_hidden)  # (B, L, 384)

    total_loss = torch.tensor(0.0, device=device)
    num_valid_positions = 0

    B, L, D = all_output.shape

    for t in range(L):
        valid_mask = (t < lengths)
        if valid_mask.sum() < 2:
            continue

        pred_t = all_output[:, t, :]
        tgt_t = target_embeddings[:, t, :]

        pred_valid = pred_t[valid_mask]
        tgt_valid = tgt_t[valid_mask]

        loss_t = sequence_mnr_loss(pred_valid, tgt_valid, temperature)
        total_loss = total_loss + loss_t
        num_valid_positions += 1

    if num_valid_positions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / num_valid_positions


def compute_last_position_loss_itemid(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_embeddings: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute MNR loss at only the last valid position."""
    B = input_ids.size(0)
    device = input_ids.device

    user_embeddings, _ = model(input_ids, lengths)

    last_indices = (lengths - 1).long()
    batch_indices = torch.arange(B, device=device)
    target_last = target_embeddings[batch_indices, last_indices]

    return sequence_mnr_loss(user_embeddings, target_last, temperature)


# ============================================================================
# Data Preparation
# ============================================================================

def build_training_data_amazon(
    config,
    seq_config,
    eval_config,
) -> Tuple[ItemIDTrainingDataset, ItemIDTrainingDataset, np.ndarray, List[str], int]:
    """Build Item-ID training data for Amazon KDD.

    Same logic as train_sequence.py but:
    - Item indices are 1-based (0 = padding for nn.Embedding)
    - Returns num_items (vocabulary size including padding)

    Returns:
        (train_dataset, val_dataset, all_item_embeddings, all_item_ids, num_items)
    """
    logger.info("Loading Amazon KDD data...")
    dataset = AmazonSessionDataset(
        data_dir=config.data_dir,
        locale=config.locale,
        min_session_length=config.min_session_length,
        item_embedding_file=config.item_embedding_file,
        item_embedding_dim=config.item_embedding_dim,
    )
    dataset.load()
    dataset.print_stats()

    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    # Create 1-based mapping (0 = padding)
    item_id_to_idx = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}
    num_items = len(all_item_ids) + 1  # +1 for padding index 0

    # Prepend a zero row for padding index 0
    all_embeddings_padded = np.vstack([
        np.zeros((1, all_embeddings.shape[1]), dtype=all_embeddings.dtype),
        all_embeddings,
    ])

    # Identify evaluation sessions to exclude
    total_sessions = len(dataset.sessions_df)
    max_eval = eval_config.max_eval_sessions_amazon
    rng = np.random.RandomState(eval_config.random_seed)

    if total_sessions > max_eval:
        eval_indices = set(rng.choice(total_sessions, max_eval, replace=False))
    else:
        eval_indices = set(range(total_sessions))

    logger.info(f"  Excluding {len(eval_indices):,} evaluation sessions")

    # Build training sequences
    train_sequences = []
    for row_idx, row in enumerate(dataset.sessions_df.iter_rows(named=True)):
        if row_idx in eval_indices:
            continue

        prev_items = row["prev_items_list"]
        next_item = row["next_item"]

        try:
            seq_indices = [item_id_to_idx[iid] for iid in prev_items]
            seq_indices.append(item_id_to_idx[next_item])
        except KeyError:
            continue

        if len(seq_indices) >= 2:
            train_sequences.append(seq_indices)

    logger.info(f"  Total training sequences: {len(train_sequences):,}")

    # Split into train / validation
    rng2 = np.random.RandomState(seq_config.random_seed + 1)
    perm = rng2.permutation(len(train_sequences))
    split_idx = int(len(train_sequences) * seq_config.train_fraction)

    train_seqs = [train_sequences[i] for i in perm[:split_idx]]
    val_seqs = [train_sequences[i] for i in perm[split_idx:]]

    logger.info(f"  Train sequences: {len(train_seqs):,}")
    logger.info(f"  Val sequences:   {len(val_seqs):,}")

    train_dataset = ItemIDTrainingDataset(
        sequences=train_seqs,
        item_embeddings=all_embeddings_padded,  # 1-based indexing
        max_seq_len=seq_config.max_seq_len_amazon,
    )
    val_dataset = ItemIDTrainingDataset(
        sequences=val_seqs,
        item_embeddings=all_embeddings_padded,
        max_seq_len=seq_config.max_seq_len_amazon,
    )

    return train_dataset, val_dataset, all_embeddings_padded, all_item_ids, num_items


def build_training_data_mind(
    config,
    seq_config,
    eval_config,
) -> Tuple[ItemIDTrainingDataset, ItemIDTrainingDataset, np.ndarray, List[str], int]:
    """Build Item-ID training data for MIND.

    Same logic as train_sequence.py but with 1-based item indices.

    Returns:
        (train_dataset, val_dataset, all_item_embeddings, all_item_ids, num_items)
    """
    logger.info("Loading MIND data...")
    dataset = MINDUserDataset(
        data_dir=config.data_dir,
        news_file=config.news_file,
        behaviors_file=config.behaviors_file,
        min_history_length=config.min_history_length,
        item_embedding_file=config.item_embedding_file,
        item_embedding_dim=config.item_embedding_dim,
    )
    dataset.load()
    dataset.print_stats()

    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    item_id_to_idx = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}
    num_items = len(all_item_ids) + 1

    # Prepend zero row for padding
    all_embeddings_padded = np.vstack([
        np.zeros((1, all_embeddings.shape[1]), dtype=all_embeddings.dtype),
        all_embeddings,
    ])

    # Get evaluation users to exclude
    eval_users = dataset.get_evaluation_users(
        max_users=eval_config.max_eval_users_mind,
        seed=eval_config.random_seed,
    )
    eval_user_ids = set(u["user_id"] for u in eval_users)
    logger.info(f"  Excluding {len(eval_user_ids):,} evaluation users")

    # Build training sequences
    user_timelines = dataset._build_user_timelines()
    valid_news = dataset._valid_news_ids

    train_sequences = []
    rng = np.random.RandomState(seq_config.random_seed)

    for uid, timeline in user_timelines.items():
        if uid in eval_user_ids:
            continue

        if len(timeline) < 1:
            continue

        last_imp = timeline[-1]
        history_ids = [nid for nid in last_imp["history_list"] if nid in valid_news]
        target_ids = [nid for nid in last_imp["clicked_articles"] if nid in valid_news]

        if len(history_ids) < config.min_history_length:
            continue
        if len(target_ids) == 0:
            continue

        try:
            history_indices = [item_id_to_idx[nid] for nid in history_ids]
        except KeyError:
            continue

        valid_targets = []
        for tid in target_ids:
            if tid in item_id_to_idx:
                valid_targets.append(item_id_to_idx[tid])

        if not valid_targets:
            continue

        target_idx = valid_targets[rng.randint(len(valid_targets))]
        seq = history_indices + [target_idx]

        if len(seq) >= 2:
            train_sequences.append(seq)

    logger.info(f"  Total training sequences: {len(train_sequences):,}")

    rng2 = np.random.RandomState(seq_config.random_seed + 1)
    perm = rng2.permutation(len(train_sequences))
    split_idx = int(len(train_sequences) * seq_config.train_fraction)

    train_seqs = [train_sequences[i] for i in perm[:split_idx]]
    val_seqs = [train_sequences[i] for i in perm[split_idx:]]

    logger.info(f"  Train sequences: {len(train_seqs):,}")
    logger.info(f"  Val sequences:   {len(val_seqs):,}")

    train_dataset = ItemIDTrainingDataset(
        sequences=train_seqs,
        item_embeddings=all_embeddings_padded,
        max_seq_len=seq_config.max_seq_len_mind,
    )
    val_dataset = ItemIDTrainingDataset(
        sequences=val_seqs,
        item_embeddings=all_embeddings_padded,
        max_seq_len=seq_config.max_seq_len_mind,
    )

    return train_dataset, val_dataset, all_embeddings_padded, all_item_ids, num_items


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config,
    use_amp: bool,
) -> float:
    """Train for one epoch. Returns average loss."""
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for input_ids, target_embs, lengths in pbar:
        input_ids = input_ids.to(device)
        target_embs = target_embs.to(device)
        lengths = lengths.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if config.multi_position_loss:
                loss = compute_multi_position_loss_itemid(
                    model, input_ids, target_embs, lengths, config.temperature
                )
            else:
                loss = compute_last_position_loss_itemid(
                    model, input_ids, target_embs, lengths, config.temperature
                )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config,
    use_amp: bool,
) -> float:
    """Compute validation loss. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, target_embs, lengths in dataloader:
            input_ids = input_ids.to(device)
            target_embs = target_embs.to(device)
            lengths = lengths.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if config.multi_position_loss:
                    loss = compute_multi_position_loss_itemid(
                        model, input_ids, target_embs, lengths, config.temperature
                    )
                else:
                    loss = compute_last_position_loss_itemid(
                        model, input_ids, target_embs, lengths, config.temperature
                    )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model_type: str,
    dataset_name: str,
    config=DEFAULT_SEQUENCE_MODEL_CONFIG,
    quick: bool = False,
) -> Path:
    """Full training pipeline for Item-ID sequence model.

    Same structure as train_sequence.py but uses ItemID model variants.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    logger.info(f"Device: {device}, AMP: {use_amp}")

    eval_config = DEFAULT_EVALUATION_CONFIG
    data_config = (
        DEFAULT_AMAZON_SESSION_CONFIG
        if dataset_name in ("amazon", "amazon_kdd")
        else DEFAULT_MIND_USER_CONFIG
    )

    if dataset_name in ("amazon", "amazon_kdd"):
        train_dataset, val_dataset, all_embeddings, all_item_ids, num_items = (
            build_training_data_amazon(data_config, config, eval_config)
        )
        max_seq_len = config.max_seq_len_amazon
        epochs = 5 if quick else config.epochs_amazon
    else:
        train_dataset, val_dataset, all_embeddings, all_item_ids, num_items = (
            build_training_data_mind(data_config, config, eval_config)
        )
        max_seq_len = config.max_seq_len_mind
        epochs = 3 if quick else config.epochs_mind

    if quick:
        max_train = min(5000, len(train_dataset))
        train_dataset.sequences = train_dataset.sequences[:max_train]
        max_val = min(1000, len(val_dataset))
        val_dataset.sequences = val_dataset.sequences[:max_val]
        logger.info(f"  Quick mode: {max_train} train, {max_val} val, {epochs} epochs")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=itemid_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=itemid_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model
    model = create_itemid_model(
        model_type=model_type,
        num_items=num_items,
        output_dim=config.input_dim,  # 384 (map to SBERT space)
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=max_seq_len,
        dropout=config.dropout,
        padding_idx=0,
    )
    model = model.to(device)

    n_params = count_parameters(model)
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING: {model_type.upper()}-ItemID on {dataset_name.upper()}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Model:           {model_type}-ItemID (learnable item embeddings)")
    logger.info(f"  Vocabulary:      {num_items:,} items (incl. padding)")
    logger.info(f"  Parameters:      {n_params:,} ({n_params * 4 / 1024 / 1024:.1f} MB)")
    logger.info(f"    Item embedding: {num_items * config.hidden_dim:,} ({num_items * config.hidden_dim * 4 / 1024 / 1024:.1f} MB)")
    logger.info(f"    Architecture:   {n_params - num_items * config.hidden_dim:,}")
    logger.info(f"  Dataset:         {dataset_name}")
    logger.info(f"  Train examples:  {len(train_dataset):,}")
    logger.info(f"  Val examples:    {len(val_dataset):,}")
    logger.info(f"  Max seq length:  {max_seq_len}")
    logger.info(f"  Batch size:      {config.batch_size}")
    logger.info(f"  Epochs:          {epochs}")
    logger.info(f"  Learning rate:   {config.learning_rate}")
    logger.info(f"  Temperature:     {config.temperature}")
    logger.info(f"  Multi-pos loss:  {config.multi_position_loss}")
    logger.info(f"  Early stopping:  patience={config.patience}")
    logger.info(f"{'=' * 80}")

    # Output directory
    output_dir = MODELS_DIR / "sequence_models_itemid" / f"{model_type}_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the item_id mapping for evaluation
    np.savez_compressed(
        output_dir / "item_id_mapping.npz",
        item_ids=np.array(all_item_ids, dtype=object),
        num_items=num_items,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    start_time = time.time()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, config, use_amp
        )

        if scheduler._step_count <= warmup_steps:
            scheduler.step()

        val_loss = validate(model, val_loader, device, config, use_amp)

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.2e}"
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "variant": "itemid",
                "config": {
                    "num_items": num_items,
                    "output_dim": config.input_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "ffn_dim": config.ffn_dim,
                    "max_seq_len": max_seq_len,
                    "dropout": config.dropout,
                    "padding_idx": 0,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "best_model.pt")
            logger.info(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info(
                    f"  Early stopping: no improvement for {config.patience} epochs"
                )
                break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    # Save training metadata
    meta = {
        "model_type": model_type,
        "variant": "itemid",
        "dataset": dataset_name,
        "num_items": num_items,
        "parameters": n_params,
        "item_embedding_params": num_items * config.hidden_dim,
        "architecture_params": n_params - num_items * config.hidden_dim,
        "epochs_completed": len(training_history),
        "epochs_planned": epochs,
        "best_val_loss": round(best_val_loss, 6),
        "training_time_seconds": round(elapsed, 1),
        "device": str(device),
        "config": {
            "num_items": num_items,
            "output_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "ffn_dim": config.ffn_dim,
            "max_seq_len": max_seq_len,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "temperature": config.temperature,
            "weight_decay": config.weight_decay,
            "multi_position_loss": config.multi_position_loss,
            "patience": config.patience,
        },
        "training_history": training_history,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved to:    {output_dir}")
    logger.info(f"Metadata saved to: {output_dir / 'training_meta.json'}")

    return output_dir


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section 11.2 Variant — Train sequence models with learnable Item IDs"
    )
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["sasrec", "gru4rec", "both"],
        help="Which model to train",
    )
    parser.add_argument(
        "--dataset", type=str, default="both",
        choices=["amazon", "mind", "both"],
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer epochs, smaller train set",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Override learning rate",
    )

    args = parser.parse_args()
    config = DEFAULT_SEQUENCE_MODEL_CONFIG

    if args.epochs:
        config.epochs_amazon = args.epochs
        config.epochs_mind = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    models_to_train = (
        ["sasrec", "gru4rec"] if args.model == "both" else [args.model]
    )
    datasets_to_train = (
        ["amazon", "mind"] if args.dataset == "both" else [args.dataset]
    )

    trained_models = {}
    for model_type in models_to_train:
        for dataset_name in datasets_to_train:
            logger.info(f"\n{'#' * 80}")
            logger.info(f"# Training {model_type.upper()}-ItemID on {dataset_name.upper()}")
            logger.info(f"{'#' * 80}")

            model_dir = train(
                model_type=model_type,
                dataset_name=dataset_name,
                config=config,
                quick=args.quick,
            )
            trained_models[f"{model_type}_itemid_{dataset_name}"] = model_dir

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE — Item-ID Models")
    print(f"{'=' * 80}")
    for key, path in trained_models.items():
        print(f"  {key:<35} -> {path}")
    print(f"\nNext step: evaluate with")
    print(f"  python evaluate_sequence_itemid.py --dataset {'both' if len(datasets_to_train) > 1 else datasets_to_train[0]}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
