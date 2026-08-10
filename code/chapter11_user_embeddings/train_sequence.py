"""
Training Script for Section 11.2: Sequence Models (SASRec / GRU4Rec)

Trains sequence models on next-item prediction with in-batch negative
sampling (MNR loss).  Uses frozen SBERT item embeddings from Chapter 10
as input.

The training loop follows Chapter 10's finetune_contrastive.py pattern:
manual PyTorch, AdamW optimizer, linear warmup, FP16 AMP, tqdm progress
bars, periodic validation, early stopping, and best-model checkpointing.

Usage:
    # Train SASRec on Amazon KDD
    python train_sequence.py --model sasrec --dataset amazon

    # Train GRU4Rec on MIND
    python train_sequence.py --model gru4rec --dataset mind

    # Train both models on both datasets
    python train_sequence.py --model both --dataset both

    # Quick mode (fewer epochs, smaller train set for smoke testing)
    python train_sequence.py --model sasrec --dataset amazon --quick
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
from models.sequence_models import create_sequence_model, count_parameters


# ============================================================================
# Training Dataset
# ============================================================================

class SequenceTrainingDataset(Dataset):
    """PyTorch Dataset for sequence model training.

    Converts sessions/users into training examples.  Each example is a
    sequence of items where the model predicts the next item at every
    position (multi-position training).

    Memory-efficient design: stores item ID indices (int32) rather than
    384-dim embeddings.  Embeddings are looked up from the shared numpy
    array on-the-fly during __getitem__.

    For Amazon:
        Session [i_1, i_2, i_3, i_4] with next_item i_5
        -> full_sequence = [i_1, i_2, i_3, i_4, i_5]
        -> input  positions: [i_1, i_2, i_3, i_4]  (predict next at each)
        -> target positions: [i_2, i_3, i_4, i_5]  (what to predict)

    For MIND:
        User history [a_1, ..., a_N] with targets [t_1, t_2]
        -> full_sequence = [a_1, ..., a_N, t_j]  (sample one target per epoch)
        -> Same input/target pattern as above
    """

    def __init__(
        self,
        sequences: List[List[int]],
        item_embeddings: np.ndarray,
        max_seq_len: int,
    ):
        """
        Args:
            sequences: List of item index sequences.  Each sequence is a
                      list of integer indices into the item_embeddings array.
                      The last element is the final target item.
            item_embeddings: (N_items, 384) shared embedding matrix.
            max_seq_len: Maximum sequence length.  Longer sequences are
                        right-truncated (keep most recent items).
        """
        self.sequences = sequences
        self.item_embeddings = item_embeddings
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return a single training example.

        Returns:
            input_embeddings: (L, 384) float32 array — items for aggregation.
                             L = min(len(sequence)-1, max_seq_len).
            target_embeddings: (L, 384) float32 array — next item at each position.
            length: int — actual input sequence length (before padding).
        """
        seq = self.sequences[idx]

        # Full sequence includes the final target:
        # input  = seq[:-1]  (items the model sees)
        # target = seq[1:]   (next item at each position)

        input_ids = seq[:-1]
        target_ids = seq[1:]

        # Right-truncate if needed (keep most recent)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            target_ids = target_ids[-self.max_seq_len:]

        length = len(input_ids)

        # Look up embeddings from the shared matrix
        input_embs = self.item_embeddings[input_ids]    # (L, 384)
        target_embs = self.item_embeddings[target_ids]  # (L, 384)

        return input_embs, target_embs, length


def sequence_collate_fn(
    batch: List[Tuple[np.ndarray, np.ndarray, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collation: pad sequences to the max length IN THE BATCH.

    Dynamic padding (not global max_seq_len) saves memory and computation.
    For Amazon with median session length 4, most batches will have
    sequences of 3-8 items, so padding to 20 would waste significant compute.

    Returns:
        input_padded:  (B, L_max, 384) float32 tensor
        target_padded: (B, L_max, 384) float32 tensor
        lengths:       (B,) int64 tensor
    """
    input_embs_list, target_embs_list, lengths_list = zip(*batch)

    lengths = torch.tensor(lengths_list, dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    emb_dim = input_embs_list[0].shape[1]

    # Pad sequences
    input_padded = torch.zeros(batch_size, max_len, emb_dim, dtype=torch.float32)
    target_padded = torch.zeros(batch_size, max_len, emb_dim, dtype=torch.float32)

    for i, (inp, tgt, ln) in enumerate(zip(input_embs_list, target_embs_list, lengths_list)):
        input_padded[i, :ln] = torch.from_numpy(inp)
        target_padded[i, :ln] = torch.from_numpy(tgt)

    return input_padded, target_padded, lengths


# ============================================================================
# Loss Function
# ============================================================================

def sequence_mnr_loss(
    predicted_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Multiple Negatives Ranking (MNR) loss for sequence models.

    Same loss family as Chapter 10's contrastive fine-tuning.  For each
    predicted user embedding, the positive is the corresponding target
    item embedding; all other targets in the batch are in-batch negatives.

    With batch_size=256, this yields 65,280 negative comparisons per batch.

    Args:
        predicted_embeddings: (B, 384) L2-normalized predicted user embeddings.
        target_embeddings:    (B, 384) L2-normalized target item embeddings.
        temperature: Controls softmax sharpness.  Lower = sharper discrimination.
                    0.05 is equivalent to Chapter 10's `similarity * 20`.

    Returns:
        Scalar loss (cross-entropy over cosine similarity matrix).
    """
    # Both should already be L2-normalized, but ensure it
    pred = F.normalize(predicted_embeddings, p=2, dim=1)
    tgt = F.normalize(target_embeddings, p=2, dim=1)

    # Cosine similarity matrix: (B, B)
    similarity = torch.mm(pred, tgt.t()) / temperature

    # Labels: diagonal entries are the correct matches
    labels = torch.arange(similarity.size(0), device=similarity.device)

    return F.cross_entropy(similarity, labels)


def compute_multi_position_loss(
    model: nn.Module,
    input_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute MNR loss at ALL valid positions (multi-position training).

    For a sequence of length L, this computes L separate MNR losses
    (one per position) and averages them.  This is critical for short
    sessions: a 4-item Amazon session produces only 3 training signals.

    The multi-position loss also teaches the model to produce good
    embeddings from partial sequences (1 item, 2 items, etc.), which
    mirrors the production scenario.

    Args:
        model: SASRec or GRU4Rec instance.
        input_embeddings: (B, L, 384) padded input sequences.
        target_embeddings: (B, L, 384) padded target sequences.
        lengths: (B,) actual sequence lengths.
        temperature: MNR loss temperature.

    Returns:
        Scalar average loss across all valid positions.
    """
    device = input_embeddings.device

    # Forward pass: get hidden states at all positions
    user_embeddings, all_hidden = model(input_embeddings, lengths)

    # all_hidden is (B, L, hidden_dim=128)
    # We need to project to 384-dim for the MNR loss
    # The output_projection is part of the model, so we apply it here
    all_output = model.output_projection(all_hidden)  # (B, L, 384)

    # Compute loss at each valid position
    total_loss = torch.tensor(0.0, device=device)
    num_valid_positions = 0

    B, L, D = all_output.shape

    for t in range(L):
        # Which sequences have a valid item at position t?
        valid_mask = (t < lengths)  # (B,)
        if valid_mask.sum() < 2:
            # Need at least 2 items for in-batch negatives
            continue

        # Gather predicted and target embeddings at position t
        pred_t = all_output[:, t, :]       # (B, 384)
        tgt_t = target_embeddings[:, t, :] # (B, 384)

        # Filter to valid sequences only
        pred_valid = pred_t[valid_mask]    # (B_valid, 384)
        tgt_valid = tgt_t[valid_mask]      # (B_valid, 384)

        # MNR loss at this position
        loss_t = sequence_mnr_loss(pred_valid, tgt_valid, temperature)
        total_loss = total_loss + loss_t
        num_valid_positions += 1

    if num_valid_positions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / num_valid_positions


def compute_last_position_loss(
    model: nn.Module,
    input_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute MNR loss at ONLY the last valid position.

    Simpler alternative to multi-position loss.  Useful as a sanity check
    or when multi-position loss is unstable.

    Args:
        model: SASRec or GRU4Rec instance.
        input_embeddings: (B, L, 384) padded input sequences.
        target_embeddings: (B, L, 384) padded target sequences.
        lengths: (B,) actual sequence lengths.
        temperature: MNR loss temperature.

    Returns:
        Scalar loss at the last position.
    """
    B = input_embeddings.size(0)
    device = input_embeddings.device

    # Forward pass
    user_embeddings, _ = model(input_embeddings, lengths)
    # user_embeddings is already at the last valid position, L2-normalized, (B, 384)

    # Get target at the last valid position for each sequence
    last_indices = (lengths - 1).long()
    batch_indices = torch.arange(B, device=device)
    target_last = target_embeddings[batch_indices, last_indices]  # (B, 384)

    return sequence_mnr_loss(user_embeddings, target_last, temperature)


# ============================================================================
# Data Preparation
# ============================================================================

def build_training_data_amazon(
    config,
    seq_config,
    eval_config,
) -> Tuple[SequenceTrainingDataset, SequenceTrainingDataset, np.ndarray, List[str]]:
    """Build training and validation datasets for Amazon KDD.

    Steps:
    1. Load all sessions via AmazonSessionDataset.
    2. Identify the evaluation sessions (same seed=42 as Section 11.1).
    3. Build training sequences from the REMAINING sessions.
    4. Split remaining into 90% train / 10% validation.

    This ensures no evaluation session appears in training data.

    Returns:
        (train_dataset, val_dataset, all_item_embeddings, all_item_ids)
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
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}

    # Identify evaluation sessions to exclude
    # We sample the same sessions as Section 11.1 (seed=42) and exclude them
    total_sessions = len(dataset.sessions_df)
    max_eval = eval_config.max_eval_sessions_amazon
    rng = np.random.RandomState(eval_config.random_seed)

    if total_sessions > max_eval:
        eval_indices = set(rng.choice(total_sessions, max_eval, replace=False))
    else:
        eval_indices = set(range(total_sessions))

    logger.info(f"  Excluding {len(eval_indices):,} evaluation sessions from training")

    # Build training sequences from remaining sessions
    train_sequences = []
    for row_idx, row in enumerate(dataset.sessions_df.iter_rows(named=True)):
        if row_idx in eval_indices:
            continue

        prev_items = row["prev_items_list"]
        next_item = row["next_item"]

        # Convert to embedding indices
        try:
            seq_indices = [item_id_to_idx[iid] for iid in prev_items]
            seq_indices.append(item_id_to_idx[next_item])
        except KeyError:
            continue  # Skip if any item lacks an embedding

        # Minimum 2 items needed (1 input + 1 target)
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

    train_dataset = SequenceTrainingDataset(
        sequences=train_seqs,
        item_embeddings=all_embeddings,
        max_seq_len=seq_config.max_seq_len_amazon,
    )
    val_dataset = SequenceTrainingDataset(
        sequences=val_seqs,
        item_embeddings=all_embeddings,
        max_seq_len=seq_config.max_seq_len_amazon,
    )

    return train_dataset, val_dataset, all_embeddings, all_item_ids


def build_training_data_mind(
    config,
    seq_config,
    eval_config,
) -> Tuple[SequenceTrainingDataset, SequenceTrainingDataset, np.ndarray, List[str]]:
    """Build training and validation datasets for MIND.

    Steps:
    1. Load all user histories via MINDUserDataset.
    2. Get all evaluation users (same seed=42 as Section 11.1).
    3. Collect their user_ids to exclude from training.
    4. Build training sequences from the REMAINING users.
    5. Split remaining into 90% train / 10% validation.

    Each user's training sequence: history articles + one target article.
    For users with multiple targets, we pick one at random per epoch.

    Returns:
        (train_dataset, val_dataset, all_item_embeddings, all_item_ids)
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
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}

    # Get evaluation users to exclude
    eval_users = dataset.get_evaluation_users(
        max_users=eval_config.max_eval_users_mind,
        seed=eval_config.random_seed,
    )
    eval_user_ids = set(u["user_id"] for u in eval_users)
    logger.info(f"  Excluding {len(eval_user_ids):,} evaluation users from training")

    # Build training sequences from all non-eval users
    # Re-process behaviors to get all eligible users
    user_timelines = dataset._build_user_timelines()
    valid_news = dataset._valid_news_ids

    train_sequences = []
    rng = np.random.RandomState(seq_config.random_seed)

    for uid, timeline in user_timelines.items():
        if uid in eval_user_ids:
            continue

        # Build history from the last impression
        if len(timeline) < 1:
            continue

        last_imp = timeline[-1]
        history_ids = [
            nid for nid in last_imp["history_list"]
            if nid in valid_news
        ]
        target_ids = [
            nid for nid in last_imp["clicked_articles"]
            if nid in valid_news
        ]

        if len(history_ids) < config.min_history_length:
            continue
        if len(target_ids) == 0:
            continue

        # Convert to embedding indices
        try:
            history_indices = [item_id_to_idx[nid] for nid in history_ids]
        except KeyError:
            continue

        # For each user, add one training sequence per target article
        # (or sample one if there are many targets)
        valid_targets = []
        for tid in target_ids:
            if tid in item_id_to_idx:
                valid_targets.append(item_id_to_idx[tid])

        if not valid_targets:
            continue

        # Sample one target per user (to balance the dataset)
        target_idx = valid_targets[rng.randint(len(valid_targets))]
        seq = history_indices + [target_idx]

        if len(seq) >= 2:
            train_sequences.append(seq)

    logger.info(f"  Total training sequences: {len(train_sequences):,}")

    # Split into train / validation (user-level: each user is in exactly one split)
    rng2 = np.random.RandomState(seq_config.random_seed + 1)
    perm = rng2.permutation(len(train_sequences))
    split_idx = int(len(train_sequences) * seq_config.train_fraction)

    train_seqs = [train_sequences[i] for i in perm[:split_idx]]
    val_seqs = [train_sequences[i] for i in perm[split_idx:]]

    logger.info(f"  Train sequences: {len(train_seqs):,}")
    logger.info(f"  Val sequences:   {len(val_seqs):,}")

    train_dataset = SequenceTrainingDataset(
        sequences=train_seqs,
        item_embeddings=all_embeddings,
        max_seq_len=seq_config.max_seq_len_mind,
    )
    val_dataset = SequenceTrainingDataset(
        sequences=val_seqs,
        item_embeddings=all_embeddings,
        max_seq_len=seq_config.max_seq_len_mind,
    )

    return train_dataset, val_dataset, all_embeddings, all_item_ids


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
    """Train for one epoch.  Returns average loss."""
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for input_embs, target_embs, lengths in pbar:
        input_embs = input_embs.to(device)
        target_embs = target_embs.to(device)
        lengths = lengths.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if config.multi_position_loss:
                loss = compute_multi_position_loss(
                    model, input_embs, target_embs, lengths, config.temperature
                )
            else:
                loss = compute_last_position_loss(
                    model, input_embs, target_embs, lengths, config.temperature
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
    """Compute validation loss.  Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_embs, target_embs, lengths in dataloader:
            input_embs = input_embs.to(device)
            target_embs = target_embs.to(device)
            lengths = lengths.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if config.multi_position_loss:
                    loss = compute_multi_position_loss(
                        model, input_embs, target_embs, lengths, config.temperature
                    )
                else:
                    loss = compute_last_position_loss(
                        model, input_embs, target_embs, lengths, config.temperature
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
    """Full training pipeline for a sequence model on a dataset.

    Follows Chapter 10's finetune_contrastive.py pattern:
    1. Build training/validation data
    2. Initialize model
    3. Set up optimizer (AdamW), scheduler (LinearLR warmup), AMP
    4. Training loop with tqdm progress bars
    5. Periodic validation
    6. Early stopping on val loss with patience
    7. Save best checkpoint + training metadata JSON

    Args:
        model_type: "sasrec" or "gru4rec"
        dataset_name: "amazon" or "mind"
        config: SequenceModelConfig
        quick: If True, reduce epochs and data for smoke testing

    Returns:
        Path to the saved model directory
    """
    # ---- Determine device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # ---- Build training data ----
    eval_config = DEFAULT_EVALUATION_CONFIG
    data_config = (
        DEFAULT_AMAZON_SESSION_CONFIG
        if dataset_name in ("amazon", "amazon_kdd")
        else DEFAULT_MIND_USER_CONFIG
    )

    if dataset_name in ("amazon", "amazon_kdd"):
        train_dataset, val_dataset, all_embeddings, all_item_ids = (
            build_training_data_amazon(data_config, config, eval_config)
        )
        max_seq_len = config.max_seq_len_amazon
        epochs = 5 if quick else config.epochs_amazon
    else:
        train_dataset, val_dataset, all_embeddings, all_item_ids = (
            build_training_data_mind(data_config, config, eval_config)
        )
        max_seq_len = config.max_seq_len_mind
        epochs = 3 if quick else config.epochs_mind

    if quick:
        # Subset the training data for smoke testing
        max_train = min(5000, len(train_dataset))
        train_dataset.sequences = train_dataset.sequences[:max_train]
        max_val = min(1000, len(val_dataset))
        val_dataset.sequences = val_dataset.sequences[:max_val]
        logger.info(f"  Quick mode: {max_train} train, {max_val} val, {epochs} epochs")

    # ---- Create data loaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=sequence_collate_fn,
        num_workers=0,      # Avoid multiprocessing overhead on Windows
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,     # Ensure consistent batch sizes for MNR loss
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # ---- Initialize model ----
    model = create_sequence_model(
        model_type=model_type,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=max_seq_len,
        dropout=config.dropout,
    )
    model = model.to(device)

    n_params = count_parameters(model)
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING: {model_type.upper()} on {dataset_name.upper()}")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Model:           {model_type}")
    logger.info(f"  Parameters:      {n_params:,} ({n_params * 4 / 1024 / 1024:.2f} MB)")
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

    # ---- Output directory ----
    output_dir = MODELS_DIR / config.output_subdir / f"{model_type}_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Optimizer & scheduler ----
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

    # FP16 mixed precision for GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training loop ----
    start_time = time.time()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_history = []

    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, config, use_amp
        )

        # Update scheduler (warmup phase)
        if scheduler._step_count <= warmup_steps:
            scheduler.step()

        # Validate
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

        # Checkpointing (save best model by val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "config": {
                    "input_dim": config.input_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "ffn_dim": config.ffn_dim,
                    "max_seq_len": max_seq_len,
                    "dropout": config.dropout,
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

    # ---- Save training metadata ----
    meta = {
        "model_type": model_type,
        "dataset": dataset_name,
        "parameters": n_params,
        "epochs_completed": len(training_history),
        "epochs_planned": epochs,
        "best_val_loss": round(best_val_loss, 6),
        "training_time_seconds": round(elapsed, 1),
        "device": str(device),
        "config": {
            "input_dim": config.input_dim,
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
        description="Section 11.2 — Train sequence models for user embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["sasrec", "gru4rec", "both"],
        help="Which model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["amazon", "mind", "both"],
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer epochs, smaller train set (for smoke testing)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )

    args = parser.parse_args()

    config = DEFAULT_SEQUENCE_MODEL_CONFIG

    # Allow CLI overrides
    if args.epochs:
        config.epochs_amazon = args.epochs
        config.epochs_mind = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    # Determine which model-dataset combinations to train
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
            logger.info(f"# Training {model_type.upper()} on {dataset_name.upper()}")
            logger.info(f"{'#' * 80}")

            model_dir = train(
                model_type=model_type,
                dataset_name=dataset_name,
                config=config,
                quick=args.quick,
            )
            trained_models[f"{model_type}_{dataset_name}"] = model_dir

    # Print summary
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE — All Models")
    print(f"{'=' * 80}")
    for key, path in trained_models.items():
        print(f"  {key:<25} -> {path}")
    print(f"\nNext step: evaluate with")
    print(f"  python evaluate_sequence.py --dataset {'both' if len(datasets_to_train) > 1 else datasets_to_train[0]}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
