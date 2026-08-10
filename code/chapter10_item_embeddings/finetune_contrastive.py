"""
Contrastive Fine-Tuning of Text Encoders for Item Embeddings

Section 10.3: Fine-tune a pre-trained sentence encoder using behavioral
signals (co-engagement / next-item pairs) so that items users interact
with sequentially are pulled closer in embedding space.

Datasets:
  - Amazon KDD 2023 (primary):  next-item pairs from 1.18M UK sessions
  - MIND (secondary):           co-click pairs from user reading histories

Training objective:
  MultipleNegativesRankingLoss (MNR) — treats every other item in the
  batch as an in-batch negative.  This is the same loss used by the
  original Sentence-BERT paper and is extremely sample-efficient: a
  batch of size B yields B*(B-1) negative pairs for free.

  Optionally, hard negatives (same category but not co-engaged) can be
  added per anchor to make the contrastive signal sharper.

Usage:
    # Fine-tune on Amazon KDD (primary)
    python finetune_contrastive.py --dataset amazon

    # Fine-tune on MIND
    python finetune_contrastive.py --dataset mind

    # Fine-tune on Amazon KDD with custom hyperparams
    python finetune_contrastive.py --dataset amazon --epochs 5 --batch_size 256
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_CONTRASTIVE_CONFIG,
    DEFAULT_AMAZON_KDD_CONFIG,
    DEFAULT_MIND_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
    EMBEDDINGS_DIR,
)
from data.amazon_kdd_dataset import AmazonKDDDataset
from data.mind_dataset import MINDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Training Pair Preparation
# ============================================================================

def build_amazon_training_pairs(
    config=DEFAULT_CONTRASTIVE_CONFIG,
    data_config=DEFAULT_AMAZON_KDD_CONFIG,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, str]]:
    """
    Build (anchor, positive) pairs from Amazon KDD next-item data.

    Each pair: (text of last-viewed item, text of next-engaged item).
    The temporal ordering within sessions ensures no data leakage —
    the "next" item always occurs after the "query" item.

    Returns:
        (train_pairs, val_pairs, item_texts)
    """
    logger.info("=" * 80)
    logger.info("Building Amazon KDD training pairs")
    logger.info("=" * 80)

    kdd_dataset = AmazonKDDDataset(
        data_dir=data_config.data_dir,
        locale=data_config.locale,
    )
    kdd_dataset.load_products()
    product_ids = set(kdd_dataset.products_df["id"])
    kdd_dataset.load_sessions(product_ids=product_ids)

    item_texts = kdd_dataset.get_item_texts(
        use_title=data_config.use_title,
        use_brand=data_config.use_brand,
        use_description=data_config.use_description,
        template=data_config.text_template,
    )

    next_item_map = kdd_dataset.get_next_item_pairs(
        min_prev_items=data_config.min_session_length - 1,
    )

    all_pairs: List[Tuple[str, str]] = []
    for query_id, target_ids in next_item_map.items():
        if query_id not in item_texts:
            continue
        q_text = item_texts[query_id]
        for target_id in target_ids:
            if target_id not in item_texts:
                continue
            all_pairs.append((q_text, item_texts[target_id]))

    logger.info(f"Total (anchor, positive) pairs: {len(all_pairs)}")

    rng = np.random.RandomState(config.random_seed)
    if len(all_pairs) > config.amazon_max_train_pairs:
        indices = rng.choice(len(all_pairs), config.amazon_max_train_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in indices]
        logger.info(f"Capped to {len(all_pairs)} pairs")

    rng.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * config.train_fraction)

    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    logger.info(f"Train pairs: {len(train_pairs)}")
    logger.info(f"Val pairs:   {len(val_pairs)}")

    return train_pairs, val_pairs, item_texts


def build_mind_training_pairs(
    config=DEFAULT_CONTRASTIVE_CONFIG,
    data_config=DEFAULT_MIND_CONFIG,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, str]]:
    """
    Build (anchor, positive) pairs from MIND co-click data.

    Each pair: (text of article A, text of article B) where A and B
    were co-clicked by at least `min_support` users.

    Returns:
        (train_pairs, val_pairs, item_texts)
    """
    logger.info("=" * 80)
    logger.info("Building MIND training pairs")
    logger.info("=" * 80)

    mind_dataset = MINDDataset(data_dir=data_config.data_dir)
    mind_dataset.load_news()
    mind_dataset.load_behaviors()

    item_texts = mind_dataset.get_item_texts(
        use_title=data_config.use_title,
        use_abstract=data_config.use_abstract,
        use_category=data_config.use_category,
        use_subcategory=data_config.use_subcategory,
        template=data_config.text_template,
    )

    coclick_pairs = mind_dataset.get_coclick_pairs(
        min_support=config.mind_min_coclick_support,
    )

    all_pairs: List[Tuple[str, str]] = []
    for nid_a, nid_b in coclick_pairs:
        if nid_a in item_texts and nid_b in item_texts:
            all_pairs.append((item_texts[nid_a], item_texts[nid_b]))

    logger.info(f"Total (anchor, positive) pairs: {len(all_pairs)}")

    rng = np.random.RandomState(config.random_seed)
    if len(all_pairs) > config.mind_max_train_pairs:
        indices = rng.choice(len(all_pairs), config.mind_max_train_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in indices]
        logger.info(f"Capped to {len(all_pairs)} pairs")

    rng.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * config.train_fraction)

    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    logger.info(f"Train pairs: {len(train_pairs)}")
    logger.info(f"Val pairs:   {len(val_pairs)}")

    return train_pairs, val_pairs, item_texts


# ============================================================================
# MNR Loss (manual implementation — avoids HuggingFace Trainer dependency)
# ============================================================================

def mnr_loss(anchor_embs: torch.Tensor, positive_embs: torch.Tensor) -> torch.Tensor:
    """
    Multiple Negatives Ranking Loss (in-batch negatives).

    For a batch of B (anchor, positive) pairs, compute cosine similarity
    between each anchor and ALL positives.  The diagonal contains the
    correct matches; every off-diagonal element is an in-batch negative.

    This yields B * (B-1) negative comparisons per batch for free.

    Equivalent to cross-entropy where the label for each row is the
    diagonal index (i.e., label[i] = i).
    """
    similarity = torch.nn.functional.cosine_similarity(
        anchor_embs.unsqueeze(1),
        positive_embs.unsqueeze(0),
        dim=2,
    )
    # Scale by temperature (20.0 is the default in sentence-transformers MNR)
    similarity = similarity * 20.0
    labels = torch.arange(similarity.size(0), device=similarity.device)
    return torch.nn.functional.cross_entropy(similarity, labels)


# ============================================================================
# Tokenization helper
# ============================================================================

def tokenize_batch(texts: List[str], tokenizer, max_length: int, device: str):
    """Tokenize a list of texts and move to device."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


# ============================================================================
# Validation
# ============================================================================

def evaluate_val(
    model: SentenceTransformer,
    val_pairs: List[Tuple[str, str]],
    max_eval: int = 5000,
) -> float:
    """
    Compute mean cosine similarity on validation pairs.

    Higher = model is placing paired items closer together.
    """
    n = min(len(val_pairs), max_eval)
    anchors = [p[0] for p in val_pairs[:n]]
    positives = [p[1] for p in val_pairs[:n]]

    a_embs = model.encode(anchors, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    p_embs = model.encode(positives, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)

    cosine_sims = np.sum(a_embs * p_embs, axis=1)
    return float(np.mean(cosine_sims))


# ============================================================================
# Training
# ============================================================================

def train(
    dataset_name: str,
    config=DEFAULT_CONTRASTIVE_CONFIG,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Fine-tune a sentence-transformer with MNR loss on behavioral pairs.

    Uses a manual PyTorch training loop for full transparency and to
    avoid version-compatibility issues with the HuggingFace Trainer.

    Args:
        dataset_name: "amazon" or "mind"
        config: ContrastiveFineTuneConfig
        output_dir: Override for model save directory

    Returns:
        Path to the saved fine-tuned model
    """
    from tqdm import tqdm

    # ---- Prepare training data ----
    if dataset_name in ("amazon", "amazon_kdd"):
        train_pairs, val_pairs, item_texts = build_amazon_training_pairs(config)
    elif dataset_name == "mind":
        train_pairs, val_pairs, item_texts = build_mind_training_pairs(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs found — check dataset paths and filters")

    # ---- Load base model ----
    logger.info(f"Loading base model: {config.base_model}")
    model = SentenceTransformer(config.base_model)
    model.max_seq_length = config.max_seq_length
    device = model.device

    tokenizer = model.tokenizer

    # ---- Output directory ----
    if output_dir is None:
        safe_base = config.base_model.replace("/", "_")
        output_dir = MODELS_DIR / config.output_subdir / f"{safe_base}_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_pairs) // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"  Base model:       {config.base_model}")
    logger.info(f"  Dataset:          {dataset_name}")
    logger.info(f"  Train pairs:      {len(train_pairs)}")
    logger.info(f"  Val pairs:        {len(val_pairs)}")
    logger.info(f"  Batch size:       {config.batch_size}")
    logger.info(f"  Epochs:           {config.epochs}")
    logger.info(f"  Learning rate:    {config.learning_rate}")
    logger.info(f"  Warmup steps:     {warmup_steps}")
    logger.info(f"  Total steps:      {total_steps}")
    logger.info(f"  Output dir:       {output_dir}")
    logger.info("=" * 80)

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # FP16 mixed precision for GPU speed
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training loop ----
    start_time = time.time()
    global_step = 0
    best_val_sim = -1.0
    rng = np.random.RandomState(config.random_seed)

    model.train()

    for epoch in range(config.epochs):
        perm = rng.permutation(len(train_pairs))
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(range(0, len(train_pairs), config.batch_size),
                     desc=f"Epoch {epoch+1}/{config.epochs}")

        for start in pbar:
            batch_indices = perm[start:start + config.batch_size]
            if len(batch_indices) < 2:
                continue

            anchor_texts = [train_pairs[i][0] for i in batch_indices]
            positive_texts = [train_pairs[i][1] for i in batch_indices]

            anchor_enc = tokenize_batch(anchor_texts, tokenizer,
                                        config.max_seq_length, str(device))
            positive_enc = tokenize_batch(positive_texts, tokenizer,
                                          config.max_seq_length, str(device))

            with torch.amp.autocast("cuda", enabled=use_amp):
                anchor_out = model(anchor_enc)["sentence_embedding"]
                positive_out = model(positive_enc)["sentence_embedding"]

                anchor_out = torch.nn.functional.normalize(anchor_out, p=2, dim=1)
                positive_out = torch.nn.functional.normalize(positive_out, p=2, dim=1)

                loss = mnr_loss(anchor_out, positive_out)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if global_step < warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            # Periodic validation
            if global_step % config.eval_steps == 0:
                model.eval()
                val_sim = evaluate_val(model, val_pairs)
                model.train()

                logger.info(f"  Step {global_step}: val_cosine_sim = {val_sim:.4f}")

                if val_sim > best_val_sim:
                    best_val_sim = val_sim
                    model.save(str(output_dir))
                    logger.info(f"  New best model saved (val_sim={val_sim:.4f})")

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1} — avg_loss: {avg_loss:.4f}")

    # Final validation and save
    model.eval()
    final_val_sim = evaluate_val(model, val_pairs)
    logger.info(f"Final val_cosine_sim = {final_val_sim:.4f}")

    if final_val_sim >= best_val_sim:
        model.save(str(output_dir))
        logger.info(f"Final model saved to {output_dir}")

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ---- Save training metadata ----
    meta = {
        "dataset": dataset_name,
        "base_model": config.base_model,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "loss_type": config.loss_type,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "total_items": len(item_texts),
        "final_val_cosine_sim": round(final_val_sim, 4),
        "best_val_cosine_sim": round(best_val_sim, 4),
        "training_time_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return output_dir


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section 10.3 — Contrastive fine-tuning of item text encoders"
    )
    parser.add_argument(
        "--dataset", type=str, default="amazon",
        choices=["amazon", "amazon_kdd", "mind"],
        help="Dataset to train on",
    )
    parser.add_argument("--base_model", type=str, default=None,
                        help="Override base model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=None,
                        help="Cap training pairs")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    config = DEFAULT_CONTRASTIVE_CONFIG

    # Allow CLI overrides
    if args.base_model:
        config.base_model = args.base_model
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_train_pairs:
        if args.dataset in ("amazon", "amazon_kdd"):
            config.amazon_max_train_pairs = args.max_train_pairs
        else:
            config.mind_max_train_pairs = args.max_train_pairs

    out = Path(args.output_dir) if args.output_dir else None

    model_dir = train(
        dataset_name=args.dataset,
        config=config,
        output_dir=out,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Fine-tuned model saved to: {model_dir}")
    print(f"\nNext step — evaluate with:")
    print(f"  python evaluate_finetuned.py --dataset {args.dataset}")
    print("=" * 80)


if __name__ == "__main__":
    main()
