"""
Section 10.4: Multi-Modal Fusion — PCA baseline + CLIP-style contrastive alignment.

Approach A (PCA): Concatenate audio + lyrics + genre features, apply PCA.
Approach B (CLIP): Train projection heads to align audio ↔ lyrics in a shared
    space using InfoNCE contrastive loss. The training signal is purely the
    correspondence between modalities — no behavioral data is used.

Usage:
    # PCA baseline only
    python multimodal_fusion.py --mode pca

    # CLIP-style contrastive training (audio ↔ lyrics)
    python multimodal_fusion.py --mode clip

    # Both (recommended)
    python multimodal_fusion.py --mode both
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_MUSIC4ALL_CONFIG, DEFAULT_MULTIMODAL_CONFIG,
    EMBEDDINGS_DIR, MODELS_DIR, METRICS_DIR,
)
from data.music4all_dataset import Music4allDataset

logger = logging.getLogger(__name__)


# ============================================================================
# PCA Baseline (Approach A)
# ============================================================================

def pca_fusion(
    dataset: Music4allDataset,
    n_components: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate all loaded modalities and apply PCA.

    Returns:
        (track_ids, fused_embeddings) — L2-normalized, shape (N, n_components).
    """
    parts = []
    labels = []
    if dataset.audio_features is not None:
        parts.append(dataset.audio_features)
        labels.append(f"audio({dataset.audio_features.shape[1]})")
    if dataset.lyrics_features is not None:
        parts.append(dataset.lyrics_features)
        labels.append(f"lyrics({dataset.lyrics_features.shape[1]})")
    if dataset.genre_features is not None:
        parts.append(dataset.genre_features)
        labels.append(f"genre({dataset.genre_features.shape[1]})")

    concat = np.concatenate(parts, axis=1)
    logger.info(f"PCA input: {' + '.join(labels)} = {concat.shape[1]} dims")

    pca = PCA(n_components=n_components, random_state=42)
    fused = pca.fit_transform(concat).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA → {n_components} dims, explained variance: {explained:.1%}")

    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    fused = fused / norms

    return dataset.track_ids, fused


# ============================================================================
# CLIP-Style Contrastive Alignment (Approach B)
# ============================================================================

class ProjectionHead(nn.Module):
    """MLP projection from modality space to shared space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CLIPFusionModel(nn.Module):
    """
    CLIP-style dual-encoder: projects audio and lyrics into a shared space.

    For a batch of N tracks, the positive pair is (audio_i, lyrics_i),
    and the (N-1) other cross-modal pairs are negatives. The loss is
    symmetric InfoNCE: audio→lyrics + lyrics→audio.
    """

    def __init__(
        self,
        audio_dim: int,
        lyrics_dim: int,
        shared_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.audio_proj = ProjectionHead(audio_dim, hidden_dim, shared_dim, dropout)
        self.lyrics_proj = ProjectionHead(lyrics_dim, hidden_dim, shared_dim, dropout)

        if learnable_temperature:
            self.log_temp = nn.Parameter(torch.tensor(np.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(np.log(temperature)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=1e-4, max=100.0)

    def forward(
        self, audio: torch.Tensor, lyrics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss, audio_projections, lyrics_projections
        """
        a = self.audio_proj(audio)    # (B, shared_dim), L2-normalized
        l = self.lyrics_proj(lyrics)  # (B, shared_dim), L2-normalized

        # Similarity matrix: (B, B)
        logits = (a @ l.T) / self.temperature

        # Symmetric InfoNCE: each diagonal element is the positive pair
        labels = torch.arange(len(logits), device=logits.device)
        loss_a2l = F.cross_entropy(logits, labels)
        loss_l2a = F.cross_entropy(logits.T, labels)
        loss = (loss_a2l + loss_l2a) / 2

        return loss, a, l


def train_clip(
    dataset: Music4allDataset,
    cfg=None,
) -> Tuple[CLIPFusionModel, Dict]:
    """
    Train the CLIP-style audio ↔ lyrics alignment model.

    Returns:
        (trained_model, training_metadata)
    """
    if cfg is None:
        cfg = DEFAULT_MULTIMODAL_CONFIG

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training CLIP fusion on {device}")

    audio = torch.tensor(dataset.audio_features, dtype=torch.float32)
    lyrics = torch.tensor(dataset.lyrics_features, dtype=torch.float32)

    n = len(audio)
    rng = np.random.RandomState(cfg.random_seed)
    perm = rng.permutation(n)
    val_size = max(1, int(n * 0.1))
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = TensorDataset(audio[train_idx], lyrics[train_idx])
    val_ds = TensorDataset(audio[val_idx], lyrics[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0)

    audio_dim = dataset.audio_features.shape[1]
    lyrics_dim = dataset.lyrics_features.shape[1]

    model = CLIPFusionModel(
        audio_dim=audio_dim,
        lyrics_dim=lyrics_dim,
        shared_dim=cfg.shared_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        temperature=cfg.temperature,
        learnable_temperature=cfg.learnable_temperature,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"  Audio proj: {audio_dim} → {cfg.hidden_dim} → {cfg.shared_dim}")
    logger.info(f"  Lyrics proj: {lyrics_dim} → {cfg.hidden_dim} → {cfg.shared_dim}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader)
    )

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "temperature": []}
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for audio_batch, lyrics_batch in train_loader:
            audio_batch = audio_batch.to(device)
            lyrics_batch = lyrics_batch.to(device)

            loss, _, _ = model(audio_batch, lyrics_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for audio_batch, lyrics_batch in val_loader:
                audio_batch = audio_batch.to(device)
                lyrics_batch = lyrics_batch.to(device)
                loss, _, _ = model(audio_batch, lyrics_batch)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        temp = model.temperature.item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["temperature"].append(temp)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " ★"

        logger.info(
            f"Epoch {epoch:2d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"temp={temp:.4f} | lr={scheduler.get_last_lr()[0]:.2e}{improved}"
        )

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.0f}s, best val_loss={best_val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "audio_dim": audio_dim,
        "lyrics_dim": lyrics_dim,
        "shared_dim": cfg.shared_dim,
        "hidden_dim": cfg.hidden_dim,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "best_val_loss": best_val_loss,
        "train_tracks": len(train_idx),
        "val_tracks": val_size,
        "total_tracks": n,
        "param_count": param_count,
        "training_seconds": elapsed,
        "history": history,
    }

    return model, metadata


def encode_clip(
    model: CLIPFusionModel,
    dataset: Music4allDataset,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all tracks through the trained CLIP model.

    Returns:
        (audio_proj, lyrics_proj, fused) — each shape (N, shared_dim), L2-normalized.
        fused = average of audio_proj and lyrics_proj, re-normalized.
    """
    model.eval()
    audio = torch.tensor(dataset.audio_features, dtype=torch.float32)
    lyrics = torch.tensor(dataset.lyrics_features, dtype=torch.float32)

    audio_projs = []
    lyrics_projs = []

    with torch.no_grad():
        for i in range(0, len(audio), batch_size):
            a = audio[i:i + batch_size].to(device)
            l = lyrics[i:i + batch_size].to(device)
            a_proj = model.audio_proj(a).cpu().numpy()
            l_proj = model.lyrics_proj(l).cpu().numpy()
            audio_projs.append(a_proj)
            lyrics_projs.append(l_proj)

    audio_proj = np.concatenate(audio_projs, axis=0)
    lyrics_proj = np.concatenate(lyrics_projs, axis=0)

    fused = (audio_proj + lyrics_proj) / 2
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    fused = fused / np.maximum(norms, 1e-8)

    return audio_proj, lyrics_proj, fused


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Section 10.4: Multi-Modal Fusion (PCA + CLIP)"
    )
    parser.add_argument("--mode", choices=["pca", "clip", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shared-dim", type=int, default=None)
    parser.add_argument("--pca-dim", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )

    data_cfg = DEFAULT_MUSIC4ALL_CONFIG
    fusion_cfg = DEFAULT_MULTIMODAL_CONFIG

    if args.epochs:
        fusion_cfg.epochs = args.epochs
    if args.batch_size:
        fusion_cfg.batch_size = args.batch_size
    if args.shared_dim:
        fusion_cfg.shared_dim = args.shared_dim
    if args.pca_dim:
        fusion_cfg.pca_dim = args.pca_dim

    # Load data
    dataset = Music4allDataset(data_cfg.data_dir, config=data_cfg)
    dataset.load_audio()
    dataset.load_lyrics()
    dataset.load_genre()
    dataset.align_modalities()

    output_dir = EMBEDDINGS_DIR / fusion_cfg.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- PCA Baseline ---
    if args.mode in ("pca", "both"):
        logger.info("=" * 60)
        logger.info("Approach A: PCA Fusion")
        logger.info("=" * 60)

        track_ids, pca_emb = pca_fusion(dataset, n_components=fusion_cfg.pca_dim)

        pca_path = output_dir / "pca_fused.npz"
        np.savez_compressed(pca_path, ids=track_ids, embeddings=pca_emb)
        logger.info(f"Saved PCA embeddings: {pca_path}")

    # --- CLIP-style Contrastive ---
    if args.mode in ("clip", "both"):
        logger.info("=" * 60)
        logger.info("Approach B: CLIP-style Audio ↔ Lyrics Alignment")
        logger.info("=" * 60)

        model, metadata = train_clip(dataset, cfg=fusion_cfg)

        model_dir = MODELS_DIR / fusion_cfg.output_subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "clip_fusion_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model: {model_path}")

        meta_path = model_dir / "clip_training_metadata.json"
        serializable = {k: v for k, v in metadata.items() if k != "history"}
        serializable["history"] = {
            k: [round(x, 6) for x in v] for k, v in metadata["history"].items()
        }
        with open(meta_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved metadata: {meta_path}")

        device = torch.device(fusion_cfg.device if torch.cuda.is_available() else "cpu")
        audio_proj, lyrics_proj, clip_fused = encode_clip(
            model, dataset, device, batch_size=fusion_cfg.batch_size
        )

        clip_path = output_dir / "clip_fused.npz"
        np.savez_compressed(
            clip_path,
            ids=dataset.track_ids,
            audio_proj=audio_proj,
            lyrics_proj=lyrics_proj,
            fused=clip_fused,
        )
        logger.info(f"Saved CLIP embeddings: {clip_path}")

    # Also save single-modality embeddings for comparison
    for name, feats in [("audio", dataset.audio_features),
                        ("lyrics", dataset.lyrics_features),
                        ("genre", dataset.genre_features)]:
        if feats is not None:
            normed = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
            path = output_dir / f"{name}_raw.npz"
            np.savez_compressed(path, ids=dataset.track_ids, embeddings=normed)
            logger.info(f"Saved {name} embeddings: {path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
