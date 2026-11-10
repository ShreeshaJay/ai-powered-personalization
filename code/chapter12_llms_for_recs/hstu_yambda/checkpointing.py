"""Save/load training checkpoints with resume support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def build_train_config(args, bundle, model_name: str = "hstu") -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "model": model_name,
        "dataset": args.dataset,
        "num_items": bundle.num_items,
        "num_actions": getattr(bundle, "num_actions", None),
        "dim": args.dim,
        "num_layers": args.num_layers,
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_negatives": args.num_negatives,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "no_amp": args.no_amp,
        "w_rating": getattr(args, "w_rating", 0.2),
        "adam_betas": getattr(args, "adam_betas", (0.9, 0.98)),
        "grad_clip": getattr(args, "grad_clip", 1.0),
        "patience": getattr(args, "patience", 10),
    }
    if hasattr(args, "max_days"):
        config["max_days"] = args.max_days
    if hasattr(args, "num_heads"):
        config["num_heads"] = args.num_heads
    return config


def validate_config_match(saved: Dict[str, Any], current: Dict[str, Any]) -> None:
    keys = ["model", "dataset", "num_items", "dim", "num_layers", "max_seq_len"]
    for key in keys:
        if key in saved and key in current and saved[key] != current[key]:
            raise ValueError(
                f"Checkpoint config mismatch on '{key}': "
                f"checkpoint={saved[key]!r}, current={current[key]!r}. "
                "Use --fresh to train from scratch."
            )


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val_ndcg: float,
    config: Dict[str, Any],
    val_metrics: Optional[Dict[str, float]] = None,
    train_loss: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "best_val_ndcg": best_val_ndcg,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "train_loss": train_loss,
        "rng_state": {
            "python": __import__("random").getstate(),
            "numpy": __import__("numpy").random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint_file(path: Path, map_location: torch.device | str | None = None) -> Dict[str, Any]:
    """Load a locally saved training checkpoint (includes optimizer/RNG state)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = load_checkpoint_file(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    rng = ckpt.get("rng_state")
    if rng:
        __import__("random").setstate(rng["python"])
        __import__("numpy").random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return ckpt


def resolve_resume_path(
    output_dir: Path,
    prefix: str,
    resume: bool,
    resume_from: Optional[Path],
    fresh: bool,
) -> Optional[Path]:
    if fresh:
        return None
    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        return resume_from
    if not resume:
        return None

    last_ckpt = output_dir / f"{prefix}_last.pt"
    if last_ckpt.exists():
        return last_ckpt
    return None


def write_resume_state(output_dir: Path, state: Dict[str, Any]) -> None:
    with (output_dir / "resume_state.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
