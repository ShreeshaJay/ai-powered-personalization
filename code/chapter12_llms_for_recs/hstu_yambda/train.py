"""Train HSTU on Yandex Yambda, MovieLens-1M, or a synthetic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import (
    build_train_config,
    load_checkpoint_file,
    load_training_checkpoint,
    resolve_resume_path,
    save_training_checkpoint,
    validate_config_match,
    write_resume_state,
)
from data_loader import save_data_bundle
from dataset import build_datasets, collate_eval, collate_train, sample_negatives
from evaluate import evaluate_full_catalog, load_bundle, output_subdir
from model import HSTURecommender
from training_utils import ADAM_BETAS, GRAD_CLIP, W_RATING, compute_hstu_loss, init_hstu_weights
from utils import Timer, append_jsonl, get_device, get_hardware_info, set_seed


def shift_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    """Inputs are positions 0..n-2 of each window, targets positions 1..n-1."""
    items = batch["items"].to(device)
    mask = batch["mask"].to(device)
    return {
        "input_items": items[:, :-1],
        "input_actions": batch["actions"].to(device)[:, :-1],
        "input_times": batch["times"].to(device)[:, :-1],
        "input_mask": mask[:, :-1],
        "target_items": items[:, 1:],
        "target_values": batch["values"].to(device)[:, 1:],
        "target_mask": mask[:, 1:],
    }


def negatives_for(target_items: torch.Tensor, target_mask: torch.Tensor, num_items: int, num_negatives: int) -> torch.Tensor:
    neg = target_items.new_zeros(*target_items.shape, num_negatives)
    neg[target_mask] = sample_negatives(target_items[target_mask], num_items, num_negatives)
    return neg


def train_one_epoch(
    model: HSTURecommender,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_items: int,
    num_negatives: int,
    use_amp: bool,
    w_rating: float = W_RATING,
    grad_clip: float = GRAD_CLIP,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "retrieval": 0.0, "rating": 0.0, "rated_fraction": 0.0}
    steps = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        b = shift_batch(batch, device)
        neg = negatives_for(b["target_items"], b["target_mask"], num_items, num_negatives)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, metrics = compute_hstu_loss(model, neg_items=neg, w_rating=w_rating, **b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key in totals:
            totals[key] += metrics[key]
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HSTU")
    parser.add_argument("--dataset", choices=["yambda", "movielens", "synthetic"], default="yambda")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-days", type=float, default=30.0)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-rating", type=float, default=W_RATING)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--num-negatives", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast on CUDA")
    parser.add_argument("--resume", action="store_true", help="Resume from <output-dir>/hstu_last.pt if it exists")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a specific checkpoint file")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and train from scratch")
    parser.add_argument("--epoch-snapshots", action="store_true",
                        help="Also keep a checkpoint per epoch under checkpoints/ (off by default)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "outputs" / output_subdir(args.dataset)

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
    timer = Timer()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle, args.data_dir = load_bundle(args.dataset, args.data_dir, args.max_days, args.max_seq_len, args.seed)
    save_data_bundle(bundle, args.output_dir / "data_stats.json")
    print(json.dumps(bundle.stats, indent=2))

    train_ds, val_ds, test_ds = build_datasets(bundle, max_seq_len=args.max_seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                              collate_fn=collate_train, num_workers=args.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_eval,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_eval,
                             num_workers=args.num_workers)

    model = HSTURecommender(
        num_items=bundle.num_items,
        num_actions=bundle.num_actions,
        dim=args.dim,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=ADAM_BETAS)
    config = build_train_config(args, bundle, model_name="hstu")

    best_ckpt = args.output_dir / "hstu_best.pt"
    last_ckpt = args.output_dir / "hstu_last.pt"
    epoch_ckpt_dir = args.output_dir / "checkpoints"
    log_path = args.output_dir / "train_log.jsonl"

    best_val_ndcg = -1.0
    best_epoch = 0
    start_epoch = 1
    epochs_without_improvement = 0
    resume_path = resolve_resume_path(
        args.output_dir,
        prefix="hstu",
        resume=args.resume or args.resume_from is not None,
        resume_from=args.resume_from,
        fresh=args.fresh,
    )
    if resume_path is not None:
        ckpt = load_training_checkpoint(resume_path, model=model, optimizer=optimizer, scaler=None, device=device)
        validate_config_match(ckpt.get("config", {}), config)
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_ndcg = float(ckpt.get("best_val_ndcg", -1.0))
        if start_epoch > args.epochs:
            print(json.dumps({"status": "already_complete", "resume_from": str(resume_path),
                              "completed_epochs": ckpt["epoch"], "target_epochs": args.epochs}, indent=2))
            return
        state = {"resume_from": str(resume_path), "start_epoch": start_epoch, "best_val_ndcg": best_val_ndcg}
        print(json.dumps({"status": "resumed", **state}, indent=2))
        write_resume_state(args.output_dir, state)
    else:
        init_hstu_weights(model)

    completed_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, bundle.num_items,
                                        args.num_negatives, use_amp, w_rating=args.w_rating, grad_clip=args.grad_clip)
        val_metrics = evaluate_full_catalog(model, val_loader, device)
        completed_epoch = epoch
        record = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        append_jsonl(log_path, record)
        print(json.dumps(record))

        is_new_best = val_metrics["ndcg@10"] > best_val_ndcg
        if is_new_best:
            best_val_ndcg = val_metrics["ndcg@10"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        ckpt_kwargs = dict(model=model, optimizer=optimizer, scaler=None, epoch=epoch, best_val_ndcg=best_val_ndcg,
                           config=config, val_metrics=val_metrics, train_loss=train_metrics["loss"])
        save_training_checkpoint(last_ckpt, **ckpt_kwargs)
        if args.epoch_snapshots:
            save_training_checkpoint(epoch_ckpt_dir / f"hstu_epoch_{epoch:03d}.pt", **ckpt_kwargs)
        if is_new_best:
            save_training_checkpoint(best_ckpt, **ckpt_kwargs)

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            print(json.dumps({"status": "early_stop", "epoch": epoch, "patience": args.patience,
                              "best_val_ndcg@10": best_val_ndcg}))
            break

    if not best_ckpt.exists():
        raise RuntimeError("No checkpoint saved; training may have failed before epoch 1 completed.")

    ckpt = load_checkpoint_file(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_full_catalog(model, test_loader, device)

    metadata = {
        "model": "hstu",
        "dataset": args.dataset,
        "data_dir": str(args.data_dir) if args.data_dir else None,
        "config": config,
        "bf16_autocast": use_amp,
        "hardware": get_hardware_info(),
        "dataset_stats": bundle.stats,
        "wall_clock_seconds": timer.elapsed(),
        "best_epoch": best_epoch or int(ckpt["epoch"]),
        "best_val_ndcg@10": best_val_ndcg,
        "test_metrics": test_metrics,
        "best_checkpoint": best_ckpt.name,
        "resumed_from": str(resume_path) if resume_path else None,
        "completed_epochs": completed_epoch,
        "stopped_early": stopped_early,
    }
    with (args.output_dir / "hstu_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. Best checkpoint: {best_ckpt}")
    print(json.dumps({"test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
