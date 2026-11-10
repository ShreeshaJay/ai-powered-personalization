"""
SASRec baseline for comparison with HSTU on the same splits.

Uses item-id embeddings + causal Transformer encoder (softmax attention),
trained with the same sampled-softmax retrieval objective as HSTU and evaluated
with the same full-catalog protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpointing import (  # noqa: E402
    build_train_config,
    load_checkpoint_file,
    load_training_checkpoint,
    resolve_resume_path,
    save_training_checkpoint,
    validate_config_match,
    write_resume_state,
)
from data_loader import save_data_bundle  # noqa: E402
from dataset import build_datasets, collate_eval, collate_train  # noqa: E402
from evaluate import evaluate_full_catalog, load_bundle, output_subdir  # noqa: E402
from train import negatives_for, shift_batch  # noqa: E402
from training_utils import ADAM_BETAS, GRAD_CLIP, init_sasrec_weights, sampled_softmax_loss  # noqa: E402
from utils import Timer, append_jsonl, get_device, get_hardware_info, set_seed  # noqa: E402


class SASRecBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class SASRecRecommender(nn.Module):
    def __init__(
        self,
        num_items: int,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.2,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.temperature = temperature
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([SASRecBlock(dim, num_heads, dim * 4, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

    def hidden_states(self, item_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Right-padded inputs; position 0 is the oldest event in the window."""
        batch_size, seq_len = item_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.item_emb(item_ids) + self.pos_emb(positions)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=item_ids.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, causal, ~attn_mask)
        return self.final_norm(x)

    def encode_sequence(self, item_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden_states(item_ids, attn_mask)
        lengths = attn_mask.sum(dim=1).long().clamp(min=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths - 1]

    def item_embeddings(self) -> torch.Tensor:
        return self.item_emb.weight


def sasrec_encode(model: SASRecRecommender, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return model.encode_sequence(batch["items"].to(device), batch["mask"].to(device))


def train_one_epoch(
    model: SASRecRecommender,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_items: int,
    num_negatives: int,
    use_amp: bool,
    grad_clip: float = GRAD_CLIP,
    chunk_size: int = 4096,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(loader, desc="Train SASRec", leave=False):
        b = shift_batch(batch, device)
        neg = negatives_for(b["target_items"], b["target_mask"], num_items, num_negatives)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            hidden = model.hidden_states(b["input_items"], b["input_mask"])
            valid = b["input_mask"] & b["target_mask"]
            h, pos, negs = hidden[valid], b["target_items"][valid], neg[valid]
            emb = model.item_embeddings()
            # chunked to bound the (positions, negatives, dim) gather; sum / count = per-position mean
            loss = h.new_zeros((), dtype=torch.float32)
            for start in range(0, h.size(0), chunk_size):
                sl = slice(start, start + chunk_size)
                loss = loss + sampled_softmax_loss(h[sl], emb[pos[sl]], emb[negs[sl]], pos[sl], negs[sl],
                                                   model.temperature).sum()
            loss = loss / max(h.size(0), 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        steps += 1
    return total_loss / max(steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SASRec baseline")
    parser.add_argument("--dataset", choices=["yambda", "movielens", "synthetic"], default="yambda")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-days", type=float, default=30.0)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--num-negatives", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast on CUDA")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--epoch-snapshots", action="store_true",
                        help="Also keep a checkpoint per epoch under checkpoints/ (off by default)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / output_subdir(args.dataset)

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
    timer = Timer()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle, args.data_dir = load_bundle(args.dataset, args.data_dir, args.max_days, args.max_seq_len, args.seed)
    save_data_bundle(bundle, args.output_dir / "data_stats.json")

    train_ds, val_ds, test_ds = build_datasets(bundle, max_seq_len=args.max_seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                              collate_fn=collate_train, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_eval)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_eval)

    model = SASRecRecommender(
        num_items=bundle.num_items,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=ADAM_BETAS)
    config = build_train_config(args, bundle, model_name="sasrec")

    best_ckpt = args.output_dir / "sasrec_best.pt"
    last_ckpt = args.output_dir / "sasrec_last.pt"
    epoch_ckpt_dir = args.output_dir / "checkpoints"
    log_path = args.output_dir / "sasrec_train_log.jsonl"

    best_val_ndcg = -1.0
    best_epoch = 0
    start_epoch = 1
    epochs_without_improvement = 0
    resume_path = resolve_resume_path(args.output_dir, prefix="sasrec", resume=args.resume or args.resume_from is not None,
                                      resume_from=args.resume_from, fresh=args.fresh)
    if resume_path is not None:
        ckpt = load_training_checkpoint(resume_path, model=model, optimizer=optimizer, scaler=None, device=device)
        validate_config_match(ckpt.get("config", {}), config)
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_ndcg = float(ckpt.get("best_val_ndcg", -1.0))
        if start_epoch > args.epochs:
            print(json.dumps({"status": "already_complete", "resume_from": str(resume_path)}, indent=2))
            return
        state = {"resume_from": str(resume_path), "start_epoch": start_epoch}
        print(json.dumps({"status": "resumed", **state}, indent=2))
        write_resume_state(args.output_dir, state)
    else:
        init_sasrec_weights(model)

    completed_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, bundle.num_items, args.num_negatives,
                                     use_amp, grad_clip=args.grad_clip)
        val_metrics = evaluate_full_catalog(model, val_loader, device, encode=sasrec_encode)
        completed_epoch = epoch
        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
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
                           config=config, val_metrics=val_metrics, train_loss=train_loss)
        save_training_checkpoint(last_ckpt, **ckpt_kwargs)
        if args.epoch_snapshots:
            save_training_checkpoint(epoch_ckpt_dir / f"sasrec_epoch_{epoch:03d}.pt", **ckpt_kwargs)
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
    test_metrics = evaluate_full_catalog(model, test_loader, device, encode=sasrec_encode)

    metadata = {
        "model": "sasrec",
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
    with (args.output_dir / "sasrec_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps({"test": test_metrics, "best_checkpoint": str(best_ckpt)}, indent=2))


if __name__ == "__main__":
    main()
