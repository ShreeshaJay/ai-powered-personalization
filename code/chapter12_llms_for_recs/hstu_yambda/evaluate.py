"""Full-catalog HR@K and NDCG@K evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import load_checkpoint_file
from dataset import build_datasets, collate_eval
from model import HSTURecommender
from training_utils import NORM_EPS
from utils import full_catalog_ranks, get_device, set_seed, summarize_ranks


def hstu_encode(model: HSTURecommender, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return model.encode_sequence(
        batch["items"].to(device), batch["actions"].to(device), batch["times"].to(device), batch["mask"].to(device)
    )


@torch.no_grad()
def evaluate_full_catalog(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int = 10,
    encode: Optional[Callable] = None,
) -> Dict[str, float]:
    """Cosine scores against every item; see ``utils.summarize_ranks`` for the metric views."""
    model.eval()
    encode = encode or hstu_encode
    item_emb = F.normalize(model.item_embeddings().detach().float(), dim=-1, eps=NORM_EPS)
    masked, unmasked, in_window = [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        seq = F.normalize(encode(model, batch, device).float(), dim=-1, eps=NORM_EPS)
        ranks = full_catalog_ranks(seq @ item_emb.T, batch["targets"].to(device), batch["items"].to(device))
        masked += ranks["masked"].tolist()
        unmasked += ranks["unmasked"].tolist()
        in_window += ranks["target_in_window"].tolist()
    return summarize_ranks(masked, unmasked, in_window, k=k)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[HSTURecommender, dict]:
    ckpt = load_checkpoint_file(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = HSTURecommender(
        num_items=config["num_items"],
        num_actions=config["num_actions"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config


def load_bundle(dataset: str, data_dir: Optional[Path], max_days: float, max_seq_len: int, seed: int = 42):
    if dataset == "synthetic":
        from synthetic_data import load_synthetic_data

        return load_synthetic_data(max_seq_len=max_seq_len, seed=seed), None
    if dataset == "movielens":
        from movielens_loader import DEFAULT_ML1M_DIR, load_movielens_data

        data_dir = data_dir or DEFAULT_ML1M_DIR
        return load_movielens_data(data_dir, max_seq_len=max_seq_len), data_dir
    from data_loader import DEFAULT_DATA_DIR, load_hstu_data

    data_dir = data_dir or DEFAULT_DATA_DIR
    return load_hstu_data(data_dir, max_days=max_days, max_seq_len=max_seq_len), data_dir


def output_subdir(dataset: str) -> str:
    return {"movielens": "ml1m", "yambda": "yambda", "synthetic": "synthetic"}[dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HSTU")
    parser.add_argument("--dataset", choices=["yambda", "movielens", "synthetic"], default="yambda")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-days", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(__file__).resolve().parent / "outputs" / output_subdir(args.dataset) / "eval_metrics.json"

    set_seed(args.seed)
    device = get_device()
    model, config = load_checkpoint(args.checkpoint, device)
    bundle, _ = load_bundle(args.dataset, args.data_dir, args.max_days, config["max_seq_len"], args.seed)
    if bundle.num_items != config["num_items"]:
        raise ValueError(f"checkpoint has {config['num_items']} items, data has {bundle.num_items}")
    _, val_ds, test_ds = build_datasets(bundle, max_seq_len=config["max_seq_len"])
    loader = DataLoader(test_ds if args.split == "test" else val_ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_eval, num_workers=0)
    metrics = evaluate_full_catalog(model, loader, device)

    payload = {"split": args.split, "checkpoint": str(args.checkpoint), "config": config, **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
