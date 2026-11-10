"""
Section 12.4: train the augmented MLP ranker for each feature set (an ablation).

    content -> content_history -> content_history_llm_item -> full

(see models/augmented_ranker.py). Every feature set is trained on the same training
examples: for each train session, the target plus hard negatives (nearest to the
history mean) and random negatives. Early stopping uses a validation loss computed on
validation *sessions*, so no session contributes to both. Test slates are scored by
evaluate_augmented.py.

Outputs:
    <work_dir>/models/ranker_<feature_set>.pt
    <work_dir>/metrics/augmented_ranker_training.json

Examples:
    python train_augmented_ranker.py --work-dir outputs/tiny --feature-sets content content_history
    python train_augmented_ranker.py --work-dir outputs/ranker                  # all four feature sets
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np
import torch

from config import FEATURE_SETS, AugmentedRankerConfig, work_paths
from data.amazon_session_loader import PreparedData
from models.augmented_ranker import AugmentedRanker
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json
from utils.ranker_data import load_feature_tables, to_tensors, training_examples

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    d = AugmentedRankerConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--max-train-sessions", type=int, default=None)
    p.add_argument("--max-val-sessions", type=int, default=20_000)
    p.add_argument("--allow-partial-coverage", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def batches(ex, order, batch_size, device):
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        rows = ex["hist_row"][idx]
        yield (torch.from_numpy(ex["cand"][idx]).to(device), torch.from_numpy(ex["hist"][rows]).to(device),
               torch.from_numpy(ex["hist_mask"][rows]).to(device), torch.from_numpy(ex["session_row"][idx]).to(device),
               torch.from_numpy(ex["label"][idx]).to(device))


def mean_loss(model, ex, batch_size, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for cand, hist, mask, sess, y in batches(ex, np.arange(len(ex["cand"])), batch_size * 4, device):
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(
                model(cand, hist, mask, sess), y, reduction="sum"))
    model.train()
    return total / len(ex["cand"])


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = AugmentedRankerConfig()
    set_seed(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)
    top_level = max(FEATURE_SETS.index(f) for f in args.feature_sets)
    tables = load_feature_tables(paths, data, need_item_llm=top_level >= 2, need_intent=top_level >= 3,
                                 min_coverage=cfg.min_llm_coverage, allow_partial=args.allow_partial_coverage)
    common = dict(content=tables["content"], session_row=tables["session_row"], max_history=cfg.max_history_items,
                  num_hard=cfg.num_hard_negatives, num_random=cfg.num_random_negatives, hard_pool=cfg.hard_negative_pool)
    train_ex = training_examples(data, "train", seed=args.seed, max_sessions=args.max_train_sessions, **common)
    val_ex = training_examples(data, "val", seed=args.seed + 1, max_sessions=args.max_val_sessions, **common)
    logger.info("Training examples: %d train, %d val (positives %.0f%%)", len(train_ex["cand"]),
                len(val_ex["cand"]), 100 * train_ex["label"].mean())

    summary = {}
    for feature_set in args.feature_sets:
        set_seed(args.seed)
        level = FEATURE_SETS.index(feature_set)
        model = AugmentedRanker(feature_set, **to_tensors(tables, level), hidden_dims=cfg.hidden_dims,
                                dropout=cfg.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(args.seed)
        best, bad, history = math.inf, 0, []
        ckpt = paths.models_dir / f"ranker_{feature_set}.pt"
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for cand, hist, mask, sess, y in batches(train_ex, rng.permutation(len(train_ex["cand"])),
                                                     args.batch_size, device):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(cand, hist, mask, sess), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item() * len(y)
            val = mean_loss(model, val_ex, args.batch_size, device)
            history.append({"epoch": epoch, "train_loss": running / len(train_ex["cand"]), "val_loss": val})
            if val < best:
                best, bad = val, 0
                torch.save({"feature_set": feature_set, "hidden_dims": list(cfg.hidden_dims),
                            "dropout": cfg.dropout, "state_dict": {k: v for k, v in model.state_dict().items()
                                                                   if k.startswith("mlp.")}}, ckpt)
            else:
                bad += 1
                if bad >= args.patience:
                    break
        summary[feature_set] = {"input_dim": model.in_dim, "epochs_completed": len(history), "best_val_loss": best,
                                "seconds": round(time.time() - start, 1), "history": history}
        logger.info("%-26s input dim %4d  epochs %2d  best val loss %.4f", feature_set, model.in_dim,
                    len(history), best)
    write_json(paths.metrics_dir / "augmented_ranker_training.json", run_record(
        "train_augmented_ranker.py", args, encoder=tables["encoder"], llm_coverage=tables["coverage"],
        llm_source=tables.get("llm_source", {}), train_examples=int(len(train_ex["cand"])),
        val_examples=int(len(val_ex["cand"])), feature_sets=summary))


if __name__ == "__main__":
    main()
