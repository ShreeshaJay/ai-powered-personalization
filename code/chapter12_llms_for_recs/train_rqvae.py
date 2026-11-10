"""
Section 12.2: train an RQ-VAE and assign a Semantic ID to every item.

Reads <work_dir>/embeddings/item_embeddings.npz (from encode_items.py), trains the
residual quantiser with k-means initialisation and dead-code reset, then writes:

    <work_dir>/models/rqvae.pt
    <work_dir>/embeddings/item_semantic_ids.npz   item_ids, codes (N, L), semantic_ids (N, L + 1)
    <work_dir>/metrics/rqvae.json                 reconstruction, per-level code usage, collisions

The last Semantic ID level is a collision suffix: items with identical codes get 0, 1, 2, ...

Examples:
    python train_rqvae.py --work-dir outputs/tiny --epochs 10
    python train_rqvae.py --work-dir outputs/synthetic --codebook-size 16 --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import RQVAEConfig, work_paths
from data.amazon_session_loader import PreparedData, load_item_embeddings
from models.rqvae import RQVAE, add_collision_suffix, code_statistics
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    d = RQVAEConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--latent-dim", type=int, default=d.latent_dim)
    p.add_argument("--hidden-dim", type=int, default=d.encoder_hidden_dim)
    p.add_argument("--num-levels", type=int, default=d.num_levels)
    p.add_argument("--codebook-size", type=int, default=d.codebook_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--commitment-weight", type=float, default=d.commitment_weight)
    p.add_argument("--dead-code-fraction", type=float, default=d.dead_code_fraction,
                   help="Reset an entry whose usage falls below this fraction of the average entry's usage")
    p.add_argument("--no-kmeans-init", action="store_true", help="Ablation: start from random codebooks")
    p.add_argument("--no-dead-code-reset", action="store_true", help="Ablation: never reset unused codes")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def evaluate(model: RQVAE, x: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    totals = {"reconstruction_loss": 0.0, "commitment_loss": 0.0}
    cos, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            out = model(xb)
            for key in totals:
                totals[key] += float(out[key]) * len(xb)
            cos += float(torch.nn.functional.cosine_similarity(out["x_hat"], xb, dim=1).sum())
            n += len(xb)
    model.train()
    return {**{k: v / n for k, v in totals.items()}, "reconstruction_cosine": cos / n}


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = RQVAEConfig()
    set_seed(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)
    embeddings, meta = load_item_embeddings(paths.item_embeddings, data.item_ids)
    logger.info("Loaded %d item embeddings (dim %d, encoder %s)", *embeddings.shape, meta["encoder"])

    x_all = torch.from_numpy(embeddings).to(device)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(embeddings))
    n_val = max(1, int(cfg.val_fraction * len(embeddings)))
    x_val, x_train = x_all[perm[:n_val]], x_all[perm[n_val:]]

    threshold = 0.0 if args.no_dead_code_reset else args.dead_code_fraction
    model = RQVAE(embeddings.shape[1], args.hidden_dim, args.latent_dim, args.num_levels,
                  args.codebook_size, cfg.ema_decay, args.commitment_weight, threshold).to(device)
    if not args.no_kmeans_init:
        sample = x_train[torch.randperm(len(x_train), generator=torch.Generator().manual_seed(args.seed))[:cfg.kmeans_init_samples].to(device)]
        model.initialize_codebooks(sample, iters=cfg.kmeans_iters, seed=args.seed)
        logger.info("Initialised %d codebooks with k-means on %d samples", args.num_levels, len(sample))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), drop_last=len(x_train) > args.batch_size)
    best_val, history = float("inf"), []
    start = time.time()
    model_path = paths.models_dir / "rqvae.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for (xb,) in loader:
            optimizer.zero_grad()
            out = model(xb)
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += out["total_loss"].item() * len(xb)
        val = evaluate(model, x_val, 8192)
        val_total = val["reconstruction_loss"] + val["commitment_loss"]
        resets = sum(q.resets for q in model.quantizers)
        history.append({"epoch": epoch, "train_loss": running / len(x_train), "val_loss": val_total,
                        "val_reconstruction_cosine": val["reconstruction_cosine"], "dead_code_resets_so_far": resets})
        logger.info("epoch %d train %.5f val %.5f val cos %.4f resets %d",
                    epoch, running / len(x_train), val_total, val["reconstruction_cosine"], resets)
        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), model_path)
    train_seconds = time.time() - start

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    codes = model.get_codes(x_all)
    semantic_ids, suffix_size = add_collision_suffix(codes)
    stats = code_statistics(codes, args.codebook_size)
    full = evaluate(model, x_all, 8192)
    level_sizes = [args.codebook_size] * args.num_levels + [suffix_size]
    np.savez_compressed(paths.semantic_ids, item_ids=np.asarray(data.item_ids), codes=codes,
                        semantic_ids=semantic_ids, level_sizes=np.asarray(level_sizes),
                        encoder=np.asarray(meta["encoder"]))

    logger.info("Distinct full codes: %d of %d items; %.1f%% of items share codes with another item; "
                "suffix vocabulary %d", stats["distinct_full_codes"], len(codes),
                100 * stats["items_sharing_codes_with_another_item"], suffix_size)
    for level in stats["levels"]:
        logger.info("level %d: %d of %d codes used, perplexity %.1f, %d distinct prefixes",
                    level["level"], level["codes_used"], level["codebook_size"], level["perplexity"],
                    level["distinct_prefixes"])
    write_json(paths.metrics_dir / "rqvae.json", run_record(
        "train_rqvae.py", args,
        config=asdict(cfg), encoder=meta["encoder"], parameters=model.count_parameters(),
        train_seconds=round(train_seconds, 1), best_val_loss=best_val,
        catalog_reconstruction=full, code_statistics=stats, level_sizes=level_sizes,
        history=history))
    logger.info("Wrote %s", paths.semantic_ids)


if __name__ == "__main__":
    main()
