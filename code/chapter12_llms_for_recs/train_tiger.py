"""
Section 12.2: train a TIGER-style generative retriever and evaluate it against baselines.

Training uses the train split; early stopping uses the validation loss; the test split is
touched once, at the end. Every method is scored on the same sampled test sessions, and
items already in the session history are removed from every method's list:

    tiger         constrained beam search over the catalog's Semantic ID prefix tree
    popularity    most frequent training targets
    content_knn   items nearest to the mean history embedding (cosine)

Results are reported for all test sessions and for the cold-target subset (targets that
appear in fewer than --cold-threshold training sessions).

Outputs:
    <work_dir>/models/tiger.pt
    <work_dir>/metrics/tiger.json

Examples:
    python train_tiger.py --work-dir outputs/tiny --epochs 3 --max-eval-sessions 300
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import asdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TIGERConfig, work_paths
from data.amazon_session_loader import PreparedData, load_item_embeddings
from data.semantic_id_dataset import SemanticIDSessions, SemanticIDTrie, collate, load_semantic_ids, sids_to_tokens
from models.tiger import TIGER
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json
from utils.faiss_index import InnerProductIndex, mean_history_queries
from utils.metrics import aggregate, metrics_from_rank, rank_in_list

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    d = TIGERConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--hidden-dim", type=int, default=d.hidden_dim)
    p.add_argument("--num-heads", type=int, default=d.num_heads)
    p.add_argument("--encoder-layers", type=int, default=d.num_encoder_layers)
    p.add_argument("--decoder-layers", type=int, default=d.num_decoder_layers)
    p.add_argument("--ffn-dim", type=int, default=d.ffn_dim)
    p.add_argument("--max-history-items", type=int, default=d.max_history_items)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--beam-width", type=int, default=d.beam_width, help="Depth of every reported list")
    p.add_argument("--history-margin", type=int, default=10,
                   help="Extra beams so that removing history items still leaves --beam-width items")
    p.add_argument("--max-eval-sessions", type=int, default=d.max_eval_sessions)
    p.add_argument("--max-val-sessions", type=int, default=20_000)
    p.add_argument("--cold-threshold", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def evaluate_loss(model: TIGER, loader: DataLoader, weights, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model.loss(batch, weights)
            total += float(loss) * len(batch["decoder_target"])
            n += len(batch["decoder_target"])
    model.train()
    return total / max(n, 1)


def filtered(ranked: Sequence[int], history: Sequence[int], k: int) -> List[int]:
    seen = set(history)
    return [i for i in ranked if i not in seen][:k]


def score_lists(lists: Dict[str, List[List[int]]], targets: np.ndarray, subsets: Dict[str, np.ndarray],
                k_values: Sequence[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, ranked in lists.items():
        per = [metrics_from_rank(rank_in_list(r, int(t)), k_values) for r, t in zip(ranked, targets)]
        out[method] = {name: aggregate([per[i] for i in np.flatnonzero(mask)]) for name, mask in subsets.items()}
    return out


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = TIGERConfig()
    set_seed(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)

    semantic_ids, level_sizes = load_semantic_ids(paths.semantic_ids, data.item_ids)
    item_tokens = sids_to_tokens(semantic_ids, level_sizes)
    trie = SemanticIDTrie(item_tokens)
    weights = list(cfg.level_weights[:len(level_sizes) - 1]) + [1.0]
    weights += [1.0] * (len(level_sizes) - len(weights))

    def dataset(split: str, limit=None, seed=0) -> SemanticIDSessions:
        frame = data.split(split)
        if limit is not None and frame.height > limit:
            frame = frame.sample(n=limit, seed=seed)
        return SemanticIDSessions(data.history_indices(frame), data.target_indices(frame),
                                  item_tokens, args.max_history_items)

    train_ds = dataset("train")
    val_ds = dataset("val", args.max_val_sessions, args.seed)
    logger.info("Train %d sessions, val %d sessions, %d items, level sizes %s",
                len(train_ds), len(val_ds), data.num_items, level_sizes)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, collate_fn=collate)

    model = TIGER(level_sizes, args.hidden_dim, args.num_heads, args.encoder_layers, args.decoder_layers,
                  args.ffn_dim, cfg.dropout, args.max_history_items).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    logger.info("TIGER parameters: %s", f"{model.count_parameters():,}")

    model_path = paths.models_dir / "tiger.pt"
    best_val, bad_epochs, history = math.inf, 0, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model.loss(batch, weights, cfg.label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(batch["decoder_target"])
            n += len(batch["decoder_target"])
        val_loss = evaluate_loss(model, val_loader, weights, device)
        history.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss})
        logger.info("epoch %d train %.4f val %.4f", epoch, running / n, val_loss)
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("Early stopping after epoch %d", epoch)
                break
    train_seconds = time.time() - start
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # ---- Test evaluation: same sessions for every method ----
    test = data.split("test")
    if test.height > args.max_eval_sessions:
        test = test.sample(n=args.max_eval_sessions, seed=args.seed)
    histories = data.history_indices(test)
    targets = data.target_indices(test)
    k_values = [k for k in cfg.k_values if k <= args.beam_width]
    depth = args.beam_width

    test_ds = SemanticIDSessions(histories, targets, item_tokens, args.max_history_items)
    tiger_lists: List[List[int]] = []
    eval_start = time.time()
    for start_idx in range(0, len(test_ds), 64):
        batch = collate([test_ds[i] for i in range(start_idx, min(start_idx + 64, len(test_ds)))])
        beams = model.constrained_beam_search(batch["encoder_input"].to(device), batch["encoder_mask"].to(device),
                                              trie, args.beam_width + args.history_margin)
        for offset, row in enumerate(beams):
            tiger_lists.append(filtered([item for item, _ in row], histories[start_idx + offset], depth))
    decode_seconds = time.time() - eval_start

    train_target_counts = data.train_target_counts()
    popular = np.argsort(-train_target_counts, kind="stable")[: depth + max(len(h) for h in histories)].tolist()
    popularity_lists = [filtered(popular, h, depth) for h in histories]

    embeddings, emb_meta = load_item_embeddings(paths.item_embeddings, data.item_ids)
    index = InnerProductIndex(embeddings)
    _, knn = index.search(mean_history_queries(embeddings, histories), depth + max(len(h) for h in histories))
    knn_lists = [filtered(row.tolist(), h, depth) for row, h in zip(knn, histories)]

    item_counts = data.train_item_counts()
    cold = item_counts[targets] < args.cold_threshold
    repeat = np.array([t in set(h) for t, h in zip(targets.tolist(), histories)])
    subsets = {"all": np.ones(len(targets), dtype=bool), "cold_target": cold, "warm_target": ~cold}
    results = score_lists({"tiger": tiger_lists, "popularity": popularity_lists, "content_knn": knn_lists},
                          targets, subsets, k_values)

    for method, by_subset in results.items():
        for subset, m in by_subset.items():
            if m.get("num_queries"):
                logger.info("%-12s %-12s n=%5d  recall@10 %.4f  ndcg@10 %.4f  mrr %.4f", method, subset,
                            m["num_queries"], m.get("recall@10", float("nan")), m.get("ndcg@10", float("nan")), m["mrr"])
    write_json(paths.metrics_dir / "tiger.json", run_record(
        "train_tiger.py", args, config=asdict(cfg), level_sizes=level_sizes, level_weights=weights,
        parameters=model.count_parameters(), item_encoder=emb_meta["encoder"],
        train_sessions=len(train_ds), val_sessions=len(val_ds), test_sessions_evaluated=len(targets),
        cold_target_sessions=int(cold.sum()), target_in_history_rate=float(repeat.mean()),
        epochs_completed=len(history), best_val_loss=best_val, train_seconds=round(train_seconds, 1),
        decode_seconds=round(decode_seconds, 1), history=history, results=results,
        notes="History items are removed from every list, so a target that repeats a history item "
              "cannot be hit (see target_in_history_rate)."))
    logger.info("Wrote %s", paths.metrics_dir / "tiger.json")


if __name__ == "__main__":
    main()
