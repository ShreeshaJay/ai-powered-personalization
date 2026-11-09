"""
Train Chapter 5's two-tower model on listen sequences and build the ANN index.

The model, dataset, training loop and evaluation are the published Chapter 5
code. Chapter 5 trains on Coveo browsing sessions with 50-dimensional
description embeddings; here the same classes train on Yambda listen
histories with Yambda's 128-dimensional audio embeddings as the item content,
so that retrieval, ranking and re-ranking share one catalog.

Outputs (retrieval/):
    two_tower.pt            state dict + constructor arguments + model_version
    content_embeddings.npy  item content embeddings that feed the user tower
    item_ids.npy            row i of both matrices belongs to item_ids[i]
    item_index.faiss        IndexFlatIP over L2-normalised item-tower outputs
    index_manifest.json     pairs the index with the model (Figure 13.4)
    popularity.json         cold-start list for users with no usable history
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import published
from common.artifacts import sha256_file, write_json

logger = logging.getLogger("ch13.build")


def build_sessions(listens: pd.DataFrame, split_timestamp: int, max_seq_len: int,
                   max_examples_per_user: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """
    One example per completed listen (played_ratio_pct >= 50) of an item the
    user has not played in the preceding `max_seq_len` listens. The history is
    those preceding listens in time order, as Chapter 5's dataset expects.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    ordered = listens.sort_values(["uid", "timestamp"], kind="stable")
    for uid, group in ordered.groupby("uid", sort=True):
        items = group["item_id"].to_numpy(np.int64)
        ratios = group["played_ratio_pct"].to_numpy()
        stamps = group["timestamp"].to_numpy(np.int64)
        positions = []
        for t in range(1, len(items)):
            window = items[max(0, t - max_seq_len):t]
            if ratios[t] >= 50 and items[t] not in window:
                positions.append(t)
        if len(positions) > max_examples_per_user:
            positions = sorted(rng.choice(positions, size=max_examples_per_user, replace=False).tolist())
        for t in positions:
            session = {
                "session_id": f"{uid}-{t}",
                "history_items": items[max(0, t - max_seq_len):t].tolist(),
                "ground_truth_item": int(items[t]),
                "max_timestamp": int(stamps[t]),
            }
            (train if stamps[t] < split_timestamp else val).append(session)
    return train, val


def popularity_hit_rate(sessions: List[Dict], ranked_items: List[int], k: int = 10) -> float:
    hits = 0
    for session in sessions:
        history = set(session["history_items"])
        top = [i for i in ranked_items if i not in history][:k]
        hits += session["ground_truth_item"] in top
    return hits / len(sessions) if sessions else 0.0


def item_vectors(model, content: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    """Item-tower outputs for every catalog item, L2-normalised (the batch job in Figure 13.4)."""
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(content), batch_size):
            batch = torch.tensor(content[start:start + batch_size], dtype=torch.float32)
            parts.append(model.encode_items(batch).cpu().numpy())
    vectors = np.ascontiguousarray(np.vstack(parts), dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def train_retrieval(data, out_dir: Path, *, split_timestamp: int, popularity: List[int],
                    baseline_popularity: List[int], model_version: str, index_version: str, epochs: int,
                    warmup_epochs: int, batch_size: int, max_examples_per_user: int, seed: int) -> dict:
    """`popularity` is the served cold-start list; `baseline_popularity` excludes the held-out plays."""
    ch5 = published.chapter5_two_tower()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(seed)

    train_sessions, val_sessions = build_sessions(
        data.listens, split_timestamp, ch5.MAX_SEQ_LEN, max_examples_per_user, seed)
    if not train_sessions or not val_sessions:
        raise RuntimeError("Not enough listen sequences to train or validate the two-tower model; "
                           "increase the number of users in the sample.")

    item_ids = data.item_ids.astype(np.int64)
    content = np.ascontiguousarray(data.content_embeddings, dtype=np.float32)
    lookup = ch5.ItemEmbeddingLookup(
        sku_to_idx={int(i): idx for idx, i in enumerate(item_ids)},
        idx_to_sku={idx: int(i) for idx, i in enumerate(item_ids)},
        embeddings=content,
    )
    dataset = ch5.TwoTowerDataset(train_sessions, lookup, max_seq_len=ch5.MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    model_args = {
        "embedding_dim": int(content.shape[1]),
        "hidden_dim": ch5.HIDDEN_DIM,
        "output_dim": ch5.USER_EMB_DIM,
        "temperature": 0.1,
        "dropout": 0.1,
        "user_tower_type": ch5.USER_TOWER_TYPE,
    }
    model = ch5.TwoTowerModel(**model_args)
    model = ch5.train_model(model, loader, val_sessions, lookup, num_epochs=epochs,
                            learning_rate=ch5.LEARNING_RATE, warmup_epochs=warmup_epochs,
                            device=torch.device("cpu"))
    model.eval()

    eval_sessions = val_sessions[:2000]
    hit_rate = ch5.quick_evaluate(model, eval_sessions, lookup, torch.device("cpu"), k=10,
                                  max_sessions=len(eval_sessions))
    baseline = popularity_hit_rate(eval_sessions, baseline_popularity, k=10)

    checkpoint = out_dir / "two_tower.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args": model_args,
        "max_seq_len": ch5.MAX_SEQ_LEN,
        "model_version": model_version,
    }, checkpoint)
    np.save(out_dir / "content_embeddings.npy", content)
    np.save(out_dir / "item_ids.npy", item_ids)

    vectors = item_vectors(model, content)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "item_index.faiss"))

    write_json(out_dir / "popularity.json", {"item_ids": [int(i) for i in popularity[:1000]]})
    files = {name: sha256_file(out_dir / name) for name in
             ("two_tower.pt", "content_embeddings.npy", "item_ids.npy", "item_index.faiss", "popularity.json")}
    manifest = {
        "index_version": index_version,
        "model_version": model_version,
        "model_file": "two_tower.pt",
        "model_sha256": files["two_tower.pt"],
        "index_file": "item_index.faiss",
        "index_type": "IndexFlatIP",
        "metric": "inner product of L2-normalised vectors (cosine)",
        "num_items": int(index.ntotal),
        "content_embedding_dim": int(content.shape[1]),
        "vector_dim": int(vectors.shape[1]),
        "max_seq_len": ch5.MAX_SEQ_LEN,
        "files": files,
    }
    write_json(out_dir / "index_manifest.json", manifest)
    metrics = {
        "train_examples": len(dataset),
        "val_examples": len(eval_sessions),
        "hit_rate_at_10": round(float(hit_rate), 4),
        "popularity_hit_rate_at_10": round(float(baseline), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }
    logger.info("Retrieval: %s", metrics)
    return metrics
