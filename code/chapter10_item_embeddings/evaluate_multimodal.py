"""
Section 10.4: Evaluate multi-modal fusion embeddings via co-listen retrieval.

Compares single-modality (audio, lyrics, genre), PCA-fused, and CLIP-fused
embeddings. For each query track, retrieves nearest neighbors in embedding
space and checks overlap with tracks co-listened by the same users.

Usage:
    python evaluate_multimodal.py
    python evaluate_multimodal.py --max-queries 10000
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_MUSIC4ALL_CONFIG, DEFAULT_MULTIMODAL_CONFIG,
    EMBEDDINGS_DIR, METRICS_DIR,
)
from data.music4all_dataset import Music4allDataset

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available — falling back to brute-force search")


def colisten_retrieval(
    track_ids: np.ndarray,
    embeddings: np.ndarray,
    colisten_pairs: List[Tuple[str, Set[str]]],
    k_values: List[int],
    max_queries: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate embeddings via co-listen retrieval.

    For each query track, find its K nearest neighbors in embedding space,
    then measure how many are co-listened tracks (ground truth).
    """
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    valid_pairs = [
        (tid, pos) for tid, pos in colisten_pairs
        if tid in id_to_idx and any(p in id_to_idx for p in pos)
    ]

    if len(valid_pairs) > max_queries:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(valid_pairs), max_queries, replace=False)
        valid_pairs = [valid_pairs[i] for i in indices]

    if not valid_pairs:
        logger.warning("No valid query tracks for evaluation")
        return {}

    query_indices = np.array([id_to_idx[tid] for tid, _ in valid_pairs])
    query_embeddings = embeddings[query_indices]

    max_k = max(k_values) + 1  # +1 because the query itself may appear

    if FAISS_AVAILABLE:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        _, nn_indices = index.search(query_embeddings.astype(np.float32), max_k)
    else:
        sims = query_embeddings @ embeddings.T
        nn_indices = np.argsort(-sims, axis=1)[:, :max_k]

    metrics = defaultdict(float)
    n_queries = len(valid_pairs)

    for i, (query_tid, positive_tids) in enumerate(valid_pairs):
        query_idx = id_to_idx[query_tid]
        pos_indices = {id_to_idx[p] for p in positive_tids if p in id_to_idx}
        n_relevant = len(pos_indices)

        if n_relevant == 0:
            continue

        retrieved = [idx for idx in nn_indices[i] if idx != query_idx]

        first_hit_rank = None
        for k in k_values:
            top_k = retrieved[:k]
            hits = sum(1 for idx in top_k if idx in pos_indices)

            metrics[f"precision@{k}"] += hits / k
            metrics[f"recall@{k}"] += hits / min(n_relevant, k)

            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, idx in enumerate(top_k)
                if idx in pos_indices
            )
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(n_relevant, k)))
            metrics[f"ndcg@{k}"] += dcg / max(idcg, 1e-8)

        for rank, idx in enumerate(retrieved[:max(k_values)]):
            if idx in pos_indices:
                first_hit_rank = rank + 1
                break
        metrics["mrr"] += 1.0 / first_hit_rank if first_hit_rank else 0.0

    for key in metrics:
        metrics[key] /= n_queries

    metrics["num_queries"] = n_queries
    metrics["avg_positives"] = np.mean([len(p) for _, p in valid_pairs])

    return dict(metrics)


def load_embeddings(name: str, emb_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load saved embeddings (ids, vectors) from an .npz file."""
    if name in ("audio", "lyrics", "genre"):
        path = emb_dir / f"{name}_raw.npz"
    elif name == "pca":
        path = emb_dir / "pca_fused.npz"
    elif name == "clip":
        path = emb_dir / "clip_fused.npz"
    else:
        return None

    if not path.exists():
        logger.warning(f"Embeddings not found: {path}")
        return None

    data = np.load(path, allow_pickle=True)
    ids = data["ids"]

    if name == "clip":
        emb = data["fused"]
    else:
        emb = data["embeddings"]

    logger.info(f"Loaded {name}: {emb.shape}")
    return ids, emb


def print_results_table(all_results: Dict[str, Dict[str, float]]):
    """Print a comparison table of all embedding methods."""
    if not all_results:
        return

    methods = list(all_results.keys())
    k_vals = sorted({
        int(k.split("@")[1]) for k in all_results[methods[0]]
        if "@" in k and k.startswith("precision")
    })

    header_cols = ["Method", "Dim"]
    for k in k_vals:
        header_cols.append(f"P@{k}")
    header_cols.extend(["MRR", "Queries"])

    print("\n" + "=" * 100)
    print("CO-LISTEN RETRIEVAL COMPARISON (Music4all)")
    print("=" * 100)

    fmt = "{:<30}" + "{:>8}" * (len(header_cols) - 1)
    print(fmt.format(*header_cols))
    print("-" * 100)

    for method, res in all_results.items():
        dim = res.get("dim", "?")
        row = [method, str(dim)]
        for k in k_vals:
            row.append(f"{res.get(f'precision@{k}', 0):.4f}")
        row.append(f"{res.get('mrr', 0):.4f}")
        row.append(f"{int(res.get('num_queries', 0))}")
        print(fmt.format(*row))

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Section 10.4: Evaluate multi-modal fusion embeddings"
    )
    parser.add_argument("--max-queries", type=int, default=5000)
    parser.add_argument("--max-users", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )

    data_cfg = DEFAULT_MUSIC4ALL_CONFIG
    fusion_cfg = DEFAULT_MULTIMODAL_CONFIG
    emb_dir = EMBEDDINGS_DIR / fusion_cfg.output_subdir

    # Load co-listen ground truth
    dataset = Music4allDataset(data_cfg.data_dir, config=data_cfg)
    dataset.load_audio()
    dataset.load_lyrics()
    dataset.load_genre()
    dataset.align_modalities()
    dataset.load_interactions()
    colisten_pairs = dataset.get_colisten_pairs(
        max_users=args.max_users, seed=fusion_cfg.random_seed
    )

    all_results = {}
    embedding_methods = ["audio", "lyrics", "genre", "pca", "clip"]

    for method in embedding_methods:
        result = load_embeddings(method, emb_dir)
        if result is None:
            continue

        ids, emb = result
        logger.info(f"\nEvaluating {method} embeddings ({emb.shape[1]}-dim)...")

        t0 = time.time()
        metrics = colisten_retrieval(
            track_ids=ids,
            embeddings=emb,
            colisten_pairs=colisten_pairs,
            k_values=fusion_cfg.k_values,
            max_queries=args.max_queries,
            seed=fusion_cfg.random_seed,
        )
        elapsed = time.time() - t0

        if metrics:
            metrics["dim"] = emb.shape[1]
            metrics["eval_seconds"] = round(elapsed, 1)
            all_results[method] = metrics
            logger.info(f"  {method}: MRR={metrics.get('mrr', 0):.4f}, "
                        f"P@10={metrics.get('precision@10', 0):.4f} ({elapsed:.1f}s)")

    print_results_table(all_results)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = METRICS_DIR / f"multimodal_comparison_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Print detailed per-method metrics
    for method, res in all_results.items():
        print(f"\n--- {method} ---")
        for key in sorted(res.keys()):
            if key not in ("dim", "eval_seconds"):
                print(f"  {key:.<40} {res[key]:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
