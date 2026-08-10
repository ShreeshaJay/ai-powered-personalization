"""
Section 11.3: LightGCN Evaluation on MIND

Evaluates trained LightGCN models on MIND next-click prediction,
with cold-start eval users and a LightGCN-native FAISS index.

Key differences from Sections 11.1/11.2 evaluation:
    1. FAISS index is built from LightGCN item embeddings (64-dim),
       NOT from SBERT embeddings (384-dim).
    2. Eval users are cold-start (NOT in the training graph).
       Their embeddings are computed by averaging the post-GCN item
       embeddings of their reading history — the standard inductive
       approach for LightGCN.
    3. All methods are compared on the same metrics (MRR, Recall@K, NDCG@K)
       using the same 10K eval users (same seed=42).

Usage:
    python evaluate_lightgcn.py                    # Evaluate best BPR model
    python evaluate_lightgcn.py --loss bpr         # Evaluate BPR model
    python evaluate_lightgcn.py --loss mnr         # Evaluate MNR model
    python evaluate_lightgcn.py --loss both        # Evaluate both
    python evaluate_lightgcn.py --quick            # Quick mode (2K users)
"""

import argparse
import json
import time
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEFAULT_MIND_USER_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_LIGHTGCN_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
)
from data.mind_user_loader import MINDUserDataset
from data.mind_graph_builder import build_bipartite_graph, build_normalized_adjacency
from models.lightgcn import LightGCN, create_lightgcn

try:
    from utils.faiss_index import build_faiss_index, brute_force_search, FAISS_AVAILABLE
except ImportError:
    FAISS_AVAILABLE = False

from utils.metrics import evaluate_ranking


# ============================================================================
# Model Loading
# ============================================================================

def load_lightgcn_model(
    model_dir: Path,
    device: torch.device,
) -> Tuple[LightGCN, Dict]:
    """Load trained LightGCN from checkpoint.

    Args:
        model_dir: Directory containing best_model.pt and training_meta.json.
        device: torch device.

    Returns:
        (model, metadata)
    """
    checkpoint_path = model_dir / "best_model.pt"
    meta_path = model_dir / "training_meta.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Load metadata
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]

    # Recreate model
    model = create_lightgcn(
        num_users=model_config["num_users"],
        num_items=model_config["num_items"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    loss_type = checkpoint.get("loss_type", "bpr")
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    logger.info(
        f"  Loaded LightGCN-{loss_type.upper()} from epoch {epoch} "
        f"(val_loss={val_loss})"
    )

    return model, metadata


# ============================================================================
# Embedding Extraction
# ============================================================================

def extract_all_item_embeddings(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    device: torch.device,
) -> np.ndarray:
    """Extract post-GCN item embeddings for building a FAISS index.

    IMPORTANT: These are LightGCN's learned 64-dim embeddings, NOT the
    384-dim SBERT embeddings used in Sections 11.1/11.2.  LightGCN
    operates in its own embedding space.  The FAISS index for evaluation
    is built from THESE embeddings, so user queries must also be in the
    same 64-dim space.

    Returns:
        (num_items, hidden_dim) numpy array, L2-normalized.
    """
    model.eval()
    with torch.no_grad():
        # Run full graph propagation to get final item embeddings
        _, item_embs = model(adj_norm)
        # L2-normalize for FAISS IndexFlatIP (cosine similarity)
        item_embs = F.normalize(item_embs, p=2, dim=1)
    return item_embs.cpu().numpy()


def extract_eval_user_embeddings(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    eval_users: List[Dict],
    item_id_to_idx: Dict[str, int],
    device: torch.device,
) -> np.ndarray:
    """Compute embeddings for cold-start eval users.

    Eval users are NOT in the training graph. Their embeddings are
    computed by averaging the post-GCN item embeddings of the items
    in their reading history. This is the standard inductive approach:

        user_emb = mean(post_GCN_item_embs[history_items])

    This works because LightGCN's propagated item embeddings capture
    collaborative signals from the training graph. Averaging a user's
    history items produces a reasonable user representation in the
    same embedding space.

    Args:
        model: Trained LightGCN model.
        adj_norm: Normalized adjacency matrix (on device).
        eval_users: List of eval user dicts with 'history_item_ids'.
        item_id_to_idx: Maps item_id string to 0-based index.
        device: torch device.

    Returns:
        (num_eval_users, hidden_dim) numpy array, L2-normalized.
    """
    model.eval()
    with torch.no_grad():
        _, all_item_embs = model(adj_norm)  # (num_items, hidden_dim)

    user_embeddings = []
    zero_fallback_count = 0

    for user in eval_users:
        history_item_ids = user["history_item_ids"]

        # Map to item indices, skip unknown items
        item_indices = []
        for iid in history_item_ids:
            if iid in item_id_to_idx:
                item_indices.append(item_id_to_idx[iid])

        if len(item_indices) == 0:
            # Fallback: zero vector (will not match anything well)
            user_embeddings.append(
                torch.zeros(model.hidden_dim, device=device)
            )
            zero_fallback_count += 1
            continue

        indices_tensor = torch.tensor(item_indices, device=device, dtype=torch.long)
        history_embs = all_item_embs[indices_tensor]  # (L, hidden_dim)
        user_emb = history_embs.mean(dim=0)            # (hidden_dim,)
        user_embeddings.append(user_emb)

    if zero_fallback_count > 0:
        logger.warning(
            f"  {zero_fallback_count} eval users had no valid history items "
            f"(zero vector fallback)"
        )

    user_emb_matrix = torch.stack(user_embeddings, dim=0)  # (N, hidden_dim)
    user_emb_matrix = F.normalize(user_emb_matrix, p=2, dim=1)
    return user_emb_matrix.cpu().numpy()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_lightgcn_on_mind(
    model: LightGCN,
    adj_norm: torch.sparse.FloatTensor,
    item_id_to_idx: Dict[str, int],
    item_ids: List[str],
    device: torch.device,
    loss_type: str = "bpr",
    sbert_init: bool = False,
    max_users: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
) -> Dict:
    """Evaluate LightGCN on MIND next-click prediction.

    Uses the same 10K eval users, same metrics, and same evaluation
    protocol as Sections 11.1 and 11.2 for fair comparison.

    The key difference: the FAISS index is built from LightGCN's
    learned item embeddings (hidden_dim), not SBERT embeddings (384-dim).

    Args:
        model: Trained LightGCN model.
        adj_norm: Normalized adjacency matrix (on device).
        item_id_to_idx: Maps item_id string to 0-based index.
        item_ids: List of all item_id strings (indexed by position).
        device: torch device.
        loss_type: "bpr" or "mnr" (for labeling in results).
        max_users: Max eval users (default: from config).
        k_values: Recall@K cutoffs.
        seed: Random seed for eval user sampling.

    Returns:
        Dict with aggregated metrics.
    """
    cfg = DEFAULT_MIND_USER_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_users is None:
        max_users = eval_cfg.max_eval_users_mind

    logger.info(f"\n  Evaluating LightGCN-{loss_type.upper()} on MIND ({max_users:,} users)...")

    # Load MIND dataset for eval users
    dataset = MINDUserDataset(
        data_dir=cfg.data_dir,
        news_file=cfg.news_file,
        behaviors_file=cfg.behaviors_file,
        min_history_length=cfg.min_history_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()

    eval_users = dataset.get_evaluation_users(max_users=max_users, seed=seed)

    # Extract all item embeddings (post-GCN, L2-normalized)
    logger.info("  Extracting LightGCN item embeddings for FAISS index...")
    t0 = time.time()
    item_embeddings = extract_all_item_embeddings(model, adj_norm, device)
    logger.info(
        f"  Item embeddings: shape={item_embeddings.shape}, "
        f"time={time.time() - t0:.1f}s"
    )

    # Build FAISS index from LightGCN item embeddings
    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(item_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None
        logger.warning("  FAISS not available, using brute-force search")

    # Extract eval user embeddings (cold-start averaging)
    logger.info("  Extracting eval user embeddings (cold-start)...")
    t0 = time.time()
    user_embeddings = extract_eval_user_embeddings(
        model, adj_norm, eval_users, item_id_to_idx, device
    )
    extract_time = time.time() - t0
    logger.info(
        f"  User embeddings: shape={user_embeddings.shape}, "
        f"time={extract_time:.1f}s"
    )

    # Retrieval evaluation
    logger.info("  Running retrieval evaluation...")
    t0 = time.time()
    all_query_metrics = []

    for i, user in enumerate(eval_users):
        user_emb = user_embeddings[i:i+1].astype(np.float32)

        max_k = max(k_values) + 10
        if faiss_index is not None:
            distances, indices = faiss_index.search(user_emb, max_k)
        else:
            distances, indices = brute_force_search(
                item_embeddings, user_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [item_ids[idx] for idx in indices[0]]
        target_ids = set(user["target_item_ids"])

        # Exclude history items from retrieval results
        history_set = set(user["history_item_ids"])
        filtered_ids = [rid for rid in retrieved_ids if rid not in history_set]
        filtered_ids = filtered_ids[:max(k_values)]

        # Compute relevance
        relevance = np.array([
            1.0 if rid in target_ids else 0.0 for rid in filtered_ids
        ])

        query_metrics = evaluate_ranking(
            scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
            relevance=relevance,
            k_values=k_values,
        )
        all_query_metrics.append(query_metrics)

    # Aggregate metrics
    agg_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        agg_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    hidden_dim = model.hidden_dim
    num_layers = model.num_layers
    agg_metrics["num_queries"] = len(eval_users)
    init_label = ", SBERT-init" if sbert_init else ""
    agg_metrics["method"] = f"LightGCN-{loss_type.upper()} ({num_layers}L, {hidden_dim}d{init_label})"
    agg_metrics["time_seconds"] = round(elapsed + extract_time, 2)
    agg_metrics["extraction_time"] = round(extract_time, 2)

    logger.info(f"\n  Results for LightGCN-{loss_type.upper()}:")
    logger.info(f"    MRR:      {agg_metrics.get('mrr', 0):.4f}")
    logger.info(f"    Recall@1: {agg_metrics.get('recall@1', 0):.4f}")
    logger.info(f"    Recall@5: {agg_metrics.get('recall@5', 0):.4f}")
    logger.info(f"    Recall@10:{agg_metrics.get('recall@10', 0):.4f}")
    logger.info(f"    Recall@50:{agg_metrics.get('recall@50', 0):.4f}")

    return agg_metrics


# ============================================================================
# Loading Previous Results for Comparison
# ============================================================================

def load_all_previous_results() -> List[Dict]:
    """Load results from Sections 11.1, 11.2, and 11.2-ItemID for comparison.

    Loads the most complete result file available (which typically includes
    all prior baselines in its 'all_results' list).
    """
    all_results = []

    # Try 11.2-ItemID results first (most complete, includes 11.1 + 11.2)
    pattern = str(METRICS_DIR / "sequence_itemid_mind_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        logger.info(f"  Loading Section 11.2-ItemID results from {Path(latest).name}")
        with open(latest) as f:
            data = json.load(f)
        if "all_results" in data:
            all_results.extend(data["all_results"])
            return all_results

    # Fallback: try 11.2 frozen results
    pattern = str(METRICS_DIR / "sequence_mind_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        logger.info(f"  Loading Section 11.2 results from {Path(latest).name}")
        with open(latest) as f:
            data = json.load(f)
        if "all_results" in data:
            all_results.extend(data["all_results"])
            return all_results

    # Fallback: try 11.1 aggregation results
    pattern = str(METRICS_DIR / "aggregation_mind_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        logger.info(f"  Loading Section 11.1 results from {Path(latest).name}")
        with open(latest) as f:
            data = json.load(f)
        if "results" in data:
            all_results.extend(data["results"])

    return all_results


# ============================================================================
# Comparison Table
# ============================================================================

def print_comparison_table(results: List[Dict], title: str):
    """Print unified comparison table across all sections."""
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")
    header = (
        f"  {'Method':<48} {'MRR':>8} {'R@1':>8} {'R@5':>8} "
        f"{'R@10':>8} {'R@20':>8} {'R@50':>8} {'Time':>7}"
    )
    print(header)
    print(f"  {'-' * 115}")

    sorted_results = sorted(results, key=lambda r: r.get("mrr", 0), reverse=True)

    for r in sorted_results:
        method = r.get("method", "Unknown")
        mrr = r.get("mrr", 0)
        r1 = r.get("recall@1", 0)
        r5 = r.get("recall@5", 0)
        r10 = r.get("recall@10", 0)
        r20 = r.get("recall@20", 0)
        r50 = r.get("recall@50", 0)
        t = r.get("time_seconds", 0)
        print(
            f"  {method:<48} {mrr:>8.4f} {r1:>8.4f} {r5:>8.4f} "
            f"{r10:>8.4f} {r20:>8.4f} {r50:>8.4f} {t:>6.1f}s"
        )

    print(f"{'=' * 120}\n")


# ============================================================================
# Save Results
# ============================================================================

def save_results(results: Dict, filename: str):
    """Save evaluation results to JSON."""
    out_path = METRICS_DIR / filename

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"  Results saved to {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGCN on MIND (Section 11.3)"
    )
    parser.add_argument(
        "--loss", type=str, default="bpr",
        choices=["bpr", "mnr", "both"],
        help="Which model variant to evaluate"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: evaluate on 2K users"
    )
    parser.add_argument(
        "--sbert_init", action="store_true",
        help="Evaluate SBERT-initialized model variant (from lightgcn_{loss}_sbert/ dirs)"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    loss_types = ["bpr", "mnr"] if args.loss == "both" else [args.loss]
    max_users = 2_000 if args.quick else None

    # Load previous results for comparison
    logger.info("\n[Loading previous results for comparison...]")
    previous_results = load_all_previous_results()
    logger.info(f"  Loaded {len(previous_results)} previous results")

    all_new_results = []

    for loss_type in loss_types:
        dir_suffix = f"lightgcn_{loss_type}_sbert" if args.sbert_init else f"lightgcn_{loss_type}"
        model_dir = MODELS_DIR / "lightgcn" / dir_suffix

        if not model_dir.exists():
            sbert_flag = " --sbert_init" if args.sbert_init else ""
            logger.warning(
                f"  Model directory not found: {model_dir}. "
                f"Run 'python train_lightgcn.py --loss {loss_type}{sbert_flag}' first."
            )
            continue

        # Load model
        logger.info(f"\n[Loading LightGCN-{loss_type.upper()}...]")
        model, metadata = load_lightgcn_model(model_dir, device)

        # Load graph mappings
        mappings_path = model_dir / "graph_mappings.npz"
        if not mappings_path.exists():
            logger.error(f"  Graph mappings not found: {mappings_path}")
            continue

        mappings = np.load(mappings_path, allow_pickle=True)
        item_ids = list(mappings["item_ids"])
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

        # Rebuild graph for forward pass (need adjacency matrix)
        logger.info("  Rebuilding graph adjacency for forward pass...")
        mind_config = DEFAULT_MIND_USER_CONFIG
        eval_config = DEFAULT_EVALUATION_CONFIG
        graph_data = build_bipartite_graph(mind_config, eval_config)

        adj_norm = build_normalized_adjacency(
            graph_data["user_item_edges"],
            graph_data["num_users"],
            graph_data["num_items"],
        )
        adj_norm = adj_norm.to(device)

        # Evaluate
        result = evaluate_lightgcn_on_mind(
            model=model,
            adj_norm=adj_norm,
            item_id_to_idx=item_id_to_idx,
            item_ids=item_ids,
            device=device,
            loss_type=loss_type,
            sbert_init=args.sbert_init,
            max_users=max_users,
        )

        all_new_results.append(result)

    # Combine with previous results and print comparison
    if all_new_results:
        all_combined = previous_results + all_new_results
        print_comparison_table(
            all_combined,
            "MIND: Section 11.1 + 11.2 + 11.3 Comparison (sorted by MRR)"
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_data = {
            "section": "11.3",
            "dataset": "mind",
            "lightgcn_results": all_new_results,
            "all_results": all_combined,
            "timestamp": datetime.now().isoformat(),
        }
        sbert_suffix = "_sbert" if args.sbert_init else ""
        save_results(save_data, f"lightgcn{sbert_suffix}_mind_{timestamp}.json")


if __name__ == "__main__":
    main()
