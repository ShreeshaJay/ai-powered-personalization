"""
Evaluation Script for Section 11.2 Variant: Sequence Models with Learnable Item IDs

Evaluates Item-ID trained models and compares them against:
  - Section 11.1 aggregation baselines
  - Section 11.2 frozen-embedding sequence models

Uses the EXACT same evaluation protocol (same sessions/users, same FAISS index,
same metrics) for a clean comparison.

Usage:
    # Evaluate all Item-ID models on both datasets
    python evaluate_sequence_itemid.py --dataset both

    # Quick mode
    python evaluate_sequence_itemid.py --dataset amazon --quick
"""

import argparse
import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from config import (
    DEFAULT_AMAZON_SESSION_CONFIG,
    DEFAULT_MIND_USER_CONFIG,
    DEFAULT_SEQUENCE_MODEL_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    MODELS_DIR,
    METRICS_DIR,
)
from data.amazon_session_loader import AmazonSessionDataset
from data.mind_user_loader import MINDUserDataset
from models.sequence_models_itemid import create_itemid_model
from models.sequence_models import count_parameters
from utils.faiss_index import build_faiss_index, FAISS_AVAILABLE, brute_force_search
from utils.metrics import evaluate_ranking, print_ranking_metrics


# ============================================================================
# User Embedding Extraction (Item-ID variant)
# ============================================================================

def extract_user_embeddings_itemid(
    model: torch.nn.Module,
    eval_data: List[Dict],
    item_id_to_idx: Dict[str, int],
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract user embeddings from a trained Item-ID model.

    Instead of looking up frozen embeddings, we pass item ID indices
    directly to the model's nn.Embedding layer.

    Args:
        model: Trained SASRecItemID or GRU4RecItemID (in eval mode).
        eval_data: List of dicts with "history_item_ids" keys.
        item_id_to_idx: Mapping from item string ID to 1-based integer index.
        max_seq_len: Maximum sequence length.
        device: torch device.
        batch_size: Batch size for inference.

    Returns:
        (N, output_dim) numpy array of L2-normalized user embeddings.
    """
    model.eval()
    all_user_embeddings = []

    with torch.no_grad():
        for start in range(0, len(eval_data), batch_size):
            batch_data = eval_data[start : start + batch_size]

            sequences = []
            lengths = []
            for item in batch_data:
                history_ids = item["history_item_ids"]
                # Convert string IDs to integer indices
                indices = []
                for iid in history_ids:
                    if iid in item_id_to_idx:
                        indices.append(item_id_to_idx[iid])
                # Truncate to max_seq_len (keep most recent)
                if len(indices) > max_seq_len:
                    indices = indices[-max_seq_len:]
                if len(indices) == 0:
                    indices = [0]  # padding fallback
                sequences.append(indices)
                lengths.append(len(indices))

            # Pad to max length in batch
            max_len = max(lengths)
            padded = np.zeros((len(batch_data), max_len), dtype=np.int64)
            for i, (seq, ln) in enumerate(zip(sequences, lengths)):
                padded[i, :ln] = seq

            input_tensor = torch.from_numpy(padded).to(device)
            length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

            user_embs, _ = model(input_tensor, length_tensor)
            all_user_embeddings.append(user_embs.cpu().numpy())

    return np.concatenate(all_user_embeddings, axis=0)


# ============================================================================
# Model Loading
# ============================================================================

def load_itemid_model(
    model_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict]:
    """Load a trained Item-ID model from checkpoint."""
    checkpoint_path = model_dir / "best_model.pt"
    meta_path = model_dir / "training_meta.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type = checkpoint["model_type"]
    model_config = checkpoint["config"]

    model = create_itemid_model(
        model_type=model_type,
        num_items=model_config["num_items"],
        output_dim=model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config.get("num_heads", 2),
        ffn_dim=model_config.get("ffn_dim", 512),
        max_seq_len=model_config["max_seq_len"],
        dropout=model_config.get("dropout", 0.0),
        padding_idx=model_config.get("padding_idx", 0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = count_parameters(model)
    logger.info(f"  Loaded {model_type}-ItemID ({n_params:,} params) from {model_dir.name}")
    logger.info(f"  Best val_loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model, metadata


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_itemid_on_amazon(
    model: torch.nn.Module,
    model_type: str,
    item_id_to_idx: Dict[str, int],
    max_seq_len: int,
    device: torch.device,
    max_sessions: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
) -> Dict:
    """Evaluate Item-ID model on Amazon KDD. Same protocol as evaluate_sequence.py."""
    cfg = DEFAULT_AMAZON_SESSION_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_sessions is None:
        max_sessions = eval_cfg.max_eval_sessions_amazon

    logger.info(f"\n  Evaluating {model_type}-ItemID on Amazon KDD ({max_sessions:,} sessions)...")

    dataset = AmazonSessionDataset(
        data_dir=cfg.data_dir,
        locale=cfg.locale,
        min_session_length=cfg.min_session_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()

    sessions = dataset.get_evaluation_sessions(max_sessions=max_sessions, seed=seed)

    # FAISS index is still built from frozen SBERT embeddings
    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(all_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None

    # Extract user embeddings using item IDs
    t0 = time.time()
    user_embeddings = extract_user_embeddings_itemid(
        model, sessions, item_id_to_idx, max_seq_len, device, batch_size=256
    )
    extract_time = time.time() - t0
    logger.info(f"  Embedding extraction: {extract_time:.1f}s")

    # Evaluate (same protocol)
    t0 = time.time()
    all_query_metrics = []

    for i, session in enumerate(sessions):
        user_emb = user_embeddings[i:i+1].astype(np.float32)

        max_k = max(k_values) + 1
        if faiss_index is not None:
            distances, indices = faiss_index.search(user_emb, max_k)
        else:
            distances, indices = brute_force_search(
                all_embeddings, user_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
        target_id = session["target_item_id"]

        history_set = set(session["history_item_ids"])
        filtered_ids = [rid for rid in retrieved_ids if rid not in history_set]
        filtered_ids = filtered_ids[:max(k_values)]

        relevance = np.array([
            1.0 if rid == target_id else 0.0 for rid in filtered_ids
        ])

        query_metrics = evaluate_ranking(
            scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
            relevance=relevance,
            k_values=k_values,
        )
        all_query_metrics.append(query_metrics)

    agg_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        agg_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    hidden_dim = DEFAULT_SEQUENCE_MODEL_CONFIG.hidden_dim
    agg_metrics["num_queries"] = len(sessions)
    agg_metrics["method"] = f"{model_type.upper()}-ItemID (2L, {hidden_dim}d)"
    agg_metrics["time_seconds"] = round(elapsed + extract_time, 2)
    agg_metrics["extraction_time"] = round(extract_time, 2)

    return agg_metrics


def evaluate_itemid_on_mind(
    model: torch.nn.Module,
    model_type: str,
    item_id_to_idx: Dict[str, int],
    max_seq_len: int,
    device: torch.device,
    max_users: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
) -> Dict:
    """Evaluate Item-ID model on MIND. Same protocol as evaluate_sequence.py."""
    cfg = DEFAULT_MIND_USER_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_users is None:
        max_users = eval_cfg.max_eval_users_mind

    logger.info(f"\n  Evaluating {model_type}-ItemID on MIND ({max_users:,} users)...")

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

    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(all_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None

    t0 = time.time()
    user_embeddings = extract_user_embeddings_itemid(
        model, eval_users, item_id_to_idx, max_seq_len, device, batch_size=256
    )
    extract_time = time.time() - t0
    logger.info(f"  Embedding extraction: {extract_time:.1f}s")

    t0 = time.time()
    all_query_metrics = []

    for i, user in enumerate(eval_users):
        user_emb = user_embeddings[i:i+1].astype(np.float32)

        max_k = max(k_values) + 10
        if faiss_index is not None:
            distances, indices = faiss_index.search(user_emb, max_k)
        else:
            distances, indices = brute_force_search(
                all_embeddings, user_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
        target_ids = set(user["target_item_ids"])

        history_set = set(user["history_item_ids"])
        filtered_ids = [rid for rid in retrieved_ids if rid not in history_set]
        filtered_ids = filtered_ids[:max(k_values)]

        relevance = np.array([
            1.0 if rid in target_ids else 0.0 for rid in filtered_ids
        ])

        query_metrics = evaluate_ranking(
            scores=np.arange(len(relevance), 0, -1, dtype=np.float64),
            relevance=relevance,
            k_values=k_values,
        )
        all_query_metrics.append(query_metrics)

    agg_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        agg_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    hidden_dim = DEFAULT_SEQUENCE_MODEL_CONFIG.hidden_dim
    agg_metrics["num_queries"] = len(eval_users)
    agg_metrics["method"] = f"{model_type.upper()}-ItemID (2L, {hidden_dim}d)"
    agg_metrics["time_seconds"] = round(elapsed + extract_time, 2)
    agg_metrics["extraction_time"] = round(extract_time, 2)

    return agg_metrics


# ============================================================================
# Loading Previous Results for Comparison
# ============================================================================

def load_previous_results(dataset_name: str) -> List[Dict]:
    """Load Section 11.1 + 11.2 results for comparison."""
    key = "amazon" if dataset_name in ("amazon", "amazon_kdd") else "mind"
    all_results = []

    # Load 11.2 frozen-embedding results (most recent)
    pattern = str(METRICS_DIR / f"sequence_{key}_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        logger.info(f"  Loading Section 11.2 results from {Path(latest).name}")
        with open(latest) as f:
            data = json.load(f)
        if "all_results" in data:
            all_results.extend(data["all_results"])
            return all_results  # This already includes 11.1 baselines

    # Fallback: load 11.1 baselines separately
    pattern = str(METRICS_DIR / f"aggregation_{key}_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        latest = files[-1]
        with open(latest) as f:
            data = json.load(f)
        if "results" in data:
            all_results.extend(data["results"])

    return all_results


# ============================================================================
# Comparison Table
# ============================================================================

def print_comparison_table(results: List[Dict], title: str):
    """Print unified comparison table."""
    print(f"\n{'=' * 115}")
    print(f"  {title}")
    print(f"{'=' * 115}")
    header = (
        f"  {'Method':<45} {'MRR':>8} {'R@1':>8} {'R@5':>8} "
        f"{'R@10':>8} {'R@20':>8} {'R@50':>8} {'Time':>7}"
    )
    print(header)
    print(f"  {'-' * 110}")

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
            f"  {method:<45} {mrr:>8.4f} {r1:>8.4f} {r5:>8.4f} "
            f"{r10:>8.4f} {r20:>8.4f} {r50:>8.4f} {t:>6.1f}s"
        )

    print(f"{'=' * 115}\n")


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
    logger.info(f"Results saved to {out_path}")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate_dataset(
    dataset_name: str,
    model_types: List[str],
    quick: bool = False,
    k_values: List[int] = [1, 5, 10, 20, 50],
) -> Dict:
    """Evaluate all Item-ID models on a dataset with full comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_config = DEFAULT_SEQUENCE_MODEL_CONFIG

    if dataset_name in ("amazon", "amazon_kdd"):
        max_seq_len = seq_config.max_seq_len_amazon
    else:
        max_seq_len = seq_config.max_seq_len_mind

    logger.info(f"\n{'=' * 80}")
    logger.info(f"ITEM-ID EVALUATION: {dataset_name.upper()}")
    logger.info(f"{'=' * 80}")

    itemid_results = []

    for model_type in model_types:
        model_dir = MODELS_DIR / "sequence_models_itemid" / f"{model_type}_{dataset_name}"

        if not (model_dir / "best_model.pt").exists():
            logger.warning(
                f"  No trained {model_type}-ItemID model found for {dataset_name}. Skipping."
            )
            continue

        model, metadata = load_itemid_model(model_dir, device)

        # Rebuild item_id_to_idx mapping (1-based)
        # We need the original item IDs to map eval data
        if dataset_name in ("amazon", "amazon_kdd"):
            cfg = DEFAULT_AMAZON_SESSION_CONFIG
            data = AmazonSessionDataset(
                data_dir=cfg.data_dir,
                locale=cfg.locale,
                min_session_length=cfg.min_session_length,
                item_embedding_file=cfg.item_embedding_file,
                item_embedding_dim=cfg.item_embedding_dim,
            )
            data.load()
            _, all_item_ids = data.get_all_item_embeddings()
        else:
            cfg = DEFAULT_MIND_USER_CONFIG
            data = MINDUserDataset(
                data_dir=cfg.data_dir,
                news_file=cfg.news_file,
                behaviors_file=cfg.behaviors_file,
                min_history_length=cfg.min_history_length,
                item_embedding_file=cfg.item_embedding_file,
                item_embedding_dim=cfg.item_embedding_dim,
            )
            data.load()
            _, all_item_ids = data.get_all_item_embeddings()

        item_id_to_idx = {iid: idx + 1 for idx, iid in enumerate(all_item_ids)}

        max_eval = None
        if quick:
            max_eval = 2000

        if dataset_name in ("amazon", "amazon_kdd"):
            result = evaluate_itemid_on_amazon(
                model, model_type, item_id_to_idx, max_seq_len, device,
                max_sessions=max_eval, k_values=k_values,
            )
        else:
            result = evaluate_itemid_on_mind(
                model, model_type, item_id_to_idx, max_seq_len, device,
                max_users=max_eval, k_values=k_values,
            )

        print_ranking_metrics(result, f"{dataset_name.upper()} - {model_type.upper()}-ItemID")
        itemid_results.append(result)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load previous results for comparison
    previous_results = load_previous_results(dataset_name)
    all_results = itemid_results + previous_results

    title = (
        f"{dataset_name.upper()} Results: Section 11.1 + 11.2 (Frozen) + "
        f"11.2 (Item-ID)"
    )
    print_comparison_table(all_results, title)

    return {
        "dataset": dataset_name,
        "itemid_results": itemid_results,
        "all_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Section 11.2 Variant — Evaluate Item-ID sequence models"
    )
    parser.add_argument(
        "--dataset", type=str, default="both",
        choices=["amazon", "mind", "both"],
        help="Which dataset to evaluate on",
    )
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["sasrec", "gru4rec", "both"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer sessions/users",
    )

    args = parser.parse_args()

    model_types = (
        ["sasrec", "gru4rec"] if args.model == "both" else [args.model]
    )
    datasets = (
        ["amazon", "mind"] if args.dataset == "both" else [args.dataset]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for dataset_name in datasets:
        results = evaluate_dataset(
            dataset_name=dataset_name,
            model_types=model_types,
            quick=args.quick,
        )
        all_results[dataset_name] = results
        save_results(
            results,
            f"sequence_itemid_{dataset_name}_{timestamp}.json",
        )

    if len(all_results) == 2:
        save_results(
            all_results,
            f"sequence_itemid_comparison_{timestamp}.json",
        )

    logger.info("\nItem-ID Evaluation complete.")


if __name__ == "__main__":
    main()
