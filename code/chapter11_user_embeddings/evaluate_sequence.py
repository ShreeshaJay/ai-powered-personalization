"""
Evaluation Script for Section 11.2: Sequence Models (SASRec / GRU4Rec)

Evaluates trained sequence models on next-item retrieval, compared against
all Section 11.1 aggregation baselines in a unified comparison table.

The evaluation protocol is identical to Section 11.1:
- Same 20K Amazon sessions / 10K MIND users (same seed=42)
- Same FAISS index over item catalog (384-dim, IndexFlatIP)
- Same metrics (MRR, Recall@K, NDCG@K)
- Results saved to outputs/metrics/ as JSON

Usage:
    # Evaluate all trained models on both datasets
    python evaluate_sequence.py --dataset both

    # Evaluate on Amazon only
    python evaluate_sequence.py --dataset amazon

    # Quick mode (fewer sessions)
    python evaluate_sequence.py --dataset amazon --quick
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
from models.sequence_models import create_sequence_model, count_parameters
from utils.faiss_index import build_faiss_index, FAISS_AVAILABLE, brute_force_search
from utils.metrics import evaluate_ranking, print_ranking_metrics


# ============================================================================
# User Embedding Extraction
# ============================================================================

def extract_user_embeddings(
    model: torch.nn.Module,
    eval_data: List[Dict],
    max_seq_len: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Extract user embeddings from a trained sequence model.

    For each session/user in eval_data:
    1. Get history_embeddings (L, 384) from the evaluation data.
    2. Truncate to max_seq_len if needed (keep most recent items).
    3. Forward pass through the model.
    4. Take the L2-normalized output at the last valid position.

    Args:
        model: Trained SASRec or GRU4Rec model (in eval mode).
        eval_data: List of dicts from get_evaluation_sessions() or
                  get_evaluation_users().
        max_seq_len: Maximum sequence length for the model.
        device: torch device.
        batch_size: Batch size for inference.

    Returns:
        (N, 384) numpy array of L2-normalized user embeddings.
    """
    model.eval()
    all_user_embeddings = []

    with torch.no_grad():
        for start in range(0, len(eval_data), batch_size):
            batch_data = eval_data[start : start + batch_size]

            # Prepare batch: variable-length sequences
            sequences = []
            lengths = []
            for item in batch_data:
                embs = item["history_embeddings"]  # (L, 384) numpy
                # Truncate to max_seq_len (keep most recent)
                if len(embs) > max_seq_len:
                    embs = embs[-max_seq_len:]
                sequences.append(embs)
                lengths.append(len(embs))

            # Pad to max length in batch
            max_len = max(lengths)
            emb_dim = sequences[0].shape[1]
            padded = np.zeros((len(batch_data), max_len, emb_dim), dtype=np.float32)
            for i, (seq, ln) in enumerate(zip(sequences, lengths)):
                padded[i, :ln] = seq

            # To tensors
            input_tensor = torch.from_numpy(padded).to(device)
            length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

            # Forward pass
            user_embs, _ = model(input_tensor, length_tensor)
            # user_embs is (B, 384), L2-normalized
            all_user_embeddings.append(user_embs.cpu().numpy())

    return np.concatenate(all_user_embeddings, axis=0)


# ============================================================================
# Model Loading
# ============================================================================

def load_trained_model(
    model_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict]:
    """Load a trained model from a checkpoint directory.

    Args:
        model_dir: Directory containing best_model.pt and training_meta.json
        device: torch device to load the model onto

    Returns:
        (model, metadata) tuple
    """
    checkpoint_path = model_dir / "best_model.pt"
    meta_path = model_dir / "training_meta.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Load metadata
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type = checkpoint["model_type"]
    model_config = checkpoint["config"]

    # Create model with saved config
    model = create_sequence_model(
        model_type=model_type,
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config.get("num_heads", 2),
        ffn_dim=model_config.get("ffn_dim", 512),
        max_seq_len=model_config["max_seq_len"],
        dropout=model_config.get("dropout", 0.0),  # No dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = count_parameters(model)
    logger.info(f"  Loaded {model_type} ({n_params:,} params) from {model_dir.name}")
    logger.info(f"  Best val_loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model, metadata


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model_on_amazon(
    model: torch.nn.Module,
    model_type: str,
    max_seq_len: int,
    device: torch.device,
    max_sessions: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
) -> Dict:
    """Evaluate a trained sequence model on Amazon KDD next-item retrieval.

    Uses the EXACT same protocol as evaluate_aggregation.py:
    1. Load 20K evaluation sessions (same seed=42).
    2. Build FAISS index over all 500K product embeddings.
    3. Extract 384-dim user embeddings via forward pass.
    4. Retrieve top-K items per session.
    5. Compute MRR, Recall@K.
    6. Exclude history items from retrieval results.
    """
    cfg = DEFAULT_AMAZON_SESSION_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_sessions is None:
        max_sessions = eval_cfg.max_eval_sessions_amazon

    logger.info(f"\n  Evaluating {model_type} on Amazon KDD ({max_sessions:,} sessions)...")

    # Load data
    dataset = AmazonSessionDataset(
        data_dir=cfg.data_dir,
        locale=cfg.locale,
        min_session_length=cfg.min_session_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()

    # Get evaluation sessions (same as Section 11.1)
    sessions = dataset.get_evaluation_sessions(max_sessions=max_sessions, seed=seed)

    # Build FAISS index
    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(all_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None

    # Extract user embeddings
    t0 = time.time()
    user_embeddings = extract_user_embeddings(
        model, sessions, max_seq_len, device, batch_size=256
    )
    extract_time = time.time() - t0
    logger.info(f"  Embedding extraction: {extract_time:.1f}s")

    # Evaluate: per-query FAISS search
    t0 = time.time()
    all_query_metrics = []

    for i, session in enumerate(sessions):
        user_emb = user_embeddings[i:i+1].astype(np.float32)  # (1, 384)

        max_k = max(k_values) + 1
        if faiss_index is not None:
            distances, indices = faiss_index.search(user_emb, max_k)
        else:
            distances, indices = brute_force_search(
                all_embeddings, user_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
        target_id = session["target_item_id"]

        # Exclude history items
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

    # Aggregate metrics
    agg_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        agg_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    agg_metrics["num_queries"] = len(sessions)
    agg_metrics["method"] = f"{model_type.upper()} (2L, {DEFAULT_SEQUENCE_MODEL_CONFIG.hidden_dim}d)"
    agg_metrics["time_seconds"] = round(elapsed + extract_time, 2)
    agg_metrics["extraction_time"] = round(extract_time, 2)

    return agg_metrics


def evaluate_model_on_mind(
    model: torch.nn.Module,
    model_type: str,
    max_seq_len: int,
    device: torch.device,
    max_users: Optional[int] = None,
    k_values: List[int] = [1, 5, 10, 20, 50],
    seed: int = 42,
) -> Dict:
    """Evaluate a trained sequence model on MIND next-click prediction.

    Same protocol as evaluate_aggregation.py's evaluate_mind().
    """
    cfg = DEFAULT_MIND_USER_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if max_users is None:
        max_users = eval_cfg.max_eval_users_mind

    logger.info(f"\n  Evaluating {model_type} on MIND ({max_users:,} users)...")

    # Load data
    dataset = MINDUserDataset(
        data_dir=cfg.data_dir,
        news_file=cfg.news_file,
        behaviors_file=cfg.behaviors_file,
        min_history_length=cfg.min_history_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()

    # Get evaluation users (same as Section 11.1)
    eval_users = dataset.get_evaluation_users(max_users=max_users, seed=seed)

    # Build FAISS index
    all_embeddings, all_item_ids = dataset.get_all_item_embeddings()
    if FAISS_AVAILABLE:
        faiss_index = build_faiss_index(all_embeddings, index_type="IndexFlatIP")
    else:
        faiss_index = None

    # Extract user embeddings
    t0 = time.time()
    user_embeddings = extract_user_embeddings(
        model, eval_users, max_seq_len, device, batch_size=256
    )
    extract_time = time.time() - t0
    logger.info(f"  Embedding extraction: {extract_time:.1f}s")

    # Evaluate: per-query FAISS search
    t0 = time.time()
    all_query_metrics = []

    for i, user in enumerate(eval_users):
        user_emb = user_embeddings[i:i+1].astype(np.float32)  # (1, 384)

        max_k = max(k_values) + 10
        if faiss_index is not None:
            distances, indices = faiss_index.search(user_emb, max_k)
        else:
            distances, indices = brute_force_search(
                all_embeddings, user_emb, k=max_k, metric="cosine"
            )

        retrieved_ids = [all_item_ids[idx] for idx in indices[0]]
        target_ids = set(user["target_item_ids"])

        # Exclude history items
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

    # Aggregate metrics
    agg_metrics = {}
    for metric_name in all_query_metrics[0].keys():
        values = [m[metric_name] for m in all_query_metrics]
        agg_metrics[metric_name] = float(np.mean(values))

    elapsed = time.time() - t0
    agg_metrics["num_queries"] = len(eval_users)
    agg_metrics["method"] = f"{model_type.upper()} (2L, {DEFAULT_SEQUENCE_MODEL_CONFIG.hidden_dim}d)"
    agg_metrics["time_seconds"] = round(elapsed + extract_time, 2)
    agg_metrics["extraction_time"] = round(extract_time, 2)

    return agg_metrics


# ============================================================================
# Baseline Results Loading
# ============================================================================

def load_baseline_results(dataset_name: str) -> Optional[List[Dict]]:
    """Load Section 11.1 aggregation results from outputs/metrics/.

    Finds the most recent aggregation results JSON file for the given
    dataset and extracts the per-method metrics.

    Args:
        dataset_name: "amazon" or "mind"

    Returns:
        List of result dicts (one per method), or None if not found.
    """
    key = "amazon" if dataset_name in ("amazon", "amazon_kdd") else "mind"
    pattern = str(METRICS_DIR / f"aggregation_{key}_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.warning(f"No Section 11.1 results found matching {pattern}")
        return None

    # Use the most recent file
    latest = files[-1]
    logger.info(f"  Loading Section 11.1 baselines from {Path(latest).name}")

    with open(latest) as f:
        data = json.load(f)

    if "results" in data:
        return data["results"]
    return None


# ============================================================================
# Comparison Table
# ============================================================================

def print_comparison_table(results: List[Dict], title: str):
    """Print a unified comparison table (same format as Section 11.1)."""
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    header = (
        f"  {'Method':<40} {'MRR':>8} {'R@1':>8} {'R@5':>8} "
        f"{'R@10':>8} {'R@20':>8} {'R@50':>8} {'Time':>7}"
    )
    print(header)
    print(f"  {'-' * 105}")

    # Sort by MRR descending
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
            f"  {method:<40} {mrr:>8.4f} {r1:>8.4f} {r5:>8.4f} "
            f"{r10:>8.4f} {r20:>8.4f} {r50:>8.4f} {t:>6.1f}s"
        )

    print(f"{'=' * 110}\n")


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
    """Evaluate all trained models on a dataset, with Section 11.1 baselines.

    Args:
        dataset_name: "amazon" or "mind"
        model_types: ["sasrec", "gru4rec"] or subset
        quick: If True, use fewer sessions/users
        k_values: Retrieval cutoffs

    Returns:
        Results dict with all methods
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_config = DEFAULT_SEQUENCE_MODEL_CONFIG
    eval_cfg = DEFAULT_EVALUATION_CONFIG

    if dataset_name in ("amazon", "amazon_kdd"):
        max_eval = 2000 if quick else eval_cfg.max_eval_sessions_amazon
        max_seq_len = seq_config.max_seq_len_amazon
    else:
        max_eval = 2000 if quick else eval_cfg.max_eval_users_mind
        max_seq_len = seq_config.max_seq_len_mind

    logger.info(f"\n{'=' * 80}")
    logger.info(f"SECTION 11.2 EVALUATION: {dataset_name.upper()}")
    logger.info(f"{'=' * 80}")

    all_results = []

    # Evaluate each trained model
    for model_type in model_types:
        model_dir = (
            MODELS_DIR / seq_config.output_subdir / f"{model_type}_{dataset_name}"
        )

        if not (model_dir / "best_model.pt").exists():
            logger.warning(
                f"  No trained {model_type} model found for {dataset_name} "
                f"at {model_dir}. Skipping."
            )
            continue

        model, metadata = load_trained_model(model_dir, device)

        if dataset_name in ("amazon", "amazon_kdd"):
            result = evaluate_model_on_amazon(
                model, model_type, max_seq_len, device,
                max_sessions=max_eval, k_values=k_values,
            )
        else:
            result = evaluate_model_on_mind(
                model, model_type, max_seq_len, device,
                max_users=max_eval, k_values=k_values,
            )

        print_ranking_metrics(result, f"{dataset_name.upper()} - {model_type.upper()}")
        all_results.append(result)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load Section 11.1 baselines for comparison
    baselines = load_baseline_results(dataset_name)
    if baselines:
        all_results.extend(baselines)
        logger.info(f"  Added {len(baselines)} Section 11.1 baselines")

    # Print unified comparison
    title = (
        f"{dataset_name.upper()} Results: Section 11.1 (Aggregation) + "
        f"Section 11.2 (Sequence Models)"
    )
    print_comparison_table(all_results, title)

    return {
        "dataset": dataset_name,
        "sequence_model_results": [
            r for r in all_results
            if "SASREC" in r.get("method", "").upper()
            or "GRU4REC" in r.get("method", "").upper()
        ],
        "all_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Section 11.2 — Evaluate sequence models for user embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["amazon", "mind", "both"],
        help="Which dataset to evaluate on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["sasrec", "gru4rec", "both"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer sessions/users (2K instead of 20K/10K)",
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
            f"sequence_{dataset_name}_{timestamp}.json",
        )

    # Save combined comparison if both datasets
    if len(all_results) == 2:
        save_results(
            all_results,
            f"sequence_comparison_{timestamp}.json",
        )

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
