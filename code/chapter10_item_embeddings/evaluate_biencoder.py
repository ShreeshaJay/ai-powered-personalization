"""
Bi-Encoder Evaluation: Query vs. Item Embeddings (IJCAI Dataset)

*** HOMEWORK EXERCISE ***

This script evaluates zero-shot text encoders on the Alibaba IJCAI-18 dataset.
Because this dataset uses hashed/anonymized category and property IDs (not
human-readable text), zero-shot text encoders perform poorly — all metrics
are near zero. This is an expected and pedagogically important result.

Students are encouraged to:
  1. Run this script and observe the near-zero performance
  2. Examine the diagnostic output (sample texts, rank diagnostics) to
     understand WHY hashed IDs defeat sub-word tokenization
  3. Compare against the MIND evaluation (evaluate_mind.py) where real
     text yields dramatically better results

For the primary Section 10.1 evaluation, see: evaluate_mind.py

Usage:
    python evaluate_biencoder.py --compare_models --sample_size 10000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_TEXT_ENCODER, DEFAULT_IJCAI_CONFIG, DEFAULT_BIENCODER_EVAL,
    METRICS_DIR, print_config
)
from data import IJCAIDataset
from models import TextEncoder, create_encoder
from utils import (
    evaluate_ranking_batch, print_ranking_metrics,
    build_faiss_index, brute_force_search
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def temporal_split(
    samples_df: pd.DataFrame,
    timestamp_col: str = "context_timestamp",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically by timestamp to prevent data leakage.
    
    IMPORTANT: Random splitting would leak future information into training data.
    In production, models are always trained on past data and evaluated on future.
    
    Args:
        samples_df: DataFrame with a timestamp column
        timestamp_col: Name of the timestamp column
        train_ratio: Fraction of earliest data for training
        val_ratio: Fraction for validation
        test_ratio: Fraction of latest data for testing
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Sort by timestamp
    sorted_df = samples_df.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(sorted_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]
    
    logger.info(f"Chronological split (by '{timestamp_col}'):")
    logger.info(f"  Train: {len(train_df)} samples "
                f"(timestamps {train_df[timestamp_col].min()} - {train_df[timestamp_col].max()})")
    logger.info(f"  Val:   {len(val_df)} samples "
                f"(timestamps {val_df[timestamp_col].min()} - {val_df[timestamp_col].max()})")
    logger.info(f"  Test:  {len(test_df)} samples "
                f"(timestamps {test_df[timestamp_col].min()} - {test_df[timestamp_col].max()})")
    
    # Sanity check: no temporal overlap
    assert train_df[timestamp_col].max() <= val_df[timestamp_col].min(), \
        "Data leakage: train timestamps overlap with validation!"
    assert val_df[timestamp_col].max() <= test_df[timestamp_col].min(), \
        "Data leakage: validation timestamps overlap with test!"
    
    return train_df, val_df, test_df


def encode_queries_and_items(
    dataset: IJCAIDataset,
    encoder: TextEncoder,
    relevance_threshold: str = "conversion",
    eval_split: str = "test",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Encode queries and items, prepare evaluation data with chronological split.
    
    Items are encoded using ALL data (item metadata is static), but evaluation
    pairs come only from the specified temporal split. This mirrors production:
    the item catalog is known, but we predict on future user queries.
    
    Args:
        dataset: Loaded IJCAIDataset
        encoder: TextEncoder instance
        relevance_threshold: "conversion" or "click"
        eval_split: Which temporal split to evaluate on ("val" or "test")
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
    
    Returns:
        Tuple of:
        - query_embeddings: (N, D) array
        - item_embeddings: (M, D) array
        - relevance_data: List of (query_idx, item_idx, relevance) tuples
        - query_indices: List of query (context) IDs
        - item_indices: List of item IDs
    """
    logger.info("Preparing bi-encoder evaluation data...")
    
    # ---- Chronological split ----
    samples_df = dataset.samples_df
    train_df, val_df, test_df = temporal_split(
        samples_df,
        timestamp_col="context_timestamp",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Select evaluation split
    if eval_split == "test":
        eval_df = test_df
    elif eval_split == "val":
        eval_df = val_df
    elif eval_split == "train":
        eval_df = train_df
    else:
        raise ValueError(f"Unknown eval_split: {eval_split}")
    
    logger.info(f"Evaluating on '{eval_split}' split: {len(eval_df)} samples")
    
    # ---- Get texts for ALL items (item catalog is static) ----
    item_texts_dict = dataset.get_item_texts()
    
    # ---- Get query texts only for the evaluation split ----
    query_texts_dict = dataset.get_query_texts()
    eval_context_ids = set(eval_df["context_id"].unique())
    eval_query_texts = {
        cid: text for cid, text in query_texts_dict.items()
        if cid in eval_context_ids
    }
    
    # Build mappings
    unique_contexts = sorted(eval_query_texts.keys())
    unique_items = sorted(item_texts_dict.keys())
    
    context_to_idx = {cid: i for i, cid in enumerate(unique_contexts)}
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    
    logger.info(f"Unique eval queries (contexts): {len(unique_contexts)}")
    logger.info(f"Unique items (full catalog): {len(unique_items)}")
    
    # Encode queries
    logger.info("Encoding evaluation queries...")
    query_texts = [eval_query_texts[cid] for cid in unique_contexts]
    query_embeddings = encoder.encode(query_texts, show_progress=True)
    
    # Encode items
    logger.info("Encoding items (full catalog)...")
    item_texts = [item_texts_dict[iid] for iid in unique_items]
    item_embeddings = encoder.encode(item_texts, show_progress=True)
    
    # Build relevance data from EVALUATION split only
    logger.info("Building relevance data from evaluation split...")
    
    relevance_data = []
    
    for _, row in eval_df.iterrows():
        context_id = row["context_id"]
        item_id = row["item_id"]
        
        if context_id not in context_to_idx or item_id not in item_to_idx:
            continue
        
        query_idx = context_to_idx[context_id]
        item_idx = item_to_idx[item_id]
        
        # Determine relevance
        if relevance_threshold == "conversion":
            relevance = int(row["is_trade"])
        elif relevance_threshold == "click":
            relevance = 1  # All samples are clicked in this dataset
        else:
            raise ValueError(f"Unknown relevance threshold: {relevance_threshold}")
        
        relevance_data.append((query_idx, item_idx, relevance))
    
    logger.info(f"Relevance data: {len(relevance_data)} query-item pairs")
    relevant_count = sum(1 for _, _, r in relevance_data if r == 1)
    logger.info(f"  Relevant (positive) pairs: {relevant_count}")
    
    return (
        query_embeddings,
        item_embeddings,
        relevance_data,
        unique_contexts,
        unique_items
    )


def evaluate_biencoder(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    relevance_data: List[Tuple[int, int, int]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100],
    use_faiss: bool = False,
    query_texts_sample: List[str] = None,
    item_texts_sample: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate bi-encoder retrieval performance.
    
    Args:
        query_embeddings: (N, D) query embeddings
        item_embeddings: (M, D) item embeddings
        relevance_data: List of (query_idx, item_idx, relevance) tuples
        k_values: K values for Precision@K, Recall@K, NDCG@K
        use_faiss: Use FAISS for fast search
        query_texts_sample: Optional query texts for diagnostic logging
        item_texts_sample: Optional item texts for diagnostic logging
    
    Returns:
        Dictionary of aggregated metrics
    """
    logger.info("Evaluating bi-encoder retrieval...")
    
    # Build query-wise relevance dictionaries
    from collections import defaultdict
    query_relevances = defaultdict(dict)
    
    for query_idx, item_idx, relevance in relevance_data:
        query_relevances[query_idx][item_idx] = relevance
    
    # Get queries with at least one relevant item
    valid_queries = [q for q in query_relevances if any(query_relevances[q].values())]
    
    logger.info(f"Evaluating on {len(valid_queries)} queries with relevant items")
    
    # Print sample query/item texts for diagnostic purposes
    if query_texts_sample and item_texts_sample:
        logger.info("\n--- Sample Query/Item Texts (diagnostic) ---")
        for i, q_idx in enumerate(valid_queries[:3]):
            if q_idx < len(query_texts_sample):
                logger.info(f"  Query[{q_idx}]: {query_texts_sample[q_idx][:120]}...")
            for item_idx, rel in query_relevances[q_idx].items():
                if rel == 1 and item_idx < len(item_texts_sample):
                    logger.info(f"  Relevant Item[{item_idx}]: {item_texts_sample[item_idx][:120]}...")
        logger.info("--- End Samples ---\n")
    
    # Compute similarities and evaluate
    if use_faiss:
        try:
            logger.info("Building FAISS index...")
            index = build_faiss_index(item_embeddings, index_type="IndexFlatIP")
            
            logger.info("Searching with FAISS...")
            max_k = max(k_values)
            
            # Search for all queries at once
            distances, indices = index.search(query_embeddings, k=max_k)
            
        except ImportError:
            logger.warning("FAISS not available, using brute-force search")
            use_faiss = False
    
    if not use_faiss:
        # Brute-force search
        logger.info("Computing similarities (brute-force)...")
        max_k = max(k_values)
        
        distances, indices = brute_force_search(
            item_embeddings,
            query_embeddings,
            k=max_k,
            metric="cosine"
        )
    
    # Diagnostic: compute actual rank of relevant items using full similarity matrix
    logger.info("--- Rank Diagnostic (first 5 positive queries) ---")
    for i, query_idx in enumerate(valid_queries[:5]):
        # Compute similarity of this query to ALL items
        query_vec = query_embeddings[query_idx:query_idx+1]
        sims = (query_vec @ item_embeddings.T).flatten()
        sorted_item_indices = np.argsort(-sims)  # descending
        
        for item_idx, rel in query_relevances[query_idx].items():
            if rel == 1:
                actual_rank = int(np.where(sorted_item_indices == item_idx)[0][0]) + 1
                sim_score = float(sims[item_idx])
                max_sim = float(sims[sorted_item_indices[0]])
                logger.info(
                    f"  Query {query_idx} -> Relevant item {item_idx}: "
                    f"rank={actual_rank}/{len(sims)}, "
                    f"sim={sim_score:.4f}, max_sim={max_sim:.4f}"
                )
    logger.info("--- End Rank Diagnostic ---\n")
    
    # Evaluate each query
    scores_list = []
    relevance_list = []
    
    for query_idx in valid_queries:
        # Get retrieved items
        retrieved_items = indices[query_idx]
        retrieved_scores = distances[query_idx]
        
        # Get relevance labels for retrieved items
        relevance_labels = np.array([
            query_relevances[query_idx].get(item_idx, 0)
            for item_idx in retrieved_items
        ])
        
        scores_list.append(retrieved_scores)
        relevance_list.append(relevance_labels)
    
    # Aggregate metrics
    logger.info("Computing metrics...")
    metrics = evaluate_ranking_batch(
        scores_list,
        relevance_list,
        k_values=k_values
    )
    
    return metrics


def compare_models_on_ijcai(
    models: List[str],
    sample_size: int = 10000,
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> pd.DataFrame:
    """
    Compare multiple text encoder models on IJCAI bi-encoder task.
    
    Returns:
        DataFrame with comparison results
    """
    logger.info("="*80)
    logger.info("Comparing Models on IJCAI Bi-Encoder Task")
    logger.info("="*80)
    
    # Load dataset once
    dataset = IJCAIDataset(
        data_dir=DEFAULT_IJCAI_CONFIG.data_dir,
        sample_size=sample_size,
        min_interactions=1,
        random_seed=42
    )
    
    dataset.load_data(split="train")
    
    results = []
    
    for model_name in models:
        logger.info(f"\nEvaluating model: {model_name}")
        
        # Initialize encoder (factory handles SBERT vs RexBERT)
        encoder = create_encoder(
            model_name=model_name,
            max_seq_length=128,
            normalize_embeddings=True
        )
        
        # Encode and evaluate (chronological split: evaluate on test set)
        (query_embs, item_embs, relevance_data,
         query_ids, item_ids) = encode_queries_and_items(
            dataset,
            encoder,
            relevance_threshold="conversion",
            eval_split="test"
        )
        
        # Get text samples for diagnostics
        item_texts_dict = dataset.get_item_texts()
        query_texts_dict = dataset.get_query_texts()
        eval_context_ids = set(
            dataset.samples_df.nlargest(
                int(len(dataset.samples_df) * 0.2), "context_timestamp"
            )["context_id"].unique()
        )
        query_texts_list = [
            query_texts_dict.get(cid, "")
            for cid in sorted(
                cid for cid in query_texts_dict if cid in eval_context_ids
            )
        ]
        item_texts_list = [
            item_texts_dict.get(iid, "")
            for iid in sorted(item_texts_dict.keys())
        ]
        
        metrics = evaluate_biencoder(
            query_embs,
            item_embs,
            relevance_data,
            k_values=k_values,
            use_faiss=True,
            query_texts_sample=query_texts_list,
            item_texts_sample=item_texts_list
        )
        
        # Add model info
        metrics["model_name"] = model_name
        metrics["embedding_dim"] = query_embs.shape[1]
        metrics["num_queries"] = len(query_embs)
        metrics["num_items"] = len(item_embs)
        
        results.append(metrics)
        
        # Print results for this model
        print_ranking_metrics(metrics, f"Results for {model_name}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    metric_cols = [c for c in df.columns if "@" in c or c == "mrr"]
    other_cols = [c for c in df.columns if c not in metric_cols]
    df = df[other_cols + sorted(metric_cols)]
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Bi-encoder evaluation for item embeddings (Section 10.1)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ijcai",
        choices=["ijcai"],
        help="Dataset to evaluate"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder model name"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use (None = all)"
    )
    
    parser.add_argument(
        "--relevance",
        type=str,
        default="conversion",
        choices=["conversion", "click"],
        help="Relevance definition"
    )
    
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 50, 100],
        help="K values for Precision@K, Recall@K, NDCG@K"
    )
    
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Use FAISS for fast similarity search"
    )
    
    parser.add_argument(
        "--compare_models",
        action="store_true",
        help="Compare multiple models"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare models mode
    if args.compare_models:
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "rexbert-base",  # E-commerce domain-specialized
        ]
        
        comparison_df = compare_models_on_ijcai(
            models=models,
            sample_size=args.sample_size if args.sample_size else 10000,
            k_values=args.k_values
        )
        
        print("\n" + "="*80)
        print("Model Comparison Results")
        print("="*80)
        print(comparison_df.to_string())
        
        # Save results
        output_file = output_dir / f"biencoder_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")
        
        return
    
    # Single model evaluation
    logger.info("="*80)
    logger.info("Bi-Encoder Evaluation")
    logger.info("="*80)
    
    # Load dataset
    dataset = IJCAIDataset(
        data_dir=DEFAULT_IJCAI_CONFIG.data_dir,
        sample_size=args.sample_size,
        min_interactions=1,
        random_seed=42
    )
    
    dataset.load_data(split="train")
    dataset.print_stats()
    
    # Initialize encoder (factory handles SBERT vs RexBERT)
    encoder = create_encoder(
        model_name=args.model,
        max_seq_length=128,
        normalize_embeddings=True
    )
    
    print(f"\nModel: {args.model}")
    print(f"Embedding dimension: {encoder.embedding_dim}")
    
    # Encode queries and items (chronological split: evaluate on test set)
    (query_embeddings, item_embeddings, relevance_data,
     query_ids, item_ids) = encode_queries_and_items(
        dataset,
        encoder,
        relevance_threshold=args.relevance,
        eval_split="test"
    )
    
    logger.info(f"\nQuery embeddings: {query_embeddings.shape}")
    logger.info(f"Item embeddings: {item_embeddings.shape}")
    
    # Evaluate
    metrics = evaluate_biencoder(
        query_embeddings,
        item_embeddings,
        relevance_data,
        k_values=args.k_values,
        use_faiss=args.use_faiss
    )
    
    # Print results
    print_ranking_metrics(metrics, "Bi-Encoder Retrieval Performance (Chronological Test Set)")
    
    # Save results
    results = {
        "model_name": args.model,
        "dataset": args.dataset,
        "sample_size": args.sample_size,
        "relevance_threshold": args.relevance,
        "split_strategy": "chronological (by context_timestamp)",
        "eval_split": "test (latest 20% by timestamp)",
        "num_queries": len(query_embeddings),
        "num_items": len(item_embeddings),
        "embedding_dim": query_embeddings.shape[1],
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = output_dir / f"biencoder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
