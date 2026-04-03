"""
Hybrid search: combining BM25, SPLADE, and Dense retrieval.

Implements two fusion strategies:
  1. Reciprocal Rank Fusion (RRF) - score-agnostic, rank-based fusion
  2. Weighted Score Fusion - min-max normalized scores with method weights

Evaluation pipeline:
  Part A: Curated pool evaluation (Qdrant-style, high label coverage)
  Part B: Full-catalog retrieval (1.2M products, Recall@K)
  Part C: Score-based analysis (boxplots, separability AUC, re-ranking NDCG)
  Part D: LLM-augmented evaluation (stub, future work)

Usage:
  python hybrid_search.py --mode evaluate --max_queries 5000
  python hybrid_search.py --mode evaluate --eval_mode curated --max_queries 2000
  python hybrid_search.py --mode demo --demo_query "wireless bluetooth headphones"
"""

import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging
import argparse
import time
import json

import torch
from tqdm import tqdm

from config import (
    HybridConfig, DEFAULT_HYBRID_CONFIG, DEFAULT_DATA_CONFIG,
    DEFAULT_BM25_CONFIG, DEFAULT_DENSE_CONFIG,
    DEFAULT_EVAL_CONFIG, METRICS_DIR, PLOTS_DIR,
    SPLADE_MODELS, SPLADEConfig,
)
from data_loader import ESCIDataset
from bm25_search import BM25SearchEngine
from splade_search import SPLADEEncoder, SPLADEIndex
from dense_search import DenseEncoder, DenseIndex
from evaluate import (
    evaluate_candidate_scoring, evaluate_retrieval,
    evaluate_curated_pool, print_curated_pool_results,
    print_evaluation_results, print_comparison_table,
    save_results_json, save_comparison_csv,
    plot_score_boxplots, plot_label_distribution_bars,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fusion Functions
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
    top_n: int = 100,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF).

    Combines multiple ranked lists into a single ranking using:
      RRF_score(d) = sum_{r in rankings} 1 / (k + rank_r(d))

    where k is a smoothing constant (default 60, from the original paper).

    This is score-agnostic: it only uses rank positions, making it robust
    to different score scales across methods.
    """
    rrf_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            rrf_scores[item_id] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]


def weighted_score_fusion(
    score_dicts: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Weighted score fusion with min-max normalization.

    For each method, normalizes scores to [0, 1] using min-max scaling,
    then computes a weighted sum across methods.
    """
    normalized = {}
    for method, scores in score_dicts.items():
        if not scores:
            normalized[method] = {}
            continue

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val < 1e-10:
            normalized[method] = {k: 0.5 for k in scores}
        else:
            normalized[method] = {
                k: (v - min_val) / range_val for k, v in scores.items()
            }

    fused = defaultdict(float)
    for method, norm_scores in normalized.items():
        w = weights.get(method, 0)
        for item_id, score in norm_scores.items():
            fused[item_id] += w * score

    return dict(fused)


# =============================================================================
# Component Loading
# =============================================================================

def load_all_components(
    device: str = "cpu",
    splade_model_keys: Optional[List[str]] = None,
):
    """Load all search indexes and encoders.

    Args:
        device: torch device
        splade_model_keys: Which SPLADE variants to load (default: ["prithivi"])

    Returns:
        Dict with keys: bm25_engine, dense_encoder, dense_index,
        splade_components (dict of {key: (encoder, index)})
    """
    if splade_model_keys is None:
        splade_model_keys = ["prithivi"]

    logger.info("Loading all search components...")
    components = {}

    # BM25
    logger.info("Loading BM25...")
    bm25_engine = BM25SearchEngine()
    if not bm25_engine.load():
        logger.warning("BM25 index not found. Run: python bm25_search.py --mode index")
        bm25_engine = None
    components["bm25_engine"] = bm25_engine

    # SPLADE variants
    components["splade_components"] = {}
    for model_key in splade_model_keys:
        if model_key not in SPLADE_MODELS:
            logger.warning(f"Unknown SPLADE model key: {model_key}")
            continue
        splade_config = SPLADE_MODELS[model_key]
        logger.info(f"Loading {splade_config.label}...")
        encoder = SPLADEEncoder(config=splade_config)
        encoder.load_model(device=device)
        index = SPLADEIndex.load(path=splade_config.cache_dir)
        if index is None:
            logger.warning(f"  {splade_config.label} index not found. "
                           f"Run: python splade_search.py --mode encode --splade_model {model_key}")
        components["splade_components"][model_key] = (encoder, index)

    # Dense
    logger.info("Loading Dense...")
    dense_encoder = DenseEncoder()
    dense_encoder.load_model(device=device)
    dense_index = DenseIndex.load()
    if dense_index is None:
        logger.warning("Dense index not found. Run: python dense_search.py --mode encode")
    components["dense_encoder"] = dense_encoder
    components["dense_index"] = dense_index

    return components


# =============================================================================
# Curated Pool Evaluation (Part A)
# =============================================================================

def run_curated_pool_evaluation(
    dataset: ESCIDataset,
    components: Dict,
    max_queries: int = 2000,
    eval_config: Any = DEFAULT_EVAL_CONFIG,
    hybrid_config: HybridConfig = DEFAULT_HYBRID_CONFIG,
) -> Dict[str, Dict[str, Any]]:
    """Part A: Curated pool evaluation (Qdrant-style).

    Creates a small product pool from ESCI-labeled products, retrieves
    within this pool, and computes NDCG/MRR/Recall with high label coverage.
    Unlabeled products in the pool are treated as Irrelevant (gain=0.0).
    """
    logger.info("=" * 60)
    logger.info("Part A: Curated Pool Evaluation (Qdrant-style)")
    logger.info("=" * 60)

    query_data, pool_product_ids = dataset.get_curated_pool_data(
        max_queries=max_queries,
        split=eval_config.curated_pool_split,
    )

    bm25_engine = components["bm25_engine"]
    dense_encoder = components["dense_encoder"]
    dense_index = components["dense_index"]
    splade_components = components["splade_components"]
    k_values = eval_config.k_values
    top_k = max(k_values)

    # Filter pool_product_ids to a list for restricting retrieval
    pool_ids_set = pool_product_ids
    curated_pool_results = {}

    # Over-retrieval factor: retrieve more than top_k to ensure enough pool hits
    # after filtering. 5x is sufficient since pool products cluster in top results.
    over_retrieve_k = top_k * 5

    # --- BM25 retrieval within curated pool ---
    if bm25_engine is not None:
        retrieved = {}
        for i, (qid, qdata) in enumerate(tqdm(query_data.items(), desc="BM25 (curated)")):
            result_ids, result_scores = bm25_engine.search(qdata["query_text"], top_k=over_retrieve_k)
            # Filter to pool only
            filtered = [pid for pid in result_ids if pid in pool_ids_set][:top_k]
            retrieved[qid] = filtered
            if i % 500 == 0:
                gc.collect()
        result = evaluate_curated_pool(retrieved, query_data, k_values, eval_config.curated_pool_gains)
        curated_pool_results["BM25"] = result
        gc.collect()

    # --- SPLADE variants retrieval within curated pool ---
    for model_key, (encoder, index) in splade_components.items():
        if index is None:
            continue
        splade_config = SPLADE_MODELS[model_key]
        method_name = splade_config.label
        retrieved = {}
        for qid, qdata in tqdm(query_data.items(), desc=f"{method_name} (curated)"):
            qv = encoder.encode_single(qdata["query_text"])
            result_ids, _ = index.search(qv, top_k=over_retrieve_k)
            filtered = [pid for pid in result_ids if pid in pool_ids_set][:top_k]
            retrieved[qid] = filtered
        result = evaluate_curated_pool(retrieved, query_data, k_values, eval_config.curated_pool_gains)
        curated_pool_results[method_name] = result

    # --- Dense retrieval within curated pool ---
    if dense_encoder is not None and dense_index is not None:
        retrieved = {}
        for qid, qdata in tqdm(query_data.items(), desc="Dense (curated)"):
            qe = dense_encoder.encode_query(qdata["query_text"])
            result_ids, _ = dense_index.search(qe, top_k=over_retrieve_k)
            filtered = [pid for pid in result_ids if pid in pool_ids_set][:top_k]
            retrieved[qid] = filtered
        result = evaluate_curated_pool(retrieved, query_data, k_values, eval_config.curated_pool_gains)
        curated_pool_results["Dense (SBERT)"] = result

    # --- Hybrid RRF (using first available SPLADE variant) ---
    base_methods = {}
    if "BM25" in curated_pool_results and bm25_engine is not None:
        base_methods["BM25"] = bm25_engine
    # Use first loaded SPLADE for hybrid
    first_splade_key = None
    for mk, (enc, idx) in splade_components.items():
        if idx is not None:
            first_splade_key = mk
            break

    if first_splade_key and dense_index is not None and len(base_methods) >= 1:
        sp_enc, sp_idx = splade_components[first_splade_key]
        retrieved = {}
        for qid, qdata in tqdm(query_data.items(), desc="Hybrid RRF (curated)"):
            ranked_lists = []
            if bm25_engine is not None:
                bids, _ = bm25_engine.search(qdata["query_text"], top_k=over_retrieve_k)
                ranked_lists.append([pid for pid in bids if pid in pool_ids_set][:top_k])
            qv = sp_enc.encode_single(qdata["query_text"])
            sids, _ = sp_idx.search(qv, top_k=over_retrieve_k)
            ranked_lists.append([pid for pid in sids if pid in pool_ids_set][:top_k])
            qe = dense_encoder.encode_query(qdata["query_text"])
            dids, _ = dense_index.search(qe, top_k=over_retrieve_k)
            ranked_lists.append([pid for pid in dids if pid in pool_ids_set][:top_k])

            rrf_result = reciprocal_rank_fusion(ranked_lists, k=hybrid_config.rrf_k, top_n=top_k)
            retrieved[qid] = [pid for pid, _ in rrf_result]
        result = evaluate_curated_pool(retrieved, query_data, k_values, eval_config.curated_pool_gains)
        curated_pool_results["Hybrid (RRF)"] = result

    # --- BM25 Rerank of Dense candidates (within curated pool) ---
    if bm25_engine is not None and dense_encoder is not None and dense_index is not None:
        retrieved = {}
        for qid, qdata in tqdm(query_data.items(), desc="BM25 Rerank Dense (curated)"):
            # Get Dense candidates
            qe = dense_encoder.encode_query(qdata["query_text"])
            dids, _ = dense_index.search(qe, top_k=over_retrieve_k)
            dense_candidates = [pid for pid in dids if pid in pool_ids_set][:top_k]
            # Get BM25 scores for these candidates
            bm25_ids, bm25_scores = bm25_engine.search(qdata["query_text"], top_k=eval_config.bm25_rerank_cache_top_k)
            bm25_score_dict = dict(zip(bm25_ids, bm25_scores.tolist()))
            # Re-rank by BM25 score
            scored = [(pid, bm25_score_dict.get(pid, 0.0)) for pid in dense_candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            retrieved[qid] = [pid for pid, _ in scored]
        result = evaluate_curated_pool(retrieved, query_data, k_values, eval_config.curated_pool_gains)
        curated_pool_results["BM25 Rerank (Dense)"] = result

    # --- BM25 Rerank of first SPLADE candidates ---
    if bm25_engine is not None and first_splade_key:
        sp_enc, sp_idx = splade_components[first_splade_key]
        sp_label = SPLADE_MODELS[first_splade_key].label
        retrieved = {}
        for qid, qdata in tqdm(query_data.items(), desc=f"BM25 Rerank {sp_label} (curated)"):
            qv = sp_enc.encode_single(qdata["query_text"])
            sids, _ = sp_idx.search(qv, top_k=over_retrieve_k)
            splade_candidates = [pid for pid in sids if pid in pool_ids_set][:top_k]
            bm25_ids, bm25_scores = bm25_engine.search(qdata["query_text"], top_k=eval_config.bm25_rerank_cache_top_k)
            bm25_score_dict = dict(zip(bm25_ids, bm25_scores.tolist()))
            scored = [(pid, bm25_score_dict.get(pid, 0.0)) for pid in splade_candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            retrieved[qid] = [pid for pid, _ in scored]
        result = evaluate_curated_pool(retrieved, query_data, k_values, eval_config.curated_pool_gains)
        curated_pool_results[f"BM25 Rerank ({sp_label})"] = result

    # Print results
    print_curated_pool_results(curated_pool_results, k_values)

    # Save results
    save_path = METRICS_DIR / "curated_pool_results.json"
    with open(save_path, "w") as f:
        json.dump(curated_pool_results, f, indent=2, default=str)
    logger.info(f"  Curated pool results saved to {save_path}")

    return curated_pool_results


# =============================================================================
# Full-Catalog Retrieval (Part B) + Score-Based Analysis (Part C)
# =============================================================================

def run_full_evaluation(
    dataset: ESCIDataset,
    components: Dict,
    max_queries: int = 5000,
    hybrid_config: HybridConfig = DEFAULT_HYBRID_CONFIG,
    eval_config: Any = DEFAULT_EVAL_CONFIG,
) -> List[Dict[str, Any]]:
    """Run Parts B+C: full-catalog retrieval and score-based analysis.

    Part B: Full-catalog retrieval (1.2M products, Recall@K, label distribution)
    Part C: Score-based analysis (boxplots, NDCG, separability AUC)
    """
    bm25_engine = components["bm25_engine"]
    dense_encoder = components["dense_encoder"]
    dense_index = components["dense_index"]
    # Use first available SPLADE for this evaluation (default: prithivi)
    splade_encoder, splade_index = None, None
    splade_label = "SPLADE"
    for mk, (enc, idx) in components["splade_components"].items():
        if idx is not None:
            splade_encoder, splade_index = enc, idx
            splade_label = SPLADE_MODELS[mk].label
            break

    all_results = []

    # =========================================================================
    # Part B: Score-based evaluation on annotated candidates
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Part B: Score-based evaluation on annotated candidates")
    logger.info("=" * 60)

    query_candidates = dataset.get_query_candidates(max_queries=max_queries)

    # BM25 scoring
    if bm25_engine is not None:
        bm25_results = evaluate_candidate_scoring(
            query_candidates, bm25_engine.score_candidates, "BM25", eval_config
        )
        print_evaluation_results(bm25_results)
        all_results.append(bm25_results)

    # SPLADE scoring (first variant)
    if splade_encoder is not None and splade_index is not None:
        def splade_scoring_fn(query_text, candidate_ids):
            qv = splade_encoder.encode_single(query_text)
            return splade_index.score_candidates(qv, candidate_ids)

        splade_results = evaluate_candidate_scoring(
            query_candidates, splade_scoring_fn, splade_label, eval_config
        )
        print_evaluation_results(splade_results)
        all_results.append(splade_results)

    # Dense scoring
    if dense_encoder is not None and dense_index is not None:
        def dense_scoring_fn(query_text, candidate_ids):
            qe = dense_encoder.encode_query(query_text)
            return dense_index.score_candidates(qe, candidate_ids)

        dense_results = evaluate_candidate_scoring(
            query_candidates, dense_scoring_fn, "Dense (SBERT)", eval_config
        )
        print_evaluation_results(dense_results)
        all_results.append(dense_results)

    # Hybrid scoring (RRF)
    if len(all_results) >= 2:
        def hybrid_rrf_scoring_fn(query_text, candidate_ids):
            method_rankings = []
            if bm25_engine is not None:
                bm25_scores = bm25_engine.score_candidates(query_text, candidate_ids)
                order = np.argsort(-bm25_scores)
                method_rankings.append([candidate_ids[i] for i in order])
            if splade_encoder is not None and splade_index is not None:
                qv = splade_encoder.encode_single(query_text)
                sp_scores = splade_index.score_candidates(qv, candidate_ids)
                order = np.argsort(-sp_scores)
                method_rankings.append([candidate_ids[i] for i in order])
            if dense_encoder is not None and dense_index is not None:
                qe = dense_encoder.encode_query(query_text)
                dn_scores = dense_index.score_candidates(qe, candidate_ids)
                order = np.argsort(-dn_scores)
                method_rankings.append([candidate_ids[i] for i in order])
            rrf_result = reciprocal_rank_fusion(
                method_rankings, k=hybrid_config.rrf_k, top_n=len(candidate_ids)
            )
            rrf_dict = {item_id: score for item_id, score in rrf_result}
            return np.array([rrf_dict.get(cid, 0.0) for cid in candidate_ids])

        hybrid_rrf_results = evaluate_candidate_scoring(
            query_candidates, hybrid_rrf_scoring_fn, "Hybrid (RRF)", eval_config
        )
        print_evaluation_results(hybrid_rrf_results)
        all_results.append(hybrid_rrf_results)

        def hybrid_weighted_scoring_fn(query_text, candidate_ids):
            score_dicts = {}
            if bm25_engine is not None:
                bm25_s = bm25_engine.score_candidates(query_text, candidate_ids)
                score_dicts["bm25"] = {cid: float(s) for cid, s in zip(candidate_ids, bm25_s)}
            if splade_encoder is not None and splade_index is not None:
                qv = splade_encoder.encode_single(query_text)
                sp_s = splade_index.score_candidates(qv, candidate_ids)
                score_dicts["splade"] = {cid: float(s) for cid, s in zip(candidate_ids, sp_s)}
            if dense_encoder is not None and dense_index is not None:
                qe = dense_encoder.encode_query(query_text)
                dn_s = dense_index.score_candidates(qe, candidate_ids)
                score_dicts["dense"] = {cid: float(s) for cid, s in zip(candidate_ids, dn_s)}
            fused = weighted_score_fusion(score_dicts, hybrid_config.score_weights)
            return np.array([fused.get(cid, 0.0) for cid in candidate_ids])

        hybrid_w_results = evaluate_candidate_scoring(
            query_candidates, hybrid_weighted_scoring_fn, "Hybrid (Weighted)", eval_config
        )
        print_evaluation_results(hybrid_w_results)
        all_results.append(hybrid_w_results)

    # =========================================================================
    # Part C: Full-catalog retrieval evaluation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Part C: Full-catalog retrieval evaluation")
    logger.info("=" * 60)

    eval_queries = dataset.get_evaluation_queries(max_queries=max_queries)
    k_values = eval_config.k_values
    top_k = max(k_values)

    method_retrieved = {}

    # BM25 retrieval + score caching for reranking
    bm25_score_cache = {}
    if bm25_engine is not None:
        retrieved = {}
        cache_top_k = eval_config.bm25_rerank_cache_top_k
        for qid, qdata in tqdm(eval_queries.items(), desc="BM25 Retrieval"):
            result_ids, result_scores = bm25_engine.search(
                qdata["query_text"], top_k=cache_top_k
            )
            retrieved[qid] = result_ids[:top_k]
            bm25_score_cache[qid] = dict(zip(result_ids, result_scores.tolist()))
        method_retrieved["BM25"] = retrieved

    # SPLADE retrieval (first variant)
    if splade_encoder is not None and splade_index is not None:
        retrieved = {}
        for qid, qdata in tqdm(eval_queries.items(), desc=f"{splade_label} Retrieval"):
            qv = splade_encoder.encode_single(qdata["query_text"])
            result_ids, _ = splade_index.search(qv, top_k=top_k)
            retrieved[qid] = result_ids
        method_retrieved[splade_label] = retrieved

    # Dense retrieval
    if dense_encoder is not None and dense_index is not None:
        retrieved = {}
        for qid, qdata in tqdm(eval_queries.items(), desc="Dense Retrieval"):
            qe = dense_encoder.encode_query(qdata["query_text"])
            result_ids, _ = dense_index.search(qe, top_k=top_k)
            retrieved[qid] = result_ids
        method_retrieved["Dense (SBERT)"] = retrieved

    # BM25 Re-ranking of Dense candidates
    if bm25_score_cache and "Dense (SBERT)" in method_retrieved:
        reranked = {}
        for qid in eval_queries:
            dense_ids = method_retrieved["Dense (SBERT)"][qid]
            cache = bm25_score_cache.get(qid, {})
            scored = [(pid, cache.get(pid, 0.0)) for pid in dense_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked[qid] = [pid for pid, _ in scored]
        method_retrieved["BM25 Rerank (Dense)"] = reranked

    # BM25 Re-ranking of SPLADE candidates
    if bm25_score_cache and splade_label in method_retrieved:
        reranked = {}
        for qid in eval_queries:
            splade_ids = method_retrieved[splade_label][qid]
            cache = bm25_score_cache.get(qid, {})
            scored = [(pid, cache.get(pid, 0.0)) for pid in splade_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            reranked[qid] = [pid for pid, _ in scored]
        method_retrieved[f"BM25 Rerank ({splade_label})"] = reranked

    # Hybrid RRF retrieval
    base_method_names = ["BM25", splade_label, "Dense (SBERT)"]
    available_base = [m for m in base_method_names if m in method_retrieved]
    if len(available_base) >= 2:
        hybrid_rrf_retrieved = {}
        for qid in tqdm(eval_queries.keys(), desc="Hybrid RRF"):
            ranked_lists = [method_retrieved[m][qid] for m in available_base]
            rrf_result = reciprocal_rank_fusion(
                ranked_lists, k=hybrid_config.rrf_k, top_n=top_k
            )
            hybrid_rrf_retrieved[qid] = [item_id for item_id, _ in rrf_result]
        method_retrieved["Hybrid (RRF)"] = hybrid_rrf_retrieved

    # Compute retrieval metrics
    for method_name, retrieved in method_retrieved.items():
        retrieval_results = evaluate_retrieval(retrieved, eval_queries, k_values)
        for r in all_results:
            if r["method"] == method_name:
                r["retrieval_metrics"] = retrieval_results
                break
        else:
            all_results.append({
                "method": method_name,
                "num_queries": retrieval_results.get("num_queries", 0),
                "ranking_metrics": {},
                "label_distribution": {},
                "per_label_scores": {},
                "separability_auc": 0.0,
                "retrieval_metrics": retrieval_results,
            })
        logger.info(f"\n  {method_name} retrieval:")
        for k in k_values:
            r_e = retrieval_results.get(f"recall_exact@{k}", 0)
            r_es = retrieval_results.get(f"recall_exact_sub@{k}", 0)
            logger.info(f"    Recall@{k} (E): {r_e:.4f}  (E+S): {r_es:.4f}")

    # =========================================================================
    # Part D: LLM-Augmented Evaluation (stub)
    # =========================================================================
    llm_labels_path = METRICS_DIR / "llm_augmented_labels.json"
    if llm_labels_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("Part D: LLM-Augmented Evaluation")
        logger.info("=" * 60)
        logger.info(f"  Loading LLM labels from {llm_labels_path}")
        # Future: load labels, build query_data, run evaluate_curated_pool
        logger.info("  (Not yet implemented — stub for future LLM-augmented eval)")
    else:
        logger.info("\n  Part D: LLM-Augmented Evaluation — skipped (no labels file)")

    # =========================================================================
    # Comparison and visualization
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Comparison and visualization")
    logger.info("=" * 60)

    # Filter to results that have ranking_metrics for comparison table
    scorable_results = [r for r in all_results if r.get("ranking_metrics")]
    if scorable_results:
        print_comparison_table(scorable_results)
        plot_score_boxplots(scorable_results)
        plot_label_distribution_bars(scorable_results, k_values)

    save_results_json(all_results)
    save_comparison_csv([r for r in all_results if r.get("ranking_metrics")])

    return all_results


# =============================================================================
# Demo Search
# =============================================================================

def run_demo_search(
    query: str,
    components: Dict,
    dataset: ESCIDataset,
    top_k: int = 10,
    hybrid_config: HybridConfig = DEFAULT_HYBRID_CONFIG,
) -> None:
    """Run a demo search query and display side-by-side results."""
    print(f"\n{'='*80}")
    print(f'  Demo Search: "{query}"')
    print(f"{'='*80}")

    bm25_engine = components["bm25_engine"]
    dense_encoder = components["dense_encoder"]
    dense_index = components["dense_index"]
    product_texts = dataset.get_product_texts()

    results = {}

    # BM25
    if bm25_engine is not None:
        bm25_ids, bm25_scores = bm25_engine.search(query, top_k=top_k)
        results["BM25"] = list(zip(bm25_ids, bm25_scores.tolist()))

    # All SPLADE variants
    for mk, (enc, idx) in components["splade_components"].items():
        if idx is None:
            continue
        label = SPLADE_MODELS[mk].label
        qv = enc.encode_single(query)
        sids, sscores = idx.search(qv, top_k=top_k)
        results[label] = list(zip(sids, sscores.tolist()))

    # Dense
    if dense_encoder is not None and dense_index is not None:
        qe = dense_encoder.encode_query(query)
        dids, dscores = dense_index.search(qe, top_k=top_k)
        results["Dense (SBERT)"] = list(zip(dids, dscores.tolist()))

    # Hybrid RRF
    if len(results) >= 2:
        ranked_lists = [[pid for pid, _ in r] for r in results.values()]
        rrf_result = reciprocal_rank_fusion(ranked_lists, k=hybrid_config.rrf_k, top_n=top_k)
        results["Hybrid (RRF)"] = rrf_result

    for method_name, method_results in results.items():
        print(f"\n  --- {method_name} ---")
        for rank, (product_id, score) in enumerate(method_results[:top_k], 1):
            text = product_texts.get(product_id, "???")[:100]
            print(f"  {rank:2d}. [{score:.4f}] {text}...")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Search: BM25 + SPLADE + Dense with Fusion"
    )
    parser.add_argument("--mode", type=str,
                        choices=["evaluate", "demo"],
                        default="evaluate", help="Mode to run")
    parser.add_argument("--eval_mode", type=str,
                        choices=["all", "curated", "full_catalog"],
                        default="all",
                        help="Evaluation mode: curated (Part A), full_catalog (Parts B+C), all")
    parser.add_argument("--max_queries", type=int, default=5000,
                        help="Max queries for evaluation (0 = all)")
    parser.add_argument("--max_products", type=int, default=0,
                        help="Max products (0 = all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--splade_models", type=str, default="prithivi",
                        help="Comma-separated SPLADE model keys (e.g., prithivi,qdrant_esci,naver)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF smoothing constant")
    parser.add_argument("--demo_query", type=str,
                        default="wireless bluetooth headphones",
                        help="Query for demo mode")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K for demo display")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    splade_model_keys = [k.strip() for k in args.splade_models.split(",")]

    # Load dataset
    data_config = DEFAULT_DATA_CONFIG
    data_config.max_products = args.max_products
    dataset = ESCIDataset(data_config)
    dataset.load_examples()
    dataset.load_products()

    # Load components
    components = load_all_components(device=device, splade_model_keys=splade_model_keys)

    hybrid_config = DEFAULT_HYBRID_CONFIG
    hybrid_config.rrf_k = args.rrf_k

    if args.mode == "evaluate":
        if args.eval_mode in ("all", "curated"):
            run_curated_pool_evaluation(
                dataset=dataset,
                components=components,
                max_queries=args.max_queries,
                eval_config=DEFAULT_EVAL_CONFIG,
                hybrid_config=hybrid_config,
            )
        if args.eval_mode in ("all", "full_catalog"):
            run_full_evaluation(
                dataset=dataset,
                components=components,
                max_queries=args.max_queries,
                hybrid_config=hybrid_config,
            )

    elif args.mode == "demo":
        run_demo_search(
            query=args.demo_query,
            components=components,
            dataset=dataset,
            top_k=args.top_k,
            hybrid_config=hybrid_config,
        )


if __name__ == "__main__":
    main()
