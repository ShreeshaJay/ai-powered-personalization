"""
Scaled LLM Labeling Pipeline: Gemini 2.5 Flash Lite for ESCI-style labels.

Incrementally labels query-product pairs retrieved by BM25/SPLADE/Dense,
storing results in a persistent JSON store keyed by (query_id, product_id).

Key features:
  - Incremental: only labels pairs not already in the store
  - Resumable: checkpoints every N pairs, safe to interrupt and restart
  - Concurrent: async API calls for 10-20x throughput
  - Phased: label top-K per method, extend K later without re-labeling
  - Cost-tracked: logs token usage and running cost estimate

Usage:
  # Phase 1: Label union of top-100 from each method for 10K queries
  # Uses retrieval-diversity sampling (Option C) by default
  python llm_labeler.py --top_k 100 --max_queries 10000 --concurrency 10

  # Phase 2: Extend to top-500 (only labels NEW pairs not in store)
  python llm_labeler.py --top_k 500 --max_queries 10000 --concurrency 10

  # Dry run: see how many new pairs would be labeled and est. cost
  python llm_labeler.py --top_k 100 --max_queries 10000 --dry_run

  # Use random sampling instead of diversity sampling
  python llm_labeler.py --top_k 100 --max_queries 10000 --sampling random

  # Resume after interruption (automatic — reads existing store)
  python llm_labeler.py --top_k 100 --max_queries 10000 --concurrency 10
"""

import json
import os
import time
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from config import (
    DEFAULT_DATA_CONFIG, DEFAULT_EVAL_CONFIG, DEFAULT_SPLADE_CONFIG,
    DEFAULT_DENSE_CONFIG, DEFAULT_BM25_CONFIG, METRICS_DIR, CACHE_DIR,
    SPLADE_MODELS, ESCI_LABEL_NAMES,
)
from data_loader import ESCIDataset

logger = logging.getLogger(__name__)

# =============================================================================
# Label Store — persistent, incremental, keyed by (query_id, product_id)
# =============================================================================

LABEL_STORE_PATH = METRICS_DIR / "llm_labels.json"


class LabelStore:
    """Persistent store for LLM-generated labels.

    Format: {
        "metadata": { "model": "...", "created": "...", "total_labeled": N },
        "labels": {
            "<query_id>": {
                "<product_id>": {
                    "label": "E|S|C|I",
                    "timestamp": "...",
                }
            }
        }
    }

    Designed for incremental labeling:
      - has(query_id, product_id) checks if already labeled
      - add(query_id, product_id, label) adds a new label
      - save() persists to disk (called periodically for checkpointing)
    """

    def __init__(self, path: Path = LABEL_STORE_PATH):
        self.path = path
        self.data: Dict = {"metadata": {}, "labels": {}}
        self._dirty_count = 0
        self.load()

    def load(self) -> None:
        """Load existing labels from disk."""
        if self.path.exists():
            with open(self.path, "r") as f:
                self.data = json.load(f)
            total = sum(len(prods) for prods in self.data.get("labels", {}).values())
            logger.info(f"  Label store loaded: {total:,} existing labels from {self.path}")
        else:
            self.data = {
                "metadata": {
                    "model": "gemini-2.5-flash-lite",
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_labeled": 0,
                },
                "labels": {},
            }
            logger.info("  Label store: starting fresh")

    def has(self, query_id: str, product_id: str) -> bool:
        """Check if a (query_id, product_id) pair is already labeled."""
        qid = str(query_id)
        return qid in self.data["labels"] and product_id in self.data["labels"][qid]

    def get(self, query_id: str, product_id: str) -> Optional[str]:
        """Get label for a pair, or None if not labeled."""
        qid = str(query_id)
        if qid in self.data["labels"] and product_id in self.data["labels"][qid]:
            return self.data["labels"][qid][product_id]["label"]
        return None

    def add(self, query_id: str, product_id: str, label: str) -> None:
        """Add a label to the store."""
        qid = str(query_id)
        if qid not in self.data["labels"]:
            self.data["labels"][qid] = {}
        self.data["labels"][qid][product_id] = {
            "label": label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._dirty_count += 1
        self.data["metadata"]["total_labeled"] = self.total_labels()

    def total_labels(self) -> int:
        """Count total labels in store."""
        return sum(len(prods) for prods in self.data["labels"].values())

    def save(self, force: bool = False) -> None:
        """Save to disk. Called periodically for checkpointing."""
        if self._dirty_count == 0 and not force:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"  Checkpoint: saved {self.total_labels():,} labels "
                    f"({self._dirty_count} new since last save)")
        self._dirty_count = 0

    def get_stats(self) -> Dict:
        """Get label distribution stats."""
        all_labels = []
        for prods in self.data["labels"].values():
            for entry in prods.values():
                all_labels.append(entry["label"])
        return dict(Counter(all_labels))


# =============================================================================
# Prompt template (same as benchmark for consistency)
# =============================================================================

LABELING_PROMPT = """You are evaluating product relevance for e-commerce search.

Given a search query and a product, classify the product into one of four categories:

- **E (Exact)**: The product is a relevant result for the query and satisfies the query.
- **S (Substitute)**: The product is somewhat relevant -- it's a functional substitute but not an exact match (e.g., wrong brand, different variant).
- **C (Complement)**: The product is complementary to the query (e.g., a case for a phone query, batteries for a toy).
- **I (Irrelevant)**: The product is not relevant to the query at all.

Query: "{query}"

Product: "{product_text}"

Respond with ONLY the single letter: E, S, C, or I."""


# =============================================================================
# Async Gemini caller with rate limiting
# =============================================================================

class GeminiLabeler:
    """Async Gemini API caller with concurrency control and rate limiting."""

    def __init__(self, concurrency: int = 10, rpm_limit: int = 1000):
        self.concurrency = concurrency
        self.rpm_limit = rpm_limit
        self.semaphore = None  # initialized in async context
        self.client = None
        self.total_calls = 0
        self.total_errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def init_client(self):
        """Initialize Gemini client (sync, called before async loop)."""
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Set it via:\n"
                "  Windows: $env:GOOGLE_API_KEY = 'your-key'\n"
                "  Linux:   export GOOGLE_API_KEY='your-key'"
            )
        self.client = genai.Client(api_key=api_key)
        logger.info(f"  Gemini client initialized (concurrency={self.concurrency})")

    async def label_one(
        self, query: str, product_text: str
    ) -> Tuple[str, float]:
        """Label a single query-product pair. Returns (label, latency)."""
        prompt = LABELING_PROMPT.format(query=query, product_text=product_text)

        async with self.semaphore:
            t0 = time.time()
            try:
                # Use sync client in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=prompt,
                        config={
                            "max_output_tokens": 5,
                            "temperature": 0.0,
                        },
                    )
                )
                latency = time.time() - t0
                text = response.text.strip().upper()
                label = text[0] if text and text[0] in "ESCI" else "?"
                self.total_calls += 1

                # Track token usage if available
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    if hasattr(usage, 'prompt_token_count'):
                        self.total_input_tokens += usage.prompt_token_count
                    if hasattr(usage, 'candidates_token_count'):
                        self.total_output_tokens += usage.candidates_token_count

                return label, latency

            except Exception as e:
                latency = time.time() - t0
                self.total_errors += 1
                # Rate limit: back off
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning(f"  Rate limited, backing off 5s...")
                    await asyncio.sleep(5)
                else:
                    logger.warning(f"  API error: {e}")
                return "?", latency

    def get_cost_estimate(self) -> float:
        """Estimate cost based on tracked token usage."""
        # Gemini 2.5 Flash Lite: $0.10/1M input, $0.40/1M output
        input_cost = self.total_input_tokens * 0.10 / 1_000_000
        output_cost = self.total_output_tokens * 0.40 / 1_000_000
        return input_cost + output_cost


# =============================================================================
# Retrieval-Diversity Query Sampling (Option C)
# =============================================================================

DIVERSITY_CACHE_PATH = METRICS_DIR / "query_disagreement_scores.json"


def _load_search_components():
    """Load all search components from cache. Returns dict of components."""
    from bm25_search import BM25SearchEngine
    from dense_search import DenseEncoder, DenseIndex
    from splade_search import SPLADEEncoder, SPLADEIndex

    components = {"bm25": None, "dense_encoder": None, "dense_index": None,
                  "splade_encoder": None, "splade_index": None}

    # BM25
    logger.info("Loading BM25 index...")
    bm25 = BM25SearchEngine()
    if bm25.load():
        components["bm25"] = bm25
    else:
        logger.warning("  BM25 index not found — skipping")

    # Dense
    logger.info("Loading Dense (SBERT) index...")
    dense_index = DenseIndex.load()
    if dense_index is not None:
        dense_encoder = DenseEncoder()
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        dense_encoder.load_model(device=device)
        components["dense_encoder"] = dense_encoder
        components["dense_index"] = dense_index
    else:
        logger.warning("  Dense index not found — skipping")

    # SPLADE (Prithivi)
    logger.info("Loading SPLADE (Prithivi) index...")
    splade_config = SPLADE_MODELS["prithivi"]
    splade_index = SPLADEIndex.load(splade_config.cache_dir)
    if splade_index is not None:
        splade_encoder = SPLADEEncoder(splade_config)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        splade_encoder.load_model(device=device)
        components["splade_encoder"] = splade_encoder
        components["splade_index"] = splade_index
    else:
        logger.warning("  SPLADE index not found — skipping")

    return components


def _search_query(query_text: str, components: Dict, top_k: int = 50
                  ) -> Dict[str, Set[str]]:
    """Run a query through all available methods. Returns {method: set(pids)}."""
    results = {}

    if components["bm25"] is not None:
        try:
            ids, _ = components["bm25"].search(query_text, top_k=top_k)
            results["BM25"] = set(ids)
        except Exception:
            pass

    if components["dense_encoder"] is not None and components["dense_index"] is not None:
        try:
            emb = components["dense_encoder"].encode_query(query_text)
            ids, _ = components["dense_index"].search(emb, top_k=top_k)
            results["Dense"] = set(ids)
        except Exception:
            pass

    if components["splade_encoder"] is not None and components["splade_index"] is not None:
        try:
            vec = components["splade_encoder"].encode_single(query_text)
            ids, _ = components["splade_index"].search(vec, top_k=top_k)
            results["SPLADE"] = set(ids)
        except Exception:
            pass

    return results


def compute_query_disagreement(
    dataset: ESCIDataset,
    all_query_ids: List,
    top_k_for_disagreement: int = 50,
    cache_path: Path = DIVERSITY_CACHE_PATH,
) -> Dict[str, float]:
    """Compute method disagreement score for each query.

    Disagreement = mean pairwise Jaccard DISTANCE of top-K result sets
    across BM25, SPLADE, Dense. Higher = methods disagree more.

    Jaccard distance = 1 - |A ∩ B| / |A ∪ B|

    Results are cached to disk so this only needs to run once.

    Returns: {query_id_str: disagreement_score}
    """
    # Check cache
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
        cached_scores = cached.get("scores", {})
        # Check if we have enough cached
        cached_ids = set(cached_scores.keys())
        needed_ids = set(str(qid) for qid in all_query_ids)
        if needed_ids.issubset(cached_ids):
            logger.info(f"  Disagreement scores loaded from cache ({len(cached_scores):,} queries)")
            return {qid: cached_scores[qid] for qid in needed_ids}
        logger.info(f"  Partial cache: {len(cached_ids & needed_ids):,}/{len(needed_ids):,} queries")
    else:
        cached_scores = {}

    # Load search components
    components = _load_search_components()
    n_methods = sum(1 for k in ["bm25", "dense_index", "splade_index"]
                    if components.get(k) is not None)
    if n_methods < 2:
        raise ValueError(f"Need at least 2 search methods for disagreement, found {n_methods}")
    logger.info(f"  Computing disagreement with {n_methods} methods, top-{top_k_for_disagreement}")

    # Get query texts
    examples_path = dataset.config.data_dir / dataset.config.examples_file
    import pandas as pd
    all_examples = pd.read_parquet(examples_path)
    query_df = all_examples[
        all_examples["product_locale"] == dataset.config.locale
    ][["query_id", "query"]].drop_duplicates("query_id")
    qid_to_text = dict(zip(query_df["query_id"], query_df["query"]))

    scores = dict(cached_scores)  # Start from cache
    queries_to_compute = [qid for qid in all_query_ids if str(qid) not in scores]

    if len(queries_to_compute) == 0:
        logger.info("  All disagreement scores already cached!")
        return {str(qid): scores[str(qid)] for qid in all_query_ids}

    logger.info(f"  Computing disagreement for {len(queries_to_compute):,} new queries...")

    for qid in tqdm(queries_to_compute, desc="Disagreement"):
        query_text = qid_to_text.get(qid, "")
        if not query_text:
            scores[str(qid)] = 0.0
            continue

        method_sets = _search_query(query_text, components, top_k=top_k_for_disagreement)

        if len(method_sets) < 2:
            scores[str(qid)] = 0.0
            continue

        # Pairwise Jaccard distance
        methods = list(method_sets.values())
        distances = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                union = len(methods[i] | methods[j])
                intersection = len(methods[i] & methods[j])
                dist = 1.0 - (intersection / union) if union > 0 else 0.0
                distances.append(dist)

        scores[str(qid)] = float(np.mean(distances))

    # Save full cache
    cache_data = {
        "metadata": {
            "top_k": top_k_for_disagreement,
            "n_methods": n_methods,
            "n_queries": len(scores),
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "scores": scores,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    logger.info(f"  Disagreement scores cached to {cache_path} ({len(scores):,} queries)")

    return {str(qid): scores[str(qid)] for qid in all_query_ids}


def select_queries_by_disagreement(
    dataset: ESCIDataset,
    max_queries: int = 10000,
    split: str = "test",
    high_disagree_fraction: float = 0.6,
    seed: int = 42,
) -> Dict[int, Dict]:
    """Select queries using retrieval-diversity sampling (Option C).

    Strategy:
      - Compute method disagreement (Jaccard distance) for all available queries
      - Sort by disagreement score
      - Top 60% of budget from high-disagreement queries (where label expansion
        matters most — methods find different products)
      - Bottom 40% uniformly sampled (ensures coverage of "easy" queries too)

    Future work (Option D): additionally stratify by product category for
    more representative coverage across the catalog.

    Returns: query_data dict compatible with get_curated_pool_data() format
    """
    import pandas as pd

    logger.info(f"Selecting {max_queries:,} queries via retrieval-diversity sampling...")

    # Load all query IDs from the split
    examples_path = dataset.config.data_dir / dataset.config.examples_file
    all_examples = pd.read_parquet(examples_path)
    split_examples = all_examples[
        (all_examples["product_locale"] == dataset.config.locale) &
        (all_examples["split"] == split)
    ][["query_id", "query", "product_id", "esci_label"]].copy()

    all_query_ids = split_examples["query_id"].unique().tolist()
    logger.info(f"  Total queries in split='{split}': {len(all_query_ids):,}")

    if max_queries >= len(all_query_ids):
        logger.info(f"  Requested {max_queries:,} >= available {len(all_query_ids):,}, using all")
        max_queries = len(all_query_ids)

    # Compute disagreement scores
    disagreement = compute_query_disagreement(
        dataset, all_query_ids, top_k_for_disagreement=50
    )

    # Sort queries by disagreement (descending)
    sorted_qids = sorted(all_query_ids, key=lambda q: disagreement.get(str(q), 0.0), reverse=True)

    # Stratified selection
    n_high = int(max_queries * high_disagree_fraction)
    n_uniform = max_queries - n_high

    # High-disagreement: take top N from sorted list
    high_disagree_qids = sorted_qids[:n_high]

    # Uniform: sample from the remaining queries
    remaining_qids = sorted_qids[n_high:]
    rng = np.random.RandomState(seed)
    if n_uniform <= len(remaining_qids):
        uniform_qids = rng.choice(remaining_qids, size=n_uniform, replace=False).tolist()
    else:
        uniform_qids = remaining_qids

    selected_ids = set(high_disagree_qids + uniform_qids)

    # Log disagreement stats
    selected_scores = [disagreement.get(str(q), 0) for q in selected_ids]
    all_scores = [disagreement.get(str(q), 0) for q in all_query_ids]
    logger.info(f"  Selected {len(selected_ids):,} queries:")
    logger.info(f"    High-disagreement: {n_high:,} (top {high_disagree_fraction*100:.0f}%)")
    logger.info(f"    Uniform sample:    {n_uniform:,}")
    logger.info(f"    Disagreement — selected mean: {np.mean(selected_scores):.3f}, "
                f"all mean: {np.mean(all_scores):.3f}")
    logger.info(f"    Disagreement — selected p50: {np.median(selected_scores):.3f}, "
                f"p90: {np.percentile(selected_scores, 90):.3f}")

    # Build query_data in same format as get_curated_pool_data
    query_groups = split_examples.groupby("query_id")
    query_data = {}
    for qid in selected_ids:
        if qid not in query_groups.groups:
            continue
        group = query_groups.get_group(qid)
        query_text = group["query"].iloc[0]
        labeled_products = dict(zip(group["product_id"], group["esci_label"]))
        query_data[qid] = {
            "query_text": query_text,
            "labeled_products": labeled_products,
        }

    logger.info(f"  Final query_data: {len(query_data):,} queries")
    return query_data


# =============================================================================
# Retrieval: get top-K product IDs per method per query
# =============================================================================

def get_retrieval_candidates(
    dataset: ESCIDataset,
    query_data: Dict,
    top_k: int = 100,
) -> Dict[str, Dict[str, List[str]]]:
    """Retrieve top-K products per query from each method.

    Returns: {query_id: {method_name: [product_ids]}}

    Loads indexes from cache (must be pre-built).
    """
    components = _load_search_components()
    results = {}

    logger.info(f"Retrieving top-{top_k} per method for {len(query_data)} queries...")

    for qid, qinfo in tqdm(query_data.items(), desc="Retrieving"):
        query_text = qinfo["query_text"]
        results[str(qid)] = {}

        if components["bm25"] is not None:
            try:
                ids, _ = components["bm25"].search(query_text, top_k=top_k)
                results[str(qid)]["BM25"] = ids
            except Exception as e:
                logger.warning(f"  BM25 error for qid={qid}: {e}")

        if components["dense_encoder"] is not None and components["dense_index"] is not None:
            try:
                emb = components["dense_encoder"].encode_query(query_text)
                ids, _ = components["dense_index"].search(emb, top_k=top_k)
                results[str(qid)]["Dense"] = ids
            except Exception as e:
                logger.warning(f"  Dense error for qid={qid}: {e}")

        if components["splade_encoder"] is not None and components["splade_index"] is not None:
            try:
                vec = components["splade_encoder"].encode_single(query_text)
                ids, _ = components["splade_index"].search(vec, top_k=top_k)
                results[str(qid)]["SPLADE"] = ids
            except Exception as e:
                logger.warning(f"  SPLADE error for qid={qid}: {e}")

    return results


def compute_labeling_pairs(
    retrieval_results: Dict[str, Dict[str, List[str]]],
    label_store: LabelStore,
) -> List[Tuple[str, str]]:
    """Compute (query_id, product_id) pairs that need labeling.

    Takes the union of all methods' top-K per query, then removes
    pairs that are already in the label store.
    """
    all_pairs = set()
    for qid, methods in retrieval_results.items():
        for method_name, product_ids in methods.items():
            for pid in product_ids:
                all_pairs.add((qid, pid))

    # Remove already-labeled pairs
    new_pairs = [
        (qid, pid) for qid, pid in all_pairs
        if not label_store.has(qid, pid)
    ]

    logger.info(f"  Total unique pairs across all methods: {len(all_pairs):,}")
    logger.info(f"  Already labeled: {len(all_pairs) - len(new_pairs):,}")
    logger.info(f"  New pairs to label: {len(new_pairs):,}")

    return new_pairs


# =============================================================================
# Main labeling loop (async)
# =============================================================================

async def run_labeling(
    pairs: List[Tuple[str, str]],
    product_texts: Dict[str, str],
    query_texts: Dict[str, str],
    label_store: LabelStore,
    labeler: GeminiLabeler,
    checkpoint_every: int = 500,
) -> None:
    """Run async labeling for all pairs."""
    labeler.semaphore = asyncio.Semaphore(labeler.concurrency)

    total = len(pairs)
    completed = 0
    skipped = 0
    t0 = time.time()

    # Process in batches for progress tracking
    batch_size = labeler.concurrency * 2  # keep pipeline full

    pbar = tqdm(total=total, desc="Labeling")

    for batch_start in range(0, total, batch_size):
        batch = pairs[batch_start:batch_start + batch_size]

        # Create async tasks for this batch
        tasks = []
        for qid, pid in batch:
            query = query_texts.get(qid, "")
            product = product_texts.get(pid, "")
            if not query or not product:
                skipped += 1
                pbar.update(1)
                continue
            tasks.append((qid, pid, labeler.label_one(query, product)))

        # Await all tasks in batch
        for qid, pid, coro in tasks:
            label, latency = await coro
            if label != "?":
                label_store.add(qid, pid, label)
            completed += 1
            pbar.update(1)

        # Checkpoint
        if completed % checkpoint_every < batch_size:
            label_store.save()
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            cost = labeler.get_cost_estimate()
            pbar.set_postfix({
                "rate": f"{rate:.1f}/s",
                "cost": f"${cost:.3f}",
                "errors": labeler.total_errors,
            })

    pbar.close()

    # Final save
    label_store.save(force=True)

    elapsed = time.time() - t0
    logger.info(f"\n  Labeling complete!")
    logger.info(f"  Labeled: {completed:,} pairs in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"  Skipped (missing text): {skipped:,}")
    logger.info(f"  API errors: {labeler.total_errors:,}")
    logger.info(f"  Throughput: {completed/elapsed:.1f} pairs/sec")
    logger.info(f"  Est. cost: ${labeler.get_cost_estimate():.4f}")
    logger.info(f"  Total labels in store: {label_store.total_labels():,}")
    logger.info(f"  Label distribution: {label_store.get_stats()}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scaled LLM Labeling Pipeline (Gemini 2.5 Flash Lite)"
    )
    parser.add_argument("--top_k", type=int, default=100,
                        help="Retrieve top-K per method (100, 500, 1000)")
    parser.add_argument("--max_queries", type=int, default=2000,
                        help="Number of queries to label")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Save checkpoint every N labels")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just count new pairs, don't label")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use for query selection")
    parser.add_argument("--sampling", type=str, default="diversity",
                        choices=["diversity", "random"],
                        help="Query sampling strategy: 'diversity' (Option C, "
                             "oversample high-disagreement queries) or 'random'")
    parser.add_argument("--label_store", type=str, default=None,
                        help="Path to label store JSON (default: outputs/metrics/llm_labels.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # --- Load label store ---
    store_path = Path(args.label_store) if args.label_store else LABEL_STORE_PATH
    label_store = LabelStore(path=store_path)

    # --- Load dataset ---
    logger.info("Loading dataset...")
    dataset = ESCIDataset()
    dataset.load_examples()
    dataset.load_products()
    product_texts = dataset.get_product_texts()

    # --- Select queries ---
    if args.sampling == "diversity":
        logger.info("Using retrieval-diversity sampling (Option C)...")
        query_data = select_queries_by_disagreement(
            dataset, max_queries=args.max_queries,
            split=args.split, seed=args.seed,
        )
    else:
        logger.info("Using random sampling...")
        query_data, _ = dataset.get_curated_pool_data(
            max_queries=args.max_queries, split=args.split
        )
    query_texts = {str(qid): qinfo["query_text"] for qid, qinfo in query_data.items()}

    logger.info(f"  Selected {len(query_data):,} queries (sampling={args.sampling})")

    if args.dry_run:
        # For dry run, try loading indexes. If unavailable, estimate.
        try:
            retrieval_results = get_retrieval_candidates(
                dataset, query_data, top_k=args.top_k
            )
            new_pairs = compute_labeling_pairs(retrieval_results, label_store)
            actual_count = len(new_pairs)
        except (ImportError, Exception) as e:
            logger.warning(f"  Could not load indexes for exact count: {e}")
            logger.info("  Using estimated counts instead.")
            # Estimate: 3 methods x top_k, with ~60% overlap = ~1.2x top_k unique per query
            est_unique_per_query = int(args.top_k * 3 * 0.4)  # ~40% unique across methods
            actual_count = est_unique_per_query * len(query_data) - label_store.total_labels()
            actual_count = max(0, actual_count)

        avg_tokens = 400  # conservative estimate
        cost_per_pair = (avg_tokens * 0.10 + 2 * 0.40) / 1_000_000
        est_cost = actual_count * cost_per_pair
        est_time_seq = actual_count / 1.8  # sequential rate from benchmark
        est_time_concurrent = est_time_seq / args.concurrency

        print(f"\n{'='*60}")
        print(f"  DRY RUN SUMMARY")
        print(f"{'='*60}")
        print(f"  Queries:           {len(query_data):,}")
        print(f"  Top-K per method:  {args.top_k}")
        print(f"  New pairs to label: {actual_count:,}")
        print(f"  Already labeled:   {label_store.total_labels():,}")
        print(f"  Est. cost:         ${est_cost:.2f}")
        print(f"  Est. time:         {est_time_concurrent/3600:.1f} hours "
              f"(concurrency={args.concurrency})")
        print(f"{'='*60}")
        return

    # --- Retrieve candidates (only for actual labeling) ---
    retrieval_results = get_retrieval_candidates(
        dataset, query_data, top_k=args.top_k
    )
    new_pairs = compute_labeling_pairs(retrieval_results, label_store)

    if len(new_pairs) == 0:
        logger.info("  No new pairs to label! All pairs already in store.")
        logger.info(f"  Total labels: {label_store.total_labels():,}")
        logger.info(f"  Distribution: {label_store.get_stats()}")
        return

    # --- Initialize Gemini ---
    labeler = GeminiLabeler(concurrency=args.concurrency)
    labeler.init_client()

    # --- Run labeling ---
    logger.info(f"\n{'='*60}")
    logger.info(f"  Starting labeling: {len(new_pairs):,} pairs")
    logger.info(f"  Concurrency: {args.concurrency}")
    logger.info(f"  Checkpoint every: {args.checkpoint_every}")
    logger.info(f"{'='*60}")

    asyncio.run(run_labeling(
        pairs=new_pairs,
        product_texts=product_texts,
        query_texts=query_texts,
        label_store=label_store,
        labeler=labeler,
        checkpoint_every=args.checkpoint_every,
    ))

    # --- Final summary ---
    stats = label_store.get_stats()
    print(f"\n{'='*60}")
    print(f"  LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total labels in store: {label_store.total_labels():,}")
    print(f"  Distribution: {stats}")
    print(f"  API calls: {labeler.total_calls:,}")
    print(f"  Errors: {labeler.total_errors:,}")
    print(f"  Cost: ${labeler.get_cost_estimate():.4f}")
    print(f"  Store: {label_store.path}")
    print(f"\n  To extend coverage, re-run with higher --top_k:")
    print(f"  python llm_labeler.py --top_k 500 --max_queries {args.max_queries}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
