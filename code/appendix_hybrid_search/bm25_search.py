"""
BM25 search using bm25s library.

bm25s pre-computes BM25 scores into scipy sparse matrices at index time,
enabling extremely fast query-time retrieval. Uses Lucene-compatible
BM25 scoring by default.

Usage:
  python bm25_search.py --mode index
  python bm25_search.py --mode evaluate --max_queries 5000
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse
import time

import bm25s

from config import (
    BM25Config, DEFAULT_BM25_CONFIG, DEFAULT_DATA_CONFIG,
    DEFAULT_EVAL_CONFIG, METRICS_DIR,
)
from data_loader import ESCIDataset
from evaluate import (
    evaluate_candidate_scoring, evaluate_retrieval,
    print_evaluation_results, save_results_json,
    plot_score_boxplots, plot_label_distribution_bars,
)

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    """BM25 search engine using bm25s library.

    bm25s stores pre-computed BM25 scores in sparse matrices, enabling
    fast top-k retrieval without iterating over all documents.
    """

    def __init__(self, config: BM25Config = DEFAULT_BM25_CONFIG):
        self.config = config
        self.bm25: Optional[bm25s.BM25] = None
        self.product_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}

    def build_index(
        self,
        product_ids: List[str],
        product_texts: List[str],
    ) -> None:
        """Build BM25 index from product texts.

        Steps:
          1. Tokenize texts using bm25s built-in tokenizer
          2. Create BM25 model with Lucene-compatible scoring
          3. Index the tokenized corpus (pre-computes sparse score matrix)
        """
        logger.info(f"Building BM25 index over {len(product_ids):,} products...")
        self.product_ids = product_ids
        self.id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}

        t0 = time.time()

        # Tokenize corpus
        logger.info("  Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(
            product_texts,
            stopwords="en",
            show_progress=True,
        )

        # Create and index
        logger.info("  Building BM25 sparse index...")
        self.bm25 = bm25s.BM25(
            method=self.config.method,
            k1=self.config.k1,
            b=self.config.b,
        )
        self.bm25.index(corpus_tokens, show_progress=True)

        elapsed = time.time() - t0
        logger.info(f"  BM25 index built in {elapsed:.1f}s")
        logger.info(f"  Vocabulary size: {self.bm25.vocab_dict.__len__():,}")

    def save(self, path: Optional[Path] = None) -> None:
        """Save BM25 index to disk using native bm25s format."""
        save_dir = path or self.config.cache_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        self.bm25.save(str(save_dir), corpus=None)

        # Save product IDs separately
        np.save(save_dir / "product_ids.npy", np.array(self.product_ids))
        logger.info(f"  BM25 index saved to {save_dir}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        load_dir = path or self.config.cache_dir
        ids_file = load_dir / "product_ids.npy"

        if not ids_file.exists():
            return False

        try:
            self.bm25 = bm25s.BM25.load(str(load_dir), load_corpus=False, mmap=True)
            self.product_ids = np.load(ids_file, allow_pickle=True).tolist()
            self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
            logger.info(f"  BM25 index loaded from {load_dir} ({len(self.product_ids):,} products)")
            return True
        except Exception as e:
            logger.warning(f"  Failed to load BM25 index: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> Tuple[List[str], np.ndarray]:
        """Search full catalog for a query. Returns (product_ids, scores)."""
        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self.bm25.retrieve(query_tokens, k=top_k)

        result_ids = [self.product_ids[idx] for idx in results[0]]
        return result_ids, scores[0]

    def score_candidates(
        self,
        query: str,
        candidate_ids: List[str],
    ) -> np.ndarray:
        """Score specific candidates for a query.

        Retrieves scores for all documents, then extracts scores
        for the specified candidates.
        """
        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)

        # Retrieve all scores (k = corpus size)
        n_docs = len(self.product_ids)
        results, scores = self.bm25.retrieve(query_tokens, k=n_docs)

        # Build index-to-score mapping
        idx_to_score = {}
        for idx, score in zip(results[0], scores[0]):
            idx_to_score[idx] = score

        # Extract candidate scores
        candidate_scores = np.zeros(len(candidate_ids), dtype=np.float64)
        for i, pid in enumerate(candidate_ids):
            idx = self.id_to_idx.get(pid)
            if idx is not None:
                candidate_scores[i] = idx_to_score.get(idx, 0.0)

        return candidate_scores


def main():
    parser = argparse.ArgumentParser(description="BM25 Search Engine")
    parser.add_argument("--mode", type=str, choices=["index", "evaluate"],
                        default="evaluate", help="Mode: index or evaluate")
    parser.add_argument("--max_queries", type=int, default=5000,
                        help="Max queries for evaluation (0 = all)")
    parser.add_argument("--max_products", type=int, default=0,
                        help="Max products to index (0 = all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load dataset
    data_config = DEFAULT_DATA_CONFIG
    data_config.max_products = args.max_products
    dataset = ESCIDataset(data_config)
    dataset.load_examples()
    dataset.load_products()

    # Build or load BM25 index
    engine = BM25SearchEngine()

    if args.mode == "index" or not engine.load():
        product_ids, product_texts = dataset.get_product_texts_list()
        engine.build_index(product_ids, product_texts)
        engine.save()

    if args.mode == "evaluate":
        logger.info("Running BM25 evaluation...")

        # Score-based evaluation (boxplots, NDCG, separability)
        query_candidates = dataset.get_query_candidates(max_queries=args.max_queries)
        scoring_results = evaluate_candidate_scoring(
            query_candidates=query_candidates,
            scoring_fn=engine.score_candidates,
            method_name="BM25",
            eval_config=DEFAULT_EVAL_CONFIG,
        )
        print_evaluation_results(scoring_results)

        # Full-catalog retrieval evaluation
        eval_queries = dataset.get_evaluation_queries(max_queries=args.max_queries)
        retrieved = {}
        from tqdm import tqdm
        for qid, qdata in tqdm(eval_queries.items(), desc="BM25 Retrieval"):
            result_ids, _ = engine.search(qdata["query_text"], top_k=100)
            retrieved[qid] = result_ids

        retrieval_results = evaluate_retrieval(
            retrieved, eval_queries, DEFAULT_EVAL_CONFIG.k_values
        )

        # Merge retrieval results into scoring results
        scoring_results["retrieval_metrics"] = retrieval_results
        save_results_json([scoring_results], METRICS_DIR / "bm25_results.json")

        # Plot
        plot_score_boxplots([scoring_results])


if __name__ == "__main__":
    main()
