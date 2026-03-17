"""
Dense semantic search using Sentence-BERT + FAISS.

Encodes all products into dense 384-dimensional embeddings using
sentence-transformers/all-MiniLM-L6-v2, then uses FAISS IndexFlatIP
for exact inner-product search on L2-normalized vectors.

Follows the encoding pattern from Chapter 10 (item embeddings).

Usage:
  python dense_search.py --mode encode --device cuda
  python dense_search.py --mode evaluate --max_queries 5000
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse
import time

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    DenseConfig, DEFAULT_DENSE_CONFIG, DEFAULT_DATA_CONFIG,
    DEFAULT_EVAL_CONFIG, METRICS_DIR,
)
from data_loader import ESCIDataset
from evaluate import (
    evaluate_candidate_scoring, evaluate_retrieval,
    print_evaluation_results, save_results_json,
    plot_score_boxplots,
)

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Dense encoder using Sentence-BERT.

    Produces L2-normalized embeddings so that inner product = cosine similarity.
    """

    def __init__(self, config: DenseConfig = DEFAULT_DENSE_CONFIG):
        self.config = config
        self.model = None

    def load_model(self, device: str = "cpu") -> None:
        """Load SentenceTransformer model."""
        logger.info(f"Loading dense model: {self.config.model_name}")
        logger.info(f"  Device: {device}")

        self.model = SentenceTransformer(
            self.config.model_name,
            device=device,
        )
        self.model.max_seq_length = self.config.max_seq_length

        logger.info(f"  Model loaded. Embedding dim: {self.config.embedding_dim}")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode texts to dense embeddings, L2-normalized.

        Returns np.ndarray of shape (N, embedding_dim).
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        logger.info(f"  Dense encoding complete: {embeddings.shape}")
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Returns (embedding_dim,) vector."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding[0]


class DenseIndex:
    """FAISS-based dense retrieval index.

    Uses IndexFlatIP (inner product) on L2-normalized embeddings,
    which is equivalent to cosine similarity search.
    """

    def __init__(
        self,
        product_ids: List[str],
        embeddings: np.ndarray,
    ):
        self.product_ids = product_ids
        self.embeddings = embeddings
        self.id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
        self.index = None

    def build_index(self) -> None:
        """Build FAISS inner-product index."""
        dim = self.embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP (dim={dim}, n={len(self.product_ids):,})")
        t0 = time.time()

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        elapsed = time.time() - t0
        logger.info(f"  FAISS index built in {elapsed:.1f}s ({self.index.ntotal:,} vectors)")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[List[str], np.ndarray]:
        """Search full catalog. Returns (product_ids, scores)."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        result_ids = [self.product_ids[idx] for idx in indices[0] if idx >= 0]
        result_scores = scores[0][:len(result_ids)]
        return result_ids, result_scores

    def score_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_ids: List[str],
    ) -> np.ndarray:
        """Score specific candidates via direct dot product."""
        candidate_indices = []
        valid_positions = []

        for i, pid in enumerate(candidate_ids):
            idx = self.id_to_idx.get(pid)
            if idx is not None:
                candidate_indices.append(idx)
                valid_positions.append(i)

        scores = np.zeros(len(candidate_ids), dtype=np.float64)

        if candidate_indices:
            candidate_embeddings = self.embeddings[candidate_indices]
            dot_scores = candidate_embeddings @ query_embedding.astype(np.float32)
            for pos, score in zip(valid_positions, dot_scores):
                scores[pos] = float(score)

        return scores

    def save(self, path: Optional[Path] = None) -> None:
        """Save embeddings and product IDs to .npz file."""
        save_dir = path or DEFAULT_DENSE_CONFIG.cache_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        save_file = save_dir / DEFAULT_DENSE_CONFIG.cache_file
        np.savez_compressed(
            save_file,
            embeddings=self.embeddings,
            product_ids=np.array(self.product_ids),
        )
        logger.info(f"  Dense index saved to {save_file}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["DenseIndex"]:
        """Load embeddings from .npz file and build FAISS index."""
        load_dir = path or DEFAULT_DENSE_CONFIG.cache_dir
        load_file = load_dir / DEFAULT_DENSE_CONFIG.cache_file

        if not load_file.exists():
            return None

        try:
            data = np.load(load_file, allow_pickle=True)
            embeddings = data["embeddings"]
            product_ids = data["product_ids"].tolist()

            index_obj = cls(product_ids, embeddings)
            index_obj.build_index()
            logger.info(f"  Dense index loaded: {embeddings.shape}")
            return index_obj
        except Exception as e:
            logger.warning(f"  Failed to load dense index: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Dense Semantic Search (SBERT + FAISS)")
    parser.add_argument("--mode", type=str,
                        choices=["encode", "evaluate"],
                        default="evaluate", help="Mode to run")
    parser.add_argument("--max_queries", type=int, default=5000,
                        help="Max queries for evaluation (0 = all)")
    parser.add_argument("--max_products", type=int, default=0,
                        help="Max products to encode (0 = all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load encoder
    encoder = DenseEncoder()
    device = args.device
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
    except ImportError:
        device = "cpu"
    encoder.load_model(device=device)

    # Load dataset
    data_config = DEFAULT_DATA_CONFIG
    data_config.max_products = args.max_products
    dataset = ESCIDataset(data_config)
    dataset.load_examples()
    dataset.load_products()

    # Encode or load embeddings
    index = DenseIndex.load()

    if args.mode == "encode" or index is None:
        product_ids, product_texts = dataset.get_product_texts_list()
        logger.info(f"Encoding {len(product_ids):,} products with SBERT...")
        t0 = time.time()
        embeddings = encoder.encode_texts(
            product_texts, batch_size=args.batch_size
        )
        elapsed = time.time() - t0
        logger.info(f"  Encoding completed in {elapsed:.1f}s")

        index = DenseIndex(product_ids, embeddings)
        index.build_index()
        index.save()

    if args.mode == "evaluate":
        logger.info("Running Dense evaluation...")

        # Score-based evaluation
        query_candidates = dataset.get_query_candidates(max_queries=args.max_queries)

        def dense_scoring_fn(query_text, candidate_ids):
            query_embedding = encoder.encode_query(query_text)
            return index.score_candidates(query_embedding, candidate_ids)

        scoring_results = evaluate_candidate_scoring(
            query_candidates=query_candidates,
            scoring_fn=dense_scoring_fn,
            method_name="Dense (SBERT)",
            eval_config=DEFAULT_EVAL_CONFIG,
        )
        print_evaluation_results(scoring_results)

        # Full-catalog retrieval evaluation
        eval_queries = dataset.get_evaluation_queries(max_queries=args.max_queries)
        retrieved = {}
        for qid, qdata in tqdm(eval_queries.items(), desc="Dense Retrieval"):
            query_embedding = encoder.encode_query(qdata["query_text"])
            result_ids, _ = index.search(query_embedding, top_k=100)
            retrieved[qid] = result_ids

        retrieval_results = evaluate_retrieval(
            retrieved, eval_queries, DEFAULT_EVAL_CONFIG.k_values
        )
        scoring_results["retrieval_metrics"] = retrieval_results
        save_results_json([scoring_results], METRICS_DIR / "dense_results.json")

        # Plot
        plot_score_boxplots([scoring_results])


if __name__ == "__main__":
    main()
