"""
FAISS Index Wrapper for Fast Similarity Search (Chapter 11)

Reused from Chapter 10 with no changes. Provides utilities for building and
querying FAISS indices for efficient nearest neighbor search over embeddings.
"""

import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "FAISS not available. Install with: pip install faiss-cpu or faiss-gpu"
    )


class FaissIndexWrapper:
    """Wrapper for FAISS index with convenient methods."""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexFlatIP",
        use_gpu: bool = False,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu

        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = 100
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        if use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                logger.warning("GPU resources not available, using CPU")
                self.use_gpu = False
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU acceleration for FAISS")

        self.is_trained = False
        self.num_vectors = 0
        logger.info(f"Initialized FAISS index: {index_type}, dim={embedding_dim}")

    def add(self, embeddings: np.ndarray):
        """Add embeddings to the index."""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if self.index_type == "IndexIVFFlat" and not self.is_trained:
            logger.info(f"Training index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self.is_trained = True

        self.index.add(embeddings)
        self.num_vectors += len(embeddings)
        logger.info(f"Added {len(embeddings)} vectors (total: {self.num_vectors})")

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors. Returns (distances, indices)."""
        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dim mismatch: expected {self.embedding_dim}, "
                f"got {query_embeddings.shape[1]}"
            )
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        return self.index.search(query_embeddings, k)

    def save(self, filepath: str):
        """Save index to disk."""
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, filepath)
        else:
            faiss.write_index(self.index, filepath)
        logger.info(f"Saved index to {filepath}")

    def load(self, filepath: str):
        """Load index from disk."""
        self.index = faiss.read_index(filepath)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.num_vectors = self.index.ntotal
        logger.info(f"Loaded index from {filepath} ({self.num_vectors} vectors)")


# ============================================================================
# Convenience Functions
# ============================================================================

def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "IndexFlatIP",
    use_gpu: bool = False,
) -> FaissIndexWrapper:
    """Build FAISS index from embeddings."""
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")
    index = FaissIndexWrapper(
        embedding_dim=embeddings.shape[1],
        index_type=index_type,
        use_gpu=use_gpu,
    )
    index.add(embeddings)
    return index


def search_faiss_index(
    index: FaissIndexWrapper,
    query_embeddings: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index."""
    return index.search(query_embeddings, k)


def brute_force_search(
    item_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force nearest neighbor search (fallback if FAISS not available)."""
    if metric == "cosine":
        similarities = query_embeddings @ item_embeddings.T
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_distances = np.take_along_axis(similarities, top_k_indices, axis=1)
    elif metric == "l2":
        num_queries = query_embeddings.shape[0]
        distances = np.zeros((num_queries, item_embeddings.shape[0]))
        for i in range(num_queries):
            diff = item_embeddings - query_embeddings[i]
            distances[i] = np.linalg.norm(diff, axis=1)
        top_k_indices = np.argsort(distances, axis=1)[:, :k]
        top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return top_k_distances, top_k_indices
