"""
FAISS Index Wrapper for Fast Similarity Search

Provides utilities for building and querying FAISS indices for efficient
nearest neighbor search over item embeddings.

Used in Section 10.1 for bi-encoder evaluation.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")


class FaissIndexWrapper:
    """
    Wrapper for FAISS index with convenient methods.
    
    Supports:
    - Inner product (for normalized embeddings = cosine similarity)
    - L2 distance
    - GPU acceleration (if available)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IndexFlatIP",
        use_gpu: bool = False
    ):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type:
                - "IndexFlatIP": Exact inner product search (for normalized vectors)
                - "IndexFlatL2": Exact L2 distance search
                - "IndexIVFFlat": Approximate search with inverted file index
            use_gpu: Use GPU acceleration (requires faiss-gpu)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Create index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IndexIVFFlat":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Move to GPU if requested
        if use_gpu:
            if not hasattr(faiss, 'StandardGpuResources'):
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
        """
        Add embeddings to the index.
        
        Args:
            embeddings: (N, D) array of embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Train index if needed (for IVF indices)
        if self.index_type == "IndexIVFFlat" and not self.is_trained:
            logger.info(f"Training index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add to index
        self.index.add(embeddings)
        self.num_vectors += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} vectors to index (total: {self.num_vectors})")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embeddings: (N, D) array of query embeddings
            k: Number of nearest neighbors to retrieve
        
        Returns:
            Tuple of:
            - distances: (N, K) array of distances/similarities
            - indices: (N, K) array of neighbor indices
        """
        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embeddings.shape[1]}"
            )
        
        # Ensure float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        return distances, indices
    
    def save(self, filepath: str):
        """Save index to disk."""
        if self.use_gpu:
            # Move to CPU before saving
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
    use_gpu: bool = False
) -> FaissIndexWrapper:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: (N, D) array of embeddings
        index_type: FAISS index type
        use_gpu: Use GPU acceleration
    
    Returns:
        Initialized and populated FaissIndexWrapper
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")
    
    embedding_dim = embeddings.shape[1]
    
    index = FaissIndexWrapper(
        embedding_dim=embedding_dim,
        index_type=index_type,
        use_gpu=use_gpu
    )
    
    index.add(embeddings)
    
    return index


def search_faiss_index(
    index: FaissIndexWrapper,
    query_embeddings: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index.
    
    Args:
        index: FaissIndexWrapper instance
        query_embeddings: (N, D) query embeddings
        k: Number of neighbors
    
    Returns:
        Tuple of (distances, indices)
    """
    return index.search(query_embeddings, k)


# ============================================================================
# Fallback: Brute-force search if FAISS not available
# ============================================================================

def brute_force_search(
    item_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force nearest neighbor search (fallback if FAISS not available).
    
    Args:
        item_embeddings: (M, D) item embeddings
        query_embeddings: (N, D) query embeddings
        k: Number of neighbors
        metric: "cosine" or "l2"
    
    Returns:
        Tuple of (distances, indices)
    """
    if metric == "cosine":
        # Compute cosine similarity (assumes normalized embeddings)
        similarities = query_embeddings @ item_embeddings.T
        
        # Get top K
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_distances = np.take_along_axis(similarities, top_k_indices, axis=1)
        
    elif metric == "l2":
        # Compute L2 distances
        # distances[i, j] = ||query[i] - item[j]||^2
        num_queries = query_embeddings.shape[0]
        num_items = item_embeddings.shape[0]
        
        distances = np.zeros((num_queries, num_items))
        for i in range(num_queries):
            diff = item_embeddings - query_embeddings[i]
            distances[i] = np.linalg.norm(diff, axis=1)
        
        # Get top K (smallest distances)
        top_k_indices = np.argsort(distances, axis=1)[:, :k]
        top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return top_k_distances, top_k_indices


# ============================================================================
# Main: Test FAISS Index
# ============================================================================

if __name__ == "__main__":
    print("Testing FAISS Index")
    print("="*80)
    
    if not FAISS_AVAILABLE:
        print("\nFAISS not available. Testing brute-force search instead.")
        print("="*80)
        
        # Generate random embeddings
        np.random.seed(42)
        num_items = 1000
        num_queries = 10
        embedding_dim = 128
        
        item_embeddings = np.random.randn(num_items, embedding_dim).astype(np.float32)
        query_embeddings = np.random.randn(num_queries, embedding_dim).astype(np.float32)
        
        # Normalize
        item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        print(f"\nItem embeddings: {item_embeddings.shape}")
        print(f"Query embeddings: {query_embeddings.shape}")
        
        # Brute-force search
        print("\nPerforming brute-force search...")
        distances, indices = brute_force_search(
            item_embeddings,
            query_embeddings,
            k=5,
            metric="cosine"
        )
        
        print(f"\nTop 5 neighbors for first query:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"  {i+1}. Item {idx}: similarity = {dist:.4f}")
        
    else:
        # Generate random embeddings
        np.random.seed(42)
        num_items = 10000
        num_queries = 100
        embedding_dim = 128
        
        item_embeddings = np.random.randn(num_items, embedding_dim).astype(np.float32)
        query_embeddings = np.random.randn(num_queries, embedding_dim).astype(np.float32)
        
        # Normalize (for inner product = cosine similarity)
        item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        print(f"\nItem embeddings: {item_embeddings.shape}")
        print(f"Query embeddings: {query_embeddings.shape}")
        
        # Build index
        print("\nBuilding FAISS index...")
        index = build_faiss_index(
            item_embeddings,
            index_type="IndexFlatIP",
            use_gpu=False
        )
        
        # Search
        print("\nSearching for nearest neighbors...")
        distances, indices = index.search(query_embeddings, k=10)
        
        print(f"\nTop 10 neighbors for first query:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"  {i+1}. Item {idx}: similarity = {dist:.4f}")
        
        print(f"\nSearch complete. Retrieved {distances.shape[1]} neighbors for {distances.shape[0]} queries.")
