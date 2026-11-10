"""
Exact inner-product nearest-neighbour search.

Uses FAISS (``IndexFlatIP``) when it is installed and falls back to blocked NumPy matrix
products otherwise, so the CPU reader path works without FAISS. Inputs are L2-normalised
here, so inner product equals cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when faiss is absent
    FAISS_AVAILABLE = False


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


class InnerProductIndex:
    def __init__(self, vectors: np.ndarray, use_faiss: Optional[bool] = None):
        self.vectors = l2_normalize(vectors)
        self.use_faiss = FAISS_AVAILABLE if use_faiss is None else (use_faiss and FAISS_AVAILABLE)
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    def search(self, queries: np.ndarray, k: int, block: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), both shaped (num_queries, k), best first."""
        queries = l2_normalize(np.atleast_2d(queries))
        k = min(k, self.size)
        if self.use_faiss:
            return self.index.search(queries, k)
        all_scores, all_idx = [], []
        for start in range(0, len(queries), block):
            sims = queries[start:start + block] @ self.vectors.T
            idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            part = np.take_along_axis(sims, idx, axis=1)
            order = np.argsort(-part, axis=1, kind="stable")
            all_idx.append(np.take_along_axis(idx, order, axis=1))
            all_scores.append(np.take_along_axis(part, order, axis=1))
        return np.vstack(all_scores), np.vstack(all_idx)


def mean_history_queries(item_vectors: np.ndarray, histories) -> np.ndarray:
    """Session query = normalised mean of the (normalised) history item vectors."""
    normed = l2_normalize(item_vectors)
    return l2_normalize(np.stack([normed[h].mean(axis=0) for h in histories]))
