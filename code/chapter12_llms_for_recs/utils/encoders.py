"""
Text encoders shared by ``encode_items.py`` and ``generate_llm_features.py``.

Item text and LLM-generated text must be embedded by the same encoder, otherwise cosine
similarities between them compare two unrelated vector spaces. The encoder name is stored
in every embedding file and checked when files are combined.

    sentence-transformers model ID or local path   (default: all-MiniLM-L6-v2, 384 dims)
    "tfidf-svd"                                     TF-IDF + truncated SVD fitted on the
                                                    item texts; no download, for tests and
                                                    offline runs
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from utils.faiss_index import l2_normalize

logger = logging.getLogger(__name__)

TFIDF_SVD = "tfidf-svd"
TFIDF_STATE = "tfidf_svd_encoder.pkl"


class TextEncoder:
    def __init__(self, name: str, model=None, device: str = "cpu", batch_size: int = 256):
        self.name = name
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t else " " for t in texts]
        if self.name == TFIDF_SVD:
            vectorizer, svd = self.model
            return l2_normalize(svd.transform(vectorizer.transform(texts)))
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 5000)
        return vecs.astype(np.float32)


def fit_encoder(name: str, corpus: Sequence[str], state_dir: Path, device: str = "cpu",
                dim: int = 64, seed: int = 42) -> TextEncoder:
    """Create the encoder; for tfidf-svd, fit it on ``corpus`` and save its state."""
    if name == TFIDF_SVD:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(min_df=1, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform([t if t else " " for t in corpus])
        n_components = max(2, min(dim, matrix.shape[1] - 1, matrix.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=seed).fit(matrix)
        state_dir.mkdir(parents=True, exist_ok=True)
        with open(state_dir / TFIDF_STATE, "wb") as f:
            pickle.dump((vectorizer, svd), f)
        return TextEncoder(name, (vectorizer, svd))
    return load_encoder(name, state_dir, device)


def load_encoder(name: str, state_dir: Path, device: str = "cpu") -> TextEncoder:
    if name == TFIDF_SVD:
        state = Path(state_dir) / TFIDF_STATE
        if not state.exists():
            raise FileNotFoundError(f"{state} not found; run encode_items.py --encoder tfidf-svd first")
        with open(state, "rb") as f:
            return TextEncoder(name, pickle.load(f))
    from sentence_transformers import SentenceTransformer

    return TextEncoder(name, SentenceTransformer(name, device=device), device=device)


def save_embeddings(path: Path, keys: Sequence, embeddings: np.ndarray, encoder: str,
                    key_name: str = "item_ids", extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key_name: np.asarray(keys), "embeddings": embeddings.astype(np.float32),
               "encoder": np.asarray(encoder)}
    if extra:
        payload.update(extra)
    np.savez(path, **payload)


def stored_encoder(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return str(data["encoder"])
