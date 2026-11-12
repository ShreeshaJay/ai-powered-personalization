"""Dense retrieval with normalized vectors and an exact flat inner-product index.

The index type is fixed to IndexFlatIP. A persisted index is loaded only
when its vector count matches the product-id list. Query encoding is batched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import DenseSettings
from fusion import HitList, host_array, ranking_from_scores


class DenseEncoder:
    def __init__(self, settings: DenseSettings | None = None) -> None:
        self.settings = settings or DenseSettings()
        self.model = None

    def load(self, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.settings.repo,
            device=device,
            revision=self.settings.revision,
            trust_remote_code=self.settings.trust_remote_code,
        )
        self.model.max_seq_length = self.settings.max_length

    def encode(self, texts: list[str], batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Dense encoder is not loaded.")
        if not texts:
            return np.zeros((0, self.settings.embedding_dim), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size or self.settings.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.settings.normalize,
            convert_to_numpy=True,
        )
        return host_array(embeddings, dtype=np.float32)

    def encode_queries(self, queries: list[str], show_progress: bool = False) -> np.ndarray:
        return self.encode(queries, batch_size=self.settings.query_batch_size, show_progress=show_progress)


class DenseIndex:
    def __init__(
        self,
        product_ids: list[str],
        embeddings: np.ndarray,
        settings: DenseSettings | None = None,
    ) -> None:
        self.settings = settings or DenseSettings()
        if self.settings.index_type != "IndexFlatIP":
            raise ValueError(
                f"Only exact IndexFlatIP is supported, received {self.settings.index_type!r}."
            )
        self.product_ids = list(product_ids)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.product_ids):
            raise ValueError("Embeddings do not match the product-id list.")
        self.id_to_row = {product_id: row for row, product_id in enumerate(self.product_ids)}
        self.index = None

    def build(self) -> None:
        import faiss

        dim = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
        if self.index.ntotal != len(self.product_ids):
            raise RuntimeError("FAISS index count does not match the product-id list.")

    def search_batch(self, query_embeddings: np.ndarray, top_k: int) -> list[HitList]:
        if self.index is None:
            raise RuntimeError("FAISS index is not built.")
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("query embeddings must have shape (n_queries, dim).")
        depth = min(top_k, len(self.product_ids))
        scores, indices = self.index.search(np.ascontiguousarray(queries), depth)
        hits: list[HitList] = []
        for row_scores, row_indices in zip(scores, indices):
            ids: list[str] = []
            values: list[float] = []
            for index, score in zip(row_indices, row_scores):
                if int(index) < 0:
                    continue
                ids.append(self.product_ids[int(index)])
                values.append(float(score))
            # IndexFlatIP is exact. Re-sort so equal scores break by product id.
            hits.append(ranking_from_scores(ids, np.asarray(values, dtype=np.float64), depth))
        return hits

    def search(self, query_embedding: np.ndarray, top_k: int) -> HitList:
        return self.search_batch(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), top_k)[0]

    def score_products(self, query_embedding: np.ndarray, product_ids: list[str]) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        scores = np.zeros(len(product_ids), dtype=np.float64)
        rows = []
        positions = []
        for position, product_id in enumerate(product_ids):
            row = self.id_to_row.get(product_id)
            if row is not None:
                rows.append(row)
                positions.append(position)
        if rows:
            dots = self.embeddings[rows] @ query
            for position, score in zip(positions, dots):
                scores[position] = float(score)
        return scores

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            self.build()
        np.savez_compressed(
            directory / self.settings.embeddings_file,
            embeddings=self.embeddings,
            product_ids=np.array(self.product_ids),
        )
        faiss.write_index(self.index, str(directory / self.settings.faiss_index_file))

    @classmethod
    def load(cls, directory: Path, settings: DenseSettings | None = None) -> "DenseIndex":
        import faiss

        settings = settings or DenseSettings()
        payload = np.load(directory / settings.embeddings_file, allow_pickle=True)
        index = cls(payload["product_ids"].tolist(), payload["embeddings"], settings)
        faiss_path = directory / settings.faiss_index_file
        if not faiss_path.exists():
            index.build()
            return index
        index.index = faiss.read_index(str(faiss_path))
        if index.index.ntotal != len(index.product_ids):
            raise ValueError("Persisted FAISS index vector count does not match product ids.")
        return index


def encode_catalog(
    encoder: DenseEncoder,
    product_ids: list[str],
    texts: list[str],
    directory: Path,
    batch_size: int,
    checkpoint_every: int = 50_000,
) -> np.ndarray:
    """Resume from a memmap checkpoint written every ``checkpoint_every`` rows."""
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / "dense_memmap.npy"
    progress_path = directory / "dense_next.txt"
    ids_path = directory / "dense_ids.json"
    n_rows = len(texts)
    dim = encoder.settings.embedding_dim
    if matrix_path.exists() and progress_path.exists() and ids_path.exists():
        saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if saved_ids != product_ids:
            raise ValueError("Dense checkpoint product ids do not match this catalog.")
        matrix = np.load(matrix_path, mmap_mode="r+")
        start = int(progress_path.read_text(encoding="utf-8").strip())
    else:
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n_rows, dim))
        start = 0
    last_checkpoint = start
    for offset in range(start, n_rows, batch_size):
        end = min(offset + batch_size, n_rows)
        matrix[offset:end] = encoder.encode(texts[offset:end], batch_size=batch_size, show_progress=False)
        crossed = (end // checkpoint_every) > (offset // checkpoint_every)
        if end == n_rows or crossed or end - last_checkpoint >= checkpoint_every:
            matrix.flush()
            progress_path.write_text(str(end), encoding="utf-8")
            ids_path.write_text(json.dumps(product_ids), encoding="utf-8")
            last_checkpoint = end
    return np.asarray(matrix, dtype=np.float32)
