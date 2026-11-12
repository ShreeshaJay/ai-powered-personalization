"""SPLADE encoding and exact sparse scoring.

Documents are stored term-major (CSC) so CPU scoring walks posting lists.
GPU scoring multiplies a torch sparse document matrix by a dense query batch.
Both paths are exact. Naver vectors are an evaluation cache only and must
not be redistributed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fusion import HitList, host_array, ranking_from_scores


def set_max_length(model, length: int) -> None:
    model.max_seq_length = int(length)
    try:
        model[0].max_seq_length = int(length)
    except (TypeError, KeyError, IndexError, AttributeError):
        return


def _cpu_float_tensor(tensor):
    """Move values to CPU float32 before any NumPy conversion."""
    import torch

    tensor = tensor.detach()
    if tensor.dtype in (torch.bfloat16, torch.float16) or tensor.device.type != "cpu":
        return tensor.to(device="cpu", dtype=torch.float32)
    return tensor


def to_csr(embedding) -> sp.csr_matrix:
    import torch

    if isinstance(embedding, sp.spmatrix):
        return embedding.tocsr()
    if isinstance(embedding, list):
        rows = [to_csr(item) for item in embedding]
        return sp.vstack(rows, format="csr") if rows else sp.csr_matrix((0, 0))
    if isinstance(embedding, torch.Tensor):
        if embedding.layout == torch.sparse_csr:
            values = _cpu_float_tensor(embedding.values()).numpy()
            columns = embedding.col_indices().detach().to(device="cpu").numpy()
            crow = embedding.crow_indices().detach().to(device="cpu").numpy()
            return sp.csr_matrix((values, columns, crow), shape=tuple(int(dim) for dim in embedding.shape))
        if embedding.is_sparse:
            coalesced = embedding.detach().coalesce()
            indices = coalesced.indices().detach().to(device="cpu").numpy()
            values = _cpu_float_tensor(coalesced.values()).numpy()
            shape = tuple(int(dim) for dim in coalesced.shape)
            if coalesced.ndim == 1:
                return sp.csr_matrix((values, indices.reshape(1, -1)), shape=(1, shape[0]))
            return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
        array = _cpu_float_tensor(embedding).numpy()
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return sp.csr_matrix(array)
    array = host_array(embedding, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return sp.csr_matrix(array)


def load_sparse_encoder(repo: str, revision: str, device: str):
    from sentence_transformers import SparseEncoder

    return SparseEncoder(
        repo,
        revision=revision,
        trust_remote_code=False,
        device=device,
    )


def encode_texts(model, texts: list[str], kind: str, batch_size: int, max_length: int) -> sp.csr_matrix:
    if not texts:
        return sp.csr_matrix((0, 0))
    set_max_length(model, max_length)
    function = model.encode_query if kind == "query" else model.encode_document
    embedding = function(
        list(texts),
        batch_size=batch_size,
        convert_to_sparse_tensor=True,
        show_progress_bar=len(texts) >= 32,
    )
    matrix = to_csr(embedding)
    if matrix.shape[0] != len(texts):
        raise RuntimeError(
            f"Sparse encoder returned {matrix.shape[0]} rows for {len(texts)} texts."
        )
    return matrix


def score_csc(docs_csc: sp.csc_matrix, term_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Walk posting lists. ``docs_csc`` is shaped (n_documents, vocabulary)."""
    scores = np.zeros(docs_csc.shape[0], dtype=np.float64)
    indptr = docs_csc.indptr
    indices = docs_csc.indices
    data = docs_csc.data
    for term, weight in zip(term_ids, weights):
        term = int(term)
        if term < 0 or term >= docs_csc.shape[1]:
            continue
        start = int(indptr[term])
        end = int(indptr[term + 1])
        scores[indices[start:end]] += float(weight) * data[start:end]
    return scores


def score_torch(docs_csc: sp.csc_matrix, query_dense: np.ndarray, device: str) -> np.ndarray:
    """Exact sparse-by-dense product. ``query_dense`` is (n_queries, vocabulary)."""
    import torch

    query_dense = host_array(query_dense, dtype=np.float32)
    if query_dense.ndim == 1:
        query_dense = query_dense.reshape(1, -1)
    coo = docs_csc.tocoo()
    if coo.nnz == 0:
        return np.zeros((query_dense.shape[0], docs_csc.shape[0]), dtype=np.float64)
    index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    value = torch.tensor(coo.data.astype(np.float32), device=device)
    sparse = torch.sparse_coo_tensor(index, value, size=tuple(coo.shape), device=device).coalesce()
    query = torch.tensor(query_dense, device=device)
    product = torch.sparse.mm(sparse, query.transpose(0, 1).contiguous())
    return host_array(product.transpose(0, 1).contiguous())


def relative_score_gap(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    return float(np.max(np.abs(left - right) / scale))


def top_ids(product_ids: list[str], scores: np.ndarray, depth: int) -> list[str]:
    return [product_id for product_id, _score in ranking_from_scores(product_ids, scores, depth)]


def encode_documents(
    model,
    product_ids: list[str],
    texts: list[str],
    directory: Path,
    batch_size: int,
    max_length: int,
    checkpoint_every: int = 50_000,
) -> sp.csc_matrix:
    """Encode documents in resumable chunks and store one CSC matrix."""
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "documents.npz"
    ids_path = directory / "product_ids.json"
    if final_path.exists() and ids_path.exists():
        saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if saved_ids == product_ids:
            return sp.load_npz(final_path).tocsc()
    blocks = []
    for start in range(0, len(texts), checkpoint_every):
        chunk_path = directory / f"chunk_{start:08d}.npz"
        end = min(start + checkpoint_every, len(texts))
        if chunk_path.exists():
            blocks.append(sp.load_npz(chunk_path).tocsr())
            continue
        block = encode_texts(model, texts[start:end], "document", batch_size, max_length)
        sp.save_npz(chunk_path, block)
        blocks.append(block)
    matrix = sp.vstack(blocks, format="csc") if blocks else sp.csc_matrix((0, 0))
    sp.save_npz(final_path, matrix)
    ids_path.write_text(json.dumps(product_ids), encoding="utf-8")
    return matrix


def encode_queries(model, queries: list[str], batch_size: int, max_length: int) -> sp.csr_matrix:
    return encode_texts(model, queries, "query", batch_size, max_length)


def search_csc(
    docs_csc: sp.csc_matrix,
    query_csr: sp.csr_matrix,
    product_ids: list[str],
    depth: int,
    device: str = "cpu",
    query_batch: int = 32,
) -> list[HitList]:
    """Exact top-k for each query row."""
    hits: list[HitList] = []
    if device == "cpu":
        for row in range(query_csr.shape[0]):
            start = int(query_csr.indptr[row])
            end = int(query_csr.indptr[row + 1])
            scores = score_csc(docs_csc, query_csr.indices[start:end], query_csr.data[start:end])
            hits.append(ranking_from_scores(product_ids, scores, depth))
        return hits
    for start in range(0, query_csr.shape[0], query_batch):
        block = query_csr[start : start + query_batch].toarray()
        scores = score_torch(docs_csc, block, device)
        for row in range(scores.shape[0]):
            hits.append(ranking_from_scores(product_ids, scores[row], depth))
    return hits


def score_rows(
    docs_csc: sp.csc_matrix,
    query_csr_row: sp.csr_matrix,
    row_indices: list[int],
) -> np.ndarray:
    """Dot one query with a subset of document rows."""
    if not row_indices:
        return np.zeros(0, dtype=np.float64)
    subset = docs_csc[row_indices]
    return np.asarray(query_csr_row.dot(subset.T).todense()).ravel()
