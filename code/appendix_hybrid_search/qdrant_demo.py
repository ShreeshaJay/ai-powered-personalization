"""Qdrant local-mode hybrid search and the parity check.

qdrant-client 1.20.0 local mode implements two details that the script does
not copy into the main fusion:

- Fusion.RRF uses constant k = 2 unless an RrfQuery sets another constant.
- Its RRF term is 1 / (k + position) with position starting at 0. The script
  uses 1 / (k + rank) with rank starting at 1. Those agree when the client
  constant is the script constant plus one. Client k = 61 matches script k = 60.
- DBSF uses the sample standard deviation and maps a zero-variance leg to 0.5.
  fusion.dbsf uses that same rule.

The default-constant query is expected to disagree with script k = 60.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from fusion import dbsf, rrf


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype(np.float32)


def _sparse_query(indices: list[int], values: list[float]):
    from qdrant_client import models

    if not indices:
        return models.SparseVector(indices=[0], values=[0.0])
    return models.SparseVector(indices=[int(i) for i in indices], values=[float(v) for v in values])


def _script_legs(dense_docs, dense_query, sparse_docs, sparse_query, product_ids, depth):
    dense_scores = dense_docs @ dense_query.astype(np.float32)
    from fusion import ranking_from_scores

    dense_hits = ranking_from_scores(product_ids, dense_scores, depth)
    if sparse_query.nnz == 0:
        sparse_hits = []
    else:
        sparse_scores = np.asarray(sparse_query.dot(sparse_docs.T).todense()).ravel()
        # Local-mode sparse search does not return a dot product of zero.
        # Those products are missing from the leg, so RRF adds nothing for them.
        kept = [index for index, score in enumerate(sparse_scores) if float(score) != 0.0]
        sparse_hits = ranking_from_scores(
            [product_ids[index] for index in kept],
            sparse_scores[kept],
            depth,
        )
    return dense_hits, sparse_hits


def run_parity(
    product_ids: list[str],
    dense: np.ndarray,
    sparse,
    query_dense: np.ndarray,
    query_sparse,
    path: Path,
    prefetch_limit: int = 1000,
    top_k: int = 10,
) -> dict:
    """Compare three client fusions with the script. ``sparse`` is a CSR or CSC matrix."""
    from qdrant_client import QdrantClient, models

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    dense = _normalize(np.asarray(dense, dtype=np.float32))
    query_dense = _normalize(np.asarray(query_dense, dtype=np.float32))
    sparse = sparse.tocsr()
    query_sparse = query_sparse.tocsr()
    dim = int(dense.shape[1])
    client = QdrantClient(path=str(path))
    client.create_collection(
        collection_name="hybrid",
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    points = []
    for row, product_id in enumerate(product_ids):
        column = sparse.getrow(row)
        points.append(
            models.PointStruct(
                id=row,
                vector={
                    "dense": dense[row].tolist(),
                    "sparse": (
                        models.SparseVector(indices=[0], values=[0.0])
                        if column.nnz == 0
                        else models.SparseVector(
                            indices=column.indices.astype(int).tolist(),
                            values=column.data.astype(float).tolist(),
                        )
                    ),
                },
                payload={"product_id": product_id},
            )
        )
    client.upsert(collection_name="hybrid", points=points)
    depth = min(prefetch_limit, len(product_ids))
    id_to_product = {row: product_id for row, product_id in enumerate(product_ids)}

    def client_hits(query, q_dense, q_sparse_row) -> list[tuple[str, float]]:
        _, cols = q_sparse_row.nonzero()
        values = q_sparse_row.data.astype(float).tolist()
        response = client.query_points(
            collection_name="hybrid",
            prefetch=[
                models.Prefetch(query=q_dense.tolist(), using="dense", limit=depth),
                models.Prefetch(
                    query=_sparse_query(cols.astype(int).tolist(), values),
                    using="sparse",
                    limit=depth,
                ),
            ],
            query=query,
            limit=top_k,
            with_payload=True,
        )
        found = []
        for point in response.points:
            payload = point.payload or {}
            found.append((payload.get("product_id", id_to_product.get(point.id)), float(point.score)))
        return found

    variants = {
        "server_default_rrf": models.FusionQuery(fusion=models.Fusion.RRF),
        "explicit_k_60": models.RrfQuery(rrf=models.Rrf(k=60)),
        "aligned_k_61": models.RrfQuery(rrf=models.Rrf(k=61)),
        "dbsf": models.FusionQuery(fusion=models.Fusion.DBSF),
    }
    script_for = {
        "server_default_rrf": lambda dense_hits, sparse_hits: rrf([dense_hits, sparse_hits], k=60),
        "explicit_k_60": lambda dense_hits, sparse_hits: rrf([dense_hits, sparse_hits], k=60),
        "aligned_k_61": lambda dense_hits, sparse_hits: rrf([dense_hits, sparse_hits], k=60),
        "dbsf": lambda dense_hits, sparse_hits: dbsf([dense_hits, sparse_hits]),
    }
    expected_match = {
        "server_default_rrf": False,
        "explicit_k_60": False,
        "aligned_k_61": True,
        "dbsf": True,
    }
    counts = {
        name: {"n": 0, "ordered_match": 0, "score_match": 0, "set_match": 0, "mismatches": []}
        for name in variants
    }
    for row in range(query_dense.shape[0]):
        dense_hits, sparse_hits = _script_legs(
            dense, query_dense[row], sparse, query_sparse[row], product_ids, depth
        )
        for name, query in variants.items():
            client_top = client_hits(query, query_dense[row], query_sparse[row])
            script_full = script_for[name](dense_hits, sparse_hits)
            script_top = script_full[:top_k]
            script_scores = dict(script_full)
            client_ids = [product_id for product_id, _score in client_top]
            script_ids = [product_id for product_id, _score in script_top]
            counts[name]["n"] += 1
            if client_ids == script_ids:
                counts[name]["ordered_match"] += 1
            if set(client_ids) == set(script_ids):
                counts[name]["set_match"] += 1
            elif len(counts[name]["mismatches"]) < 20:
                counts[name]["mismatches"].append({"client": client_ids, "script": script_ids})
            if client_top and all(
                abs(score - script_scores.get(product_id, score + 1.0)) <= 1e-5
                for product_id, score in client_top
            ):
                counts[name]["score_match"] += 1
    report = {"top_k": top_k, "prefetch_limit": depth, "n_queries": int(query_dense.shape[0]), "variants": {}}
    for name, stats in counts.items():
        ordered = stats["ordered_match"] / stats["n"] if stats["n"] else 0.0
        scored = stats["score_match"] / stats["n"] if stats["n"] else 0.0
        if name == "aligned_k_61":
            passed = ordered >= 0.99 and scored >= 0.99
        elif name == "dbsf":
            passed = ordered >= 0.99 and scored >= 0.99
        elif name == "explicit_k_60":
            # The same top-k order can occur. The scores still differ by the rank origin.
            passed = scored < 0.99
        else:
            passed = ordered < 0.99
        report["variants"][name] = {
            **stats,
            "ordered_fraction": ordered,
            "score_fraction": scored,
            "expected_match": expected_match[name],
            "pass": passed,
        }
    report["pass"] = all(item["pass"] for item in report["variants"].values())
    report["note"] = (
        "Client RRF uses 1/(k + position) with position starting at 0 and a default "
        "constant of 2. Script RRF uses 1/(k + rank) with rank starting at 1. "
        "Client k = 61 matches script k = 60 in the fused scores. "
        "Client k = 60 uses a different rank origin, so the scores differ even when the top-k order agrees. "
        "A sparse dot product of zero is omitted, matching local-mode sparse search."
    )
    close = getattr(client, "close", None)
    if callable(close):
        close()
    return report
