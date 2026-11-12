"""Build indexes, write TREC runs, and record a manifest for each retriever.

A step is skipped when both of its run files already exist and are non-empty.
Encoding checkpoints are kept beside the runs, so an interrupted encode resumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from config import PLAN_PATH, DenseSettings, load_plan, package_versions, sha256_file, sha256_text
from data_loader import ESCIData, QueryJudgments
from dense_search import DenseEncoder, DenseIndex, encode_catalog
from evaluate import write_json
from fusion import Run, ranking_from_scores, write_run
from bm25_search import BM25Index
from splade_search import encode_documents, encode_queries, load_sparse_encoder, score_rows, search_csc


def non_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def runs_ready(run_dir: Path, name: str) -> bool:
    return non_empty(run_dir / f"{name}.p1.trec") and non_empty(run_dir / f"{name}.p2.trec")


def query_map(data: ESCIData, query_ids: list[str]) -> dict[str, QueryJudgments]:
    missing = [query_id for query_id in query_ids if query_id not in data.queries]
    if missing:
        raise KeyError(f"{len(missing)} query ids are not in the loaded catalog, for example {missing[:3]}.")
    return {query_id: data.queries[query_id] for query_id in query_ids}


def write_manifest(path: Path, payload: dict) -> None:
    body = dict(payload)
    body["package_versions"] = package_versions()
    body["analysis_plan_sha256"] = sha256_file(PLAN_PATH)
    write_json(path, body)


def _write_pair(run_dir: Path, name: str, p1: Run, p2: Run, depth: int) -> None:
    write_run(run_dir / f"{name}.p1.trec", p1, name, depth=None)
    write_run(run_dir / f"{name}.p2.trec", p2, name, depth=depth)


def retrieve_bm25(
    data: ESCIData,
    queries: dict[str, QueryJudgments],
    product_ids: list[str],
    output_dir: Path,
    split: str,
    depth: int,
) -> Path:
    run_dir = output_dir / "runs" / split
    if runs_ready(run_dir, "bm25"):
        return run_dir / "bm25.p2.trec"
    started = time.perf_counter()
    ids, texts = data.catalog.texts_in_order(product_ids)
    index = BM25Index()
    index.build(ids, texts)
    index.save(output_dir / "indexes" / "bm25")
    p1: Run = {}
    p2: Run = {}
    for query_id, item in queries.items():
        p1[query_id] = ranking_from_scores(item.product_order, index.score_products(item.query_text, item.product_order))
        p2[query_id] = index.search(item.query_text, depth)
    _write_pair(run_dir, "bm25", p1, p2, depth)
    write_manifest(
        output_dir / "manifests" / split / "bm25.json",
        {
            "system": "bm25",
            "split": split,
            "n_products": len(ids),
            "n_queries": len(queries),
            "catalog_sha256": data.catalog.catalog_sha256,
            "template_sha256": data.catalog.template_sha256,
            "text_sha256": data.catalog.text_sha256,
            "text_role": "standard",
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return run_dir / "bm25.p2.trec"


def _p1_from_rows(queries: dict[str, QueryJudgments], score_one) -> Run:
    run: Run = {}
    for query_id, item in queries.items():
        run[query_id] = ranking_from_scores(item.product_order, score_one(item))
    return run


def retrieve_dense(
    data: ESCIData,
    queries: dict[str, QueryJudgments],
    product_ids: list[str],
    output_dir: Path,
    split: str,
    depth: int,
    device: str,
    batch_size: int | None = None,
) -> Path:
    run_dir = output_dir / "runs" / split
    if runs_ready(run_dir, "dense"):
        return run_dir / "dense.p2.trec"
    started = time.perf_counter()
    plan = load_plan()["models"]["dense"]
    settings = DenseSettings(
        repo=plan["repo"],
        revision=plan["revision"],
        trust_remote_code=bool(plan["trust_remote_code"]),
        embedding_dim=int(plan["embedding_dim"]),
        max_length=int(plan["max_length"]),
        normalize=bool(plan["normalize"]),
        index_type=plan["index"],
        batch_size=int(batch_size or plan["l4_batch_size"]),
        query_batch_size=int(plan["query_batch_size"]),
        faiss_index_file=plan["faiss_index_file"],
    )
    if device == "cpu":
        settings.batch_size = min(settings.batch_size, 16)
    encoder = DenseEncoder(settings)
    encoder.load(device)
    ids, texts = data.catalog.texts_in_order(product_ids)
    cache = output_dir / "indexes" / "dense"
    embeddings = encode_catalog(
        encoder,
        ids,
        texts,
        cache,
        batch_size=settings.batch_size,
        checkpoint_every=int(load_plan()["retrieval"]["checkpoint_every_products"]),
    )
    index = DenseIndex(ids, embeddings, settings)
    index.build()
    index.save(cache)
    ordered = list(queries.items())
    query_vectors = encoder.encode_queries([item.query_text for _, item in ordered])
    p2_hits = index.search_batch(query_vectors, depth)
    p2 = {query_id: hits for (query_id, _item), hits in zip(ordered, p2_hits)}
    p1 = {
        query_id: ranking_from_scores(item.product_order, index.score_products(vector, item.product_order))
        for (query_id, item), vector in zip(ordered, query_vectors)
    }
    _write_pair(run_dir, "dense", p1, p2, depth)
    write_manifest(
        output_dir / "manifests" / split / "dense.json",
        {
            "system": "dense",
            "split": split,
            "repo": settings.repo,
            "revision": settings.revision,
            "trust_remote_code": False,
            "device": device,
            "n_products": len(ids),
            "n_queries": len(queries),
            "catalog_sha256": data.catalog.catalog_sha256,
            "template_sha256": data.catalog.template_sha256,
            "text_sha256": data.catalog.text_sha256,
            "text_role": "standard",
            "index_type": "IndexFlatIP",
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return run_dir / "dense.p2.trec"


def retrieve_splade(
    model_key: str,
    data: ESCIData,
    queries: dict[str, QueryJudgments],
    product_ids: list[str],
    output_dir: Path,
    split: str,
    depth: int,
    device: str,
    batch_size: int | None = None,
) -> Path:
    run_dir = output_dir / "runs" / split
    if runs_ready(run_dir, model_key):
        return run_dir / f"{model_key}.p2.trec"
    started = time.perf_counter()
    spec = load_plan()["models"][model_key]
    model = load_sparse_encoder(spec["repo"], spec["revision"], device)
    batch = int(batch_size or spec["l4_batch_size"])
    if device == "cpu":
        batch = min(batch, 8)
    ids, texts = data.catalog.texts_in_order(product_ids)
    cache = output_dir / "indexes" / model_key
    docs = encode_documents(
        model,
        ids,
        texts,
        cache,
        batch_size=batch,
        max_length=int(spec["document_max_length"]),
        checkpoint_every=int(load_plan()["retrieval"]["checkpoint_every_products"]),
    )
    ordered = list(queries.items())
    query_matrix = encode_queries(
        model,
        [item.query_text for _, item in ordered],
        batch,
        int(spec["query_max_length"]),
    )
    p2_hits = search_csc(docs, query_matrix, ids, depth, device=device, query_batch=batch)
    p2 = {query_id: hits for (query_id, _item), hits in zip(ordered, p2_hits)}
    id_to_row = {product_id: row for row, product_id in enumerate(ids)}
    p1_docs = docs
    p1_rows = id_to_row
    text_role = "standard"
    text_hash = data.catalog.text_sha256
    if model_key == "splade_esci":
        judged = sorted({product_id for item in queries.values() for product_id in item.labels})
        sensitivity = [data.catalog.qdrant_texts[product_id] for product_id in judged]
        p1_docs = encode_documents(
            model,
            judged,
            sensitivity,
            cache / "p1_sensitivity",
            batch_size=batch,
            max_length=int(spec["document_max_length"]),
            checkpoint_every=int(load_plan()["retrieval"]["checkpoint_every_products"]),
        )
        p1_rows = {product_id: row for row, product_id in enumerate(judged)}
        text_role = "qdrant_sensitivity"
        text_hash = sha256_text("\n".join(sensitivity) + ("\n" if sensitivity else ""))
    p1: Run = {}
    for row, (query_id, item) in enumerate(ordered):
        positions = []
        matrix_rows = []
        for position, product_id in enumerate(item.product_order):
            mapped = p1_rows.get(product_id)
            if mapped is not None:
                positions.append(position)
                matrix_rows.append(mapped)
        scores = np.zeros(len(item.product_order), dtype=np.float64)
        if matrix_rows:
            found = score_rows(p1_docs, query_matrix[row], matrix_rows)
            scores[positions] = found
        p1[query_id] = ranking_from_scores(item.product_order, scores)
    _write_pair(run_dir, model_key, p1, p2, depth)
    write_manifest(
        output_dir / "manifests" / split / f"{model_key}.json",
        {
            "system": model_key,
            "split": split,
            "repo": spec["repo"],
            "revision": spec["revision"],
            "trust_remote_code": False,
            "licence": spec["licence"],
            "device": device,
            "query_max_length": int(spec["query_max_length"]),
            "document_max_length": int(spec["document_max_length"]),
            "n_products": len(ids),
            "n_queries": len(queries),
            "catalog_sha256": data.catalog.catalog_sha256,
            "template_sha256": data.catalog.template_sha256,
            "text_sha256": text_hash,
            "text_role": text_role,
            "p1_text_role": text_role,
            "redistribute_vectors": bool(spec.get("redistribute_vectors", True)),
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return run_dir / f"{model_key}.p2.trec"


def load_runs(run_dir: Path, names: list[str], kind: str) -> dict[str, Run]:
    from fusion import read_run

    loaded = {}
    for name in names:
        path = run_dir / f"{name}.{kind}.trec"
        if path.exists():
            loaded[name] = read_run(path)
    return loaded


def read_selection(output_dir: Path, split: str = "development") -> dict:
    path = output_dir / "tuning" / f"selection_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"No frozen selection at {path.name}. Tune on development first.")
    return json.loads(path.read_text(encoding="utf-8"))
