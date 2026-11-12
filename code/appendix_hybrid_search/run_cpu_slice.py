"""Teaching slice that runs every module on a tiny schema fixture.

The timings this script writes are measurements of that fixture on the
machine where it is run. They are not catalog timings and they are not
comparable to a full ESCI run. Model downloads happen on the first run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp

from analyze import evaluate_split, slice_membership
from config import resolve_device
from data_loader import load_esci
from finetune_splade import train_one_seed
from fusion import read_run
from llm_labels_convert import convert, select_query_ids
from qdrant_demo import run_parity
from rerank_cross_encoder import rerank_run
from run_retrieval import query_map, retrieve_bm25, retrieve_dense, retrieve_splade
from splits import build_splits, write_bundle
from tests.support import write_tiny_esci
from tune_fusion import apply_frozen, tune


def _record(rows: list[dict], name: str, started: float, status: str, detail: str, process: psutil.Process) -> None:
    info = process.memory_info()
    rows.append(
        {
            "step": name,
            "seconds": round(time.perf_counter() - started, 3),
            "status": status,
            "detail": detail,
            "rss_bytes": int(info.rss),
            "peak_wset_bytes": int(getattr(info, "peak_wset", info.rss)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every module on a tiny teaching fixture.")
    parser.add_argument("--output-dir", default="outputs/cpu_slice")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = resolve_device(args.device)
    output = Path(args.output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    process = psutil.Process()
    rows: list[dict] = []
    holder: dict = {}

    def step(name: str, function) -> None:
        started = time.perf_counter()
        try:
            detail = function() or ""
            _record(rows, name, started, "ok", str(detail), process)
        except Exception as exc:
            _record(rows, name, started, "error", f"{type(exc).__name__}: {exc}", process)
            (output / f"{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    def prepare() -> str:
        fixture = Path(tempfile.mkdtemp(prefix="appendixb_fixture_"))
        holder["fixture"] = fixture
        data = load_esci(write_tiny_esci(fixture))
        bundle = build_splits(
            data,
            prefix_texts=["red shoes"],
            prefix_source="teaching-fixture",
            development_size=1,
            development_seed=20261110,
            exercise3_pairs=1,
        )
        write_bundle(output / "splits" / "splits.json", bundle)
        holder["data"] = data
        holder["bundle"] = bundle
        return f"products={len(data.catalog.product_ids)} test={len(bundle.test_ids)} development={len(bundle.development_ids)}"

    def bm25() -> str:
        data = holder["data"]
        bundle = holder["bundle"]
        ids = list(bundle.development_ids) + list(bundle.test_ids)
        retrieve_bm25(data, query_map(data, ids), data.catalog.product_ids, output, "teaching", depth=8)
        return "bm25"

    def qdrant() -> str:
        rng = np.random.default_rng(0)
        product_ids = [f"d{i}" for i in range(12)]
        dense = rng.normal(size=(12, 8)).astype(np.float32)
        sparse = sp.random(12, 20, density=0.2, format="csr", random_state=0)
        query_dense = rng.normal(size=(4, 8)).astype(np.float32)
        query_sparse = sp.random(4, 20, density=0.2, format="csr", random_state=1)
        report = run_parity(
            product_ids,
            dense,
            sparse,
            query_dense,
            query_sparse,
            output / "qdrant",
            prefetch_limit=12,
            top_k=5,
        )
        (output / "qdrant_parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["pass"]:
            raise RuntimeError("Qdrant parity report did not pass.")
        return "pass"

    def dense() -> str:
        data = holder["data"]
        bundle = holder["bundle"]
        ids = list(bundle.development_ids) + list(bundle.test_ids)
        retrieve_dense(data, query_map(data, ids), data.catalog.product_ids, output, "teaching", 8, device, batch_size=4)
        return "dense"

    def splade() -> str:
        data = holder["data"]
        bundle = holder["bundle"]
        ids = list(bundle.development_ids) + list(bundle.test_ids)
        retrieve_splade(
            "splade_esci",
            data,
            query_map(data, ids),
            data.catalog.product_ids,
            output,
            "teaching",
            8,
            device,
            batch_size=2,
        )
        return "splade_esci"

    def fusion() -> str:
        data = holder["data"]
        bundle = holder["bundle"]
        from run_retrieval import load_runs

        dev = query_map(data, bundle.development_ids)
        runs = load_runs(output / "runs" / "teaching", ["bm25", "dense", "splade_esci"], "p2")
        # tune() reads the split name only for depth sensitivity and the file name.
        dev_runs = {name: {qid: run.get(qid, []) for qid in dev} for name, run in runs.items()}
        selection = tune(dev_runs, dev, output, split="teaching")
        all_ids = list(bundle.development_ids) + list(bundle.test_ids)
        for kind in ("p1", "p2"):
            kind_runs = load_runs(output / "runs" / "teaching", ["bm25", "dense", "splade_esci"], kind)
            fused = apply_frozen(kind_runs, all_ids, selection, depth=8)
            from fusion import write_run

            for name, run in fused.items():
                write_run(output / "runs" / "teaching" / f"{name}.{kind}.trec", run, name, depth=None if kind == "p1" else 8)
        holder["selection"] = selection
        return selection["best_hybrid"]

    def evaluate() -> str:
        data = holder["data"]
        bundle = holder["bundle"]
        queries = query_map(data, list(bundle.development_ids) + list(bundle.test_ids))
        names = ["bm25", "dense", "splade_esci", holder["selection"]["best_hybrid"], holder["selection"]["best_single"]]
        evaluate_split(output, "teaching", queries, names, holder["selection"])
        return "teaching"

    def slices() -> str:
        counts = {name: 0 for name in ("digits", "negation", "long", "short", "all")}
        for item in holder["data"].queries.values():
            for name, flag in slice_membership(item.query_text).items():
                counts[name] += int(flag)
        return json.dumps(counts)

    def rerank() -> str:
        data = holder["data"]
        name = holder["selection"]["best_hybrid"]
        queries = query_map(data, list(holder["bundle"].development_ids) + list(holder["bundle"].test_ids))
        reranked = rerank_run(
            read_run(output / "runs" / "teaching" / f"{name}.p2.trec"),
            queries,
            data.catalog.texts,
            depth=4,
            device=device,
            batch_size=2,
        )
        from fusion import write_run

        write_run(output / "runs" / "teaching" / f"{name}_cross_encoder.p2.trec", reranked, f"{name}_cross_encoder", depth=4)
        return "rerank"

    def finetune() -> str:
        data = holder["data"]
        pairs = []
        for item in data.queries.values():
            if item.split != "train":
                continue
            for product_id, label in item.labels.items():
                if label in {"E", "S"}:
                    pairs.append((item.query_text, data.catalog.texts[product_id]))
        record = train_one_seed(pairs[:4], output / "finetune", seed=20261111, device=device, max_steps=1)
        return f"router={record['router_mapping_used']}"

    def labels() -> str:
        payload = {
            "metadata": {"sampling": {"max_queries": 2, "seed": 42, "high_disagree_fraction": 0.6, "top_k": 2}},
            "labels": {"q5": {"p1": {"label": "E", "timestamp": "2026-01-01"}}},
        }
        label_path = output / "sample_labels.json"
        label_path.write_text(json.dumps(payload), encoding="utf-8")
        scores = {"q1": 0.1, "q2": 0.9, "q3": 0.4, "q4": 0.2}
        (output / "sample_disagreement.json").write_text(json.dumps({"scores": scores}), encoding="utf-8")
        chosen = select_query_ids(list(scores), scores, 2, 0.6, 42)
        result = convert(label_path, output, output / "sample_disagreement.json", "teaching")
        return f"selected={chosen} {result['reason']}"

    step("fixture_and_splits", prepare)
    step("bm25", bm25)
    step("qdrant_parity", qdrant)
    step("dense", dense)
    step("splade_esci", splade)
    step("fusion", fusion)
    step("evaluate", evaluate)
    step("slices", slices)
    step("cross_encoder", rerank)
    step("finetune_one_step", finetune)
    step("label_convert", labels)
    total = round(sum(row["seconds"] for row in rows), 3)
    peak = max(row["peak_wset_bytes"] for row in rows) if rows else 0
    report = {
        "label": "teaching measurement on a schema fixture",
        "device": device,
        "steps": rows,
        "total_seconds": total,
        "peak_wset_bytes": peak,
    }
    (output / "cpu_slice_timings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if holder.get("fixture"):
        shutil.rmtree(holder["fixture"], ignore_errors=True)


if __name__ == "__main__":
    main()
