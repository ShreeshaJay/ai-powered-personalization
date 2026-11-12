"""Headless runner for the full-catalog path.

Stages are s0, s1, s2, s3, s4 and s5. A step whose output files already exist
is skipped. After every step the runner may run a command supplied with
--sync-cmd. The command may contain the placeholders {stage}, {step} and
{output}. No destination is stored in this file.

s0 runs the unit tests and, when a data directory is given, a smoke limited
by --s0-products and --s0-queries. s1 encodes and retrieves. s2 builds BM25,
tunes fusion on development, and scores. s3 and s4 are optional. s5 scores
the saved run files again in a separate directory and compares summary hashes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from analyze import evaluate_split
from config import EXAMPLES_FILE, PRODUCTS_FILE, load_plan, resolve_data_dir, resolve_device, sha256_file
from data_loader import load_esci
from evaluate import write_json
from fusion import write_run
from run_retrieval import (
    load_runs,
    query_map,
    read_selection,
    retrieve_bm25,
    retrieve_dense,
    retrieve_splade,
    runs_ready,
    write_manifest,
)
from splits import build_splits, load_prefix_texts, write_bundle
from tune_fusion import apply_frozen, tune


def sync(template: str | None, stage: str, step: str, output: Path) -> None:
    if not template:
        return
    command = (
        template.replace("{stage}", stage).replace("{step}", step).replace("{output}", str(output))
    )
    completed = subprocess.run(command, shell=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Sync command for {stage}/{step} exited {completed.returncode}.")


def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def ensure_splits(data, output: Path, prefix_source: str) -> dict:
    path = output / "splits" / "splits.json"
    if _done(path):
        return json.loads(path.read_text(encoding="utf-8"))
    plan = load_plan()
    prefix, source = load_prefix_texts(prefix_source, revision=plan["splits"]["qdrant_prefix"]["revision"])
    bundle = build_splits(
        data,
        prefix,
        source,
        development_size=int(plan["splits"]["development"]["size"]),
        development_seed=int(plan["splits"]["development"]["seed"]),
        exercise3_pairs=int(plan["splits"]["exercise3_reservation"]["pairs"]),
    )
    write_bundle(path, bundle)
    return json.loads(path.read_text(encoding="utf-8"))


def _systems() -> list[str]:
    return ["bm25", "dense", "splade_naver", "splade_prithivi", "splade_esci"]


def _timed_step(rows: list[dict], name: str, function) -> None:
    import psutil

    process = psutil.Process()
    started = time.perf_counter()
    function()
    info = process.memory_info()
    rows.append(
        {
            "step": name,
            "seconds": round(time.perf_counter() - started, 3),
            "rss_bytes": int(info.rss),
            "peak_wset_bytes": int(getattr(info, "peak_wset", info.rss)),
        }
    )


def run_s0(args, output: Path) -> None:
    rows: list[dict] = []

    def tests() -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("Unit tests failed.")

    _timed_step(rows, "unit_tests", tests)
    if not args.data_dir and not args.run_s0_smoke:
        write_manifest(output / "manifests" / "s0.json", {"steps": rows, "reported": False})
        return
    data_dir = resolve_data_dir(args.data_dir)
    data = load_esci(data_dir)
    product_ids = data.catalog.product_ids[: args.s0_products]
    test_ids = sorted(qid for qid, item in data.queries.items() if item.split == "test")[: args.s0_queries]
    queries = query_map(data, test_ids)
    smoke = output / "s0_smoke"
    device = resolve_device(args.device)
    depth = min(100, len(product_ids))
    _timed_step(rows, "bm25", lambda: retrieve_bm25(data, queries, product_ids, smoke, "smoke", depth))
    _timed_step(
        rows,
        "dense",
        lambda: retrieve_dense(data, queries, product_ids, smoke, "smoke", depth, device, batch_size=8),
    )
    for model_key in ("splade_naver", "splade_prithivi", "splade_esci"):
        _timed_step(
            rows,
            model_key,
            lambda model_key=model_key: retrieve_splade(
                model_key, data, queries, product_ids, smoke, "smoke", depth, device, batch_size=4
            ),
        )
    write_manifest(
        smoke / "manifests" / "s0.json",
        {
            "n_products": len(product_ids),
            "n_queries": len(queries),
            "device": device,
            "reported": False,
            "catalog_sha256": data.catalog.catalog_sha256,
            "template_sha256": data.catalog.template_sha256,
            "text_sha256": data.catalog.text_sha256,
            "examples_sha256": sha256_file(data_dir / EXAMPLES_FILE),
            "products_sha256": sha256_file(data_dir / PRODUCTS_FILE),
            "steps": rows,
            "wall_seconds": round(sum(row["seconds"] for row in rows), 3),
        },
    )


def run_s1(args, output: Path, data, splits: dict) -> None:
    device = resolve_device(args.device)
    depth = int(load_plan()["retrieval"]["run_depth"])
    product_ids = data.catalog.product_ids
    if args.max_products:
        product_ids = product_ids[: args.max_products]
    for split in ("development", "test"):
        ids = splits["development_ids"] if split == "development" else splits["test_ids"]
        if args.max_queries:
            ids = ids[: args.max_queries]
        queries = query_map(data, ids)
        retrieve_dense(data, queries, product_ids, output, split, depth, device)
        for model_key in ("splade_naver", "splade_prithivi", "splade_esci"):
            retrieve_splade(model_key, data, queries, product_ids, output, split, depth, device)


def _write_fused(output: Path, split: str, fused: dict, depth: int, kind: str) -> None:
    run_dir = output / "runs" / split
    for name, run in fused.items():
        path = run_dir / f"{name}.{kind}.trec"
        if _done(path):
            continue
        write_run(path, run, name, depth=None if kind == "p1" else depth)


def run_s2(args, output: Path, data, splits: dict) -> None:
    depth = int(load_plan()["retrieval"]["run_depth"])
    product_ids = data.catalog.product_ids
    if args.max_products:
        product_ids = product_ids[: args.max_products]
    for split, ids in (("development", splits["development_ids"]), ("test", splits["test_ids"])):
        chosen = ids[: args.max_queries] if args.max_queries else ids
        queries = query_map(data, chosen)
        if not runs_ready(output / "runs" / split, "bm25"):
            retrieve_bm25(data, queries, product_ids, output, split, depth)
    dev_queries = query_map(data, splits["development_ids"][: args.max_queries] if args.max_queries else splits["development_ids"])
    selection_path = output / "tuning" / "selection_development.json"
    if not _done(selection_path):
        runs = load_runs(output / "runs" / "development", _systems(), "p2")
        tune(runs, dev_queries, output, split="development")
    selection = read_selection(output, "development")
    for split, ids in (("development", splits["development_ids"]), ("test", splits["test_ids"])):
        chosen = ids[: args.max_queries] if args.max_queries else ids
        queries = query_map(data, chosen)
        for kind in ("p1", "p2"):
            runs = load_runs(output / "runs" / split, _systems(), kind)
            fused = apply_frozen(runs, list(queries), selection, depth=depth)
            _write_fused(output, split, fused, depth, kind)
        names = _systems() + [
            "f1_rrf_bm25_dense",
            "f2_rrf_bm25_splade_naver_dense",
            "f2_rrf_bm25_splade_prithivi_dense",
            "f2_rrf_bm25_splade_esci_dense",
            "f3_rrf_splade_naver_dense",
            "f3_rrf_splade_prithivi_dense",
            "f3_rrf_splade_esci_dense",
            "f4_convex_bm25_dense",
            "f5_convex_three",
            "f6_dbsf_three",
            "f7_rrf_tuned",
            selection["best_hybrid"],
            selection["best_single"],
        ]
        summary = output / "summaries" / f"{split}.json"
        if not _done(summary):
            evaluate_split(output, split, queries, names, selection)
    if args.labels:
        from llm_labels_convert import convert

        disagreement = Path(args.disagreement) if args.disagreement else None
        convert(Path(args.labels), output, disagreement, "test")


def run_s3(args, output: Path, data, splits: dict) -> None:
    from rerank_cross_encoder import rerank_run

    selection = read_selection(output, "development")
    name = selection["best_hybrid"]
    device = resolve_device(args.device)
    depth = int(load_plan()["models"]["cross_encoder"]["rerank_depth"])
    for split, ids in (("development", splits["development_ids"]), ("test", splits["test_ids"])):
        destination = output / "runs" / split / f"{name}_cross_encoder.p2.trec"
        if _done(destination):
            continue
        source = output / "runs" / split / f"{name}.p2.trec"
        from fusion import read_run

        queries = query_map(data, ids)
        reranked = rerank_run(read_run(source), queries, data.catalog.texts, depth, device, 32)
        write_run(destination, reranked, f"{name}_cross_encoder", depth=depth)


def run_s4(args, output: Path, data, splits: dict) -> None:
    from finetune_splade import train_one_seed

    pairs = []
    for row in splits.get("exercise3_pairs", []):
        query = data.queries[row["query_id"]].query_text
        positive = data.catalog.texts[row["product_id"]]
        pairs.append((query, positive))
    if not pairs:
        raise RuntimeError("The split file has no Exercise 3 pairs.")
    device = resolve_device(args.device)
    seeds = list(load_plan()["exercise3"]["seeds"])
    for seed in seeds:
        manifest = output / "finetune" / f"seed_{seed}" / "train_manifest.json"
        if _done(manifest):
            continue
        train_one_seed(pairs, output / "finetune", int(seed), device, max_steps=args.max_steps)


def run_s5(args, output: Path, data, splits: dict) -> None:
    source_summaries = output / "summaries"
    if not any(source_summaries.glob("*.json")):
        raise RuntimeError("s5 needs the summaries written by s2.")
    repro = output / "reproduction"
    if repro.exists() and not args.force_s5:
        report_path = repro / "reproduction_report.json"
        if _done(report_path):
            return
    if repro.exists():
        shutil.rmtree(repro)
    for split in ("development", "test"):
        source = output / "runs" / split
        destination = repro / "runs" / split
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.glob("*.trec"):
            shutil.copy2(path, destination / path.name)
    selection = read_selection(output, "development")
    (repro / "tuning").mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "tuning" / "selection_development.json", repro / "tuning" / "selection_development.json")
    for split, ids in (("development", splits["development_ids"]), ("test", splits["test_ids"])):
        queries = query_map(data, ids[: args.max_queries] if args.max_queries else ids)
        names = [path.name.split(".")[0] for path in (repro / "runs" / split).glob("*.p2.trec")]
        evaluate_split(repro, split, queries, names, selection)
    comparisons = []
    match = True
    for path in sorted(source_summaries.glob("*.json")):
        other = repro / "summaries" / path.name
        same = other.exists() and sha256_file(path) == sha256_file(other)
        comparisons.append({"file": path.name, "match": same})
        match = match and same
    write_json(repro / "reproduction_report.json", {"match": match, "files": comparisons})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid-search stages. Existing outputs are skipped.")
    parser.add_argument("--stages", default="s1,s2,s5")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefix-source", default="tasksource/esci")
    parser.add_argument("--sync-cmd", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--disagreement", default=None)
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--s0-products", type=int, default=20000)
    parser.add_argument("--s0-queries", type=int, default=200)
    parser.add_argument("--run-s0-smoke", action="store_true")
    parser.add_argument("--force-s5", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    data = None
    splits = None
    if any(stage != "s0" for stage in stages):
        data = load_esci(resolve_data_dir(args.data_dir))
        splits = ensure_splits(data, output, args.prefix_source)
    for stage in stages:
        step = stage
        if stage == "s0":
            run_s0(args, output)
        elif stage == "s1":
            run_s1(args, output, data, splits)
        elif stage == "s2":
            run_s2(args, output, data, splits)
        elif stage == "s3":
            run_s3(args, output, data, splits)
        elif stage == "s4":
            run_s4(args, output, data, splits)
        elif stage == "s5":
            run_s5(args, output, data, splits)
        else:
            raise ValueError(f"Unknown stage {stage}.")
        sync(args.sync_cmd, stage, step, output)
        print(f"Finished {stage}")


if __name__ == "__main__":
    main()
