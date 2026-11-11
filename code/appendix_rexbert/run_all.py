"""Run a profile end to end and write run_manifest.json.

    python run_all.py --profile gpu_full --runs R1,R2,R3,R4

Completed runs are skipped when their metrics JSON already exists.
--sync_cmd is an optional shell template run after each completed run.
Placeholders are {run}, {metrics_dir}, and {profile}. It defaults to none.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from config import (
    CHAPTER10_QUERY_SEED,
    MODELS,
    PACKAGE_DIR,
    PROFILES,
    RUN_IDS,
    SAMPLE_SEED,
    SPLIT_TEST,
    SPLIT_VAL,
    Profile,
    set_seed,
)
from data import (
    chapter10_query_ids,
    prepare_dataset,
    print_data_card,
    resolve_data_dir,
    sample_catalog,
    sample_indices,
)
from evaluate import (
    METRIC_NAMES,
    build_ip_index,
    multi_target_at_k,
    per_query_from_ranks,
    ranks_for_single_targets,
    seed_averaged_difference,
    self_excluded_neighbors,
    summarize_metric_matrix,
)
from make_figures import write_figures

logger = logging.getLogger("appendix_rexbert")

BANNED_FRAGMENTS = ("one" + "drive", "c:" + "\\use" + "rs", "c:/" + "use" + "rs")
IGNORE_COMPARE_KEYS = {
    "wall_clock_seconds",
    "peak_working_set_mb",
    "rss_mb",
    "startup_data_load_seconds",
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def assert_public(value) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        for fragment in BANNED_FRAGMENTS:
            if fragment in lowered:
                raise ValueError("Refusing to write a local machine path into metrics")
    elif isinstance(value, dict):
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


def dump_json(path: Path, payload: dict) -> None:
    payload = to_jsonable(payload)
    assert_public(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_output_root(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        raise SystemExit("output_root must be a relative path under outputs/.")
    root = (PACKAGE_DIR / path).resolve()
    outputs = (PACKAGE_DIR / "outputs").resolve()
    if root != outputs and outputs not in root.parents:
        raise SystemExit("output_root must stay under outputs/.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def metrics_file(root: Path, profile: str, run_id: str) -> Path:
    return root / "metrics" / profile / f"{run_id}.json"


def memory_mb() -> dict[str, float | None]:
    rss = peak = None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        # psapi.GetProcessMemoryInfo returns failure on this Windows build.
        # kernel32.K32GetProcessMemoryInfo is the supported export.
        query = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        ok = query(handle, ctypes.byref(counters), counters.cb)
        if ok:
            rss = counters.WorkingSetSize / (1024 * 1024)
            peak = counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        rss = peak = None
    if peak is None:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            rss = peak
        except Exception:
            pass
    return {"rss_mb": None if rss is None else round(rss, 1), "peak_working_set_mb": None if peak is None else round(peak, 1)}


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "torch",
        "transformers",
        "sentence-transformers",
        "accelerate",
        "faiss-cpu",
        "numpy",
        "pandas",
        "matplotlib",
        "datasets",
        "huggingface-hub",
        "scikit-learn",
        "scipy",
    ]
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def git_head() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PACKAGE_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def preregistration_sha256() -> str | None:
    path = PACKAGE_DIR / "PREREGISTRATION.md"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        loaded = load_json(path)
    except json.JSONDecodeError:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def carried_commitment(previous: dict | None) -> bool | None:
    """Return a boolean already stored in an earlier manifest, otherwise None."""
    if not isinstance(previous, dict):
        return None
    value = previous.get("preregistration_committed_before_r3")
    if isinstance(value, bool):
        return value
    return None


def gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "name": None}
    return {"available": True, "name": torch.cuda.get_device_name(0)}


def apply_sync(template: str | None, run_id: str, metrics_dir: Path, profile: str) -> int | None:
    if not template:
        return None
    relative = metrics_dir.resolve().relative_to(PACKAGE_DIR).as_posix()
    command = template.format(run=run_id, metrics_dir=relative, profile=profile)
    completed = subprocess.run(command, shell=True, check=False, cwd=PACKAGE_DIR)
    return int(completed.returncode)


def build_plan(prepared, profile: Profile) -> dict:
    test_available = np.flatnonzero(prepared.session_frame_mask(SPLIT_TEST))
    test_index = sample_indices(test_available, profile.n_test_queries, SAMPLE_SEED)
    test_query = [prepared.query_ids[int(index)] for index in test_index]
    test_target = [prepared.target_ids[int(index)] for index in test_index]
    ch10_queries = chapter10_query_ids(
        prepared.catalog_ids,
        prepared.chapter10_map,
        profile.chapter10_queries,
        CHAPTER10_QUERY_SEED,
    )
    must: list[str] = []
    for query, target in zip(test_query, test_target):
        must.extend((query, target))
    for query in ch10_queries:
        must.append(query)
        must.extend(prepared.chapter10_map.get(query, ()))
    val_available = np.flatnonzero(prepared.session_frame_mask(SPLIT_VAL))
    val_index = sample_indices(val_available, profile.n_val_sessions, SAMPLE_SEED)
    for index in val_index.tolist():
        must.extend((prepared.query_ids[int(index)], prepared.target_ids[int(index)]))
    catalog_ids, catalog_info = sample_catalog(prepared.catalog_ids, must, profile.n_catalog, SAMPLE_SEED)
    return {
        "test_query": test_query,
        "test_target": test_target,
        "n_test_available": int(len(test_available)),
        "ch10_queries": ch10_queries,
        "val_index": val_index,
        "catalog_ids": catalog_ids,
        "catalog_info": catalog_info,
    }


def _summary_row(label: str, group: str, summary: dict, cold_summary: dict) -> dict:
    row = {"label": label, "group": group}
    for name in ("recall_at_10", "recall_at_50", "mrr", "mrr_at_100", "ndcg_at_10"):
        block = summary.get(name, {})
        row[name] = block.get("mean")
        if name in ("recall_at_50", "mrr"):
            row[f"{name}_low"] = block.get("low")
            row[f"{name}_high"] = block.get("high")
    cold = cold_summary.get("recall_at_50", {})
    row["cold_recall_at_50"] = cold.get("mean")
    row["cold_recall_at_50_low"] = cold.get("low")
    row["cold_recall_at_50_high"] = cold.get("high")
    return row


def _evaluate_pairs(embeddings: np.ndarray, catalog_ids: list[str], query_ids: list[str], target_ids: list[str]):
    id_to_row = {product_id: index for index, product_id in enumerate(catalog_ids)}
    query_rows = []
    target_rows = []
    for query, target in zip(query_ids, target_ids):
        query_row = id_to_row.get(query)
        target_row = id_to_row.get(target)
        if query_row is None or target_row is None or query_row == target_row:
            raise RuntimeError("An evaluation pair is missing from the catalog or is degenerate")
        query_rows.append(query_row)
        target_rows.append(target_row)
    ranks = ranks_for_single_targets(
        embeddings, np.asarray(query_rows, dtype=np.int64), np.asarray(target_rows, dtype=np.int64)
    )
    return per_query_from_ranks(ranks)


def _evaluate_chapter10(embeddings: np.ndarray, catalog_ids: list[str], query_ids: list[str], next_item_map: dict):
    id_to_row = {product_id: index for index, product_id in enumerate(catalog_ids)}
    kept_queries = [query for query in query_ids if query in id_to_row]
    if not kept_queries:
        return {name: np.zeros(0) for name in METRIC_NAMES} | {"hit_rate_at_10": np.zeros(0), "n_targets": np.zeros(0)}
    matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = build_ip_index(matrix)
    query_rows = np.asarray([id_to_row[query] for query in kept_queries], dtype=np.int64)
    search_k = min(101, matrix.shape[0])
    _, neighbors = index.search(matrix[query_rows], search_k)
    per_query = {name: np.empty(len(kept_queries), dtype=np.float64) for name in METRIC_NAMES}
    per_query["hit_rate_at_10"] = np.empty(len(kept_queries), dtype=np.float64)
    per_query["n_targets"] = np.empty(len(kept_queries), dtype=np.float64)
    for row_index, query in enumerate(kept_queries):
        query_row = int(query_rows[row_index])
        filtered = self_excluded_neighbors(neighbors[row_index], query_row, 100)
        targets = [id_to_row[item] for item in next_item_map.get(query, ()) if item in id_to_row]
        target_set = set(targets)
        metrics = multi_target_at_k(filtered, target_set)
        best = int(metrics["best_rank_in_list"])
        if best == 0 and targets:
            sims = matrix @ matrix[query_row]
            sims[query_row] = -np.inf
            best_score = max(float(sims[item]) for item in targets)
            best = int(np.sum(sims > best_score)) + 1
        per_query["recall_at_10"][row_index] = metrics["recall_at_10"]
        per_query["recall_at_50"][row_index] = metrics["recall_at_50"]
        per_query["mrr_at_100"][row_index] = (1.0 / best) if 0 < best <= 100 else 0.0
        per_query["mrr"][row_index] = (1.0 / best) if best > 0 else 0.0
        per_query["ndcg_at_10"][row_index] = metrics["ndcg_at_10"]
        per_query["hit_rate_at_10"][row_index] = metrics["hit_rate_at_10"]
        per_query["n_targets"][row_index] = float(len(targets))
    return per_query


def _direction(minilm: dict, rexbert: dict) -> dict:
    def mean(summary, key):
        return summary[key]["mean"]

    higher_mrr = mean(minilm, "mrr") > mean(rexbert, "mrr")
    higher_recall = mean(minilm, "recall_at_10") > mean(rexbert, "recall_at_10")
    return {
        "statement": "Chapter 10 paragraph 86 says general-purpose SBERT outperforms RexBERT",
        "minilm_higher_mrr": bool(higher_mrr),
        "minilm_higher_recall_at_10": bool(higher_recall),
        "matches_chapter10_paragraph_86": bool(higher_mrr and higher_recall),
    }


def _encode(spec_key: str, texts: list[str], batch_size: int, device: str, model=None):
    from encoders import build_sentence_encoder, encode_texts

    own = model is None
    if own:
        model = build_sentence_encoder(MODELS[spec_key], device=device)
    vectors = encode_texts(model, texts, batch_size)
    if own:
        del model
        gc.collect()
    return vectors


def _validation_evaluator(prepared, profile: Profile, val_index: np.ndarray, batch_size: int):
    from train import build_selection_evaluator

    queries = {}
    relevant = {}
    query_items = {}
    must = []
    for session_index in val_index.tolist():
        query_id = f"s{int(session_index)}"
        query_item = prepared.query_ids[int(session_index)]
        target_item = prepared.target_ids[int(session_index)]
        queries[query_id] = prepared.texts[query_item]
        relevant[query_id] = {target_item}
        query_items[query_id] = query_item
        must.extend((query_item, target_item))
    corpus_ids, info = sample_catalog(prepared.catalog_ids, must, profile.n_val_corpus, SAMPLE_SEED)
    corpus = {product_id: prepared.texts[product_id] for product_id in corpus_ids}
    evaluator = build_selection_evaluator(queries, corpus, relevant, query_items, batch_size)
    return evaluator, info


def _slice_masks(prepared, query_ids: list[str], target_ids: list[str]) -> dict[str, np.ndarray]:
    warm = np.asarray([target in prepared.seen_train_items for target in target_ids], dtype=bool)
    seen = np.asarray(
        [(query, target) in prepared.pair_keys_before_cap for query, target in zip(query_ids, target_ids)],
        dtype=bool,
    )
    return {"all": np.ones(len(query_ids), dtype=bool), "warm": warm, "cold": ~warm, "seen_pair": seen, "unseen_pair": ~seen}


def run_profile(args, profile: Profile, root: Path) -> dict:
    metrics_dir = root / "metrics" / profile.name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "cache" / profile.name
    # Stay on CPU until a model run. R0 must not import torch, so its peak
    # working set is the data load rather than the library.
    device = "cpu"
    device_ready = False
    set_seed(profile.seeds[0])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    prepared = None
    plan = None
    embeddings: dict[tuple[str, str], np.ndarray] = {}

    def ensure_data() -> float:
        nonlocal prepared, plan
        if prepared is not None:
            return 0.0
        started = time.perf_counter()
        data_dir = resolve_data_dir(args.data_dir)
        prepared = prepare_dataset(data_dir, pair_cap=profile.n_train_pairs)
        plan = build_plan(prepared, profile)
        print_data_card(prepared.card)
        return time.perf_counter() - started

    def resolve_device_once() -> None:
        nonlocal device, device_ready
        if device_ready:
            return
        started_import = time.perf_counter()
        import torch

        manifest["gpu"] = gpu_info()
        manifest["torch_import_seconds"] = round(time.perf_counter() - started_import, 3)
        if profile.request_bf16 and torch.cuda.is_available():
            device = "cuda"
        set_seed(profile.seeds[0])
        device_ready = True

    manifest_path = root / "run_manifest.json"
    previous_manifest = _read_manifest(manifest_path)
    previous_gpu = previous_manifest.get("gpu")
    if not isinstance(previous_gpu, dict):
        previous_gpu = {"available": None, "name": None}
    prior_runs = previous_manifest.get("runs")
    kept_runs = {}
    if isinstance(prior_runs, dict):
        kept_runs = {
            key: value
            for key, value in prior_runs.items()
            if key in RUN_IDS and isinstance(value, dict)
        }
    manifest = {
        "profile": profile.name,
        "reported_protocol": profile.reported,
        "git_commit": git_head(),
        "packages": package_versions(),
        "model_revisions": {key: spec.revision for key, spec in MODELS.items()},
        "gpu": previous_gpu,
        "seeds": list(profile.seeds),
        "preregistration_sha256": preregistration_sha256(),
        "preregistration_committed_before_r3": carried_commitment(previous_manifest),
        "runs": kept_runs,
    }
    if isinstance(previous_manifest.get("dataset_sha256"), dict):
        manifest["dataset_sha256"] = previous_manifest["dataset_sha256"]
    if "torch_import_seconds" in previous_manifest:
        manifest["torch_import_seconds"] = previous_manifest["torch_import_seconds"]
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    unknown = [item for item in runs if item not in RUN_IDS]
    if unknown:
        raise SystemExit("Unknown run id.")
    if any(item in runs for item in ("R1", "R2", "R3", "R4")):
        r0_path = metrics_file(root, profile.name, "R0")
        if "R0" not in runs and (args.force or not r0_path.exists()):
            runs = ["R0", *runs]

    def save_manifest() -> None:
        dump_json(manifest_path, manifest)

    startup_load = None
    r0_will_run = "R0" in runs and (args.force or not metrics_file(root, profile.name, "R0").exists())
    if not r0_will_run and any(item in runs for item in ("R1", "R2", "R3", "R4")):
        startup_load = round(ensure_data(), 3)
        manifest["startup_data_load_seconds"] = startup_load

    for run_id in runs:
        path = metrics_file(root, profile.name, run_id)
        if path.exists() and not args.force:
            logger.info("Skipping %s; metrics file exists", run_id)
            prior_metrics: dict = {}
            try:
                loaded_metrics = load_json(path)
                if isinstance(loaded_metrics, dict):
                    prior_metrics = loaded_metrics
            except json.JSONDecodeError:
                prior_metrics = {}
            manifest["runs"][run_id] = {
                "status": "skipped",
                "wall_clock_seconds": prior_metrics.get("wall_clock_seconds"),
                "peak_working_set_mb": prior_metrics.get("peak_working_set_mb"),
                "rss_mb": prior_metrics.get("rss_mb"),
                "metrics_file": f"metrics/{profile.name}/{run_id}.json",
                "sync_exit_code": None,
            }
            if run_id == "R0" and isinstance(prior_metrics.get("file_hashes"), dict):
                manifest["dataset_sha256"] = prior_metrics["file_hashes"]
            save_manifest()
            continue
        logger.info("Starting %s", run_id)
        started = time.perf_counter()
        try:
            if run_id == "R0":
                ensure_data()
                payload = {
                    "run": "R0",
                    "data_card": prepared.card,
                    "file_hashes": prepared.file_hashes,
                    "n_test_queries": len(plan["test_query"]),
                    "n_test_available": plan["n_test_available"],
                    "n_catalog": len(plan["catalog_ids"]),
                    "catalog_info": plan["catalog_info"],
                    "n_chapter10_queries": len(plan["ch10_queries"]),
                    "n_val_sessions_sampled": int(len(plan["val_index"])),
                    "profile": profile.name,
                    "reported_protocol": profile.reported,
                }
            elif run_id == "R1":
                resolve_device_once()
                ensure_data()
                texts = [prepared.texts[product_id] for product_id in plan["catalog_ids"]]
                models_out = {}
                masks = _slice_masks(prepared, plan["test_query"], plan["test_target"])
                for model_key in profile.zero_shot_models:
                    logger.info("Encoding zero-shot %s", model_key)
                    vectors = _encode(model_key, texts, profile.encode_batch_size, device)
                    embeddings[(model_key, "zero_shot")] = vectors
                    appendix = _evaluate_pairs(vectors, plan["catalog_ids"], plan["test_query"], plan["test_target"])
                    chapter10 = _evaluate_chapter10(
                        vectors, plan["catalog_ids"], plan["ch10_queries"], prepared.chapter10_map
                    )
                    models_out[model_key] = {
                        "appendix_protocol": {
                            name: summarize_metric_matrix(appendix, mask, profile.bootstrap_resamples)
                            for name, mask in masks.items()
                        },
                        "chapter10_protocol": summarize_metric_matrix(chapter10, n_resamples=profile.bootstrap_resamples),
                        "chapter10_mean_targets_per_query": float(np.mean(chapter10["n_targets"]))
                        if len(chapter10["n_targets"])
                        else None,
                        "embedding_dim": int(vectors.shape[1]),
                    }
                direction = None
                if "minilm" in models_out and "rexbert_micro" in models_out:
                    direction = _direction(
                        models_out["minilm"]["chapter10_protocol"],
                        models_out["rexbert_micro"]["chapter10_protocol"],
                    )
                    appendix_direction = _direction(
                        models_out["minilm"]["appendix_protocol"]["all"],
                        models_out["rexbert_micro"]["appendix_protocol"]["all"],
                    )
                    direction["protocol"] = "chapter10"
                    direction["appendix_protocol_minilm_higher_mrr"] = appendix_direction["minilm_higher_mrr"]
                    direction["corpus_is_full_catalog"] = bool(plan["catalog_info"].get("full_catalog"))
                    direction["this_profile_is_the_reported_protocol"] = profile.reported
                    if "rexbert_base" in models_out:
                        direction["minilm_vs_rexbert_base"] = _direction(
                            models_out["minilm"]["chapter10_protocol"],
                            models_out["rexbert_base"]["chapter10_protocol"],
                        )
                payload = {
                    "run": "R1",
                    "protocol_note": "appendix session-level protocol and Chapter 10 protocol are separate",
                    "models": models_out,
                    "q1_direction": direction,
                    "n_catalog": len(plan["catalog_ids"]),
                    "n_test_queries": len(plan["test_query"]),
                    "reported_protocol": profile.reported,
                }
            elif run_id == "R2":
                from encoders import build_sentence_encoder
                from train import select_learning_rate, train_encoder, use_bf16

                resolve_device_once()
                ensure_data()
                selected = {}
                sweep_rows = []
                if profile.run_lr_sweep:
                    anchors = [prepared.texts[item] for item in prepared.pair_query]
                    positives = [prepared.texts[item] for item in prepared.pair_positive]
                    n_sweep = min(profile.sweep_pairs, len(anchors))
                    rng = np.random.RandomState(0)
                    chosen = np.sort(rng.choice(len(anchors), size=n_sweep, replace=False))
                    sweep_anchors = [anchors[int(index)] for index in chosen]
                    sweep_positives = [positives[int(index)] for index in chosen]
                    bf16 = use_bf16(profile)
                    for model_key in profile.finetune_models:
                        scores = {}
                        for rate in profile.learning_rates:
                            logger.info("Sweep %s lr=%s", model_key, rate)
                            evaluator, _info = _validation_evaluator(
                                prepared, profile, plan["val_index"], profile.encode_batch_size
                            )
                            model = build_sentence_encoder(MODELS[model_key], device=device)
                            result = train_encoder(
                                model,
                                sweep_anchors,
                                sweep_positives,
                                evaluator,
                                cache_dir / "sweep" / model_key / str(rate),
                                epochs=1,
                                batch_size=profile.batch_size,
                                learning_rate=rate,
                                seed=0,
                                bf16=bf16,
                            )
                            score = result["best_validation_mrr_at_100"] or 0.0
                            scores[rate] = score
                            sweep_rows.append({"model": model_key, "learning_rate": rate, "validation_mrr_at_100": score})
                            del model
                            gc.collect()
                        selected[model_key] = select_learning_rate(scores)
                else:
                    for model_key in profile.finetune_models:
                        selected[model_key] = profile.learning_rates[0]
                payload = {
                    "run": "R2",
                    "sweep_executed": profile.run_lr_sweep,
                    "selected_learning_rate": selected,
                    "rows": sweep_rows,
                    "reported_protocol": profile.reported,
                }
            elif run_id == "R3":
                from encoders import build_sentence_encoder
                from train import train_encoder, use_bf16

                resolve_device_once()
                ensure_data()
                r2_path = metrics_file(root, profile.name, "R2")
                if profile.run_lr_sweep and not r2_path.exists():
                    raise RuntimeError("R2 metrics are required before R3 when the learning-rate sweep is on")
                if r2_path.exists():
                    selected = load_json(r2_path)["selected_learning_rate"]
                else:
                    selected = {model_key: profile.learning_rates[0] for model_key in profile.finetune_models}
                anchors = [prepared.texts[item] for item in prepared.pair_query]
                positives = [prepared.texts[item] for item in prepared.pair_positive]
                bf16 = use_bf16(profile)
                trained = []
                for model_key in profile.finetune_models:
                    rate = float(selected[model_key])
                    for seed in profile.seeds:
                        logger.info("Fine-tuning %s seed %s", model_key, seed)
                        evaluator, val_info = _validation_evaluator(
                            prepared, profile, plan["val_index"], profile.encode_batch_size
                        )
                        model = build_sentence_encoder(MODELS[model_key], device=device)
                        result = train_encoder(
                            model,
                            anchors,
                            positives,
                            evaluator,
                            cache_dir / "models" / model_key / f"seed{seed}",
                            epochs=profile.epochs,
                            batch_size=profile.batch_size,
                            learning_rate=rate,
                            seed=int(seed),
                            bf16=bf16,
                        )
                        result["model"] = model_key
                        result["relative_model_dir"] = f"cache/{profile.name}/models/{model_key}/seed{seed}/best"
                        result["validation_corpus"] = val_info
                        trained.append(result)
                        del model
                        gc.collect()
                payload = {
                    "run": "R3",
                    "selected_learning_rate": selected,
                    "trained": trained,
                    "n_pairs": len(anchors),
                    "epochs": profile.epochs,
                    "seeds": list(profile.seeds),
                    "bf16": bf16,
                    "reported_protocol": profile.reported,
                }
            elif run_id == "R4":
                from diagnostics import fill_mask_demo, paired_cosine, sample_index_pairs

                resolve_device_once()
                ensure_data()
                texts = [prepared.texts[product_id] for product_id in plan["catalog_ids"]]
                masks = _slice_masks(prepared, plan["test_query"], plan["test_target"])
                from sentence_transformers import SentenceTransformer

                per_model_seed: dict[str, dict[int, dict[str, np.ndarray]]] = {}
                for model_key in profile.finetune_models:
                    per_model_seed[model_key] = {}
                    for seed in profile.seeds:
                        model_dir = cache_dir / "models" / model_key / f"seed{seed}" / "best"
                        if not model_dir.exists():
                            raise RuntimeError("Fine-tuned checkpoint is missing. Run R3 before R4.")
                        logger.info("Encoding fine-tuned %s seed %s", model_key, seed)
                        model = SentenceTransformer(str(model_dir), trust_remote_code=False, device=device)
                        vectors = _encode(model_key, texts, profile.encode_batch_size, device, model=model)
                        embeddings[(model_key, f"seed{seed}")] = vectors
                        per_model_seed[model_key][int(seed)] = _evaluate_pairs(
                            vectors, plan["catalog_ids"], plan["test_query"], plan["test_target"]
                        )
                        del model
                        gc.collect()
                for model_key in ("minilm", "rexbert_micro", "ettin_17m"):
                    if model_key not in profile.zero_shot_models:
                        continue
                    if (model_key, "zero_shot") not in embeddings:
                        embeddings[(model_key, "zero_shot")] = _encode(
                            model_key, texts, profile.encode_batch_size, device
                        )
                from baselines import ranks_for_pairs as _ranks

                baseline_metrics = {}
                for kind in ("popularity", "covisitation"):
                    ranks = _ranks(
                        plan["test_query"],
                        plan["test_target"],
                        plan["catalog_ids"],
                        prepared.popularity,
                        prepared.transitions,
                        kind,
                    )
                    if np.any(ranks < 1):
                        raise RuntimeError("Baseline rank was undefined for an evaluation pair")
                    baseline_metrics[kind] = per_query_from_ranks(ranks)

                def pack(metrics_by_name):
                    return {
                        name: summarize_metric_matrix(metrics_by_name, mask, profile.bootstrap_resamples)
                        for name, mask in masks.items()
                    }

                rows = []
                model_payload = {}
                for kind, metrics in baseline_metrics.items():
                    packed = pack(metrics)
                    label = "Popularity" if kind == "popularity" else "Co-visitation"
                    rows.append(_summary_row(label, "baseline", packed["all"], packed["cold"]))
                    model_payload[kind] = packed
                for model_key in profile.zero_shot_models:
                    if (model_key, "zero_shot") not in embeddings:
                        continue
                    metrics = _evaluate_pairs(
                        embeddings[(model_key, "zero_shot")],
                        plan["catalog_ids"],
                        plan["test_query"],
                        plan["test_target"],
                    )
                    packed = pack(metrics)
                    rows.append(_summary_row(f"{model_key} zero-shot", "zero_shot", packed["all"], packed["cold"]))
                    model_payload[f"{model_key}_zero_shot"] = packed
                decisions = {}
                for model_key, by_seed in per_model_seed.items():
                    seed_summaries = []
                    averaged = {}
                    for name in METRIC_NAMES:
                        stacked = [by_seed[int(seed)][name] for seed in profile.seeds]
                        averaged[name] = np.mean(np.stack(stacked), axis=0)
                    packed = pack(averaged)
                    rows.append(
                        _summary_row(f"{model_key} fine-tuned", "fine_tuned", packed["all"], packed["cold"])
                    )
                    model_payload[f"{model_key}_fine_tuned"] = packed
                    for seed, metrics in by_seed.items():
                        seed_summaries.append({"seed": seed, "all": summarize_metric_matrix(metrics, n_resamples=profile.bootstrap_resamples)})
                    model_payload[f"{model_key}_fine_tuned_seeds"] = seed_summaries
                if "rexbert_micro" in per_model_seed and "ettin_17m" in per_model_seed:
                    for metric_name in ("mrr", "recall_at_50"):
                        decisions[f"q3_rexbert_minus_ettin_{metric_name}"] = seed_averaged_difference(
                            [per_model_seed["rexbert_micro"][int(seed)][metric_name] for seed in profile.seeds],
                            [per_model_seed["ettin_17m"][int(seed)][metric_name] for seed in profile.seeds],
                            n_resamples=profile.bootstrap_resamples,
                        )
                if "rexbert_micro" in per_model_seed and ( "rexbert_micro", "zero_shot") in embeddings:
                    zero = _evaluate_pairs(
                        embeddings[("rexbert_micro", "zero_shot")],
                        plan["catalog_ids"],
                        plan["test_query"],
                        plan["test_target"],
                    )
                    decisions["q2_rexbert_finetuned_minus_zeroshot_mrr"] = seed_averaged_difference(
                        [per_model_seed["rexbert_micro"][int(seed)]["mrr"] for seed in profile.seeds],
                        [zero["mrr"] for _seed in profile.seeds],
                        n_resamples=profile.bootstrap_resamples,
                    )
                q3 = decisions.get("q3_rexbert_minus_ettin_mrr", {})
                if q3.get("supported") and q3.get("sign") == "positive":
                    branch = "a"
                elif q3.get("supported") and q3.get("sign") == "negative":
                    branch = "c"
                else:
                    branch = "b"
                decisions["q3_branch_on_mrr"] = branch
                decisions["primary_metric"] = "mrr"
                decisions["rule"] = (
                    "Supported when the paired 95% interval of the seed-averaged per-query "
                    "difference excludes zero and all required seeds agree in sign."
                )
                id_to_row = {product_id: index for index, product_id in enumerate(plan["catalog_ids"])}
                n_pos = min(profile.n_cosine_pairs, len(plan["test_query"]))
                pos_left = np.asarray([id_to_row[item] for item in plan["test_query"][:n_pos]], dtype=np.int64)
                pos_right = np.asarray([id_to_row[item] for item in plan["test_target"][:n_pos]], dtype=np.int64)
                rand_left, rand_right = sample_index_pairs(
                    len(plan["catalog_ids"]), min(profile.n_cosine_pairs, max(len(plan["catalog_ids"]) - 1, 0)), SAMPLE_SEED
                )
                cosine_series = {
                    "rexbert_micro_zero_shot": ("rexbert_micro", "zero_shot"),
                    "rexbert_micro_fine_tuned_seed0": ("rexbert_micro", "seed0"),
                    "minilm_zero_shot": ("minilm", "zero_shot"),
                    "ettin_17m_zero_shot": ("ettin_17m", "zero_shot"),
                }
                diagnostics = {"synthetic": False, "random_pairs": {}, "positive_pairs": {}, "demonstration": True}
                for series_name, embed_key in cosine_series.items():
                    if embed_key not in embeddings:
                        continue
                    diagnostics["random_pairs"][series_name] = paired_cosine(embeddings[embed_key], rand_left, rand_right)
                    diagnostics["positive_pairs"][series_name] = paired_cosine(embeddings[embed_key], pos_left, pos_right)
                try:
                    micro = MODELS["rexbert_micro"]
                    diagnostics["fill_mask_demo"] = fill_mask_demo(micro.hub_id, micro.revision)
                except Exception:
                    logger.exception("Fill-mask demo failed")
                    diagnostics["fill_mask_demo_error"] = "fill-mask demo failed"
                dump_json(metrics_dir / "diagnostics.json", diagnostics)
                dump_json(metrics_dir / "figure_a4.json", {"synthetic": False, "rows": rows, "reported_protocol": profile.reported})
                dump_json(metrics_dir / "decision.json", decisions)
                _write_summary_csv(metrics_dir / "summary.csv", rows)
                figure_dir = root / "figures" / profile.name
                write_figures(metrics_dir, figure_dir)
                payload = {
                    "run": "R4",
                    "models": model_payload,
                    "decision": decisions,
                    "n_test_queries": len(plan["test_query"]),
                    "n_catalog": len(plan["catalog_ids"]),
                    "reported_protocol": profile.reported,
                    "slice_counts": {name: int(mask.sum()) for name, mask in masks.items()},
                }
            else:
                raise RuntimeError("Unknown run")
        except Exception as exc:
            manifest["runs"][run_id] = {"status": "failed", "error_type": type(exc).__name__}
            save_manifest()
            raise
        elapsed = time.perf_counter() - started
        mem = memory_mb()
        payload["wall_clock_seconds"] = round(elapsed, 3)
        payload["peak_working_set_mb"] = mem["peak_working_set_mb"]
        payload["rss_mb"] = mem["rss_mb"]
        dump_json(path, payload)
        exit_code = apply_sync(args.sync_cmd, run_id, metrics_dir, profile.name)
        manifest["dataset_sha256"] = prepared.file_hashes if prepared is not None else None
        manifest["runs"][run_id] = {
            "status": "completed",
            "wall_clock_seconds": round(elapsed, 3),
            "peak_working_set_mb": mem["peak_working_set_mb"],
            "rss_mb": mem["rss_mb"],
            "metrics_file": f"metrics/{profile.name}/{run_id}.json",
            "sync_exit_code": exit_code,
        }
        save_manifest()
        logger.info("Finished %s in %.1f min", run_id, elapsed / 60.0)
        if run_id == "R0" and any(item in runs for item in ("R1", "R2", "R3", "R4")):
            resolve_device_once()
    save_manifest()
    return manifest


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fields = [
        "label",
        "group",
        "recall_at_10",
        "recall_at_50",
        "recall_at_50_low",
        "recall_at_50_high",
        "mrr",
        "mrr_low",
        "mrr_high",
        "ndcg_at_10",
        "cold_recall_at_50",
        "cold_recall_at_50_low",
        "cold_recall_at_50_high",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compare_trees(left: dict, right: dict) -> list[tuple[str, float]]:
    diffs: list[tuple[str, float]] = []

    def walk(a, b, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) & set(b)):
                if key in IGNORE_COMPARE_KEYS:
                    continue
                walk(a[key], b[key], f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            diffs.append((prefix, abs(float(a) - float(b))))

    walk(left, right, "")
    return diffs


def compare_directories(left: Path, right: Path) -> dict:
    report = {}
    for name in ("R0.json", "R1.json", "R4.json", "decision.json"):
        left_path = left / name
        right_path = right / name
        if not left_path.exists() or not right_path.exists():
            report[name] = {"status": "missing"}
            continue
        diffs = compare_trees(load_json(left_path), load_json(right_path))
        max_diff = max((item[1] for item in diffs), default=0.0)
        report[name] = {"status": "compared", "max_abs_diff": max_diff, "n_numeric": len(diffs)}
    return report


def check_models() -> dict:
    from encoders import verify_backbone

    payload = {"models": [verify_backbone(key) for key in ("rexbert_micro", "rexbert_base", "ettin_17m")]}
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the RexBERT appendix profile.")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="cpu_small or gpu_full")
    parser.add_argument("--runs", default="R0,R1,R2,R3,R4")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sync_cmd", default=None)
    parser.add_argument("--check_models", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    os.chdir(PACKAGE_DIR)
    if args.compare:
        left = Path(args.compare[0])
        right = Path(args.compare[1])
        print(json.dumps(compare_directories(left, right), indent=2))
        return
    root = resolve_output_root(args.output_root)
    if args.check_models:
        payload = check_models()
        dump_json(root / "metrics" / "model_check.json", payload)
        print(json.dumps(to_jsonable(payload), indent=2))
        return
    if not args.profile:
        raise SystemExit("Pass --profile cpu_small or --profile gpu_full.")
    run_profile(args, PROFILES[args.profile], root)


if __name__ == "__main__":
    main()
