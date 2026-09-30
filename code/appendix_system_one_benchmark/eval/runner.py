"""Resumable benchmark runner with cached predictions and standard metrics."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.base import ItemPrediction
from adapters.schemas import PROMPT_VERSION, TASK_FIELDS
from adjudicate_pilots import sha256
from eval.datasets import labeled_pairs, load_examples, write_jsonl
from eval.metrics import exact_match, field_metrics, latency_metrics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "outputs" / "benchmark_cache.sqlite"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "benchmark"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            cache_key TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            task TEXT NOT NULL,
            adapter TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prediction_json TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            elapsed_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_lookup
            ON predictions(task, adapter, model_id, item_id);
        """
    )
    connection.commit()
    return connection


def prediction_cache_key(
    adapter_name: str,
    model_id: str,
    task: str,
    item_id: str,
    prompt_version: str,
) -> str:
    return sha256("|".join([adapter_name, model_id, task, item_id, prompt_version]))


def load_cached(
    connection: sqlite3.Connection,
    adapter_name: str,
    model_id: str,
    task: str,
    item_ids: list[str],
    prompt_version: str,
) -> dict[str, ItemPrediction]:
    cached: dict[str, ItemPrediction] = {}
    for item_id in item_ids:
        key = prediction_cache_key(adapter_name, model_id, task, item_id, prompt_version)
        row = connection.execute(
            "SELECT prediction_json FROM predictions WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row:
            cached[item_id] = prediction_from_dict(json.loads(row[0]))
    return cached


def store_prediction(
    connection: sqlite3.Connection,
    adapter_name: str,
    model_id: str,
    prompt_version: str,
    prediction: ItemPrediction,
) -> None:
    key = prediction_cache_key(
        adapter_name, model_id, prediction.task, prediction.item_id, prompt_version
    )
    connection.execute(
        """
        INSERT OR REPLACE INTO predictions(
            cache_key, item_id, task, adapter, model_id, prompt_version,
            prediction_json, input_tokens, elapsed_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key,
            prediction.item_id,
            prediction.task,
            adapter_name,
            model_id,
            prompt_version,
            json.dumps(prediction.to_dict(), ensure_ascii=False, sort_keys=True),
            prediction.input_tokens,
            prediction.elapsed_ms,
            utc_now(),
        ),
    )


def prediction_from_dict(payload: dict[str, Any]) -> ItemPrediction:
    from adapters.base import FieldPrediction

    fields = {
        name: FieldPrediction(
            field=name,
            predicted=values["predicted"],
            probabilities=values["probabilities"],
            confidence=float(values["confidence"]),
        )
        for name, values in payload["fields"].items()
    }
    return ItemPrediction(
        item_id=payload["item_id"],
        task=payload["task"],
        fields=fields,
        input_tokens=int(payload.get("input_tokens") or 0),
        elapsed_ms=float(payload.get("elapsed_ms") or 0.0),
    )


def gpu_memory_snapshot(device: str) -> dict[str, float] | None:
    if device != "cuda":
        return None
    torch_snapshot = _torch_gpu_snapshot()
    if torch_snapshot and torch_snapshot.get("peak_reserved_mb", 0) > 0:
        return torch_snapshot
    smi_snapshot = nvidia_smi_snapshot()
    if smi_snapshot:
        return smi_snapshot
    return torch_snapshot


def _torch_gpu_snapshot() -> dict[str, float] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
        "device_total_mb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**2), 1
        ),
        "source": "torch",
    }


def nvidia_smi_snapshot() -> dict[str, float] | None:
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return None
    used_mb = float(parts[0])
    total_mb = float(parts[1])
    return {
        "peak_allocated_mb": used_mb,
        "peak_reserved_mb": used_mb,
        "device_total_mb": total_mb,
        "source": "nvidia-smi",
    }


def evaluate_task(
    examples: list[dict[str, Any]],
    predictions: list[ItemPrediction],
    task: str,
) -> dict[str, Any]:
    fields = TASK_FIELDS[task]
    field_reports = {}
    for field in fields:
        pairs = labeled_pairs(examples, predictions, field)
        if not pairs:
            field_reports[field] = {"items": 0}
            continue
        y_true = [row[0] for row in pairs]
        y_pred = [row[1] for row in pairs]
        probabilities = [row[2] for row in pairs]
        confidences = [row[3] for row in pairs]
        report = field_metrics(y_true, y_pred, probabilities, confidences)
        by_stratum: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        for reference, predicted, _probs, _conf, example in pairs:
            stratum = example.get("sampling_context", {}).get("source_stratum") or example.get(
                "label_source", "unknown"
            )
            grouped[stratum].append((reference, predicted))
        for stratum, rows in grouped.items():
            correct = sum(
                str(reference) == str(predicted)
                if not isinstance(reference, bool)
                else reference == predicted
                for reference, predicted in rows
            )
            by_stratum[stratum] = {
                "items": len(rows),
                "accuracy": round(correct / len(rows), 4),
            }
        report["accuracy_by_stratum"] = dict(sorted(by_stratum.items()))
        if task == "esci":
            by_source: dict[str, dict[str, Any]] = {}
            source_groups: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
            for reference, predicted, _probs, _conf, example in pairs:
                source_groups[example.get("label_source", "unknown")].append(
                    (reference, predicted)
                )
            for source, rows in source_groups.items():
                correct = sum(reference == predicted for reference, predicted in rows)
                by_source[source] = {
                    "items": len(rows),
                    "accuracy": round(correct / len(rows), 4),
                }
            report["accuracy_by_label_source"] = dict(sorted(by_source.items()))
        field_reports[field] = report

    return {
        "items": len(examples),
        "fields": field_reports,
        "complete_exact_match": exact_match(examples, predictions, fields),
        "latency": latency_metrics([prediction.elapsed_ms for prediction in predictions]),
    }


def run_task(
    adapter: Any,
    task: str,
    dataset: str,
    output_dir: Path,
    cache: Path,
    batch_size: int,
    max_items: int | None,
    require_complete: bool,
    device: str,
) -> dict[str, Any]:
    examples = load_examples(
        task,
        dataset=dataset,
        require_complete=require_complete,
        max_items=max_items,
    )
    connection = connect_cache(cache)
    adapter_name = getattr(adapter, "name", adapter.__class__.__name__)
    model_id = getattr(adapter, "cache_model_id", getattr(adapter, "model_id", adapter_name))
    cached = load_cached(
        connection, adapter_name, model_id, task, [row["item_id"] for row in examples], PROMPT_VERSION
    )
    pending = [row for row in examples if row["item_id"] not in cached]
    started = time.perf_counter()
    if hasattr(adapter, "load") and pending:
        adapter.load()
        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        predictions = adapter.predict_batch(task, chunk, batch_size=batch_size)
        for prediction in predictions:
            cached[prediction.item_id] = prediction
            store_prediction(connection, adapter_name, model_id, PROMPT_VERSION, prediction)
        connection.commit()

    ordered = [cached[row["item_id"]] for row in examples]
    elapsed = time.perf_counter() - started
    metrics = evaluate_task(examples, ordered, task)
    metrics["adapter"] = adapter_name
    metrics["model_id"] = model_id
    metrics["prompt_version"] = PROMPT_VERSION
    metrics["dataset"] = dataset
    metrics["task"] = task
    metrics["cached_items"] = len(examples) - len(pending)
    metrics["new_items"] = len(pending)
    metrics["wall_seconds"] = round(elapsed, 3)
    metrics["memory"] = gpu_memory_snapshot(device)
    input_tokens = sum(prediction.input_tokens for prediction in ordered)
    metrics["input_tokens"] = input_tokens
    rate = getattr(adapter, "input_usd_per_million", None)
    if rate is not None:
        metrics["cost_usd"] = round(input_tokens * float(rate) / 1_000_000, 6)
    write_jsonl(
        output_dir / "predictions" / f"{task}.jsonl",
        [prediction.to_dict() for prediction in ordered],
    )
    metrics_path = output_dir / "metrics" / f"{task}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    connection.close()
    return metrics
