"""Run a small local Laya ESCI-style throughput and VRAM smoke test."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

import torch


BASE_STATES = [
    "Query: iphone 13\nProduct: Apple iPhone 13 128GB unlocked smartphone",
    "Query: iphone 13\nProduct: Samsung Galaxy S23 Android smartphone",
    "Query: iphone 13\nProduct: Protective case designed for Apple iPhone 13",
    "Query: iphone 13\nProduct: Stainless steel garden hose nozzle",
    "Query: black running shoes women size 8\nProduct: Women's black running shoe, size 8",
    "Query: black running shoes women size 8\nProduct: Women's navy walking shoe, size 8",
    "Query: black running shoes women size 8\nProduct: Replacement athletic shoe laces",
    "Query: black running shoes women size 8\nProduct: Men's brown leather wallet",
]
BASE_EXPECTED = ["E", "S", "C", "I", "E", "S", "C", "I"]

QUESTIONS = {
    "esci_label": {
        "type": "choice",
        "instructions": (
            "Classify the query-product relationship using Amazon ESCI relevance."
        ),
        "criteria": {
            "E": "Exact: directly satisfies the query and its explicit constraints",
            "S": "Substitute: plausible alternative for the same main purpose",
            "C": "Complement: useful with the requested item but not a replacement",
            "I": "Irrelevant: neither satisfies nor meaningfully complements the query",
        },
    }
}


def make_states(batch_size: int) -> list[str]:
    return [BASE_STATES[index % len(BASE_STATES)] for index in range(batch_size)]


def make_expected(batch_size: int) -> list[str]:
    return [BASE_EXPECTED[index % len(BASE_EXPECTED)] for index in range(batch_size)]


def run_smoke(
    model_id: str, subfolder: str | None, batch_size: int, device: str
) -> dict[str, Any]:
    import laya

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    agent = laya.load(
        model_id,
        device=device,
        subfolder=subfolder,
    )
    load_seconds = time.perf_counter() - load_started

    agent.predict(BASE_STATES[0], QUESTIONS)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    states = make_states(batch_size)
    started = time.perf_counter()
    results = agent.predict_batch(states, QUESTIONS, batch_size=batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    answers = [
        result["answers"]["esci_label"]["choice"]
        for result in results
    ]
    expected = make_expected(batch_size)
    correct = sum(
        predicted == reference
        for predicted, reference in zip(answers, expected, strict=True)
    )
    report: dict[str, Any] = {
        "model_id": model_id,
        "subfolder": subfolder,
        "device": device,
        "batch_size": batch_size,
        "load_seconds": round(load_seconds, 4),
        "batch_seconds": round(elapsed, 4),
        "items_per_second": round(batch_size / elapsed, 3),
        "answers": answers,
        "expected": expected,
        "sanity_correct": correct,
        "sanity_accuracy": round(correct / batch_size, 4),
    }
    if device == "cuda":
        report["peak_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024**2), 1
        )
        report["peak_reserved_mb"] = round(
            torch.cuda.max_memory_reserved() / (1024**2), 1
        )
        report["device_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**2), 1
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--subfolder")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    report = run_smoke(
        model_id=args.model_id,
        subfolder=args.subfolder,
        batch_size=args.batch_size,
        device=args.device,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
