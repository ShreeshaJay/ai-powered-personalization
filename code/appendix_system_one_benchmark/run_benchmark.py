"""Run the System One search benchmark for one model adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adapters.jev import JevAdapter
from adapters.jevlite import JevLiteAdapter
from adapters.kev import KevAdapter
from adapters.laya import LayaAdapter
from adapters.majority import MajorityAdapter
from eval.runner import DEFAULT_CACHE, DEFAULT_OUTPUT_DIR, run_task
from write_benchmark_report import write_report


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = (
    "compatibility",
    "query_segmentation",
    "brand_category",
)


def output_slug(model: str, model_id: str) -> str:
    if model == "laya":
        return "laya_typed_decisions"
    if model == "jev":
        return "jev_1_13_0"
    if model == "jevlite":
        return "jevlite"
    if model == "kev":
        lowered = model_id.lower()
        if "kev-4b" in lowered or lowered.endswith("4b"):
            return "kev_4b"
        if "kev-9b" in lowered or lowered.endswith("9b"):
            return "kev_9b"
        return "kev_0_8b"
    return model


def build_adapter(args: argparse.Namespace) -> Any:
    if args.model == "majority":
        return MajorityAdapter()
    if args.model == "laya":
        return LayaAdapter(
            model_id=args.model_id,
            subfolder=args.subfolder,
            device=args.device,
            max_len=args.max_len,
        )
    if args.model == "jev":
        return JevAdapter(
            model_id=args.model_id if args.model_id != "convaiinnovations/laya" else "jev-1.13.0",
            concurrency=args.concurrency,
            budget_usd=args.budget_usd,
        )
    if args.model == "kev":
        return KevAdapter(
            run=args.model_id if args.model_id != "convaiinnovations/laya" else "jaredpalmer/kev-0.8b",
            concurrency=args.concurrency if args.concurrency != 8 else 4,
        )
    if args.model == "jevlite":
        return JevLiteAdapter(
            run=args.model_id if args.model_id != "convaiinnovations/laya" else "vagmi/jev-lite",
            concurrency=1 if args.concurrency == 8 else args.concurrency,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("majority", "laya", "jev", "kev", "jevlite"), default="laya")
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default="typed-decisions")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks: compatibility,query_segmentation,brand_category,esci",
    )
    parser.add_argument("--dataset", choices=("pilot", "reference"), default="reference")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Evaluate only items with complete dual-judge agreement",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    output_dir = args.output_dir
    if output_dir is None:
        slug = output_slug(args.model, args.model_id)
        output_dir = DEFAULT_OUTPUT_DIR / slug
    adapter = build_adapter(args)
    summary = {
        "model": args.model,
        "model_id": getattr(adapter, "cache_model_id", getattr(adapter, "model_id", args.model)),
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "device": args.device,
        "tasks": {},
    }
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("model") == args.model:
            summary["tasks"].update(previous.get("tasks") or {})
    for task in tasks:
        print(f"Running {args.model} on {task}...", flush=True)
        summary["tasks"][task] = run_task(
            adapter=adapter,
            task=task,
            dataset=args.dataset,
            output_dir=output_dir,
            cache=args.cache,
            batch_size=args.batch_size,
            max_items=args.max_items or None,
            require_complete=args.require_complete,
            device=args.device if args.model in {"laya", "kev", "jevlite"} else "cpu",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "BENCHMARK_RESULTS.md"
    write_report(summary, report_path)
    print(json.dumps({"output_dir": str(output_dir), "tasks": list(summary["tasks"])}, indent=2))


if __name__ == "__main__":
    main()
