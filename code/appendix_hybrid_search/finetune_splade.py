"""Optional Exercise 3 fine-tune of the Prithivi SPLADE checkpoint.

One epoch, batch 32, learning rate 2e-5, SpladeLoss around
SparseMultipleNegativesRankingLoss, query regularizer 5e-5, document
regularizer 3e-5, no-duplicates sampler, warmup_steps 0.1. fp16 is used
only when the device is CUDA. The training columns are anchor and positive,
with router mapping anchor to query and positive to document.

A caller may set max_steps to run a short check. The full exercise leaves
max_steps unset and trains for one epoch on the reserved pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_plan, resolve_device
from run_retrieval import write_manifest
from splade_search import load_sparse_encoder, set_max_length


def train_one_seed(
    pairs: list[tuple[str, str]],
    output_dir: Path,
    seed: int,
    device: str,
    max_steps: int | None = None,
) -> dict:
    spec = load_plan()["exercise3"]
    base = load_plan()["models"][spec["base"]]
    from datasets import Dataset
    from sentence_transformers.sparse_encoder import (
        SparseEncoderTrainer,
        SparseEncoderTrainingArguments,
    )
    from sentence_transformers.sparse_encoder.losses import (
        SparseMultipleNegativesRankingLoss,
        SpladeLoss,
    )
    from sentence_transformers.base.sampler import BatchSamplers

    model = load_sparse_encoder(base["repo"], base["revision"], device)
    set_max_length(model, int(base["document_max_length"]))
    dataset = Dataset.from_dict(
        {
            "anchor": [query for query, _product in pairs],
            "positive": [product for _query, product in pairs],
        }
    )
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model),
        query_regularizer_weight=float(spec["query_regularizer_weight"]),
        document_regularizer_weight=float(spec["document_regularizer_weight"]),
    )
    use_fp16 = bool(spec["fp16_on_cuda"]) and device == "cuda"
    arguments = {
        "output_dir": str(output_dir / f"seed_{seed}"),
        "num_train_epochs": int(spec["epochs"]),
        "per_device_train_batch_size": int(spec["batch_size"]) if max_steps is None else min(2, len(pairs)),
        "learning_rate": float(spec["learning_rate"]),
        "warmup_steps": float(spec["warmup_steps"]),
        "fp16": use_fp16,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "router_mapping": {"anchor": "query", "positive": "document"},
        "seed": int(seed),
        "save_strategy": "no",
        "report_to": "none",
        "logging_steps": 1,
    }
    if max_steps is not None:
        arguments["max_steps"] = int(max_steps)
        arguments["per_device_train_batch_size"] = min(2, len(pairs))
    router_used = True
    try:
        trainer = SparseEncoderTrainer(
            model=model,
            args=SparseEncoderTrainingArguments(**arguments),
            train_dataset=dataset,
            loss=loss,
        )
        trainer.train()
    except Exception as exc:
        if "router" not in str(exc).lower():
            raise
        router_used = False
        arguments["router_mapping"] = None
        trainer = SparseEncoderTrainer(
            model=model,
            args=SparseEncoderTrainingArguments(**arguments),
            train_dataset=dataset,
            loss=loss,
        )
        trainer.train()
    model.save_pretrained(str(output_dir / f"seed_{seed}" / "model"))
    record = {
        "seed": seed,
        "n_pairs": len(pairs),
        "max_steps": max_steps,
        "fp16": use_fp16,
        "router_mapping_used": router_used,
        "base_repo": base["repo"],
        "base_revision": base["revision"],
        "query_regularizer_weight": spec["query_regularizer_weight"],
        "document_regularizer_weight": spec["document_regularizer_weight"],
        "warmup_steps": spec["warmup_steps"],
        "batch_sampler": "no_duplicates",
        "device": device,
    }
    write_manifest(output_dir / f"seed_{seed}" / "train_manifest.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Prithivi SPLADE on reserved pairs.")
    parser.add_argument("--pairs", required=True, help="JSON list of {query, positive} objects.")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    device = resolve_device(args.device)
    raw = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    pairs = [(row["query"], row["positive"]) for row in raw]
    seeds = [args.seed] if args.seed is not None else list(load_plan()["exercise3"]["seeds"])
    output = Path(args.output_dir)
    for seed in seeds:
        record = train_one_seed(pairs, output, int(seed), device, args.max_steps)
        print(json.dumps({"seed": record["seed"], "router_mapping_used": record["router_mapping_used"]}))


if __name__ == "__main__":
    main()
