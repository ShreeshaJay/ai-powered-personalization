"""Read-only local hardware preflight for System One benchmark models."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


MODEL_PROFILES = [
    {
        "model": "Laya / laya-typed-decisions",
        "configuration": "bf16 or fp16",
        "estimated_peak_vram_gb": "2-4",
        "initial_batch": 8,
        "status_at_8gb": "expected_to_fit",
    },
    {
        "model": "Kev 0.8B / Simple-Jev 0.8B",
        "configuration": "bf16 or 8-bit",
        "estimated_peak_vram_gb": "3-5",
        "initial_batch": 4,
        "status_at_8gb": "expected_to_fit",
    },
    {
        "model": "SemIf or Kev 4B",
        "configuration": "4-bit quantized",
        "estimated_peak_vram_gb": "5-8",
        "initial_batch": 1,
        "status_at_8gb": "borderline",
    },
    {
        "model": "JevLite (Gemma 4 E4B adapter)",
        "configuration": "exact trained 4-bit NF4 path",
        "estimated_peak_vram_gb": "6-8+",
        "initial_batch": 1,
        "status_at_8gb": "borderline",
    },
    {
        "model": "Nimble 9B / Kev 9B",
        "configuration": "4-bit with possible CPU offload",
        "estimated_peak_vram_gb": "9-12+",
        "initial_batch": 1,
        "status_at_8gb": "not_recommended_locally",
    },
    {
        "model": "Decider 35B-A3B",
        "configuration": "bf16",
        "estimated_peak_vram_gb": "65+",
        "initial_batch": 1,
        "status_at_8gb": "requires_colab_or_other_large_gpu",
    },
]


def query_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=20
    )
    first_gpu = completed.stdout.strip().splitlines()[0]
    name, total_mb, free_mb, driver = [part.strip() for part in first_gpu.split(",")]
    return {
        "name": name,
        "memory_total_mb": int(total_mb),
        "memory_free_mb": int(free_mb),
        "driver_version": driver,
    }


def build_report() -> dict[str, Any]:
    try:
        gpu = query_nvidia_smi()
        error = None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        gpu = None
        error = str(exc)

    return {
        "gpu": gpu,
        "gpu_probe_error": error,
        "models": MODEL_PROFILES,
        "local_policy": {
            "sequence": [
                "Run Laya at batch 8.",
                "Measure peak allocated and reserved VRAM after warm-up.",
                "Try batch 16 only if at least 2 GB remains free.",
                "Run 4B and JevLite models quantized at batch 1.",
                "Stop on CUDA OOM, OS instability, sustained thermal throttling, or swap pressure.",
                "Move only the failing workload to Colab.",
            ],
            "note": (
                "VRAM ranges are planning heuristics. Actual use depends on sequence "
                "length, attention implementation, quantization backend, and driver."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    print(json.dumps(report, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
