"""Zip the files Colab needs to evaluate JevLite and Kev-4B."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmark" / "colab_bundle.zip"

CODE_FILES = (
    "adjudicate_pilots.py",
    "run_benchmark.py",
    "write_benchmark_report.py",
    "write_comparison_report.py",
    "pack_colab_bundle.py",
    "colab/run_open_models.py",
    "adapters/__init__.py",
    "adapters/base.py",
    "adapters/jev.py",
    "adapters/jevlite.py",
    "adapters/kev.py",
    "adapters/laya.py",
    "adapters/majority.py",
    "adapters/schemas.py",
    "eval/__init__.py",
    "eval/datasets.py",
    "eval/metrics.py",
    "eval/runner.py",
    "eval/esci_slice.py",
)

DATA_GLOBS = (
    "references/compatibility_reference_2000.jsonl",
    "references/query_segmentation_reference_2000.jsonl",
    "references/brand_category_reference_2000.jsonl",
    "references/esci_eval_slice.jsonl",
    "references/consensus/compatibility_consensus.jsonl",
    "references/consensus/query_segmentation_consensus.jsonl",
    "references/consensus/brand_category_consensus.jsonl",
    "outputs/benchmark/majority/summary.json",
    "outputs/benchmark/laya_typed_decisions/summary.json",
    "outputs/benchmark/jev_1_13_0/summary.json",
    "outputs/benchmark/kev_0_8b/summary.json",
)


def collect_files() -> list[Path]:
    files: list[Path] = []
    for relative in (*CODE_FILES, *DATA_GLOBS):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundle file: {relative}")
        files.append(path)
    return files


def write_bundle(output: Path) -> Path:
    files = collect_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_bundle(args.output)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Wrote {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
