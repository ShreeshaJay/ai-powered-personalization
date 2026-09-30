"""Build a stratified ESCI evaluation slice from existing labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.esci_slice import DEFAULT_OUTPUT, build_esci_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-per-label", type=int, default=2000)
    parser.add_argument("--synthetic-per-label", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_esci_slice(
        human_per_label=args.human_per_label,
        synthetic_per_label=args.synthetic_per_label,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
