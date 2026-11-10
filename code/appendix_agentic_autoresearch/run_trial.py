"""Train one configuration and write validation predictions.

The harness scores the predictions. This script does not compute them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.train import train_trial


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Appendix C trial")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        train_trial(
            run_dir=args.run_dir,
            trial_dir=args.trial_dir,
            config=config,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
