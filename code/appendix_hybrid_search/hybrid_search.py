"""Command line entry point for the hybrid-search appendix.

Retrieval is exact judged-list scoring (P1) or a depth-limited run (P2).
Lists are not filtered down to a pool after retrieval. Fusion is built for
every SPLADE variant, not only the first one. The device defaults to auto:
CUDA when it is available, otherwise CPU. An explicit CUDA request fails
when CUDA is absent.
"""

from __future__ import annotations

import argparse
import sys

from run_pipeline import main as pipeline_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search with vector databases.")
    parser.add_argument(
        "stages",
        nargs="?",
        default="s1,s2,s5",
        help="Comma-separated stages: s0, s1, s2, s3, s4, s5.",
    )
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--stages", args.stages, *rest]
    pipeline_main()


if __name__ == "__main__":
    main()
