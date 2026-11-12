"""Exercise 1. Build a fixed slice of test queries and distractor products.

The default sizes are 1,000 test queries and 30,000 distractors, seed
20261110. Pass smaller sizes to try the same function on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_plan, resolve_data_dir
from data_loader import load_esci
from splits import exercise_slice, load_prefix_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Exercise 1 product and query slice.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="outputs/splits/exercise_slice.json")
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--distractors", type=int, default=None)
    parser.add_argument("--prefix-source", default="tasksource/esci")
    args = parser.parse_args()
    plan = load_plan()
    spec = plan["splits"]["exercise1_slice"]
    data = load_esci(resolve_data_dir(args.data_dir))
    prefix, _source = load_prefix_texts(
        args.prefix_source,
        revision=plan["splits"]["qdrant_prefix"]["revision"],
    )
    info = exercise_slice(
        data,
        n_test_queries=args.queries or int(spec["test_queries"]),
        n_distractors=args.distractors or int(spec["distractor_products"]),
        seed=int(spec["seed"]),
        prefix_texts=prefix,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Queries: {len(info['query_ids'])}. Products: {len(info['product_ids'])}.")


if __name__ == "__main__":
    main()
