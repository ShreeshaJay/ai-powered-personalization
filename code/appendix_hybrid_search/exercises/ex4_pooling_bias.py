"""Exercise 4. Pooling bias on a saved label file.

The label file is not included. Pass a file produced outside this folder.
The script removes products that appear only in the BM25 top 100, then
recomputes judged@10 and condensed nDCG@10. If the file is missing, it
stops without inventing labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import QueryJudgments
from evaluate import macro, score_p2
from fusion import read_run
from llm_labels_convert import labels_for_queries, load_label_file


def without_bm25_only(labels: dict[str, dict[str, str]], runs: dict[str, dict], depth: int = 100) -> dict[str, QueryJudgments]:
    queries = {}
    for query_id, pairs in labels.items():
        bm25_ids = {product_id for product_id, _score in runs["bm25"].get(query_id, [])[:depth]}
        others: set[str] = set()
        for name, run in runs.items():
            if name == "bm25":
                continue
            others.update(product_id for product_id, _score in run.get(query_id, [])[:depth])
        bm25_only = bm25_ids - others
        kept = {product_id: label for product_id, label in pairs.items() if product_id not in bm25_only}
        queries[query_id] = QueryJudgments(query_id, query_id, "test", kept, list(kept))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a label pool with and without BM25-only products.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--prithivi", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--output", default="outputs/exercise4.json")
    args = parser.parse_args()
    label_path = Path(args.labels)
    if not label_path.exists():
        raise SystemExit(f"Label file not found: {label_path.name}. This folder does not include labels.")
    labels = labels_for_queries(load_label_file(label_path))
    runs = {
        "bm25": read_run(Path(args.bm25)),
        "splade_prithivi": read_run(Path(args.prithivi)),
        "dense": read_run(Path(args.dense)),
    }
    gains = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
    original = {}
    for query_id, pairs in labels.items():
        original[query_id] = QueryJudgments(query_id, query_id, "test", pairs, list(pairs))
    before = score_p2(runs["bm25"], original, gains)
    after = score_p2(runs["bm25"], without_bm25_only(labels, runs), gains)
    payload = {
        "before": {"judged_at_10": macro(before, "judged_at_10"), "condensed_ndcg_at_10": macro(before, "condensed_ndcg_at_10")},
        "after_removing_bm25_only": {
            "judged_at_10": macro(after, "judged_at_10"),
            "condensed_ndcg_at_10": macro(after, "condensed_ndcg_at_10"),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
