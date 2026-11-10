"""
Section 12.6: build the candidate slates that every slate ranker is scored on.

For each requested split and mode, samples up to --max-sessions sessions (seeded) and
writes <work_dir>/slates/<split>_<mode>_n<size>.jsonl. The zero-shot LLM, TALLRec and the
baselines all read the same files, so their numbers are comparable.

It also scores four non-learned baselines on each slate file and writes
<work_dir>/metrics/slate_baselines.json:

    display_order        candidates in the shuffled order shown to the LLM
    random_order         a fresh seeded shuffle
    popularity           training popularity, highest first
    history_similarity   cosine similarity to the mean history embedding

On 'retrieved' slates the history-similarity order is exactly the retriever's order,
and the target is present only when retrieval found it (see target_in_slate_rate).

Examples:
    python build_slates.py --work-dir outputs/tiny --max-sessions 300
    python build_slates.py --work-dir outputs/tiny --splits train --modes retrieved --max-sessions 2000
"""

from __future__ import annotations

import argparse
import json
import logging

from config import SlateConfig, work_paths
from data.amazon_session_loader import PreparedData, load_item_embeddings
from utils.common import run_record, setup_logging, write_json
from utils.slates import (
    SLATE_MODES,
    baseline_rankings,
    build_slates,
    evaluate_slate_rankings,
    slate_summary,
    write_slates,
)

logger = logging.getLogger(__name__)
K_VALUES = [1, 5, 10]


def parse_args() -> argparse.Namespace:
    d = SlateConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--modes", nargs="+", default=list(d.modes), choices=list(SLATE_MODES))
    p.add_argument("--size", type=int, default=d.size)
    p.add_argument("--max-sessions", type=int, default=d.max_sessions)
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    paths = work_paths(args.work_dir).ensure()
    data = PreparedData.from_paths(paths)
    vectors, meta = load_item_embeddings(paths.item_embeddings, data.item_ids)
    popularity = data.train_item_counts()
    popularity_by_id = dict(zip(data.item_ids, popularity.tolist()))

    report = {}
    for split in args.splits:
        frame = data.split(split)
        if frame.height > args.max_sessions:
            frame = frame.sample(n=args.max_sessions, seed=args.seed)
        frame = frame.sort("session_id")
        for mode in args.modes:
            slates = build_slates(
                mode=mode, size=args.size, session_ids=frame["session_id"].to_list(), split=split,
                histories=data.history_indices(frame), targets=data.target_indices(frame),
                item_ids=data.item_ids, item_vectors=vectors, popularity=popularity, seed=args.seed,
            )
            path = paths.slate_file(split, mode, args.size)
            write_slates(path, slates)
            summary = slate_summary(slates, K_VALUES)
            baselines = {name: evaluate_slate_rankings(slates, ranked, K_VALUES)
                         for name, ranked in baseline_rankings(slates, popularity_by_id, args.seed).items()}
            report[f"{split}_{mode}"] = {"file": path.name, **summary, "baselines": baselines}
            logger.info("%s %-9s n=%d target_in_slate=%.3f  history_similarity recall@10=%.3f  popularity recall@10=%.3f",
                        split, mode, len(slates), summary["target_in_slate_rate"],
                        baselines["history_similarity"]["all_sessions"]["recall@10"],
                        baselines["popularity"]["all_sessions"]["recall@10"])
    out_path = paths.metrics_dir / "slate_baselines.json"
    if out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8")).get("slates", {})
        report = {**previous, **report}
    write_json(out_path, run_record("build_slates.py", args, item_encoder=meta["encoder"], slates=report))
    logger.info("Wrote slates to %s", paths.slates_dir)


if __name__ == "__main__":
    main()
