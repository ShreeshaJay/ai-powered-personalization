"""
Prepare Amazon KDD Cup 2023 sessions once, for every Chapter 12 script.

Writes, under <work_dir>/data/:
    sessions.parquet   session_id, history (list of item IDs), target, split
    items.parquet      item_id, title, brand, desc, text (the encoder input)
    prepare_meta.json  filter counts and the arguments used

session_id is the session's row number in sessions_train.csv, so every later artifact
(slates, LLM intents, ranker features) can be joined back to the right session.

Examples:
    # Laptop-sized: 3,000 sessions, the items they touch plus 3,000 distractors
    python prepare_data.py --work-dir outputs/tiny --max-sessions 3000 --catalog referenced --extra-random-items 3000

    # Full UK catalog (GPU runs; see CLOUD_RUN.md)
    python prepare_data.py --work-dir outputs/full

    # No download needed: synthetic data with the same schema, for a pipeline check
    python prepare_data.py --work-dir outputs/synthetic --synthetic
"""

from __future__ import annotations

import argparse
import logging
import time

from config import AMAZON_KDD_DATA_PATH, PrepareConfig, work_paths
from data.amazon_session_loader import (
    load_products,
    load_raw_sessions,
    prepare_sessions_and_items,
    write_meta,
)
from data.synthetic import make_synthetic
from utils.common import run_record, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    d = PrepareConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None, help="Output folder (default: CH12_WORK_DIR or outputs/default)")
    p.add_argument("--data-dir", default=str(AMAZON_KDD_DATA_PATH), help="Folder with sessions_train.csv and products_train.csv")
    p.add_argument("--locale", default=d.locale)
    p.add_argument("--min-prev-items", type=int, default=d.min_prev_items)
    p.add_argument("--max-sessions", type=int, default=d.max_sessions)
    p.add_argument("--catalog", choices=["all", "referenced"], default=d.catalog,
                   help="'all' keeps every product in the locale; 'referenced' keeps items in the kept sessions")
    p.add_argument("--extra-random-items", type=int, default=d.extra_random_items,
                   help="With --catalog referenced, add this many unreferenced items as distractors")
    p.add_argument("--desc-max-chars", type=int, default=d.desc_max_chars)
    p.add_argument("--synthetic", action="store_true", help="Use generated data instead of the KDD files")
    p.add_argument("--synthetic-items", type=int, default=600)
    p.add_argument("--synthetic-sessions", type=int, default=3000)
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    paths = work_paths(args.work_dir).ensure()
    start = time.time()

    if args.synthetic:
        sessions, products = make_synthetic(args.synthetic_items, args.synthetic_sessions, seed=args.seed)
        source = "synthetic"
    else:
        logger.info("Reading %s (locale %s)", args.data_dir, args.locale)
        products = load_products(args.data_dir, args.locale)
        sessions = load_raw_sessions(args.data_dir, args.locale)
        source = "amazon_kdd_2023"
    logger.info("Loaded %d sessions and %d products in %.1fs", sessions.height, products.height, time.time() - start)

    sessions, items, meta = prepare_sessions_and_items(
        sessions, products,
        min_prev_items=args.min_prev_items,
        max_sessions=args.max_sessions,
        split_fractions=PrepareConfig().split_fractions,
        catalog=args.catalog,
        extra_random_items=args.extra_random_items,
        desc_max_chars=args.desc_max_chars,
        seed=args.seed,
    )
    sessions.write_parquet(paths.sessions)
    items.write_parquet(paths.items)
    meta["source"] = source
    write_meta(paths.prepare_meta, run_record("prepare_data.py", args, summary=meta))
    for key, value in meta.items():
        logger.info("%s: %s", key, value)
    logger.info("Wrote %s and %s in %.1fs", paths.sessions, paths.items, time.time() - start)


if __name__ == "__main__":
    main()
