"""
Synthetic user histories with the Yambda schema, for tests and a no-download CPU run.

Users drift through item "neighbourhoods": the next item is usually close to the
previous one, so a sequential model has something to learn. Not a benchmark.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from data_loader import ACTION_MAP, NUM_ACTIONS, HSTUDataBundle, build_leave_one_out_splits


def make_synthetic_events(num_users: int = 300, num_items: int = 500, mean_len: int = 40,
                          seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    actions = list(ACTION_MAP)
    rows = {"uid": [], "item_id": [], "action_id": [], "seconds": [], "value": [], "event_type": []}
    for uid in range(num_users):
        length = max(5, int(rng.poisson(mean_len)))
        item = int(rng.integers(num_items))
        t = int(rng.integers(0, 86_400 * 30))
        for _ in range(length):
            item = int((item + rng.integers(-3, 4)) % num_items) if rng.random() < 0.8 else int(rng.integers(num_items))
            t += int(rng.exponential(600))
            event = actions[0] if rng.random() < 0.9 else actions[int(rng.integers(1, len(actions)))]
            rows["uid"].append(uid)
            rows["item_id"].append(item + 1000)
            rows["action_id"].append(ACTION_MAP[event])
            rows["seconds"].append(t)
            rows["value"].append(float(rng.random()) if event == "listen" else float("nan"))
            rows["event_type"].append(event)
    return pl.DataFrame(rows)


def load_synthetic_data(max_seq_len: int = 50, seed: int = 42, **kwargs) -> HSTUDataBundle:
    bundle = build_leave_one_out_splits(make_synthetic_events(seed=seed, **kwargs), max_seq_len=max_seq_len,
                                        dataset_name="synthetic")
    bundle.num_actions = NUM_ACTIONS
    return bundle
