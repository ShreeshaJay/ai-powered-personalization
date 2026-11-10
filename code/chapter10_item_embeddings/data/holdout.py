"""
Session- and user-level hold-out for contrastive fine-tuning (Section 10.6).

finetune_contrastive.py learns from (anchor, positive) pairs, and
evaluate_finetuned.py scores retrieval against the same kind of pairs. The
two must come from disjoint sessions (Amazon KDD) or users (MIND): pairs
drawn from one pool and split afterwards would score the fine-tuned model
on behaviour it was trained on.

A key's side depends only on the key and the seed, so both scripts pick
the same hold-out without sharing state.
"""

import hashlib
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def is_holdout(keys: Iterable, fraction: float, seed: int = 42) -> np.ndarray:
    """Boolean mask, True for keys in the evaluation hold-out."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"hold-out fraction must be in (0, 1), got {fraction}")
    mask = []
    for key in keys:
        digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
        mask.append(int.from_bytes(digest, "big") / 2.0**64 < fraction)
    return np.array(mask, dtype=bool)


def split_amazon_sessions(
    sessions_df: pd.DataFrame, fraction: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(train sessions, hold-out sessions).

    The sessions file has no session id, so a session's key is its content
    (prev_items and next_item). Identical sessions share a key and land on
    the same side.
    """
    keys = sessions_df["prev_items"].astype(str) + "|" + sessions_df["next_item"].astype(str)
    held = is_holdout(keys, fraction, seed)
    return sessions_df[~held], sessions_df[held]


def split_mind_users(
    behaviors_df: pd.DataFrame, fraction: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(train users' impressions, hold-out users' impressions), split by user_id."""
    held = is_holdout(behaviors_df["user_id"].astype(str), fraction, seed)
    return behaviors_df[~held], behaviors_df[held]
