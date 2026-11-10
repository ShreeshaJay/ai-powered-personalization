"""
Feature tables and training examples for the augmented ranker (Section 12.4).

``load_feature_tables`` aligns every table to the work directory's item order and session
IDs and refuses to continue when:
    * the LLM vectors were embedded by a different encoder than the item vectors, or
    * LLM coverage of the ranker's items or sessions is below the configured minimum
      (unless partial coverage is explicitly allowed; missing rows are then zero vectors
      with a 0 flag the ranker can see).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np
import torch

from data.amazon_session_loader import PreparedData, load_item_embeddings
from utils.faiss_index import InnerProductIndex, mean_history_queries

logger = logging.getLogger(__name__)


def load_feature_tables(paths, data: PreparedData, need_item_llm: bool, need_intent: bool,
                        min_coverage: float, allow_partial: bool) -> Dict:
    content, meta = load_item_embeddings(paths.item_embeddings, data.item_ids)
    tables: Dict = {"content": content, "encoder": meta["encoder"], "coverage": {}}
    session_ids = data.sessions["session_id"].to_list()
    tables["session_row"] = {sid: i for i, sid in enumerate(session_ids)}

    def check(npz, what: str) -> None:
        enc = str(npz["encoder"])
        if enc != meta["encoder"]:
            raise ValueError(f"{what} were embedded with {enc}, item vectors with {meta['encoder']}; "
                             "re-run generate_llm_features.py after encode_items.py")
        tables.setdefault("llm_source", {})[what] = str(npz["llm"]) if "llm" in npz else "unknown"

    def guard(what: str, coverage: float) -> None:
        tables["coverage"][what] = coverage
        if coverage < min_coverage and not allow_partial:
            raise ValueError(f"{what} cover {coverage:.1%} of the ranker's universe (minimum {min_coverage:.0%}). "
                             "Generate the missing rows or pass --allow-partial-coverage.")
        if coverage < 1.0:
            logger.warning("%s coverage %.2f%%; missing rows are zero vectors with a 0 flag", what, 100 * coverage)

    if need_item_llm:
        if not paths.llm_item_embeddings.exists():
            raise FileNotFoundError(f"{paths.llm_item_embeddings} missing; run generate_llm_features.py")
        npz = np.load(paths.llm_item_embeddings, allow_pickle=False)
        check(npz, "LLM item-knowledge vectors")
        vectors = npz["embeddings"]  # NpzFile re-reads the archive on every key access
        table = np.zeros_like(content)
        present = np.zeros(len(content), dtype=np.float32)
        for row, iid in enumerate(npz["item_ids"].tolist()):
            j = data.item_index.get(iid)
            if j is not None:
                table[j] = vectors[row]
                present[j] = 1.0
        tables["llm_item"], tables["llm_item_present"] = table, present
        guard("LLM item-knowledge vectors", float(present.mean()))

    if need_intent:
        if not paths.llm_intent_embeddings.exists():
            raise FileNotFoundError(f"{paths.llm_intent_embeddings} missing; run generate_llm_features.py")
        npz = np.load(paths.llm_intent_embeddings, allow_pickle=False)
        check(npz, "LLM session-intent vectors")
        rows = tables["session_row"]
        vectors = npz["embeddings"]
        table = np.zeros((len(session_ids) + 1, content.shape[1]), dtype=np.float32)
        present = np.zeros(len(session_ids) + 1, dtype=np.float32)
        for r, sid in enumerate(npz["session_ids"].tolist()):
            j = rows.get(int(sid))
            if j is not None:
                table[j] = vectors[r]
                present[j] = 1.0
        tables["intent"], tables["intent_present"] = table, present
        guard("LLM session-intent vectors", float(present[:-1].mean()))
    return tables


def to_tensors(tables: Dict, feature_set_level: int) -> Dict[str, torch.Tensor]:
    out = {"content": torch.from_numpy(tables["content"])}
    if feature_set_level >= 2:
        out["llm_item"] = torch.from_numpy(tables["llm_item"])
        out["llm_item_present"] = torch.from_numpy(tables["llm_item_present"])
    if feature_set_level >= 3:
        out["intent"] = torch.from_numpy(tables["intent"])
        out["intent_present"] = torch.from_numpy(tables["intent_present"])
    return out


def training_examples(data: PreparedData, split: str, content: np.ndarray, session_row: Dict[int, int],
                      max_history: int, num_hard: int, num_random: int, hard_pool: int,
                      seed: int, max_sessions: int | None = None) -> Dict[str, np.ndarray]:
    """One positive (the target) plus hard and random negatives per session.

    Hard negatives are drawn from the items nearest to the mean history embedding, after
    removing history items and the target. Every example keeps its session's row, so the
    train / validation split is by session.
    """
    rng = np.random.default_rng(seed)
    frame = data.split(split)
    if max_sessions is not None and frame.height > max_sessions:
        frame = frame.sample(n=max_sessions, seed=seed)
    histories = data.history_indices(frame)
    targets = data.target_indices(frame)
    rows = [session_row[int(s)] for s in frame["session_id"].to_list()]
    longest = max(len(h) for h in histories)
    _, near = InnerProductIndex(content).search(mean_history_queries(content, histories), hard_pool + longest + 1)

    cand, label, hist_rows, sess = [], [], [], []
    n_items = len(content)
    for i, (h, t) in enumerate(zip(histories, targets.tolist())):
        exclude = set(h) | {t}
        pool = [j for j in near[i].tolist() if j not in exclude][:hard_pool]
        hard = rng.choice(pool, size=min(num_hard, len(pool)), replace=False).tolist() if pool else []
        rand: List[int] = []
        while len(rand) < num_random:
            j = int(rng.integers(n_items))
            if j not in exclude and j not in hard and j not in rand:
                rand.append(j)
        for c, y in [(t, 1.0)] + [(j, 0.0) for j in hard + rand]:
            cand.append(c)
            label.append(y)
            hist_rows.append(i)
            sess.append(rows[i])
    hist, mask = pad_histories(histories, max_history)
    return {"cand": np.asarray(cand, dtype=np.int64), "label": np.asarray(label, dtype=np.float32),
            "hist_row": np.asarray(hist_rows, dtype=np.int64), "session_row": np.asarray(sess, dtype=np.int64),
            "hist": hist, "hist_mask": mask}


def pad_histories(histories: Sequence[Sequence[int]], max_history: int):
    """Right-pad the most recent ``max_history`` items; the last valid slot is the latest item."""
    hist = np.zeros((len(histories), max_history), dtype=np.int64)
    mask = np.zeros((len(histories), max_history), dtype=np.float32)
    for i, h in enumerate(histories):
        h = list(h)[-max_history:]
        hist[i, :len(h)] = h
        mask[i, :len(h)] = 1.0
    return hist, mask
