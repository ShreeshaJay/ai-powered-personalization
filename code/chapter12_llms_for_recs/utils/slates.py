"""
Candidate slates shared by every slate ranker in the chapter (LLM zero-shot, TALLRec,
augmented MLP ranker and the simple baselines).

How a slate is built changes what each number means, so the mode is part of every result:

    random     target + uniformly sampled items. Easy: the target is usually the only
               relevant-looking item.
    popular    target + items sampled in proportion to training popularity.
    retrieved  the top-N items by cosine similarity to the mean history embedding. The
               target is NOT inserted; sessions where retrieval missed it cannot be ranked
               correctly by anyone, and the retrieval hit rate is reported alongside.
    injected   the retrieved slate with the target forced into it when retrieval missed.
               Kept only as a diagnostic: the injected target is, by construction, less
               similar to the history than every other candidate, so a ranker that follows
               similarity is pushed below random. Do not report it as a ranking result.

Candidates are stored in a shuffled *display order*; ``similarity_rank`` records where each
candidate falls under the history-similarity retriever, so the similarity baseline can be
scored on exactly the same slate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from utils.faiss_index import InnerProductIndex, l2_normalize
from utils.metrics import aggregate, metrics_from_rank, random_slate_expectation, rank_in_list

SLATE_MODES = ("random", "popular", "retrieved", "injected")


def _sample_excluding(rng: np.random.Generator, num_items: int, k: int, exclude: set,
                      p: Optional[np.ndarray] = None) -> List[int]:
    """Draw k distinct item indices not in ``exclude`` (uniform, or with probabilities p)."""
    chosen: List[int] = []
    seen = set(exclude)
    available = num_items - len(seen)
    if available < k:
        raise ValueError(f"cannot draw {k} negatives from {available} available items")
    while len(chosen) < k:
        draw = rng.choice(num_items, size=max(2 * (k - len(chosen)), 8), replace=True, p=p)
        for idx in draw.tolist():
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
                if len(chosen) == k:
                    break
    return chosen


def build_slates(
    *,
    mode: str,
    size: int,
    session_ids: Sequence[int],
    split: str,
    histories: Sequence[Sequence[int]],
    targets: Sequence[int],
    item_ids: Sequence[str],
    item_vectors: np.ndarray,
    popularity: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[Dict]:
    """Build one slate per session. ``histories`` and ``targets`` are item indices."""
    if mode not in SLATE_MODES:
        raise ValueError(f"mode must be one of {SLATE_MODES}")
    num_items = len(item_ids)
    if size > num_items:
        raise ValueError(f"slate size {size} exceeds catalog size {num_items}")
    rng = np.random.default_rng(seed)
    normed = l2_normalize(item_vectors)
    queries = l2_normalize(np.stack([normed[list(h)].mean(axis=0) for h in histories]))

    retrieved_lists: Optional[np.ndarray] = None
    if mode in ("retrieved", "injected"):
        max_hist = max(len(set(h)) for h in histories)
        index = InnerProductIndex(normed)
        _, retrieved_lists = index.search(queries, min(size + max_hist, num_items))

    pop_p = None
    if mode == "popular":
        if popularity is None:
            raise ValueError("popular mode needs training popularity counts")
        weights = popularity.astype(np.float64)
        if weights.sum() == 0:
            raise ValueError("popularity counts are all zero")
        pop_p = weights / weights.sum()

    slates = []
    for row, (sid, history, target) in enumerate(zip(session_ids, histories, targets)):
        hist_set = set(int(h) for h in history)
        target = int(target)
        if mode in ("retrieved", "injected"):
            ranked = [int(i) for i in retrieved_lists[row] if int(i) not in hist_set][:size]
            if mode == "injected" and target not in ranked:
                ranked[-1] = target
            candidates = ranked
        else:
            if mode == "random":
                negatives = _sample_excluding(rng, num_items, size - 1, hist_set | {target})
            else:
                support = int(np.count_nonzero(pop_p))
                p = pop_p if support - len(hist_set | {target}) >= size - 1 else None
                negatives = _sample_excluding(rng, num_items, size - 1, hist_set | {target}, p=p)
            candidates = negatives + [target]
        display = [candidates[i] for i in rng.permutation(len(candidates))]
        sims = normed[display] @ queries[row]
        sim_order = np.argsort(-sims, kind="stable")
        sim_rank = np.empty(len(display), dtype=np.int64)
        sim_rank[sim_order] = np.arange(1, len(display) + 1)
        slates.append({
            "session_id": int(sid),
            "split": split,
            "mode": mode,
            "history": [item_ids[h] for h in history],
            "target": item_ids[target],
            "candidates": [item_ids[c] for c in display],
            "similarity_rank": sim_rank.tolist(),
            "target_in_slate": target in display,
        })
    return slates


def write_slates(path: Path, slates: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for slate in slates:
            f.write(json.dumps(slate) + "\n")


def read_slates(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_slates.py first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_slate_rankings(slates: Sequence[Dict], rankings: Sequence[Sequence[str]],
                            k_values: Sequence[int]) -> Dict[str, Dict[str, float]]:
    """Score ranked candidate lists against slate targets.

    Returns ``all_sessions`` (a missed retrieval counts as a miss) and ``target_in_slate``
    (only sessions whose slate contains the target, the ranker's own job).
    """
    per_all, per_in = [], []
    for slate, ranked in zip(slates, rankings):
        rank = rank_in_list(ranked, slate["target"]) if slate["target_in_slate"] else math.inf
        m = metrics_from_rank(rank, k_values)
        per_all.append(m)
        if slate["target_in_slate"]:
            per_in.append(m)
    return {"all_sessions": aggregate(per_all), "target_in_slate": aggregate(per_in)}


def baseline_rankings(slates: Sequence[Dict], popularity_by_id: Dict[str, int],
                      seed: int = 42) -> Dict[str, List[List[str]]]:
    """Display order, a seeded random order, popularity order and similarity order per slate."""
    rng = np.random.default_rng(seed)
    out: Dict[str, List[List[str]]] = {"display_order": [], "random_order": [],
                                       "popularity": [], "history_similarity": []}
    for slate in slates:
        cands = slate["candidates"]
        out["display_order"].append(list(cands))
        out["random_order"].append([cands[i] for i in rng.permutation(len(cands))])
        out["popularity"].append(sorted(cands, key=lambda c: -popularity_by_id.get(c, 0)))
        order = np.argsort(slate["similarity_rank"], kind="stable")
        out["history_similarity"].append([cands[i] for i in order])
    return out


def slate_summary(slates: Sequence[Dict], k_values: Sequence[int]) -> Dict:
    sizes = [len(s["candidates"]) for s in slates]
    in_slate = [s["target_in_slate"] for s in slates]
    size = int(np.median(sizes)) if sizes else 0
    return {
        "num_slates": len(slates),
        "mode": slates[0]["mode"] if slates else None,
        "slate_size": size,
        "target_in_slate_rate": float(np.mean(in_slate)) if slates else 0.0,
        "uniform_random_expectation_given_target_in_slate": random_slate_expectation(size, k_values) if size else {},
    }
