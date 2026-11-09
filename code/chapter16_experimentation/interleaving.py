"""
Chapter 16: Experimentation and Bandits - interleaving (Section 16.7)
=====================================================================
Team-draft interleaving (Radlinski, Kurup, and Joachims, 2008), in a readable
single-impression form and a vectorized form for simulating many sessions,
plus balanced interleaving for comparison.

Team-draft, as described in Section 16.7.2:
    1. Each round, a coin flip decides which ranker picks first.
    2. Each ranker adds its highest-ranked item not already chosen, and the
       item is tagged with that ranker's team.
    3. Clicks are credited to the team that contributed the clicked item.
    4. The impression is a win for the team with more clicks, else a tie.
    5. Wins are aggregated across impressions with a binomial (sign) test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats

TEAM_A = 0
TEAM_B = 1


@dataclass
class TeamDraftResult:
    items: List
    teams: List[int]
    first_picker: List[int] = field(default_factory=list)  # one entry per round


def team_draft_interleave(
    ranking_a: Sequence,
    ranking_b: Sequence,
    k: int,
    rng: Optional[np.random.Generator] = None,
    coins: Optional[Sequence[int]] = None,
) -> TeamDraftResult:
    """Interleave two rankings into one list of length k by team draft.

    `coins[r]` (0 = A first, 1 = B first) can be supplied to make the result
    deterministic; otherwise a fair coin is drawn from `rng` each round.
    """
    if coins is None and rng is None:
        raise ValueError("Provide rng or coins")
    items: List = []
    teams: List[int] = []
    firsts: List[int] = []
    chosen = set()
    pa = pb = 0
    rnd = 0
    max_len = min(k, len(set(ranking_a) | set(ranking_b)))
    while len(items) < max_len:
        first = int(coins[rnd]) if coins is not None else int(rng.integers(2))
        firsts.append(first)
        for team in (first, 1 - first):
            if len(items) >= max_len:
                break
            ranking = ranking_a if team == TEAM_A else ranking_b
            ptr = pa if team == TEAM_A else pb
            while ptr < len(ranking) and ranking[ptr] in chosen:
                ptr += 1
            if ptr < len(ranking):
                items.append(ranking[ptr])
                teams.append(team)
                chosen.add(ranking[ptr])
                ptr += 1
            if team == TEAM_A:
                pa = ptr
            else:
                pb = ptr
        rnd += 1
    return TeamDraftResult(items, teams, firsts)


def team_draft_outcome(teams: Sequence[int], clicked: Sequence[bool]) -> int:
    """+1 if team A earned more clicks, -1 if team B did, 0 for a tie."""
    teams = np.asarray(teams)
    clicked = np.asarray(clicked, dtype=bool)
    a = int(np.sum(clicked & (teams == TEAM_A)))
    b = int(np.sum(clicked & (teams == TEAM_B)))
    return int(np.sign(a - b))


def team_draft_batch(top_a: np.ndarray, top_b: np.ndarray, k: int, coins: np.ndarray):
    """Vectorized team draft over many sessions.

    Parameters
    ----------
    top_a, top_b : int arrays (n_sessions, >= k)
        Each ranker's ranked candidate indices, best first. Since at most k
        items are chosen in total, each ranker's pick always lies in its own top k.
    coins : int array (n_sessions, ceil(k / 2))
        0 means A picks first in that round, 1 means B picks first.

    Returns
    -------
    items : (n_sessions, k) candidate indices of the interleaved list
    teams : (n_sessions, k) TEAM_A or TEAM_B for each position
    """
    n = top_a.shape[0]
    top_a = top_a[:, :k]
    top_b = top_b[:, :k]
    rows = np.arange(n)
    items = np.full((n, k), -1, dtype=np.int64)
    teams = np.full((n, k), -1, dtype=np.int8)
    ptr = np.zeros((2, n), dtype=np.int64)
    tops = (top_a, top_b)
    pos = 0
    rnd = 0
    while pos < k:
        first = coins[:, rnd].astype(np.int64)
        for slot in range(2):
            if pos >= k:
                break
            team = first if slot == 0 else 1 - first
            # Advance each session's pointer for its picking team past chosen items.
            for t in (TEAM_A, TEAM_B):
                sel = team == t
                if not sel.any():
                    continue
                r = rows[sel]
                p = ptr[t, sel]
                top = tops[t]
                while True:
                    cand = top[r, np.minimum(p, k - 1)]
                    already = (items[r, :pos] == cand[:, None]).any(axis=1) & (p < k)
                    if not already.any():
                        break
                    p = p + already
                items[r, pos] = cand
                teams[r, pos] = t
                ptr[t, sel] = p + 1
            pos += 1
        rnd += 1
    return items, teams


def balanced_interleave(ranking_a: Sequence, ranking_b: Sequence, k: int, a_first: bool):
    """Balanced interleaving (Joachims, 2002): alternate with a fixed priority ranker."""
    items: List = []
    ka = kb = 0
    while len(items) < k and (ka < len(ranking_a) or kb < len(ranking_b)):
        take_a = (ka < kb) or (ka == kb and a_first)
        if kb >= len(ranking_b):
            take_a = True
        elif ka >= len(ranking_a):
            take_a = False
        if take_a:
            if ranking_a[ka] not in items:
                items.append(ranking_a[ka])
            ka += 1
        else:
            if ranking_b[kb] not in items:
                items.append(ranking_b[kb])
            kb += 1
    return items


def balanced_outcome(ranking_a: Sequence, ranking_b: Sequence, interleaved: Sequence, clicked: Sequence[bool]) -> int:
    """Balanced-interleaving credit (Joachims, 2002).

    Let c be the lowest clicked item in the interleaved list and d the smaller
    of its ranks in A and B. Each ranker is credited with the clicks that lie
    in its own top d; the ranker with more credited clicks wins.
    """
    clicked_items = [it for it, c in zip(interleaved, clicked) if c]
    if not clicked_items:
        return 0
    rank_a = {it: i for i, it in enumerate(ranking_a)}
    rank_b = {it: i for i, it in enumerate(ranking_b)}
    inf = len(ranking_a) + len(ranking_b)
    lowest = clicked_items[-1]
    depth = min(rank_a.get(lowest, inf), rank_b.get(lowest, inf)) + 1
    ha = sum(1 for c in clicked_items if rank_a.get(c, inf) < depth)
    hb = sum(1 for c in clicked_items if rank_b.get(c, inf) < depth)
    return int(np.sign(ha - hb))


def interleaving_sign_test(outcomes: np.ndarray, alpha: float = 0.05):
    """Binomial test of wins for A among non-tied impressions.

    `scipy.stats.binomtest` tests whether a count of successes is consistent
    with a success probability of 0.5. Returns (wins_a, wins_b, p_value, delta),
    where delta = (wins_a - wins_b) / (wins_a + wins_b).
    """
    outcomes = np.asarray(outcomes)
    wa = int(np.sum(outcomes > 0))
    wb = int(np.sum(outcomes < 0))
    if wa + wb == 0:
        return wa, wb, 1.0, 0.0
    p = stats.binomtest(wa, wa + wb, 0.5).pvalue
    return wa, wb, float(p), (wa - wb) / (wa + wb)
