"""
Warm-up inputs, scored once at the end of loading and before readiness.

The first XGBoost, PyTorch or FAISS call in a new process pays one-off
initialisation costs; scoring a throwaway request first keeps that cost off
the first user request, which would otherwise hit its stage timeout.
"""

import time
from typing import Callable, Iterable, List

from contracts import RankingCandidate, RankingRequest, RequestContext


def ranking_request(user_features: Iterable[str], candidate_features: Iterable[str], item_ids: List[int],
                    feature_set_version: str) -> RankingRequest:
    candidate_features = list(candidate_features)
    return RankingRequest(
        user_id=0, context=RequestContext(timestamp=0, is_organic=0),
        user_features={name: 0.0 for name in user_features},
        candidates=[RankingCandidate(item_id=int(i), features={name: 0.0 for name in candidate_features})
                    for i in item_ids],
        feature_set_version=feature_set_version)


def timed(call: Callable[[], object]) -> float:
    started = time.perf_counter()
    call()
    return round((time.perf_counter() - started) * 1000.0, 1)
