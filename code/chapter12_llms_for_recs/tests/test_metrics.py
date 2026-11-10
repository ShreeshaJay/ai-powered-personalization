import math

import numpy as np
import pytest

from utils.metrics import aggregate, metrics_from_rank, random_slate_expectation, rank_in_list, rank_of_target


def test_metrics_at_known_ranks():
    first = metrics_from_rank(1, [1, 5])
    assert first == {"recall@1": 1.0, "ndcg@1": 1.0, "recall@5": 1.0, "ndcg@5": 1.0, "mrr": 1.0}
    third = metrics_from_rank(3, [1, 5])
    assert third["recall@1"] == 0.0 and third["ndcg@1"] == 0.0
    assert third["recall@5"] == 1.0
    assert third["ndcg@5"] == pytest.approx(0.5)  # 1 / log2(4)
    assert third["mrr"] == pytest.approx(1 / 3)
    missing = metrics_from_rank(math.inf, [10])
    assert missing == {"recall@10": 0.0, "ndcg@10": 0.0, "mrr": 0.0}


def test_aggregate_means_and_counts():
    out = aggregate([metrics_from_rank(1, [1]), metrics_from_rank(math.inf, [1])])
    assert out["recall@1"] == 0.5 and out["mrr"] == 0.5 and out["num_queries"] == 2


def test_rank_helpers():
    assert rank_in_list(["a", "b", "c"], "c") == 3
    assert rank_in_list(["a"], "z") == math.inf
    scores = np.array([0.1, 0.9, 0.5, 0.9])
    assert rank_of_target(scores, 2) == 3
    # ties count against the target: index 3 ties with index 1, which comes first
    assert rank_of_target(scores, 3) == 2
    assert rank_of_target(scores, 1) == 1


def test_random_slate_expectation():
    exp = random_slate_expectation(20, [10, 20])
    assert exp["recall@10"] == pytest.approx(0.5)
    assert exp["recall@20"] == pytest.approx(1.0)
    harmonic = sum(1 / r for r in range(1, 21))
    assert exp["mrr"] == pytest.approx(harmonic / 20)


def test_random_expectation_matches_simulation():
    rng = np.random.default_rng(0)
    ranks = rng.integers(1, 21, size=20_000)
    sim = aggregate([metrics_from_rank(float(r), [10]) for r in ranks])
    exp = random_slate_expectation(20, [10])
    assert sim["recall@10"] == pytest.approx(exp["recall@10"], abs=0.01)
    assert sim["ndcg@10"] == pytest.approx(exp["ndcg@10"], abs=0.01)
