"""Hand-computed retrieval metrics, self-match exclusion, and baselines."""

import math

import numpy as np
import pytest

from baselines import popularity_positions, popularity_rank, ranks_for_pairs
from evaluate import (
    check_unit_norm,
    multi_target_at_k,
    ndcg_from_binary_rels,
    ranks_for_single_targets,
    single_target_metrics,
    supported_difference,
)


def test_single_target_metrics_match_hand_calculation():
    rank1 = single_target_metrics(1)
    assert rank1["recall_at_10"] == 1.0
    assert rank1["recall_at_50"] == 1.0
    assert rank1["mrr"] == 1.0
    assert rank1["ndcg_at_10"] == 1.0
    rank2 = single_target_metrics(2)
    assert rank2["mrr"] == 0.5
    assert rank2["ndcg_at_10"] == pytest.approx(1.0 / math.log2(3))
    rank60 = single_target_metrics(60)
    assert rank60["recall_at_10"] == 0.0
    assert rank60["recall_at_50"] == 0.0
    assert rank60["mrr_at_100"] == pytest.approx(1.0 / 60)
    assert rank60["ndcg_at_10"] == 0.0
    assert single_target_metrics(0)["mrr"] == 0.0


def test_recall_equals_hit_rate_only_for_one_target():
    neighbors = [0, 1, 2, 3]
    one = multi_target_at_k(neighbors, {1})
    assert one["recall_at_10"] == one["hit_rate_at_10"] == 1.0
    two = multi_target_at_k(neighbors, {1, 5})
    assert two["hit_rate_at_10"] == 1.0
    assert two["recall_at_10"] == pytest.approx(0.5)
    rels = [1, 1, 0]
    assert ndcg_from_binary_rels(rels, 2, 10) == pytest.approx(1.0)


def test_self_match_is_excluded_and_norms_are_checked():
    raw = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, math.sqrt(1.0 - 0.81), 0.0],
            [0.1, 0.0, math.sqrt(1.0 - 0.01)],
            [0.2, 0.0, math.sqrt(1.0 - 0.04)],
        ],
        dtype=np.float32,
    )
    embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert check_unit_norm(embeddings) < 1e-3
    ranks = ranks_for_single_targets(
        embeddings,
        np.array([0, 0, 0]),
        np.array([1, 3, 2]),
    )
    assert ranks.tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        check_unit_norm(np.array([[2.0, 0.0]], dtype=np.float32))


def test_popularity_and_covisitation_toy():
    catalog = ["a", "b", "c", "d"]
    popularity = {"b": 2}
    transitions = {"a": {"b": 2, "c": 1}}
    # Query a, targets b, c, d. Ranks 1, 2, 3 from the traced example.
    ranks = ranks_for_pairs(
        ["a", "a", "a", "b"],
        ["b", "c", "d", "d"],
        catalog,
        popularity,
        transitions,
        "covisitation",
    )
    assert ranks.tolist() == [1, 2, 3, 3]
    pop_ranks = ranks_for_pairs(
        ["a", "a"],
        ["b", "c"],
        catalog,
        popularity,
        transitions,
        "popularity",
    )
    position = popularity_positions(np.array([0, 2, 0, 0]))
    assert popularity_rank(0, 1, position) == pop_ranks[0] == 1
    assert pop_ranks[1] == 2


def test_decision_rule():
    supported = supported_difference([0.02, 0.01, 0.03], 0.005, 0.04)
    assert supported["supported"] and supported["sign"] == "positive"
    too_few_seeds = supported_difference([0.02], 0.005, 0.04)
    assert not too_few_seeds["supported"]
    crosses_zero = supported_difference([0.02, 0.01, 0.03], -0.01, 0.04)
    assert not crosses_zero["supported"]
    mixed = supported_difference([0.02, -0.01, 0.03], 0.001, 0.02)
    assert not mixed["supported"] and mixed["sign"] == "not_agreed"
    negative = supported_difference([-0.02, -0.01, -0.03], -0.05, -0.001)
    assert negative["supported"] and negative["sign"] == "negative"
