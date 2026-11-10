import math

import numpy as np
import pytest

from utils.slates import baseline_rankings, build_slates, evaluate_slate_rankings, slate_summary

ITEM_IDS = [f"i{j}" for j in range(30)]


def vectors():
    """Items 0-9 point along x, 10-19 along y, 20-29 along z (with small jitter)."""
    rng = np.random.default_rng(0)
    base = np.repeat(np.eye(3), 10, axis=0)
    return (base + 0.05 * rng.standard_normal(base.shape)).astype(np.float32)


def make(mode, histories, targets, size=5, popularity=None):
    return build_slates(mode=mode, size=size, session_ids=list(range(len(targets))), split="test",
                        histories=histories, targets=targets, item_ids=ITEM_IDS, item_vectors=vectors(),
                        popularity=popularity, seed=0)


@pytest.mark.parametrize("mode", ["random", "popular"])
def test_sampled_slates_contain_target_and_exclude_history(mode):
    pop = np.arange(30, dtype=np.int64)
    slates = make(mode, [[0, 1, 2]] * 50, [25] * 50, popularity=pop)
    for s in slates:
        assert s["target_in_slate"] and s["target"] in s["candidates"]
        assert len(s["candidates"]) == 5 == len(set(s["candidates"]))
        assert not set(s["candidates"]) & {"i0", "i1", "i2"}
    positions = [s["candidates"].index("i25") for s in slates]
    assert len(set(positions)) > 1  # display order is shuffled, not "target last"


def test_retrieved_slates_never_inject_the_target():
    slates = make("retrieved", [[0, 1, 2]], [25])
    s = slates[0]
    assert not s["target_in_slate"]
    assert "i25" not in s["candidates"]
    assert all(int(c[1:]) < 10 for c in s["candidates"])  # nearest items share the history's direction
    assert not set(s["candidates"]) & {"i0", "i1", "i2"}


def test_injected_mode_forces_the_target_in():
    s = make("injected", [[0, 1, 2]], [25])[0]
    assert s["target_in_slate"] and "i25" in s["candidates"]
    # the injected target is the least similar candidate, which is why this mode is only a diagnostic
    assert s["similarity_rank"][s["candidates"].index("i25")] == len(s["candidates"])


def test_similarity_rank_is_a_permutation():
    for s in make("random", [[0, 1, 2]] * 5, [5] * 5):
        assert sorted(s["similarity_rank"]) == list(range(1, 6))


def test_slates_are_deterministic():
    a = make("random", [[0, 1, 2]] * 5, [5] * 5)
    b = make("random", [[0, 1, 2]] * 5, [5] * 5)
    assert a == b


def test_evaluation_separates_missed_retrieval():
    slates = [
        {"target": "a", "candidates": ["a", "b"], "target_in_slate": True, "similarity_rank": [1, 2]},
        {"target": "z", "candidates": ["a", "b"], "target_in_slate": False, "similarity_rank": [1, 2]},
    ]
    out = evaluate_slate_rankings(slates, [["a", "b"], ["b", "a"]], [1])
    assert out["all_sessions"]["recall@1"] == pytest.approx(0.5)
    assert out["target_in_slate"]["recall@1"] == pytest.approx(1.0)
    assert out["target_in_slate"]["num_queries"] == 1


def test_baselines_use_the_same_candidates():
    slates = make("random", [[0, 1, 2]] * 3, [5] * 3)
    pop = {iid: j for j, iid in enumerate(ITEM_IDS)}
    ranked = baseline_rankings(slates, pop, seed=0)
    for name, lists in ranked.items():
        for s, r in zip(slates, lists):
            assert sorted(r) == sorted(s["candidates"]), name
    for s, r in zip(slates, ranked["popularity"]):
        assert [pop[c] for c in r] == sorted((pop[c] for c in r), reverse=True)


def test_summary_reports_random_expectation():
    slates = make("random", [[0, 1, 2]] * 4, [5] * 4, size=10)
    summary = slate_summary(slates, [5])
    assert summary["target_in_slate_rate"] == 1.0
    assert summary["uniform_random_expectation_given_target_in_slate"]["recall@5"] == pytest.approx(0.5)
    assert math.isclose(summary["slate_size"], 10)
