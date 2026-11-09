"""Known-answer tests for team-draft and balanced interleaving."""

import numpy as np
import pytest
from scipy import stats

from interleaving import (
    TEAM_A, TEAM_B, balanced_interleave, balanced_outcome, interleaving_sign_test,
    team_draft_batch, team_draft_interleave, team_draft_outcome,
)


def test_team_draft_hand_example():
    # Round 1: A first -> A takes 1, B takes 2. Round 2: B first -> B takes 4, A takes 3.
    res = team_draft_interleave([1, 2, 3, 4], [2, 1, 4, 3], k=4, coins=[0, 1])
    assert res.items == [1, 2, 4, 3]
    assert res.teams == [TEAM_A, TEAM_B, TEAM_B, TEAM_A]


def test_team_draft_skips_items_already_chosen():
    # Both rank item 7 first; whoever picks second must take its next unchosen item.
    res = team_draft_interleave([7, 8, 9], [7, 9, 8], k=3, coins=[1, 0])
    assert res.items == [7, 8, 9]
    assert res.teams == [TEAM_B, TEAM_A, TEAM_A]


def test_batch_matches_reference_implementation():
    rng = np.random.default_rng(0)
    n, m, k = 300, 15, 6
    top_a = np.argsort(rng.random((n, m)), axis=1)
    top_b = np.argsort(rng.random((n, m)), axis=1)
    top_b[: n // 3] = top_a[: n // 3]                   # include identical rankings
    coins = rng.integers(0, 2, (n, (k + 1) // 2))
    items, teams = team_draft_batch(top_a, top_b, k, coins)
    for s in range(n):
        ref = team_draft_interleave(list(top_a[s]), list(top_b[s]), k, coins=coins[s])
        assert list(items[s]) == ref.items
        assert list(teams[s]) == ref.teams
        assert len(set(items[s])) == k


def test_identical_rankers_give_balanced_teams_and_even_wins():
    rng = np.random.default_rng(1)
    n, m, k = 20_000, 20, 10
    ranking = np.argsort(rng.random((n, m)), axis=1)
    items, teams = team_draft_batch(ranking, ranking.copy(), k, rng.integers(0, 2, (n, k // 2)))
    np.testing.assert_array_equal((teams == TEAM_A).sum(axis=1), k // 2)   # exactly balanced at even k
    clicks = rng.random((n, k)) < 0.3 / np.arange(1, k + 1)
    outcome = np.sign((clicks & (teams == TEAM_A)).sum(1) - (clicks & (teams == TEAM_B)).sum(1))
    wa, wb, p, _ = interleaving_sign_test(outcome)
    assert wa / (wa + wb) == pytest.approx(0.5, abs=0.02)


def test_team_draft_is_fair_when_clicks_depend_only_on_position():
    # Different rankers, but users click by position alone: the coin flips must
    # give each team every position equally often, so neither team should win.
    rng = np.random.default_rng(2)
    n, m, k = 40_000, 30, 10
    top_a = np.argsort(rng.random((n, m)), axis=1)
    top_b = np.argsort(rng.random((n, m)), axis=1)
    _, teams = team_draft_batch(top_a, top_b, k, rng.integers(0, 2, (n, (k + 1) // 2)))
    clicks = rng.random((n, k)) < 0.5 / np.arange(1, k + 1)
    outcome = np.sign((clicks & (teams == TEAM_A)).sum(1) - (clicks & (teams == TEAM_B)).sum(1))
    wa, wb, p, _ = interleaving_sign_test(outcome)
    assert wa / (wa + wb) == pytest.approx(0.5, abs=0.015)
    np.testing.assert_allclose((teams == TEAM_A).mean(axis=0), 0.5, atol=0.015)


def test_team_draft_outcome():
    teams = [TEAM_A, TEAM_B, TEAM_A, TEAM_B]
    assert team_draft_outcome(teams, [1, 0, 1, 1]) == 1
    assert team_draft_outcome(teams, [0, 1, 0, 0]) == -1
    assert team_draft_outcome(teams, [1, 1, 0, 0]) == 0
    assert team_draft_outcome(teams, [0, 0, 0, 0]) == 0


def test_balanced_interleaving_and_credit():
    a, b = ["a", "b", "c", "d"], ["b", "e", "a", "f"]
    shown = balanced_interleave(a, b, k=4, a_first=True)
    assert shown == ["a", "b", "e", "c"]
    # Lowest click is "e": rank 2 in B, absent from A -> compare top-2 prefixes.
    assert balanced_outcome(a, b, shown, [False, False, True, False]) == -1
    assert balanced_outcome(a, b, shown, [True, False, False, False]) == 1
    # "b" is rank 1 in B and rank 2 in A, so the depth is 1 and only B is credited.
    assert balanced_outcome(a, b, shown, [False, True, False, False]) == -1
    assert balanced_outcome(a, b, shown, [True, True, False, False]) == 0


def test_balanced_interleaving_continues_when_one_list_runs_out():
    assert balanced_interleave([1, 2, 3, 4, 5], [6], k=5, a_first=False) == [6, 1, 2, 3, 4]
    assert balanced_interleave([6], [1, 2, 3, 4, 5], k=5, a_first=True) == [6, 1, 2, 3, 4]


def test_sign_test_matches_binomtest():
    outcome = np.r_[np.ones(60), -np.ones(40), np.zeros(25)]
    wa, wb, p, delta = interleaving_sign_test(outcome)
    assert (wa, wb) == (60, 40)
    assert p == pytest.approx(stats.binomtest(60, 100, 0.5).pvalue)
    assert delta == pytest.approx(0.2)
