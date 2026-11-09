"""Known-answer and regret-behaviour tests for the bandit policies."""

import numpy as np
import pytest

from bandits import (
    UCB1, BanditStream, BernoulliBandit, EpsilonGreedy, LinUCB, ThompsonBernoulli, UniformRandom, run_policy,
)


def test_ucb1_index_closed_form_and_initial_round_robin():
    pol = UCB1(3)
    assert [pol.select() for _ in range(1)] == [0]
    for arm, r in [(0, 1), (1, 0), (2, 1)]:
        assert pol.select() == arm          # every arm is tried once before the index is used
        pol.update(None, arm, r)
    pol.update(None, 0, 0); pol.update(None, 1, 1)
    # counts [2, 2, 1], sums [1, 1, 1], t = 5
    expected = np.array([0.5, 0.5, 1.0]) + np.sqrt(2 * np.log(5) / np.array([2, 2, 1]))
    np.testing.assert_allclose(pol.index(), expected)


def test_thompson_posterior_counts():
    pol = ThompsonBernoulli(2, np.random.default_rng(0))
    for r in [1, 1, 0, 1]:
        pol.update(None, 0, r)
    pol.update(None, 1, 0)
    np.testing.assert_array_equal(pol.alpha, [4, 1])
    np.testing.assert_array_equal(pol.beta, [2, 2])


def test_epsilon_greedy_extremes():
    rng = np.random.default_rng(1)
    greedy = EpsilonGreedy(3, 0.0, rng)
    greedy.update(None, 0, 0.2); greedy.update(None, 1, 0.9); greedy.update(None, 2, 0.5)
    assert all(greedy.select() == 1 for _ in range(50))
    explore = EpsilonGreedy(4, 1.0, rng)
    explore.update(None, 0, 1.0)
    picks = np.bincount([explore.select() for _ in range(8000)], minlength=4) / 8000
    np.testing.assert_allclose(picks, 0.25, atol=0.02)


def test_linucb_sherman_morrison_matches_ridge_closed_form():
    rng = np.random.default_rng(2)
    d, ridge = 4, 2.0
    pol = LinUCB(2, d, alpha=1.0, ridge=ridge)
    X = rng.normal(size=(60, d))
    r = rng.normal(size=60)
    for x, y in zip(X, r):
        pol.update(x, 0, y)
    A = ridge * np.eye(d) + X.T @ X
    np.testing.assert_allclose(pol.A_inv[0], np.linalg.inv(A), atol=1e-10)
    np.testing.assert_allclose(pol.theta[0], np.linalg.solve(A, X.T @ r), atol=1e-10)
    np.testing.assert_allclose(pol.A_inv[1], np.eye(d) / ridge)        # untouched arm


def _linear_stream(T, rng, K=5, d=4, noise=0.1):
    theta = rng.normal(size=(K, d))
    X = np.hstack([np.ones((T, 1)), rng.normal(size=(T, d - 1))])
    expected = X @ theta.T
    return BanditStream(X, expected, expected + rng.normal(0, noise, expected.shape))


def test_linucb_regret_is_sublinear_in_linear_environment():
    rng = np.random.default_rng(3)
    T = 4000
    stream = _linear_stream(T, rng)
    regret = np.cumsum(run_policy(LinUCB(5, 4, alpha=1.0), stream))
    first, second = regret[T // 2 - 1], regret[-1] - regret[T // 2 - 1]
    assert second < 0.3 * first
    uniform = np.cumsum(run_policy(UniformRandom(5, rng), stream))
    assert regret[-1] < 0.1 * uniform[-1]


def test_ucb1_and_thompson_regret_flatten_while_uniform_grows_linearly():
    env = BernoulliBandit(np.array([0.5, 0.3, 0.25]))
    stream = env.stream(6000, np.random.default_rng(4))
    for pol in (UCB1(3), ThompsonBernoulli(3, np.random.default_rng(5))):
        c = np.cumsum(run_policy(pol, stream))
        assert c[-1] - c[2999] < 0.6 * c[2999]
    u = np.cumsum(run_policy(UniformRandom(3, np.random.default_rng(6)), stream))
    assert u[-1] - u[2999] == pytest.approx(u[2999], rel=0.1)
    # Expected per-round regret of uniform play is mean(best - means).
    assert u[-1] / 6000 == pytest.approx(np.mean(0.5 - env.means), rel=0.1)
