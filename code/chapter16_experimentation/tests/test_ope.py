"""Known-answer tests for the off-policy estimators."""

import numpy as np
import pytest

import ope

# A four-event log small enough to check by hand. Target always plays action 0.
R = np.array([1.0, 0.0, 1.0, 1.0])
A = np.array([0, 1, 0, 1])
P = np.array([0.5, 0.5, 0.8, 0.2])
TARGET = np.tile([1.0, 0.0], (4, 1))
Q = np.tile([0.6, 0.4], (4, 1))


def test_hand_computed_values():
    # w = [2, 0, 1.25, 0]
    assert ope.ips(R, A, P, TARGET) == pytest.approx(3.25 / 4)
    assert ope.snips(R, A, P, TARGET) == pytest.approx(1.0)
    assert ope.clipped_ips(R, A, P, TARGET, max_weight=1.5) == pytest.approx(2.75 / 4)
    assert ope.direct_method(R, A, P, TARGET, Q) == pytest.approx(0.6)
    # residuals r - q(a) = [0.4, -0.4, 0.4, 0.6]; weighted = [0.8, 0, 0.5, 0]
    assert ope.doubly_robust(R, A, P, TARGET, Q) == pytest.approx(0.6 + 1.3 / 4)
    assert ope.replay(R, A, P, TARGET) == pytest.approx(1.0)


def _log(n, rng, logger=(0.5, 0.3, 0.2), means=(0.2, 0.5, 0.1)):
    logger, means = np.array(logger), np.array(means)
    a = rng.choice(3, size=n, p=logger)
    r = (rng.random(n) < means[a]).astype(float)
    return r, a, logger[a], logger, means


def test_ips_equals_on_policy_mean_when_target_is_logger():
    rng = np.random.default_rng(0)
    r, a, p, logger, _ = _log(1000, rng)
    target = np.tile(logger, (1000, 1))
    assert ope.ips(r, a, p, target) == pytest.approx(r.mean())
    assert ope.snips(r, a, p, target) == pytest.approx(r.mean())


def test_dr_equals_dm_when_reward_model_is_exact():
    rng = np.random.default_rng(1)
    n = 500
    q = rng.random((n, 3))
    a = rng.integers(0, 3, n)
    r = q[np.arange(n), a]                     # deterministic rewards equal to the model
    target = rng.dirichlet(np.ones(3), n)
    p = np.full(n, 1 / 3)
    assert ope.doubly_robust(r, a, p, target, q) == pytest.approx(ope.direct_method(r, a, p, target, q))


def test_ips_snips_dr_unbiased_against_closed_form_value():
    rng = np.random.default_rng(2)
    target = np.tile([0.0, 1.0, 0.0], (400, 1))     # always action 1: true value 0.5
    wrong_q = np.tile([0.9, 0.1, 0.9], (400, 1))    # badly wrong reward model
    est = {"ips": [], "snips": [], "dr": [], "dm": []}
    for _ in range(3000):
        r, a, p, _, _ = _log(400, rng)
        est["ips"].append(ope.ips(r, a, p, target))
        est["snips"].append(ope.snips(r, a, p, target))
        est["dr"].append(ope.doubly_robust(r, a, p, target, wrong_q))
        est["dm"].append(ope.direct_method(r, a, p, target, wrong_q))
    for k in ("ips", "snips", "dr"):
        v = np.array(est[k])
        assert abs(v.mean() - 0.5) < 4 * v.std() / np.sqrt(v.size), k
    # DR is protected by correct propensities; DM inherits the model's error.
    assert np.mean(est["dm"]) == pytest.approx(0.1)


def test_clipping_above_max_weight_is_a_no_op():
    assert ope.clipped_ips(R, A, P, TARGET, max_weight=100) == pytest.approx(ope.ips(R, A, P, TARGET))


def test_replay_equals_snips_under_uniform_logging():
    rng = np.random.default_rng(3)
    n, K = 2000, 4
    a = rng.integers(0, K, n)
    r = rng.random(n)
    target = ope.one_hot(rng.integers(0, K, n), K)
    p = np.full(n, 1 / K)
    assert ope.replay(r, a, p, target) == pytest.approx(ope.snips(r, a, p, target))


def test_epsilon_greedy_probs_and_sampling():
    greedy = np.array([2, 0, 1])
    probs = ope.epsilon_greedy_probs(greedy, 4, 0.2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 2] == pytest.approx(0.8 + 0.05) and probs[0, 0] == pytest.approx(0.05)
    rng = np.random.default_rng(4)
    draws = ope.sample_actions(np.tile([0.1, 0.2, 0.3, 0.4], (40_000, 1)), rng)
    np.testing.assert_allclose(np.bincount(draws, minlength=4) / 40_000, [0.1, 0.2, 0.3, 0.4], atol=0.01)


def test_ridge_per_arm_recovers_linear_rewards():
    rng = np.random.default_rng(5)
    X = np.hstack([np.ones((3000, 1)), rng.normal(size=(3000, 3))])
    theta = rng.normal(size=(2, 4))
    a = rng.integers(0, 2, 3000)
    r = np.einsum("nd,nd->n", X, theta[a])
    np.testing.assert_allclose(ope.fit_ridge_per_arm(X, a, r, 2, ridge=1e-8), theta, atol=1e-6)
