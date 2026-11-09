"""
Chapter 16: Experimentation and Bandits - off-policy evaluation (Section 16.9)
==============================================================================
Estimators of a target policy's value from logs collected by a different
(logging) policy. Every estimator shares one signature:

    estimator(rewards, actions, pscores, target_probs, q_hat=None, **kwargs) -> float

    rewards       (n,)    logged reward r_i
    actions       (n,)    logged action a_i
    pscores       (n,)    logging propensity pi_0(a_i | x_i), recorded at decision time
    target_probs  (n, K)  target policy pi_e(a | x_i) for every action
    q_hat         (n, K)  reward-model predictions q(x_i, a), for DM and DR

Estimators:
- Replay        (Li et al., 2011), valid only under uniformly random logging
- IPS           (1/n) sum w_i r_i, with w_i = pi_e(a_i|x_i) / pi_0(a_i|x_i)
- Clipped IPS   IPS with weights capped at M
- SNIPS         sum w_i r_i / sum w_i            (Swaminathan and Joachims, 2015)
- DM            (1/n) sum_i sum_a pi_e(a|x_i) q(x_i, a)
- DR            DM + (1/n) sum w_i (r_i - q(x_i, a_i))   (Dudik, Langford, and Li, 2011)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from stats_utils import effective_sample_size


def importance_weights(actions: np.ndarray, pscores: np.ndarray, target_probs: np.ndarray) -> np.ndarray:
    n = actions.shape[0]
    return target_probs[np.arange(n), actions] / pscores


def replay(rewards, actions, pscores, target_probs, q_hat=None) -> float:
    """Mean reward on the logged events where the (deterministic) target agrees with the log."""
    target_actions = np.argmax(target_probs, axis=1)
    match = target_actions == actions
    if not match.any():
        return float("nan")
    return float(np.mean(rewards[match]))


def ips(rewards, actions, pscores, target_probs, q_hat=None) -> float:
    w = importance_weights(actions, pscores, target_probs)
    return float(np.mean(w * rewards))


def clipped_ips(rewards, actions, pscores, target_probs, q_hat=None, max_weight: float = 10.0) -> float:
    w = np.minimum(importance_weights(actions, pscores, target_probs), max_weight)
    return float(np.mean(w * rewards))


def snips(rewards, actions, pscores, target_probs, q_hat=None) -> float:
    w = importance_weights(actions, pscores, target_probs)
    total = np.sum(w)
    if total == 0:
        return float("nan")
    return float(np.sum(w * rewards) / total)


def direct_method(rewards, actions, pscores, target_probs, q_hat=None) -> float:
    if q_hat is None:
        raise ValueError("DM needs q_hat")
    return float(np.mean(np.sum(target_probs * q_hat, axis=1)))


def doubly_robust(rewards, actions, pscores, target_probs, q_hat=None) -> float:
    if q_hat is None:
        raise ValueError("DR needs q_hat")
    n = actions.shape[0]
    w = importance_weights(actions, pscores, target_probs)
    dm_term = np.sum(target_probs * q_hat, axis=1)
    residual = rewards - q_hat[np.arange(n), actions]
    return float(np.mean(dm_term + w * residual))


def evaluate_all(rewards, actions, pscores, target_probs, q_hat, clip_levels=(10.0, 100.0)) -> Dict[str, float]:
    """Run every estimator on one log; also report the effective sample size."""
    out = {
        "IPS": ips(rewards, actions, pscores, target_probs),
        "SNIPS": snips(rewards, actions, pscores, target_probs),
        "DM": direct_method(rewards, actions, pscores, target_probs, q_hat),
        "DR": doubly_robust(rewards, actions, pscores, target_probs, q_hat),
    }
    for m in clip_levels:
        out[f"Clipped IPS (M={int(m)})"] = clipped_ips(rewards, actions, pscores, target_probs, max_weight=m)
    out["ESS"] = effective_sample_size(importance_weights(actions, pscores, target_probs))
    return out


# =============================================================================
# Policies and reward models used to build logs
# =============================================================================

def one_hot(actions: np.ndarray, n_arms: int) -> np.ndarray:
    out = np.zeros((actions.shape[0], n_arms))
    out[np.arange(actions.shape[0]), actions] = 1.0
    return out


def epsilon_greedy_probs(greedy_actions: np.ndarray, n_arms: int, epsilon: float) -> np.ndarray:
    """pi(a|x) = epsilon / K + (1 - epsilon) * 1[a = greedy(x)]."""
    return epsilon / n_arms + (1.0 - epsilon) * one_hot(greedy_actions, n_arms)


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one action per row from a matrix of action probabilities."""
    u = rng.random(probs.shape[0])[:, None]
    return np.minimum((np.cumsum(probs, axis=1) < u).sum(axis=1), probs.shape[1] - 1)


def fit_ridge_per_arm(X: np.ndarray, actions: np.ndarray, rewards: np.ndarray, n_arms: int, ridge: float = 1.0) -> np.ndarray:
    """Per-arm ridge regression of reward on context; returns theta (K, d)."""
    d = X.shape[1]
    theta = np.zeros((n_arms, d))
    for a in range(n_arms):
        sel = actions == a
        Xa = X[sel]
        A = Xa.T @ Xa + ridge * np.eye(d)
        theta[a] = np.linalg.solve(A, Xa.T @ rewards[sel])
    return theta


def predict_rewards(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Linear reward-model predictions clipped to the valid range [0, 1]."""
    return np.clip(X @ theta.T, 0.0, 1.0)
