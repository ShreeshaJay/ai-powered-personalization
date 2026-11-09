"""
Chapter 16: Experimentation and Bandits - bandit policies (Section 16.8)
========================================================================
Multi-armed and contextual bandit policies with a common interface:

    policy.select(x) -> arm index
    policy.update(x, arm, reward)

Context-free policies ignore x. Implemented:
- UniformRandom
- EpsilonGreedy
- UCB1, index mean_a + sqrt(2 ln t / n_a)        (Auer, Cesa-Bianchi, and Fischer, 2002)
- ThompsonBernoulli, Beta-Bernoulli posterior   (Chapelle and Li, 2011; Russo et al., 2018)
- LinUCB with disjoint per-arm ridge models     (Li et al., 2010)

The environments draw every context and every arm's reward up front, so all
policies in a comparison face the same sequence (common random numbers), and
regret is measured against the known reward probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# Environments
# =============================================================================

@dataclass
class BanditStream:
    """A pre-drawn sequence of contexts, expected rewards, and realized rewards."""
    contexts: np.ndarray      # (T, d)
    expected: np.ndarray      # (T, K) true reward probability of every arm
    rewards: np.ndarray       # (T, K) realized Bernoulli reward of every arm


class BernoulliBandit:
    """Context-free Bernoulli arms with fixed means."""

    def __init__(self, means: np.ndarray):
        self.means = np.asarray(means, dtype=float)
        self.n_arms = self.means.size
        self.dim = 1

    @classmethod
    def random(cls, n_arms: int, rng: np.random.Generator, low: float = 0.02, high: float = 0.10):
        """Low, recommendation-like click rates spread over [low, high]."""
        return cls(rng.uniform(low, high, size=n_arms))

    def stream(self, T: int, rng: np.random.Generator) -> BanditStream:
        expected = np.broadcast_to(self.means, (T, self.n_arms)).copy()
        rewards = (rng.random((T, self.n_arms)) < expected).astype(float)
        return BanditStream(np.ones((T, 1)), expected, rewards)


class LogisticContextualBandit:
    """Shelf-selection environment: reward ~ Bernoulli(sigmoid(theta_a . z + b_a)).

    The context is x = [1, z] with z ~ N(0, I_d), so the intercept is the first
    feature. Biases are centred on logit(base_rate) so rewards are rare, as
    clicks on a recommendation shelf usually are.
    """

    def __init__(self, n_arms: int, dim: int, rng: np.random.Generator,
                 base_rate: float = 0.05, weight_scale: float = 0.6, bias_scale: float = 0.3):
        self.n_arms = n_arms
        self.feature_dim = dim
        self.dim = dim + 1
        self.theta = rng.normal(0.0, weight_scale / np.sqrt(dim), size=(n_arms, dim))
        self.bias = np.log(base_rate / (1 - base_rate)) + rng.normal(0.0, bias_scale, size=n_arms)

    def sample_contexts(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(n, self.feature_dim))
        return np.hstack([np.ones((n, 1)), z])

    def expected_rewards(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X[:, 1:] @ self.theta.T + self.bias)

    def stream(self, T: int, rng: np.random.Generator) -> BanditStream:
        X = self.sample_contexts(T, rng)
        expected = self.expected_rewards(X)
        rewards = (rng.random(expected.shape) < expected).astype(float)
        return BanditStream(X, expected, rewards)


# =============================================================================
# Context-free policies
# =============================================================================

class UniformRandom:
    name = "Uniform random"

    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n_arms = n_arms
        self.rng = rng

    def select(self, x=None) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, x, arm: int, reward: float) -> None:
        pass


class _MeanTracker:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    @property
    def means(self) -> np.ndarray:
        return np.divide(self.sums, self.counts, out=np.zeros(self.n_arms), where=self.counts > 0)

    def update(self, x, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


def _argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


class EpsilonGreedy(_MeanTracker):
    name = "Epsilon-greedy"

    def __init__(self, n_arms: int, epsilon: float, rng: np.random.Generator):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.rng = rng

    def select(self, x=None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return _argmax_random_tie(self.means, self.rng)


class UCB1(_MeanTracker):
    name = "UCB1"

    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.t = 0

    def index(self) -> np.ndarray:
        """mean_a + sqrt(2 ln t / n_a); infinite for arms never played."""
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = np.sqrt(2.0 * np.log(max(self.t, 1)) / self.counts)
        return np.where(self.counts > 0, self.means + bonus, np.inf)

    def select(self, x=None) -> int:
        idx = self.index()
        return int(np.argmax(idx))

    def update(self, x, arm: int, reward: float) -> None:
        super().update(x, arm, reward)
        self.t += 1


class ThompsonBernoulli:
    name = "Thompson sampling"

    def __init__(self, n_arms: int, rng: np.random.Generator, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.rng = rng
        self.alpha = np.full(n_arms, prior_alpha, dtype=float)
        self.beta = np.full(n_arms, prior_beta, dtype=float)

    def select(self, x=None) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, x, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


# =============================================================================
# Contextual policy
# =============================================================================

class LinUCB:
    """Disjoint LinUCB: one ridge regression per arm plus an optimism bonus.

    score_a(x) = theta_a . x + alpha * sqrt(x' A_a^{-1} x),
    A_a = lambda I + sum x x',  b_a = sum r x,  theta_a = A_a^{-1} b_a.
    A_a^{-1} is maintained with the Sherman-Morrison rank-one update, so each
    step costs O(d^2) per arm instead of a matrix inversion.
    """

    name = "LinUCB"

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, ridge: float = 1.0):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.A_inv = np.repeat(np.eye(dim)[None, :, :] / ridge, n_arms, axis=0)
        self.b = np.zeros((n_arms, dim))

    @property
    def theta(self) -> np.ndarray:
        return np.einsum("kij,kj->ki", self.A_inv, self.b)

    def scores(self, x: np.ndarray) -> np.ndarray:
        Ax = self.A_inv @ x                     # (K, d)
        mean = self.theta @ x
        width = np.sqrt(np.maximum(Ax @ x, 0.0))
        return mean + self.alpha * width

    def select(self, x: np.ndarray) -> int:
        return int(np.argmax(self.scores(x)))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        Ainv = self.A_inv[arm]
        Ax = Ainv @ x
        self.A_inv[arm] = Ainv - np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[arm] += reward * x

    def greedy_actions(self, X: np.ndarray) -> np.ndarray:
        """Exploit-only actions of the current model, used as a frozen target policy."""
        return np.argmax(X @ self.theta.T, axis=1)


# =============================================================================
# Simulation loop
# =============================================================================

def run_policy(policy, stream: BanditStream) -> np.ndarray:
    """Play `policy` on a pre-drawn stream; return the per-round expected regret.

    Expected (pseudo-) regret at round t is max_a p_a(x_t) - p_{a_t}(x_t),
    measured against the best arm for that context.
    """
    T = stream.contexts.shape[0]
    regret = np.empty(T)
    best = stream.expected.max(axis=1)
    for t in range(T):
        x = stream.contexts[t]
        a = policy.select(x)
        policy.update(x, a, stream.rewards[t, a])
        regret[t] = best[t] - stream.expected[t, a]
    return regret
