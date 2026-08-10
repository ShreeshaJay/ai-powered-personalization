"""
Aggregation-Based User Embedding Methods (Section 11.1)

Implements five progressively sophisticated strategies for constructing a user
embedding from a sequence of item embeddings, without any training.

Progression:
1. Simple Mean       -- equal weight to every item (bag-of-items)
2. Last-K Mean       -- only the K most recent items (recency prior)
3. Exponential Decay -- smooth positional/temporal decay (recent > old)
4. TF-IDF Weighted   -- upweight rare items, downweight popular ones
5. TF-IDF + Recency  -- combine rarity and recency signals

All methods:
- Accept a (L, D) matrix of item embeddings (chronologically ordered).
- Return a single (D,) user embedding vector.
- Optionally L2-normalize the output (required for FAISS inner product search).

Design note: these are *training-free* baselines. Their purpose is to
establish how far you can get with pure aggregation, setting the bar that
learned models in Sections 11.2 and 11.3 must beat.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class UserEmbeddingAggregator(ABC):
    """Abstract base class for user embedding aggregation methods."""

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, L2-normalize the output embedding.
                       Required for FAISS IndexFlatIP (cosine similarity).
        """
        self.normalize = normalize

    @abstractmethod
    def aggregate(
        self,
        item_embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Aggregate item embeddings into a single user embedding.

        Args:
            item_embeddings: (L, D) array of item embeddings in chronological
                            order (index 0 = oldest, index L-1 = most recent).
            **kwargs: Method-specific parameters (positions, timestamps, etc.)

        Returns:
            (D,) user embedding vector.
        """
        pass

    def _maybe_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize if configured to do so."""
        norm = np.linalg.norm(embedding)
        if self.normalize and norm > 0:
            return embedding / norm
        return embedding

    def aggregate_batch(
        self,
        sessions: List[Dict],
        key: str = "history_embeddings",
        **kwargs,
    ) -> np.ndarray:
        """Aggregate a batch of sessions/users into user embeddings.

        Args:
            sessions: List of dicts, each containing item embeddings under `key`.
            key: Key in each dict for the (L, D) item embedding array.
            **kwargs: Passed through to aggregate().

        Returns:
            (N, D) array of user embeddings.
        """
        user_embs = []
        for session in sessions:
            emb = self.aggregate(session[key], **kwargs)
            user_embs.append(emb)
        return np.stack(user_embs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and result tables."""
        pass


class SimpleMeanAggregator(UserEmbeddingAggregator):
    """Simple mean of all item embeddings in the history.

    user_emb = (1/L) * sum(item_embs[i] for i in range(L))

    This is the "bag of items" assumption: every item contributes equally,
    regardless of when it was interacted with or how popular it is.

    Strengths:
    - Dead simple, zero hyperparameters.
    - Reasonable when history is short and all items are equally relevant.

    Weaknesses:
    - Old interests dilute current intent (e.g., a user who browsed laptops
      last week and phone cases today gets a blurred embedding).
    - Popular items (clicked by everyone) dominate the centroid.
    """

    @property
    def name(self) -> str:
        return "Simple Mean"

    def aggregate(
        self,
        item_embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        if len(item_embeddings) == 0:
            return np.zeros(item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0)
        user_emb = np.mean(item_embeddings, axis=0)
        return self._maybe_normalize(user_emb)


class LastKMeanAggregator(UserEmbeddingAggregator):
    """Mean of the K most recent item embeddings.

    user_emb = (1/min(K, L)) * sum(item_embs[-K:])

    Uses only the tail of the interaction history, imposing a hard recency
    cutoff. When K >= L, this degrades to SimpleMean.

    The value of K should be grounded in data:
    - For Amazon sessions (median ~3-5 items): K=2-3 is meaningful.
    - For MIND user histories (median ~20 items): K=5-50 spans a wide range.

    Strengths:
    - Captures recent intent without being diluted by ancient history.
    - Single hyperparameter with intuitive meaning.

    Weaknesses:
    - Hard cutoff: item K+1 has weight 0, item K has weight 1/K. No graceful
      transition.
    - Ignores potentially valuable long-term preferences.
    """

    def __init__(self, k: int, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.k = k

    @property
    def name(self) -> str:
        return f"Last-{self.k} Mean"

    def aggregate(
        self,
        item_embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        if len(item_embeddings) == 0:
            return np.zeros(item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0)
        # Take last K items (or all if fewer than K)
        recent = item_embeddings[-self.k:]
        user_emb = np.mean(recent, axis=0)
        return self._maybe_normalize(user_emb)


class ExponentialDecayAggregator(UserEmbeddingAggregator):
    """Exponentially decayed weighted mean based on position or time.

    Position-based (Amazon, no timestamps):
        weight_i = exp(-lambda * (L-1 - i))
        where i=0 is oldest, i=L-1 is most recent => most recent gets weight 1.

    Time-based (MIND, has timestamps):
        weight_i = exp(-lambda * (t_ref - t_i) / 3600)
        where t_ref is the reference time and t_i is the click time.

    The decay rate lambda is derived from a half-life parameter:
        lambda = ln(2) / half_life
    so that an item at distance `half_life` from the most recent gets weight 0.5.

    For position-based decay:
        half_life = median session length (data-driven).
        An item at the median position gets weight ~0.5.

    Strengths:
    - Smooth transition: recent items matter more, but old items still contribute.
    - Single interpretable hyperparameter (half-life).

    Weaknesses:
    - Assumes exponential importance decay, which may not hold for all domains.
    - Position-based is only a proxy for time when timestamps are unavailable.
    """

    def __init__(
        self,
        half_life: float,
        mode: str = "position",
        normalize: bool = True,
    ):
        """
        Args:
            half_life: For "position" mode: number of positions back where
                      weight = 0.5. For "time" mode: hours until weight = 0.5.
            mode: "position" (index-based) or "time" (timestamp-based).
        """
        super().__init__(normalize=normalize)
        self.half_life = half_life
        self.mode = mode
        self.decay_rate = np.log(2) / half_life if half_life > 0 else 0.0

    @property
    def name(self) -> str:
        return f"Exp Decay (hl={self.half_life:.1f}, {self.mode})"

    def aggregate(
        self,
        item_embeddings: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        L = len(item_embeddings)
        if L == 0:
            return np.zeros(item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0)

        if self.mode == "position":
            # Position-based: distance from most recent (L-1)
            distances = np.arange(L - 1, -1, -1, dtype=np.float64)  # [L-1, L-2, ..., 0]
            weights = np.exp(-self.decay_rate * distances)
        elif self.mode == "time" and timestamps is not None:
            # Time-based: distance in hours from reference time
            ref_time = timestamps[-1]  # Most recent timestamp
            hours_ago = np.array([
                (ref_time - t).total_seconds() / 3600.0
                for t in timestamps
            ])
            weights = np.exp(-self.decay_rate * hours_ago)
        else:
            # Fallback to position-based
            distances = np.arange(L - 1, -1, -1, dtype=np.float64)
            weights = np.exp(-self.decay_rate * distances)

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Weighted mean
        user_emb = np.average(item_embeddings, axis=0, weights=weights)
        return self._maybe_normalize(user_emb)


class TFIDFWeightedAggregator(UserEmbeddingAggregator):
    """TF-IDF weighted mean of item embeddings.

    weight_i = idf(item_i)
    user_emb = weighted_mean(item_embs, weights)

    IDF = log((N + smooth) / (df_item + smooth))
    where N = total sessions/users, df_item = sessions/users containing item.

    TF is binary (1 if item in history) since most items appear once per
    session or user history.

    Pedagogical insight: A user who clicks a mainstream article (read by 80%
    of users) shouldn't have their embedding dominated by that article. The
    niche robotics article they also clicked is a much stronger signal of
    their unique interests. TF-IDF upweights exactly those distinctive items.

    Strengths:
    - Naturally downweights ubiquitous items (homepage articles, bestsellers).
    - No temporal assumptions.

    Weaknesses:
    - Ignores recency entirely.
    - Items not seen in training data get default IDF (may over/under-weight).
    """

    def __init__(
        self,
        idf_scores: Dict[str, float],
        default_idf: float = 1.0,
        normalize: bool = True,
    ):
        """
        Args:
            idf_scores: Dict mapping item_id -> IDF score.
                       Computed from training data by the evaluation script.
            default_idf: IDF for items not in the training vocabulary.
        """
        super().__init__(normalize=normalize)
        self.idf_scores = idf_scores
        self.default_idf = default_idf

    @property
    def name(self) -> str:
        return "TF-IDF Weighted"

    def aggregate(
        self,
        item_embeddings: np.ndarray,
        item_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> np.ndarray:
        L = len(item_embeddings)
        if L == 0:
            return np.zeros(item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0)

        if item_ids is None:
            # Without item IDs, fall back to uniform weighting
            logger.warning("TF-IDF aggregator called without item_ids; using uniform weights")
            return self._maybe_normalize(np.mean(item_embeddings, axis=0))

        # Look up IDF weights
        weights = np.array([
            self.idf_scores.get(iid, self.default_idf)
            for iid in item_ids
        ])

        # Normalize weights to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(L) / L

        user_emb = np.average(item_embeddings, axis=0, weights=weights)
        return self._maybe_normalize(user_emb)


class TFIDFRecencyAggregator(UserEmbeddingAggregator):
    """Combined TF-IDF + exponential recency weighting.

    weight_i = idf(item_i) * exp(-lambda * distance_i)
    user_emb = weighted_mean(item_embs, weights)

    Combines both signals:
    - Rare items are more informative (TF-IDF).
    - Recent items are more relevant to current intent (recency).

    This is the strongest non-learned aggregation baseline: a rare item
    that was interacted with recently gets the highest weight.

    Strengths:
    - Best of both worlds: rarity + recency.
    - Establishes the ceiling for training-free methods.

    Weaknesses:
    - Two hyperparameters (half_life + IDF vocabulary).
    - Still cannot learn *which* items in the sequence are actually
      predictive of the target — that requires attention (Section 11.2).
    """

    def __init__(
        self,
        idf_scores: Dict[str, float],
        half_life: float,
        mode: str = "position",
        default_idf: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.idf_scores = idf_scores
        self.half_life = half_life
        self.mode = mode
        self.default_idf = default_idf
        self.decay_rate = np.log(2) / half_life if half_life > 0 else 0.0

    @property
    def name(self) -> str:
        return f"TF-IDF + Recency (hl={self.half_life:.1f})"

    def aggregate(
        self,
        item_embeddings: np.ndarray,
        item_ids: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        L = len(item_embeddings)
        if L == 0:
            return np.zeros(item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0)

        # IDF weights
        if item_ids is not None:
            idf_weights = np.array([
                self.idf_scores.get(iid, self.default_idf)
                for iid in item_ids
            ])
        else:
            idf_weights = np.ones(L)

        # Recency weights
        if self.mode == "time" and timestamps is not None:
            ref_time = timestamps[-1]
            hours_ago = np.array([
                (ref_time - t).total_seconds() / 3600.0
                for t in timestamps
            ])
            recency_weights = np.exp(-self.decay_rate * hours_ago)
        else:
            distances = np.arange(L - 1, -1, -1, dtype=np.float64)
            recency_weights = np.exp(-self.decay_rate * distances)

        # Combined weights
        weights = idf_weights * recency_weights

        # Normalize to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(L) / L

        user_emb = np.average(item_embeddings, axis=0, weights=weights)
        return self._maybe_normalize(user_emb)


# ============================================================================
# Factory & Utility Functions
# ============================================================================

def compute_idf_scores(
    item_frequency: Dict[str, int],
    total_documents: int,
    smooth: float = 1.0,
) -> Dict[str, float]:
    """Compute IDF scores from item frequency counts.

    IDF(item) = log((N + smooth) / (df(item) + smooth))

    Args:
        item_frequency: Dict mapping item_id -> document frequency.
        total_documents: Total number of sessions/users (N).
        smooth: Additive smoothing to prevent log(0).

    Returns:
        Dict mapping item_id -> IDF score.
    """
    idf = {}
    for item_id, df in item_frequency.items():
        idf[item_id] = float(np.log((total_documents + smooth) / (df + smooth)))
    return idf


def create_aggregators(
    idf_scores: Optional[Dict[str, float]] = None,
    last_k_values: List[int] = [3, 5, 10],
    decay_half_life: float = 3.0,
    decay_mode: str = "position",
) -> List[UserEmbeddingAggregator]:
    """Create all aggregation methods for evaluation.

    Args:
        idf_scores: Pre-computed IDF scores (required for TF-IDF methods).
        last_k_values: K values for Last-K Mean.
        decay_half_life: Half-life for exponential decay.
        decay_mode: "position" or "time" for decay methods.

    Returns:
        List of aggregator instances.
    """
    aggregators = [
        SimpleMeanAggregator(),
    ]

    for k in last_k_values:
        aggregators.append(LastKMeanAggregator(k=k))

    aggregators.append(
        ExponentialDecayAggregator(
            half_life=decay_half_life,
            mode=decay_mode,
        )
    )

    if idf_scores is not None:
        aggregators.append(
            TFIDFWeightedAggregator(idf_scores=idf_scores)
        )
        aggregators.append(
            TFIDFRecencyAggregator(
                idf_scores=idf_scores,
                half_life=decay_half_life,
                mode=decay_mode,
            )
        )
    else:
        logger.warning(
            "IDF scores not provided; skipping TF-IDF aggregators. "
            "Call compute_idf_scores() first."
        )

    return aggregators
