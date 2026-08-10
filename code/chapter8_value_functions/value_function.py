"""
Multi-Objective Value Functions for Recommender Systems

This module implements value functions that blend multiple model objectives
(listen probability, like probability, engagement depth) into a single
composite score for ranking.

Key Concepts:
- Pointwise scores → Listwise optimization
- Business-aligned weighting of different objectives
- Conditional probability composition (P(like|listen) only matters if P(listen) > 0)
- Calibration-aware weights for handling miscalibrated model predictions

Reference: https://shaped.ai/blog/the-anatomy-of-a-modern-ranking-architectures-part-4
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Union
from pathlib import Path
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Calibration Utilities
# ============================================================================

@dataclass
class CalibrationFactors:
    """
    Calibration factors computed from model evaluation.
    
    Each factor is computed as: actual_positive_rate / avg_predicted_probability
    
    Interpretation:
        - factor > 1: Model under-predicts (predictions too low)
        - factor < 1: Model over-predicts (predictions too high)
        - factor ≈ 1: Model is well-calibrated
    
    Example from Yambda MMoE training:
        - engagement: 0.92 / 0.50 ≈ 1.84 (under-predicting)
        - like: 0.0072 / 0.125 ≈ 0.058 (over-predicting 17x!)
        - dislike: 0.002 / 0.064 ≈ 0.031 (over-predicting 32x!)
    """
    engagement: float = 1.0
    completion: float = 1.0
    play_ratio: float = 1.0
    like: float = 1.0
    dislike: float = 1.0
    
    @classmethod
    def from_chapter7_calibration(cls, calibration_path: Union[str, Path]) -> 'CalibrationFactors':
        """
        Load calibration factors from Chapter 7's calibration.json output.
        
        The calibration.json contains per-task calibration curves. We compute
        a simple mean calibration factor for each task.
        
        Args:
            calibration_path: Path to calibration.json from Chapter 7
            
        Returns:
            CalibrationFactors with computed factors
        """
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        
        factors = {}
        for task in ['engagement', 'completion', 'like', 'dislike']:
            if task not in calib_data:
                factors[task] = 1.0
                continue
                
            task_data = calib_data[task]
            positive_rate = task_data.get('positive_rate', 0)
            mean_preds = task_data.get('mean_predicted', [])
            
            if mean_preds and positive_rate > 0:
                # Use overall positive rate / average of mean predictions
                avg_prediction = np.mean(mean_preds)
                if avg_prediction > 1e-6:
                    factors[task] = positive_rate / avg_prediction
                else:
                    factors[task] = 1.0
            else:
                factors[task] = 1.0
        
        # play_ratio is typically well-calibrated for regression
        factors['play_ratio'] = 1.0
        
        return cls(**factors)
    
    def __repr__(self) -> str:
        return (
            f"CalibrationFactors(\n"
            f"  engagement={self.engagement:.4f},  # {'under' if self.engagement > 1 else 'over'}-predicting\n"
            f"  completion={self.completion:.4f},  # {'under' if self.completion > 1 else 'over'}-predicting\n"
            f"  play_ratio={self.play_ratio:.4f},\n"
            f"  like={self.like:.4f},        # {'under' if self.like > 1 else 'over'}-predicting by {1/self.like if self.like < 1 else self.like:.1f}x\n"
            f"  dislike={self.dislike:.4f},     # {'under' if self.dislike > 1 else 'over'}-predicting by {1/self.dislike if self.dislike < 1 else self.dislike:.1f}x\n"
            f")"
        )


def compute_calibrated_weights(
    base_weights: 'ValueFunctionConfig',
    calibration: CalibrationFactors,
) -> 'ValueFunctionConfig':
    """
    Compute effective weights that absorb calibration factors.
    
    For ranking purposes, applying calibration to predictions:
        value = w * P(x) * calibration_factor
    
    Is mathematically equivalent to adjusting the weight:
        value = (w * calibration_factor) * P(x)
        value = effective_w * P(x)
    
    This is the recommended approach because:
    1. Simpler: No need to modify predictions at inference time
    2. Avoids probabilities going > 1 after scaling
    3. Ranking order is preserved (calibration is a global constant)
    
    Args:
        base_weights: Original ValueFunctionConfig with desired relative weights
        calibration: CalibrationFactors from model evaluation
        
    Returns:
        New ValueFunctionConfig with calibration-adjusted weights
        
    Example:
        >>> base = ValueFunctionConfig(w_listen=1.0, w_like=2.0, w_dislike_penalty=1.0)
        >>> calib = CalibrationFactors(engagement=1.84, like=0.058, dislike=0.031)
        >>> adjusted = compute_calibrated_weights(base, calib)
        >>> print(f"Effective w_like: {adjusted.w_like:.4f}")  # 2.0 * 0.058 = 0.116
    """
    return ValueFunctionConfig(
        w_listen=base_weights.w_listen * calibration.engagement,
        w_like=base_weights.w_like * calibration.like,
        w_engagement=base_weights.w_engagement * calibration.play_ratio,
        w_dislike_penalty=base_weights.w_dislike_penalty * calibration.dislike,
        min_listen_threshold=base_weights.min_listen_threshold,
        normalize_scores=base_weights.normalize_scores,
    )


@dataclass
class ValueFunctionConfig:
    """Configuration for multi-objective value function.
    
    Attributes:
        w_listen: Weight for P(listen) - base engagement probability
        w_like: Weight for P(like|listen) - explicit positive feedback
        w_engagement: Weight for E[play_ratio] - engagement depth
        w_dislike_penalty: Penalty weight for P(dislike|listen) - negative signal
        min_listen_threshold: Minimum P(listen) to consider item viable
        normalize_scores: Whether to normalize final scores to [0, 1]
    
    Note on Calibration:
        If your multi-task model predictions are miscalibrated (e.g., P(like) 
        over-predicts by 17x), you have two options:
        
        1. **Post-hoc calibration**: Apply Platt scaling or isotonic regression
           to each task's predictions before computing values.
           
        2. **Absorb into weights** (recommended for ranking): Multiply the weight
           by the calibration factor. For ranking, this is mathematically equivalent
           since calibration is a global constant per task.
           
        Example: If P(like) over-predicts by 17x (actual_rate / avg_pred = 0.058):
            - Original: w_like=2.0, value contribution = 2.0 * P(like)
            - Calibrated weight: w_like=2.0 * 0.058 = 0.116
            - Same ranking, but weights reflect true signal strength
            
        Use `compute_calibrated_weights()` to automatically adjust weights
        based on Chapter 7's calibration.json output.
    """
    w_listen: float = 1.0
    w_like: float = 2.0
    w_engagement: float = 1.5
    w_dislike_penalty: float = 1.0
    min_listen_threshold: float = 0.0
    normalize_scores: bool = True
    
    def __post_init__(self):
        """Validate weights are non-negative."""
        for attr in ['w_listen', 'w_like', 'w_engagement', 'w_dislike_penalty']:
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be non-negative, got {getattr(self, attr)}")


class ValueFunction:
    """
    Multi-objective value function for ranking candidates.
    
    Combines multiple model outputs into a single business-aligned score.
    
    The default formulation:
        Value = w_listen * P(listen) 
              + w_like * P(listen) * P(like|listen)
              + w_engagement * P(listen) * E[play_ratio]
              - w_dislike * P(listen) * P(dislike|listen)
    
    The conditional composition (multiplying by P(listen)) ensures that
    downstream signals only contribute when there's a reasonable chance
    of initial engagement.
    
    Example:
        >>> config = ValueFunctionConfig(w_listen=1.0, w_like=2.0, w_engagement=1.5)
        >>> vf = ValueFunction(config)
        >>> scores = vf.compute_value(
        ...     p_listen=np.array([0.8, 0.6, 0.3]),
        ...     p_like=np.array([0.2, 0.7, 0.9]),
        ...     e_engagement=np.array([0.6, 0.8, 0.4])
        ... )
    """
    
    def __init__(self, config: Optional[ValueFunctionConfig] = None):
        """Initialize value function with configuration.
        
        Args:
            config: ValueFunctionConfig instance. Uses defaults if None.
        """
        self.config = config or ValueFunctionConfig()
        logger.info(f"Initialized ValueFunction with config: {self.config}")
    
    def compute_value(
        self,
        p_listen: np.ndarray,
        p_like: Optional[np.ndarray] = None,
        e_engagement: Optional[np.ndarray] = None,
        p_dislike: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute composite value scores from multi-task model outputs.
        
        Args:
            p_listen: P(listen) predictions, shape (n_candidates,)
            p_like: P(like|listen) predictions, shape (n_candidates,)
            e_engagement: E[play_ratio] predictions in [0, 1], shape (n_candidates,)
            p_dislike: P(dislike|listen) predictions, shape (n_candidates,)
        
        Returns:
            Composite value scores, shape (n_candidates,)
        
        Note:
            Missing objectives (None) are treated as zero contribution.
            This allows graceful degradation if a model tower fails.
        """
        n = len(p_listen)
        
        # Validate inputs
        p_listen = np.asarray(p_listen, dtype=np.float32)
        assert p_listen.shape == (n,), f"p_listen shape mismatch: {p_listen.shape}"
        
        # Clamp probabilities to valid range
        p_listen = np.clip(p_listen, 0, 1)
        
        # Start with base listen probability
        value = self.config.w_listen * p_listen
        
        # Add like contribution (conditional on listen)
        if p_like is not None:
            p_like = np.clip(np.asarray(p_like, dtype=np.float32), 0, 1)
            value += self.config.w_like * p_listen * p_like
        
        # Add engagement depth contribution (conditional on listen)
        if e_engagement is not None:
            e_engagement = np.clip(np.asarray(e_engagement, dtype=np.float32), 0, 1)
            value += self.config.w_engagement * p_listen * e_engagement
        
        # Subtract dislike penalty (conditional on listen)
        if p_dislike is not None:
            p_dislike = np.clip(np.asarray(p_dislike, dtype=np.float32), 0, 1)
            value -= self.config.w_dislike_penalty * p_listen * p_dislike
        
        # Apply minimum listen threshold
        if self.config.min_listen_threshold > 0:
            mask = p_listen < self.config.min_listen_threshold
            value[mask] = -np.inf  # Effectively filter out low-listen items
        
        # Normalize to [0, 1] if configured
        if self.config.normalize_scores:
            value = self._normalize(value)
        
        return value
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        # Handle -inf from threshold filtering
        valid_mask = np.isfinite(scores)
        if not valid_mask.any():
            return scores
        
        min_val = scores[valid_mask].min()
        max_val = scores[valid_mask].max()
        
        if max_val - min_val < 1e-8:
            # All scores are the same
            return np.where(valid_mask, 0.5, scores)
        
        normalized = (scores - min_val) / (max_val - min_val)
        normalized = np.where(valid_mask, normalized, -np.inf)
        return normalized
    
    def explain_score(
        self,
        p_listen: float,
        p_like: float = 0.0,
        e_engagement: float = 0.0,
        p_dislike: float = 0.0,
    ) -> Dict[str, float]:
        """
        Decompose a value score into its components for interpretability.
        
        Useful for debugging and explaining recommendations to stakeholders.
        
        Args:
            p_listen: P(listen) for single item
            p_like: P(like|listen) for single item
            e_engagement: E[play_ratio] for single item
            p_dislike: P(dislike|listen) for single item
        
        Returns:
            Dictionary with component contributions and total
        """
        components = {
            'listen_contribution': self.config.w_listen * p_listen,
            'like_contribution': self.config.w_like * p_listen * p_like,
            'engagement_contribution': self.config.w_engagement * p_listen * e_engagement,
            'dislike_penalty': -self.config.w_dislike_penalty * p_listen * p_dislike,
        }
        components['total'] = sum(components.values())
        components['inputs'] = {
            'p_listen': p_listen,
            'p_like': p_like,
            'e_engagement': e_engagement,
            'p_dislike': p_dislike,
        }
        return components


# ============================================================================
# Predefined Value Function Configurations
# ============================================================================

def get_conservative_config() -> ValueFunctionConfig:
    """
    Conservative value function prioritizing engagement depth.
    
    Use when: Platform maturity, optimizing for session time.
    """
    return ValueFunctionConfig(
        w_listen=1.0,
        w_like=1.0,
        w_engagement=3.0,  # High weight on engagement depth
        w_dislike_penalty=2.0,  # Strongly penalize predicted dislikes
    )


def get_growth_config() -> ValueFunctionConfig:
    """
    Growth-focused value function prioritizing explicit positive signals.
    
    Use when: Building user loyalty, optimizing for retention.
    """
    return ValueFunctionConfig(
        w_listen=0.5,
        w_like=4.0,  # High weight on likes (retention signal)
        w_engagement=1.0,
        w_dislike_penalty=1.0,
    )


def get_exploration_config() -> ValueFunctionConfig:
    """
    Exploration-friendly value function with lower thresholds.
    
    Use when: Testing new content, cold-start scenarios.
    """
    return ValueFunctionConfig(
        w_listen=2.0,  # Emphasize base engagement
        w_like=1.0,
        w_engagement=1.0,
        w_dislike_penalty=0.5,  # Lower penalty to allow exploration
        min_listen_threshold=0.0,  # No threshold filtering
    )


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def batch_compute_values(
    model_outputs: Dict[str, np.ndarray],
    value_function: ValueFunction,
) -> np.ndarray:
    """
    Convenience function to compute values from model output dictionary.
    
    Args:
        model_outputs: Dictionary with keys matching multi-task model outputs:
            - 'p_listen': Required
            - 'p_like': Optional
            - 'e_engagement': Optional  
            - 'p_dislike': Optional
        value_function: ValueFunction instance
    
    Returns:
        Value scores array
    """
    return value_function.compute_value(
        p_listen=model_outputs['p_listen'],
        p_like=model_outputs.get('p_like'),
        e_engagement=model_outputs.get('e_engagement'),
        p_dislike=model_outputs.get('p_dislike'),
    )


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 60)
    print("Value Function Demonstration")
    print("=" * 60)
    
    # Create sample predictions
    np.random.seed(42)
    n_candidates = 10
    
    p_listen = np.random.uniform(0.3, 0.9, n_candidates)
    p_like = np.random.uniform(0.1, 0.8, n_candidates)
    e_engagement = np.random.uniform(0.4, 0.95, n_candidates)
    p_dislike = np.random.uniform(0.0, 0.3, n_candidates)
    
    # Compare different configurations
    configs = {
        'Default': ValueFunctionConfig(),
        'Conservative': get_conservative_config(),
        'Growth': get_growth_config(),
        'Exploration': get_exploration_config(),
    }
    
    print("\nSample predictions (first 5 candidates):")
    print(f"  P(listen):    {p_listen[:5].round(3)}")
    print(f"  P(like):      {p_like[:5].round(3)}")
    print(f"  E[engagement]:{e_engagement[:5].round(3)}")
    print(f"  P(dislike):   {p_dislike[:5].round(3)}")
    
    print("\nValue scores by configuration:")
    for name, config in configs.items():
        vf = ValueFunction(config)
        scores = vf.compute_value(p_listen, p_like, e_engagement, p_dislike)
        ranking = np.argsort(-scores)[:5]  # Top 5
        print(f"\n  {name}:")
        print(f"    Weights: listen={config.w_listen}, like={config.w_like}, "
              f"engagement={config.w_engagement}, dislike_penalty={config.w_dislike_penalty}")
        print(f"    Top 5 ranking: {ranking}")
        print(f"    Top 5 scores:  {scores[ranking].round(3)}")
    
    # Score explanation
    print("\n" + "=" * 60)
    print("Score Explanation (Candidate 0)")
    print("=" * 60)
    vf = ValueFunction(ValueFunctionConfig())
    explanation = vf.explain_score(
        p_listen=p_listen[0],
        p_like=p_like[0],
        e_engagement=e_engagement[0],
        p_dislike=p_dislike[0],
    )
    print(f"  Inputs: {explanation['inputs']}")
    print(f"  Listen contribution:     {explanation['listen_contribution']:.4f}")
    print(f"  Like contribution:       {explanation['like_contribution']:.4f}")
    print(f"  Engagement contribution: {explanation['engagement_contribution']:.4f}")
    print(f"  Dislike penalty:         {explanation['dislike_penalty']:.4f}")
    print(f"  Total (before norm):     {explanation['total']:.4f}")
    
    # ========================================================================
    # Calibration-Aware Weights Demonstration
    # ========================================================================
    print("\n" + "=" * 60)
    print("Calibration-Aware Weights Demonstration")
    print("=" * 60)
    
    # Typical calibration factors from Yambda MMoE training
    print("\nSimulated calibration factors (from model evaluation):")
    calib = CalibrationFactors(
        engagement=1.84,   # Under-predicting: actual=0.92, avg_pred=0.50
        completion=1.60,   # Under-predicting: actual=0.61, avg_pred=0.38
        play_ratio=1.0,    # Well calibrated (regression)
        like=0.058,        # Over-predicting 17x: actual=0.0072, avg_pred=0.125
        dislike=0.031,     # Over-predicting 32x: actual=0.002, avg_pred=0.064
    )
    print(calib)
    
    # Show weight adjustment
    base_config = ValueFunctionConfig(
        w_listen=1.0,
        w_like=2.0,
        w_engagement=1.5,
        w_dislike_penalty=1.0,
    )
    calibrated_config = compute_calibrated_weights(base_config, calib)
    
    print("\nBase weights vs Calibration-adjusted weights:")
    print(f"  w_listen:         {base_config.w_listen:.2f} → {calibrated_config.w_listen:.4f}")
    print(f"  w_like:           {base_config.w_like:.2f} → {calibrated_config.w_like:.4f}")
    print(f"  w_engagement:     {base_config.w_engagement:.2f} → {calibrated_config.w_engagement:.4f}")
    print(f"  w_dislike_penalty:{base_config.w_dislike_penalty:.2f} → {calibrated_config.w_dislike_penalty:.4f}")
    
    # Compare rankings
    print("\nRanking comparison (should be identical for ranking purposes):")
    vf_base = ValueFunction(base_config)
    vf_calibrated = ValueFunction(calibrated_config)
    
    # With calibrated predictions (simulate what the model outputs)
    scores_base = vf_base.compute_value(p_listen, p_like, e_engagement, p_dislike)
    scores_calib = vf_calibrated.compute_value(p_listen, p_like, e_engagement, p_dislike)
    
    ranking_base = np.argsort(-scores_base)
    ranking_calib = np.argsort(-scores_calib)
    
    print(f"  Base weights ranking:       {ranking_base[:5]}")
    print(f"  Calibrated weights ranking: {ranking_calib[:5]}")
    
    # Key insight: rankings can differ because calibration changes relative importance
    print("\n  → Note: Rankings MAY differ because calibration-adjusted weights")
    print("    change the relative importance of each objective!")
    print("    This is the INTENDED effect: rare events (like) should contribute")
    print("    less to the value score when the model over-predicts them.")

