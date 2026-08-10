"""
Delayed Feedback Modeling for Adtech
====================================

In advertising, conversions often occur hours or days after a click.
This creates a data censoring problem where recent samples appear negative
but may convert later.

The Problem:
    - User clicks ad at time T
    - User converts at time T + Δ (e.g., 12 hours later)
    - If we train on data collected at T + 6h, this looks like no conversion
    - This biases CVR estimates DOWNWARD for recent data

Approaches:
    1. Attribution Window Cutoff: Only use samples older than window
       - Pro: Clean labels
       - Con: Loses recent data, model is always stale
       
    2. Importance Weighting: Weight samples by observation probability
       - Pro: Uses all data
       - Con: Assumes delay distribution is known/estimated
       
    3. Delay Model (DFM): Explicitly model P(delay = t | will_convert)
       - Pro: Most principled
       - Con: Most complex, requires delay observations

This module implements importance weighting and utilities for delay modeling.

Reference:
    Chapelle et al. "Modeling Delayed Feedback in Display Advertising" (KDD 2014)

Usage:
    from utils.delayed_feedback import (
        compute_observation_weights,
        apply_delayed_feedback_correction,
        estimate_delay_distribution,
    )
    
    # Compute sample weights based on observation time
    weights = compute_observation_weights(
        click_times, collection_time, attribution_window_hours=24
    )
    
    # Use weights in training
    loss = weighted_bce_loss(preds, labels, weights)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy import stats


def compute_observation_weights(
    click_times: np.ndarray,
    collection_time: float,
    attribution_window_hours: float = 24.0,
    min_weight: float = 0.1,
    converted: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute importance weights based on observation time.
    
    The intuition: A sample observed 1 hour after click has only had
    1/24 of the attribution window to convert. Its "no conversion" label
    is less reliable than a sample observed after 24 hours.
    
    Weight formula:
        w = min(1, time_since_click / attribution_window)
        
    For converted samples: w = 1.0 (we observed the conversion)
    
    Args:
        click_times: Unix timestamps of clicks
        collection_time: Unix timestamp of data collection
        attribution_window_hours: Attribution window in hours
        min_weight: Minimum weight for recent samples
        converted: Optional binary array (converted samples get weight=1)
        
    Returns:
        Array of sample weights in [min_weight, 1.0]
    """
    click_times = np.asarray(click_times).flatten()
    
    # Time since click in hours
    time_since_click_hours = (collection_time - click_times) / 3600.0
    
    # Weight based on fraction of attribution window observed
    weights = np.clip(time_since_click_hours / attribution_window_hours, min_weight, 1.0)
    
    # Converted samples always get full weight
    if converted is not None:
        converted = np.asarray(converted).flatten()
        weights[converted == 1] = 1.0
    
    return weights


def estimate_delay_distribution(
    click_times: np.ndarray,
    conversion_times: np.ndarray,
    converted: np.ndarray,
    n_bins: int = 24,
    max_delay_hours: float = 72.0,
) -> Dict[str, Any]:
    """Estimate the delay distribution from observed conversions.
    
    Models the distribution P(delay | will_convert) from data where we
    observe the conversion time.
    
    Common findings:
        - Most conversions happen within hours of click
        - Distribution is heavy-tailed (some very late conversions)
        - Often well-modeled by exponential or log-normal
    
    Args:
        click_times: Unix timestamps of clicks
        conversion_times: Unix timestamps of conversions (NaN if not converted)
        converted: Binary array indicating which samples converted
        n_bins: Number of bins for histogram
        max_delay_hours: Maximum delay to consider
        
    Returns:
        Dictionary with delay distribution statistics
    """
    click_times = np.asarray(click_times).flatten()
    conversion_times = np.asarray(conversion_times).flatten()
    converted = np.asarray(converted).flatten().astype(bool)
    
    # Compute delays for converted samples
    delays_hours = (conversion_times[converted] - click_times[converted]) / 3600.0
    
    # Filter to valid delays
    valid_mask = (delays_hours >= 0) & (delays_hours <= max_delay_hours) & (~np.isnan(delays_hours))
    delays_hours = delays_hours[valid_mask]
    
    if len(delays_hours) == 0:
        return {
            'n_conversions': 0,
            'mean_delay_hours': None,
            'median_delay_hours': None,
        }
    
    # Compute statistics
    result = {
        'n_conversions': len(delays_hours),
        'mean_delay_hours': float(np.mean(delays_hours)),
        'median_delay_hours': float(np.median(delays_hours)),
        'std_delay_hours': float(np.std(delays_hours)),
        'p25_delay_hours': float(np.percentile(delays_hours, 25)),
        'p75_delay_hours': float(np.percentile(delays_hours, 75)),
        'p90_delay_hours': float(np.percentile(delays_hours, 90)),
        'p99_delay_hours': float(np.percentile(delays_hours, 99)),
    }
    
    # Histogram
    bin_edges = np.linspace(0, max_delay_hours, n_bins + 1)
    hist, _ = np.histogram(delays_hours, bins=bin_edges, density=True)
    
    result['histogram'] = {
        'bin_edges': bin_edges.tolist(),
        'density': hist.tolist(),
    }
    
    # Fit exponential distribution
    try:
        # MLE for exponential: lambda = 1 / mean
        exp_rate = 1.0 / np.mean(delays_hours)
        result['exponential_rate'] = float(exp_rate)
        result['exponential_mean'] = float(1.0 / exp_rate)
    except Exception:
        result['exponential_rate'] = None
    
    # Cumulative distribution (useful for attribution window selection)
    sorted_delays = np.sort(delays_hours)
    cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
    
    # Find attribution windows for different coverage levels
    for coverage in [0.5, 0.75, 0.9, 0.95, 0.99]:
        idx = np.searchsorted(cdf, coverage)
        if idx < len(sorted_delays):
            result[f'window_{int(coverage*100)}pct_hours'] = float(sorted_delays[idx])
    
    return result


def apply_attribution_window_cutoff(
    click_times: np.ndarray,
    labels: np.ndarray,
    collection_time: float,
    attribution_window_hours: float = 24.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter samples to only those with reliable labels.
    
    Key insight: Observed conversions are CERTAIN - no need to wait for 
    attribution window. Only NEGATIVES within the attribution window are 
    unreliable (they might convert later).
    
    Keep samples that are EITHER:
        1. Old enough (full attribution window has passed), OR
        2. Already converted (observed conversion is certain)
    
    Args:
        click_times: Unix timestamps of clicks
        labels: Conversion labels (1 = converted, 0 = not converted)
        collection_time: Unix timestamp of data collection
        attribution_window_hours: Attribution window in hours
        
    Returns:
        Tuple of (filtered_mask, filtered_labels)
    """
    click_times = np.asarray(click_times).flatten()
    labels = np.asarray(labels).flatten()
    
    time_since_click_hours = (collection_time - click_times) / 3600.0
    
    # Keep: (1) samples with full attribution window, OR (2) observed conversions
    # Only discard recent NEGATIVES - they might still convert
    valid_mask = (time_since_click_hours >= attribution_window_hours) | (labels == 1)
    
    return valid_mask, labels[valid_mask]


@dataclass
class DelayedFeedbackCorrector:
    """Class for applying delayed feedback corrections during training.
    
    Usage:
        corrector = DelayedFeedbackCorrector(
            attribution_window_hours=24,
            method='importance_weighting'
        )
        
        # During training
        weights = corrector.get_weights(click_times, collection_time, converted)
        loss = weighted_bce_loss(preds, labels, weights)
    """
    attribution_window_hours: float = 24.0
    method: str = 'importance_weighting'  # 'importance_weighting' or 'cutoff'
    min_weight: float = 0.1
    
    def get_weights(
        self,
        click_times: np.ndarray,
        collection_time: float,
        converted: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get sample weights for delayed feedback correction.
        
        Args:
            click_times: Unix timestamps of clicks
            collection_time: Unix timestamp of data collection
            converted: Optional conversion labels
            
        Returns:
            Sample weights (all 1.0 for cutoff method, variable for importance weighting)
        """
        if self.method == 'cutoff':
            # For cutoff, all kept samples have weight 1
            return np.ones(len(click_times))
        
        elif self.method == 'importance_weighting':
            return compute_observation_weights(
                click_times=click_times,
                collection_time=collection_time,
                attribution_window_hours=self.attribution_window_hours,
                min_weight=self.min_weight,
                converted=converted,
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_valid_mask(
        self,
        click_times: np.ndarray,
        collection_time: float,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get mask of valid samples (for cutoff method).
        
        For cutoff method: keeps samples that are either old enough OR converted.
        Only recent negatives are discarded (they might still convert).
        
        Args:
            click_times: Unix timestamps of clicks
            collection_time: Unix timestamp of data collection
            labels: Conversion labels (required for cutoff method to preserve positives)
            
        Returns:
            Boolean mask of valid samples
        """
        if self.method == 'cutoff':
            time_since_click_hours = (collection_time - click_times) / 3600.0
            old_enough = time_since_click_hours >= self.attribution_window_hours
            
            if labels is not None:
                # Keep: old samples OR observed conversions
                return old_enough | (np.asarray(labels) == 1)
            else:
                # Without labels, can only filter by time
                return old_enough
        else:
            # Importance weighting keeps all samples
            return np.ones(len(click_times), dtype=bool)


def compute_weighted_aggregate_features(
    group_ids: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    min_samples: int = 10,
    smoothing_prior: float = 0.01,
    smoothing_strength: float = 10.0,
) -> Dict[Any, float]:
    """Compute delay-corrected aggregate features (e.g., historical CVR per item).
    
    CRITICAL: When computing features like "item historical CVR" or "user historical CVR",
    you must apply the same delayed feedback correction as for labels. Otherwise,
    items/users with recent traffic will appear to have lower CVR.
    
    This function computes weighted averages per group, with Bayesian smoothing
    for groups with few observations.
    
    Args:
        group_ids: Group identifiers (e.g., item_id, user_id, category_id)
        labels: Conversion labels (0 or 1)
        weights: Observation weights from compute_observation_weights()
        min_samples: Minimum weighted samples to compute feature (otherwise use prior)
        smoothing_prior: Prior CVR for Bayesian smoothing
        smoothing_strength: Strength of prior (higher = more regularization)
        
    Returns:
        Dictionary mapping group_id -> weighted CVR estimate
        
    Example:
        # Compute item-level historical CVR with delay correction
        weights = compute_observation_weights(click_times, collection_time, ...)
        item_cvr = compute_weighted_aggregate_features(
            item_ids, conversions, weights
        )
        
        # Use as feature
        df['item_historical_cvr'] = df['item_id'].map(item_cvr)
    """
    group_ids = np.asarray(group_ids).flatten()
    labels = np.asarray(labels).flatten()
    weights = np.asarray(weights).flatten()
    
    unique_groups = np.unique(group_ids)
    group_cvr = {}
    
    for group in unique_groups:
        mask = group_ids == group
        group_labels = labels[mask]
        group_weights = weights[mask]
        
        weighted_sum = (group_labels * group_weights).sum()
        weight_total = group_weights.sum()
        
        if weight_total >= min_samples:
            # Bayesian smoothing: (weighted_conversions + alpha * prior) / (weighted_impressions + alpha)
            smoothed_cvr = (weighted_sum + smoothing_strength * smoothing_prior) / (weight_total + smoothing_strength)
        else:
            # Too few observations, use prior
            smoothed_cvr = smoothing_prior
        
        group_cvr[group] = float(smoothed_cvr)
    
    return group_cvr


def simulate_delayed_feedback(
    n_samples: int = 10000,
    true_cvr: float = 0.02,
    mean_delay_hours: float = 6.0,
    collection_delay_hours: float = 12.0,
    attribution_window_hours: float = 24.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Simulate a delayed feedback scenario for testing.
    
    Simulates:
        1. Clicks at random times in the past
        2. Conversions with exponential delay distribution
        3. Observation at collection_time (some conversions not yet observed)
    
    Args:
        n_samples: Number of samples
        true_cvr: True conversion rate
        mean_delay_hours: Mean conversion delay
        collection_delay_hours: How long after clicks we collect data
        attribution_window_hours: Attribution window for labeling
        seed: Random seed
        
    Returns:
        Dictionary with simulated data:
            - click_times
            - conversion_times (NaN if not converted)
            - true_labels (actual conversion, may not be observed)
            - observed_labels (what we observe at collection time)
    """
    np.random.seed(seed)
    
    # Click times: uniformly distributed in past 48 hours
    # (relative to some reference time = 0)
    collection_time = 0.0
    click_times = np.random.uniform(-48 * 3600, -collection_delay_hours * 3600, n_samples)
    
    # True conversions
    will_convert = np.random.rand(n_samples) < true_cvr
    
    # Conversion delays (exponential distribution)
    conversion_delays_hours = np.random.exponential(mean_delay_hours, n_samples)
    conversion_times = np.where(
        will_convert,
        click_times + conversion_delays_hours * 3600,
        np.nan
    )
    
    # True labels
    true_labels = will_convert.astype(float)
    
    # Observed labels: Only observe conversions that happened before collection_time
    observed_conversions = will_convert & (conversion_times <= collection_time)
    observed_labels = observed_conversions.astype(float)
    
    # Time since click at collection
    time_since_click_hours = (collection_time - click_times) / 3600.0
    
    return {
        'click_times': click_times,
        'conversion_times': conversion_times,
        'true_labels': true_labels,
        'observed_labels': observed_labels,
        'time_since_click_hours': time_since_click_hours,
        'collection_time': collection_time,
        'true_cvr': true_cvr,
        'observed_cvr': observed_labels.mean(),
    }


def analyze_delayed_feedback_bias(
    simulation: Dict[str, np.ndarray],
    attribution_window_hours: float = 24.0,
) -> Dict[str, float]:
    """Analyze the bias introduced by delayed feedback.
    
    Compares true CVR vs. observed CVR under different correction methods.
    
    Args:
        simulation: Output from simulate_delayed_feedback()
        attribution_window_hours: Attribution window to use
        
    Returns:
        Dictionary with bias analysis
    """
    true_labels = simulation['true_labels']
    observed_labels = simulation['observed_labels']
    click_times = simulation['click_times']
    collection_time = simulation['collection_time']
    time_since_click = simulation['time_since_click_hours']
    
    results = {
        'true_cvr': float(true_labels.mean()),
        'observed_cvr': float(observed_labels.mean()),
        'bias_raw': float(observed_labels.mean() - true_labels.mean()),
        'bias_relative': float((observed_labels.mean() - true_labels.mean()) / true_labels.mean()) if true_labels.mean() > 0 else 0,
    }
    
    # Method 1: Attribution window cutoff
    valid_mask = time_since_click >= attribution_window_hours
    if valid_mask.sum() > 0:
        cutoff_cvr = observed_labels[valid_mask].mean()
        results['cutoff_cvr'] = float(cutoff_cvr)
        results['cutoff_bias'] = float(cutoff_cvr - true_labels[valid_mask].mean())
        results['cutoff_samples_kept'] = float(valid_mask.mean())
    
    # Method 2: Importance weighting
    weights = compute_observation_weights(
        click_times, collection_time, 
        attribution_window_hours=attribution_window_hours,
        converted=observed_labels,
    )
    
    # Weighted CVR estimate
    weighted_cvr = (observed_labels * weights).sum() / weights.sum()
    results['weighted_cvr'] = float(weighted_cvr)
    results['weighted_bias'] = float(weighted_cvr - true_labels.mean())
    
    # Per-recency analysis
    recency_bins = [0, 6, 12, 24, 48]
    for i in range(len(recency_bins) - 1):
        low, high = recency_bins[i], recency_bins[i+1]
        mask = (time_since_click >= low) & (time_since_click < high)
        if mask.sum() > 0:
            results[f'cvr_{low}_{high}h_true'] = float(true_labels[mask].mean())
            results[f'cvr_{low}_{high}h_observed'] = float(observed_labels[mask].mean())
    
    return results


if __name__ == "__main__":
    # Test delayed feedback utilities
    print("Testing delayed feedback modeling...")
    
    # Simulate delayed feedback scenario
    print("\n" + "=" * 50)
    print("Simulating delayed feedback...")
    
    sim = simulate_delayed_feedback(
        n_samples=100000,
        true_cvr=0.02,
        mean_delay_hours=6.0,
        collection_delay_hours=12.0,
    )
    
    print(f"\nSimulation results:")
    print(f"  True CVR: {sim['true_cvr']:.4f}")
    print(f"  Observed CVR: {sim['observed_cvr']:.4f}")
    print(f"  Bias: {sim['observed_cvr'] - sim['true_cvr']:.4f}")
    print(f"  Relative bias: {(sim['observed_cvr'] / sim['true_cvr'] - 1)*100:.1f}%")
    
    # Analyze bias with different methods
    print("\n" + "=" * 50)
    print("Analyzing delayed feedback bias...")
    
    analysis = analyze_delayed_feedback_bias(sim, attribution_window_hours=24.0)
    
    print("\nMethod comparison:")
    print(f"  Raw (no correction):     CVR={analysis['observed_cvr']:.4f}, bias={analysis['bias_relative']*100:+.1f}%")
    if 'cutoff_cvr' in analysis:
        print(f"  Cutoff (24h window):     CVR={analysis['cutoff_cvr']:.4f}, bias={analysis['cutoff_bias']/analysis['true_cvr']*100:+.1f}%, kept={analysis['cutoff_samples_kept']:.1%}")
    print(f"  Importance weighting:    CVR={analysis['weighted_cvr']:.4f}, bias={analysis['weighted_bias']/analysis['true_cvr']*100:+.1f}%")
    
    # Test delay distribution estimation
    print("\n" + "=" * 50)
    print("Estimating delay distribution...")
    
    # Use only converted samples with observed delays
    converted_mask = sim['true_labels'] == 1
    delay_dist = estimate_delay_distribution(
        sim['click_times'],
        sim['conversion_times'],
        sim['true_labels'],
    )
    
    print(f"\nDelay distribution (from {delay_dist['n_conversions']} conversions):")
    print(f"  Mean delay: {delay_dist['mean_delay_hours']:.1f} hours")
    print(f"  Median delay: {delay_dist['median_delay_hours']:.1f} hours")
    print(f"  90th percentile: {delay_dist['p90_delay_hours']:.1f} hours")
    print(f"  Exponential rate: {delay_dist['exponential_rate']:.4f} (mean={delay_dist['exponential_mean']:.1f}h)")
    
    # Test corrector class
    print("\n" + "=" * 50)
    print("Testing DelayedFeedbackCorrector...")
    
    corrector = DelayedFeedbackCorrector(
        attribution_window_hours=24.0,
        method='importance_weighting'
    )
    
    weights = corrector.get_weights(
        sim['click_times'][:100],
        sim['collection_time'],
        sim['observed_labels'][:100]
    )
    
    print(f"\nSample weights (first 10):")
    print(f"  {weights[:10]}")
    print(f"  Min weight: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")
    
    print("\nDelayed feedback test passed!")

