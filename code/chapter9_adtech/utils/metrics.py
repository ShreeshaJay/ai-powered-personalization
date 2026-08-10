"""
Adtech-Specific Evaluation Metrics
==================================

This module provides specialized metrics for evaluating adtech models,
going beyond standard classification metrics.

Metrics:
    1. GAUC (Grouped AUC): AUC computed per-user, then averaged
       - Better for personalization evaluation
       - Handles user-level ranking quality
       
    2. Lift Curves: How much better is targeting vs. random?
       - Lift@k = (CVR in top k%) / (overall CVR)
       - Critical for campaign planning
       
    3. Revenue Metrics: Business-oriented evaluation
       - eCPM, ROI under different bid strategies

Usage:
    from utils.metrics import compute_adtech_metrics, compute_lift_curve
    
    metrics = compute_adtech_metrics(preds, labels, user_ids=users)
    lift_data = compute_lift_curve(preds, labels)
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc


def compute_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute AUC-ROC, handling edge cases."""
    labels = np.asarray(labels).flatten()
    predictions = np.asarray(predictions).flatten()
    
    # Check for degenerate cases
    if len(np.unique(labels)) < 2:
        return 0.5
    
    return roc_auc_score(labels, predictions)


def compute_logloss(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute log loss with clipping for numerical stability."""
    labels = np.asarray(labels).flatten()
    predictions = np.asarray(predictions).flatten()
    
    # Clip predictions
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    
    return log_loss(labels, predictions)


def compute_gauc(
    predictions: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    min_samples: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Grouped AUC (GAUC).
    
    GAUC computes AUC separately for each user, then averages.
    This better reflects per-user ranking quality than global AUC.
    
    GAUC = sum_u (AUC_u * n_u) / sum_u (n_u)
    
    Where AUC_u is the AUC for user u and n_u is the number of samples.
    
    Args:
        predictions: Predicted probabilities
        labels: True labels
        user_ids: User identifiers for grouping
        min_samples: Minimum samples per user to include in GAUC
        
    Returns:
        Tuple of (GAUC value, statistics dictionary)
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    user_ids = np.asarray(user_ids).flatten()
    
    # Group by user
    unique_users = np.unique(user_ids)
    
    weighted_auc_sum = 0.0
    weight_sum = 0.0
    valid_users = 0
    skipped_users = 0
    user_aucs = []
    
    for user in unique_users:
        mask = user_ids == user
        user_labels = labels[mask]
        user_preds = predictions[mask]
        
        # Skip if too few samples or only one class
        if len(user_labels) < min_samples:
            skipped_users += 1
            continue
        
        if len(np.unique(user_labels)) < 2:
            skipped_users += 1
            continue
        
        # Compute user AUC
        user_auc = roc_auc_score(user_labels, user_preds)
        user_aucs.append(user_auc)
        
        # Weighted sum
        weight = len(user_labels)
        weighted_auc_sum += user_auc * weight
        weight_sum += weight
        valid_users += 1
    
    gauc = weighted_auc_sum / weight_sum if weight_sum > 0 else 0.5
    
    stats = {
        'n_users': len(unique_users),
        'valid_users': valid_users,
        'skipped_users': skipped_users,
        'mean_user_auc': np.mean(user_aucs) if user_aucs else 0.5,
        'std_user_auc': np.std(user_aucs) if user_aucs else 0.0,
        'min_user_auc': np.min(user_aucs) if user_aucs else 0.5,
        'max_user_auc': np.max(user_aucs) if user_aucs else 0.5,
    }
    
    return gauc, stats


def compute_lift_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    percentiles: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute lift curve metrics.
    
    Lift measures how much better targeting is vs. random selection.
    
    Lift@k = (Conversion rate in top k%) / (Overall conversion rate)
    
    A lift of 2.0 at 10% means the top 10% by predicted CVR converts
    at twice the overall rate.
    
    Args:
        predictions: Predicted probabilities (higher = more likely to convert)
        labels: True conversion labels
        percentiles: Percentiles to compute lift at (default: [1, 5, 10, 20, 50])
        
    Returns:
        Dictionary with lift values and cumulative conversion data
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    
    if percentiles is None:
        percentiles = [1, 5, 10, 20, 30, 50, 100]
    
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(-predictions)
    sorted_labels = labels[sorted_indices]
    
    n_total = len(labels)
    overall_cvr = labels.mean()
    
    lift_data = {
        'percentiles': percentiles,
        'lift_values': [],
        'cvr_values': [],
        'n_samples': [],
        'overall_cvr': float(overall_cvr),
    }
    
    for pct in percentiles:
        n_top = max(1, int(n_total * pct / 100))
        top_labels = sorted_labels[:n_top]
        top_cvr = top_labels.mean()
        lift = top_cvr / overall_cvr if overall_cvr > 0 else 1.0
        
        lift_data['lift_values'].append(float(lift))
        lift_data['cvr_values'].append(float(top_cvr))
        lift_data['n_samples'].append(n_top)
    
    # Compute cumulative metrics for plotting
    cumulative_conversions = np.cumsum(sorted_labels)
    cumulative_samples = np.arange(1, n_total + 1)
    cumulative_cvr = cumulative_conversions / cumulative_samples
    
    # Sample at 100 points for smooth curve
    sample_points = np.linspace(0, n_total - 1, 100).astype(int)
    lift_data['curve'] = {
        'fraction': (sample_points + 1) / n_total,
        'lift': (cumulative_cvr[sample_points] / overall_cvr).tolist() if overall_cvr > 0 else [1.0] * len(sample_points),
        'cvr': cumulative_cvr[sample_points].tolist(),
    }
    
    return lift_data


def compute_popularity_scores(
    item_ids: np.ndarray,
    labels: np.ndarray,
    smoothing: float = 10.0,
    prior_mean: Optional[float] = None,
    category_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[Any, float]]:
    """Compute popularity-based scores for each sample.
    
    Popularity score for an item = smoothed historical conversion rate.
    This serves as a strong non-personalized baseline.
    
    Uses Bayesian smoothing (Beta-Binomial conjugate prior):
        smoothed_cvr = (conversions + α × prior_mean) / (impressions + α)
    
    Args:
        item_ids: Item identifier for each sample
        labels: True conversion labels (used to compute historical rates)
        smoothing: Prior strength α (pseudo-count). Controls shrinkage toward prior.
                   - α=10: Item needs ~10 impressions before observed CVR dominates
                   - α=100: More conservative, needs ~100 impressions
                   Rule of thumb: α ≈ median impressions per item
        prior_mean: Prior CVR to shrink toward. If None, uses global CVR (empirical Bayes).
                    Can also pass category-level CVRs via category_ids for hierarchical prior.
        category_ids: Optional category for each sample. If provided along with prior_mean=None,
                      uses category-level CVR as prior (hierarchical empirical Bayes).
                   
    Returns:
        Tuple of:
        - popularity_scores: Score for each sample based on item popularity
        - item_popularity: Dictionary mapping item_id -> popularity score
        
    Note:
        In production, popularity should be computed on TRAINING data only,
        then applied to test data. This function computes on the provided data
        for simplicity - be careful of data leakage in real experiments!
        
    Example choosing smoothing:
        # Method 1: Based on typical item volume
        item_counts = df.groupby('item_id').size()
        smoothing = item_counts.median()  # Items 50% data, 50% prior at median volume
        
        # Method 2: Conservative for bidding
        smoothing = item_counts.quantile(0.75)  # Need more data to trust
    """
    item_ids = np.asarray(item_ids).flatten()
    labels = np.asarray(labels).flatten()
    
    # Compute global CVR as default prior
    global_cvr = labels.mean()
    
    # Determine prior mean for each item
    if prior_mean is not None:
        # Fixed prior mean provided
        item_prior = {item: prior_mean for item in np.unique(item_ids)}
    elif category_ids is not None:
        # Hierarchical prior: use category-level CVR
        category_ids = np.asarray(category_ids).flatten()
        unique_categories = np.unique(category_ids)
        category_cvr = {}
        for cat in unique_categories:
            cat_mask = category_ids == cat
            category_cvr[cat] = labels[cat_mask].mean() if cat_mask.sum() > 0 else global_cvr
        
        # Map item -> category -> category CVR
        item_to_category = {}
        for item, cat in zip(item_ids, category_ids):
            item_to_category[item] = cat
        item_prior = {item: category_cvr[item_to_category[item]] for item in np.unique(item_ids)}
    else:
        # Default: empirical Bayes with global CVR
        item_prior = {item: global_cvr for item in np.unique(item_ids)}
    
    # Compute per-item smoothed CVR
    unique_items = np.unique(item_ids)
    item_popularity = {}
    
    for item in unique_items:
        mask = item_ids == item
        item_conversions = labels[mask].sum()
        item_impressions = mask.sum()
        
        # Bayesian smoothing: (conversions + α × prior) / (impressions + α)
        # Weight on observed = n/(n+α), weight on prior = α/(n+α)
        smoothed_cvr = (item_conversions + smoothing * item_prior[item]) / (item_impressions + smoothing)
        item_popularity[item] = smoothed_cvr
    
    # Map back to per-sample scores
    popularity_scores = np.array([item_popularity[item] for item in item_ids])
    
    return popularity_scores, item_popularity


def compute_lift_comparison(
    model_predictions: np.ndarray,
    labels: np.ndarray,
    item_ids: Optional[np.ndarray] = None,
    percentiles: Optional[List[float]] = None,
    smoothing: float = 10.0,
) -> Dict[str, Any]:
    """Compare lift curves: Model vs Popularity vs Random baselines.
    
    This answers the key business question: "Does personalization add value
    beyond just showing popular items?"
    
    Args:
        model_predictions: Model's predicted probabilities
        labels: True conversion labels
        item_ids: Optional item identifiers for popularity baseline.
                  If None, only model vs random is compared.
        percentiles: Percentiles to compute lift at
        smoothing: Smoothing for popularity calculation
        
    Returns:
        Dictionary with lift values for each method at each percentile,
        plus summary statistics.
        
    Example output:
        {
            'percentiles': [1, 5, 10, 20],
            'model': {'lift': [4.2, 3.5, 2.8, 2.1], 'cvr': [...]},
            'popularity': {'lift': [3.8, 3.1, 2.5, 1.9], 'cvr': [...]},
            'random': {'lift': [1.0, 1.0, 1.0, 1.0], 'cvr': [...]},
            'model_vs_popularity_gain': [0.4, 0.4, 0.3, 0.2],  # Lift difference
            'overall_cvr': 0.02,
        }
    """
    model_predictions = np.asarray(model_predictions).flatten()
    labels = np.asarray(labels).flatten()
    
    if percentiles is None:
        percentiles = [1, 5, 10, 20, 30, 50]
    
    n_total = len(labels)
    overall_cvr = labels.mean()
    
    results = {
        'percentiles': percentiles,
        'overall_cvr': float(overall_cvr),
        'n_samples': n_total,
        'model': {'lift': [], 'cvr': []},
        'random': {'lift': [], 'cvr': []},
    }
    
    # Model lift
    model_sorted_idx = np.argsort(-model_predictions)
    model_sorted_labels = labels[model_sorted_idx]
    
    for pct in percentiles:
        n_top = max(1, int(n_total * pct / 100))
        top_cvr = model_sorted_labels[:n_top].mean()
        lift = top_cvr / overall_cvr if overall_cvr > 0 else 1.0
        results['model']['lift'].append(float(lift))
        results['model']['cvr'].append(float(top_cvr))
        
        # Random baseline: lift is always 1.0 (by definition)
        results['random']['lift'].append(1.0)
        results['random']['cvr'].append(float(overall_cvr))
    
    # Popularity baseline (if item_ids provided)
    if item_ids is not None:
        item_ids = np.asarray(item_ids).flatten()
        popularity_scores, item_pop_dict = compute_popularity_scores(
            item_ids, labels, smoothing=smoothing
        )
        
        pop_sorted_idx = np.argsort(-popularity_scores)
        pop_sorted_labels = labels[pop_sorted_idx]
        
        results['popularity'] = {'lift': [], 'cvr': []}
        results['model_vs_popularity_gain'] = []
        
        for i, pct in enumerate(percentiles):
            n_top = max(1, int(n_total * pct / 100))
            top_cvr = pop_sorted_labels[:n_top].mean()
            lift = top_cvr / overall_cvr if overall_cvr > 0 else 1.0
            results['popularity']['lift'].append(float(lift))
            results['popularity']['cvr'].append(float(top_cvr))
            
            # How much does model beat popularity?
            gain = results['model']['lift'][i] - lift
            results['model_vs_popularity_gain'].append(float(gain))
        
        # Additional popularity statistics
        results['popularity_stats'] = {
            'n_unique_items': len(item_pop_dict),
            'min_item_popularity': float(min(item_pop_dict.values())),
            'max_item_popularity': float(max(item_pop_dict.values())),
            'mean_item_popularity': float(np.mean(list(item_pop_dict.values()))),
        }
    
    return results


def print_lift_comparison(comparison: Dict[str, Any]) -> str:
    """Format lift comparison as a readable table.
    
    Args:
        comparison: Output from compute_lift_comparison()
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("LIFT CURVE COMPARISON")
    lines.append(f"Overall CVR: {comparison['overall_cvr']:.4f} ({comparison['overall_cvr']*100:.2f}%)")
    lines.append(f"Total samples: {comparison['n_samples']:,}")
    lines.append("=" * 70)
    lines.append("")
    
    # Header
    has_popularity = 'popularity' in comparison
    if has_popularity:
        lines.append(f"{'Top %':<10} {'Model Lift':<12} {'Pop Lift':<12} {'Random':<10} {'Model-Pop':<12}")
        lines.append("-" * 60)
    else:
        lines.append(f"{'Top %':<10} {'Model Lift':<12} {'Random':<10}")
        lines.append("-" * 35)
    
    # Data rows
    for i, pct in enumerate(comparison['percentiles']):
        model_lift = comparison['model']['lift'][i]
        random_lift = comparison['random']['lift'][i]
        
        if has_popularity:
            pop_lift = comparison['popularity']['lift'][i]
            gain = comparison['model_vs_popularity_gain'][i]
            gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
            lines.append(f"{pct:>5}%     {model_lift:<12.2f} {pop_lift:<12.2f} {random_lift:<10.2f} {gain_str:<12}")
        else:
            lines.append(f"{pct:>5}%     {model_lift:<12.2f} {random_lift:<10.2f}")
    
    lines.append("")
    
    # Interpretation
    lines.append("Interpretation:")
    lines.append(f"  - Model Lift@10%: Top 10% by model score converts {comparison['model']['lift'][comparison['percentiles'].index(10)] if 10 in comparison['percentiles'] else 'N/A'}x better than random")
    
    if has_popularity:
        pop_lift_10 = comparison['popularity']['lift'][comparison['percentiles'].index(10)] if 10 in comparison['percentiles'] else None
        if pop_lift_10:
            lines.append(f"  - Popularity Lift@10%: Top 10% by item popularity converts {pop_lift_10:.2f}x better than random")
            gain_10 = comparison['model_vs_popularity_gain'][comparison['percentiles'].index(10)] if 10 in comparison['percentiles'] else 0
            if gain_10 > 0:
                lines.append(f"  - Personalization adds {gain_10:.2f} lift points over popularity baseline")
            else:
                lines.append(f"  - WARNING: Model underperforms popularity by {abs(gain_10):.2f} lift points!")
    
    return "\n".join(lines)


def compute_auc_pr(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve.
    
    Better than AUC-ROC for highly imbalanced datasets (common in CVR).
    """
    labels = np.asarray(labels).flatten()
    predictions = np.asarray(predictions).flatten()
    
    if len(np.unique(labels)) < 2:
        return 0.0
    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return auc(recall, precision)


def compute_adtech_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    user_ids: Optional[np.ndarray] = None,
    click_labels: Optional[np.ndarray] = None,
    compute_lift: bool = True,
) -> Dict[str, Any]:
    """Compute comprehensive adtech metrics.
    
    Args:
        predictions: Predicted probabilities (CVR or CTCVR)
        labels: True conversion labels
        user_ids: Optional user IDs for GAUC computation
        click_labels: Optional click labels for CTR metrics
        compute_lift: Whether to compute lift curve
        
    Returns:
        Dictionary with all computed metrics
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    
    metrics = {}
    
    # Basic metrics
    metrics['auc'] = compute_auc(labels, predictions)
    metrics['logloss'] = compute_logloss(labels, predictions)
    metrics['auc_pr'] = compute_auc_pr(labels, predictions)
    
    # Positive rate
    metrics['positive_rate'] = float(labels.mean())
    metrics['mean_prediction'] = float(predictions.mean())
    
    # GAUC if user_ids provided
    if user_ids is not None:
        gauc, gauc_stats = compute_gauc(predictions, labels, user_ids)
        metrics['gauc'] = gauc
        metrics['gauc_stats'] = gauc_stats
    
    # CTR metrics if click labels provided
    if click_labels is not None:
        click_labels = np.asarray(click_labels).flatten()
        clicked_mask = click_labels == 1
        
        metrics['click_rate'] = float(click_labels.mean())
        
        if clicked_mask.sum() > 0:
            metrics['post_click_cvr'] = float(labels[clicked_mask].mean())
            
            # CVR AUC on clicked samples
            if len(np.unique(labels[clicked_mask])) >= 2:
                metrics['cvr_auc_clicked'] = roc_auc_score(
                    labels[clicked_mask], 
                    predictions[clicked_mask]
                )
    
    # Lift curve
    if compute_lift:
        lift_data = compute_lift_curve(predictions, labels)
        metrics['lift'] = {
            'percentiles': lift_data['percentiles'],
            'values': lift_data['lift_values'],
            'cvr_at_percentile': lift_data['cvr_values'],
        }
        metrics['lift_at_10pct'] = lift_data['lift_values'][lift_data['percentiles'].index(10)] if 10 in lift_data['percentiles'] else None
        metrics['lift_at_20pct'] = lift_data['lift_values'][lift_data['percentiles'].index(20)] if 20 in lift_data['percentiles'] else None
    
    return metrics


def compute_bid_simulation_metrics(
    cvr_predictions: np.ndarray,
    conversion_values: np.ndarray,
    true_conversions: np.ndarray,
    budget: float,
    bid_strategies: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Simulate bidding strategies and compute revenue metrics.
    
    This simulates a simplified bidding scenario to evaluate how
    model quality affects business outcomes.
    
    Simplified model:
        - Each impression has a fixed cost (bid)
        - If conversion happens, we get conversion_value
        - Goal: Maximize ROI = (revenue - cost) / cost
    
    Args:
        cvr_predictions: Predicted conversion rates
        conversion_values: Value of each conversion (could be constant)
        true_conversions: True conversion labels (0/1)
        budget: Total budget available
        bid_strategies: List of strategies to evaluate
        
    Returns:
        Dictionary with metrics for each strategy
    """
    if bid_strategies is None:
        bid_strategies = ['value_based', 'top_k', 'random']
    
    n_samples = len(cvr_predictions)
    conversion_values = np.asarray(conversion_values).flatten()
    
    results = {}
    
    for strategy in bid_strategies:
        if strategy == 'value_based':
            # Bid proportional to expected value
            expected_values = cvr_predictions * conversion_values
            # Normalize bids to fit budget
            bid_weights = expected_values / expected_values.sum()
            bids = bid_weights * budget
            
        elif strategy == 'top_k':
            # Bid on top k% by CVR prediction
            k = 0.2  # Top 20%
            threshold_idx = int(n_samples * (1 - k))
            threshold = np.sort(cvr_predictions)[threshold_idx]
            mask = cvr_predictions >= threshold
            bids = np.zeros(n_samples)
            bids[mask] = budget / mask.sum()  # Equal bid for selected
            
        elif strategy == 'random':
            # Random bidding (baseline)
            bids = np.ones(n_samples) * (budget / n_samples)
        
        else:
            continue
        
        # Compute metrics
        total_cost = bids.sum()
        conversions = true_conversions * (bids > 0)  # Only count if we bid
        total_conversions = conversions.sum()
        total_revenue = (conversions * conversion_values).sum()
        
        roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
        cpa = total_cost / total_conversions if total_conversions > 0 else float('inf')
        
        results[strategy] = {
            'total_cost': float(total_cost),
            'total_revenue': float(total_revenue),
            'total_conversions': int(total_conversions),
            'roi': float(roi),
            'cpa': float(cpa),
            'conversion_rate': float(total_conversions / (bids > 0).sum()) if (bids > 0).sum() > 0 else 0,
        }
    
    return results


def print_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> str:
    """Format metrics comparison as a table.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        metric_keys: Which metrics to include (default: all common keys)
        
    Returns:
        Formatted table string
    """
    if not metrics_dict:
        return "No metrics to compare"
    
    # Get common keys
    all_keys = set()
    for metrics in metrics_dict.values():
        all_keys.update(k for k, v in metrics.items() if isinstance(v, (int, float)))
    
    if metric_keys is None:
        metric_keys = sorted(all_keys)
    
    # Build table
    model_names = list(metrics_dict.keys())
    
    # Header
    col_width = max(15, max(len(name) for name in model_names) + 2)
    header = f"{'Metric':<20} | " + " | ".join(f"{name:>{col_width}}" for name in model_names)
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for key in metric_keys:
        row = f"{key:<20} | "
        values = []
        for name in model_names:
            metrics = metrics_dict[name]
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    values.append(f"{val:>{col_width}.4f}")
                else:
                    values.append(f"{val:>{col_width}}")
            else:
                values.append(f"{'N/A':>{col_width}}")
        row += " | ".join(values)
        lines.append(row)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics utilities
    print("Testing adtech metrics...")
    
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 10000
    n_users = 500
    
    # True CVR varies by user
    user_ids = np.random.randint(0, n_users, n_samples)
    user_cvr = np.random.beta(0.5, 10, n_users)  # Most users have low CVR
    true_probs = user_cvr[user_ids]
    
    labels = (np.random.rand(n_samples) < true_probs).astype(float)
    
    # Model predictions (noisy version of true)
    predictions = np.clip(true_probs + np.random.randn(n_samples) * 0.1, 0.01, 0.99)
    
    # Click labels (higher than CVR)
    click_probs = np.clip(true_probs * 5, 0, 0.5)
    click_labels = (np.random.rand(n_samples) < click_probs).astype(float)
    
    print(f"\nSynthetic data: {n_samples} samples, {n_users} users")
    print(f"  Overall CVR: {labels.mean():.4f}")
    print(f"  Click rate: {click_labels.mean():.4f}")
    
    # Compute metrics
    print("\n" + "=" * 50)
    print("Computing adtech metrics...")
    
    metrics = compute_adtech_metrics(
        predictions, labels,
        user_ids=user_ids,
        click_labels=click_labels,
    )
    
    print("\nResults:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        elif key == 'lift':
            print(f"  {key}:")
            for pct, lift in zip(value['percentiles'], value['values']):
                print(f"    @{pct}%: {lift:.2f}x")
    
    # Test GAUC
    print("\n" + "=" * 50)
    print("GAUC Analysis:")
    gauc, stats = compute_gauc(predictions, labels, user_ids)
    print(f"  GAUC: {gauc:.4f}")
    print(f"  Global AUC: {compute_auc(labels, predictions):.4f}")
    print(f"  Valid users: {stats['valid_users']}/{stats['n_users']}")
    print(f"  Mean user AUC: {stats['mean_user_auc']:.4f} ± {stats['std_user_auc']:.4f}")
    
    # Test lift curve
    print("\n" + "=" * 50)
    print("Lift Curve:")
    lift_data = compute_lift_curve(predictions, labels)
    print(f"  Overall CVR: {lift_data['overall_cvr']:.4f}")
    for pct, lift, cvr in zip(lift_data['percentiles'], lift_data['lift_values'], lift_data['cvr_values']):
        print(f"  Top {pct:>3}%: Lift={lift:.2f}x, CVR={cvr:.4f}")
    
    # Test lift comparison with popularity baseline
    print("\n" + "=" * 50)
    print("Lift Comparison: Model vs Popularity vs Random")
    
    # Generate item IDs with varying popularity
    n_items = 200
    item_ids = np.random.randint(0, n_items, n_samples)
    
    # Make some items more popular (higher base CVR)
    item_base_cvr = np.random.beta(0.5, 10, n_items)
    
    # Labels now depend on both user and item
    item_effect = item_base_cvr[item_ids]
    combined_probs = np.clip(true_probs * 0.5 + item_effect * 0.5, 0, 1)
    labels_with_items = (np.random.rand(n_samples) < combined_probs).astype(float)
    
    # Model predictions (knows about both user and item effects)
    model_preds = np.clip(combined_probs + np.random.randn(n_samples) * 0.05, 0.01, 0.99)
    
    comparison = compute_lift_comparison(
        model_predictions=model_preds,
        labels=labels_with_items,
        item_ids=item_ids,
        percentiles=[1, 5, 10, 20, 50],
    )
    
    print(print_lift_comparison(comparison))
    
    # Test bid simulation
    print("\n" + "=" * 50)
    print("Bid Simulation:")
    conversion_values = np.ones(n_samples) * 10.0  # $10 per conversion
    bid_metrics = compute_bid_simulation_metrics(
        predictions, conversion_values, labels, budget=1000
    )
    
    for strategy, metrics in bid_metrics.items():
        print(f"\n  {strategy}:")
        print(f"    ROI: {metrics['roi']:.2%}")
        print(f"    CPA: ${metrics['cpa']:.2f}")
        print(f"    Conversions: {metrics['total_conversions']}")
    
    print("\nMetrics test passed!")

