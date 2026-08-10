"""
Model Calibration Utilities for Adtech
======================================

Calibration is critical in adtech because predicted probabilities directly
influence bidding decisions. A miscalibrated model will:
- Overbid (overconfident): Waste budget on low-value impressions
- Underbid (underconfident): Miss high-value opportunities

This module provides calibration techniques and evaluation metrics.

Calibration Methods:
    1. Platt Scaling: Logistic regression on predicted probabilities
    2. Isotonic Regression: Non-parametric monotonic calibration
    3. Temperature Scaling: Simple scalar adjustment (neural network specific)

Evaluation:
    - Expected Calibration Error (ECE): Measures calibration quality
    - Calibration curves: Visual inspection of calibration

Usage:
    from utils.calibration import Calibrator, compute_ece
    
    # Fit calibrator
    calibrator = Calibrator(method='isotonic')
    calibrator.fit(val_preds, val_labels)
    
    # Calibrate predictions
    calibrated_preds = calibrator.calibrate(test_preds)
    
    # Evaluate
    ece_before = compute_ece(test_preds, test_labels)
    ece_after = compute_ece(calibrated_preds, test_labels)
"""

import numpy as np
from typing import Tuple, Optional, Literal, Dict, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.optim as optim


def compute_ece(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match observed frequencies.
    Lower is better (0 = perfectly calibrated).
    
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    
    where:
        B_b = samples in bin b
        acc(B_b) = accuracy (true positive rate) in bin b
        conf(B_b) = mean predicted probability in bin b
    
    Args:
        predictions: Predicted probabilities in [0, 1]
        labels: Binary ground truth labels
        n_bins: Number of calibration bins
        
    Returns:
        Tuple of (ECE value, bin statistics dictionary)
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        if i == n_bins - 1:
            # Include upper boundary for last bin
            mask = (predictions >= lower) & (predictions <= upper)
        else:
            mask = (predictions >= lower) & (predictions < upper)
        
        n_samples = mask.sum()
        
        if n_samples > 0:
            bin_conf = predictions[mask].mean()  # Mean predicted probability
            bin_acc = labels[mask].mean()  # Actual positive rate
            
            bin_ece = n_samples * abs(bin_conf - bin_acc)
            ece += bin_ece
            
            bin_stats.append({
                'bin': i,
                'lower': lower,
                'upper': upper,
                'n_samples': int(n_samples),
                'confidence': float(bin_conf),
                'accuracy': float(bin_acc),
                'gap': float(abs(bin_conf - bin_acc)),
            })
        else:
            bin_stats.append({
                'bin': i,
                'lower': lower,
                'upper': upper,
                'n_samples': 0,
                'confidence': None,
                'accuracy': None,
                'gap': None,
            })
    
    ece = ece / len(predictions)
    
    return float(ece), {'bins': bin_stats, 'n_bins': n_bins}


def calibration_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve data.
    
    Returns mean predicted probability and true positive rate per bin.
    Perfect calibration: predicted = actual (diagonal line).
    
    Args:
        predictions: Predicted probabilities
        labels: Binary labels
        n_bins: Number of bins
        
    Returns:
        Tuple of (mean_predicted, true_rate, bin_counts)
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    mean_predicted = []
    true_rate = []
    bin_counts = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            mask = (predictions >= lower) & (predictions <= upper)
        else:
            mask = (predictions >= lower) & (predictions < upper)
        
        n_samples = mask.sum()
        bin_counts.append(n_samples)
        
        if n_samples > 0:
            mean_predicted.append(predictions[mask].mean())
            true_rate.append(labels[mask].mean())
        else:
            mean_predicted.append((lower + upper) / 2)
            true_rate.append(np.nan)
    
    return np.array(mean_predicted), np.array(true_rate), np.array(bin_counts)


def platt_scaling(
    val_predictions: np.ndarray,
    val_labels: np.ndarray,
    test_predictions: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, LogisticRegression]:
    """Apply Platt scaling calibration.
    
    Fits a logistic regression: calibrated = sigmoid(A * pred + B)
    
    Good for:
        - General calibration of any classifier
        - Monotonic transformation
        - Well-behaved with enough validation data
    
    Args:
        val_predictions: Validation set predictions for fitting
        val_labels: Validation set labels
        test_predictions: Test predictions to calibrate (optional)
        
    Returns:
        Tuple of (calibrated predictions, fitted calibrator)
    """
    val_predictions = np.asarray(val_predictions).reshape(-1, 1)
    val_labels = np.asarray(val_labels).flatten()
    
    # Fit logistic regression
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(val_predictions, val_labels)
    
    if test_predictions is not None:
        test_predictions = np.asarray(test_predictions).reshape(-1, 1)
        calibrated = calibrator.predict_proba(test_predictions)[:, 1]
        return calibrated, calibrator
    else:
        calibrated = calibrator.predict_proba(val_predictions)[:, 1]
        return calibrated, calibrator


def isotonic_calibration(
    val_predictions: np.ndarray,
    val_labels: np.ndarray,
    test_predictions: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, IsotonicRegression]:
    """Apply isotonic regression calibration.
    
    Non-parametric calibration that learns a monotonic mapping.
    
    Good for:
        - Flexible, non-parametric calibration
        - Guaranteed monotonicity
        - Works well when relationship is non-linear
    
    Caution:
        - Can overfit with small validation sets
        - May produce step-like calibration functions
    
    Args:
        val_predictions: Validation set predictions for fitting
        val_labels: Validation set labels
        test_predictions: Test predictions to calibrate (optional)
        
    Returns:
        Tuple of (calibrated predictions, fitted calibrator)
    """
    val_predictions = np.asarray(val_predictions).flatten()
    val_labels = np.asarray(val_labels).flatten()
    
    # Fit isotonic regression
    calibrator = IsotonicRegression(
        out_of_bounds='clip',
        y_min=0.0,
        y_max=1.0,
    )
    calibrator.fit(val_predictions, val_labels)
    
    if test_predictions is not None:
        test_predictions = np.asarray(test_predictions).flatten()
        calibrated = calibrator.predict(test_predictions)
        return calibrated, calibrator
    else:
        calibrated = calibrator.predict(val_predictions)
        return calibrated, calibrator


def temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: Optional[np.ndarray] = None,
    max_iter: int = 100,
    lr: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """Apply temperature scaling calibration.
    
    Learns a single temperature parameter T such that:
        calibrated = sigmoid(logit / T)
    
    - T > 1: Softens predictions (less confident)
    - T < 1: Sharpens predictions (more confident)
    
    Good for:
        - Neural network calibration
        - Simple, interpretable
        - Preserves ranking (doesn't change order)
    
    Requires:
        - Logits (pre-sigmoid) rather than probabilities
    
    Args:
        val_logits: Validation logits (pre-sigmoid)
        val_labels: Validation labels
        test_logits: Test logits to calibrate (optional)
        max_iter: Maximum optimization iterations
        lr: Learning rate
        
    Returns:
        Tuple of (calibrated predictions, learned temperature)
    """
    val_logits = torch.tensor(val_logits, dtype=torch.float32).flatten()
    val_labels = torch.tensor(val_labels, dtype=torch.float32).flatten()
    
    # Initialize temperature
    temperature = nn.Parameter(torch.ones(1))
    
    # Optimize temperature
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = val_logits / temperature
        loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, val_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    T = float(temperature.item())
    
    # Apply calibration
    if test_logits is not None:
        test_logits = torch.tensor(test_logits, dtype=torch.float32).flatten()
        calibrated = torch.sigmoid(test_logits / T).numpy()
        return calibrated, T
    else:
        calibrated = torch.sigmoid(val_logits / T).numpy()
        return calibrated, T


@dataclass
class Calibrator:
    """Unified calibration interface.
    
    Supports multiple calibration methods with a consistent API.
    
    Usage:
        calibrator = Calibrator(method='isotonic')
        calibrator.fit(val_preds, val_labels)
        calibrated = calibrator.calibrate(test_preds)
    """
    method: Literal['platt', 'isotonic', 'temperature'] = 'isotonic'
    
    def __post_init__(self):
        self._fitted = False
        self._calibrator = None
        self._temperature = None
    
    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> 'Calibrator':
        """Fit the calibrator on validation data.
        
        Args:
            predictions: Predicted probabilities
            labels: True labels
            logits: Pre-sigmoid logits (required for temperature scaling)
            
        Returns:
            Self for chaining
        """
        if self.method == 'platt':
            _, self._calibrator = platt_scaling(predictions, labels)
        elif self.method == 'isotonic':
            _, self._calibrator = isotonic_calibration(predictions, labels)
        elif self.method == 'temperature':
            if logits is None:
                raise ValueError("Temperature scaling requires logits (pre-sigmoid)")
            _, self._temperature = temperature_scaling(logits, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._fitted = True
        return self
    
    def calibrate(
        self,
        predictions: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calibrate predictions.
        
        Args:
            predictions: Predictions to calibrate
            logits: Pre-sigmoid logits (required for temperature scaling)
            
        Returns:
            Calibrated predictions
        """
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'platt':
            predictions = np.asarray(predictions).reshape(-1, 1)
            return self._calibrator.predict_proba(predictions)[:, 1]
        elif self.method == 'isotonic':
            predictions = np.asarray(predictions).flatten()
            return self._calibrator.predict(predictions)
        elif self.method == 'temperature':
            if logits is None:
                raise ValueError("Temperature scaling requires logits")
            logits = torch.tensor(logits, dtype=torch.float32).flatten()
            return torch.sigmoid(logits / self._temperature).numpy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get calibration parameters for serialization."""
        if self.method == 'platt':
            return {
                'method': 'platt',
                'coef': self._calibrator.coef_.tolist(),
                'intercept': self._calibrator.intercept_.tolist(),
            }
        elif self.method == 'isotonic':
            return {
                'method': 'isotonic',
                'X_thresholds': self._calibrator.X_thresholds_.tolist() if self._calibrator.X_thresholds_ is not None else None,
                'y_thresholds': self._calibrator.y_thresholds_.tolist() if self._calibrator.y_thresholds_ is not None else None,
            }
        elif self.method == 'temperature':
            return {
                'method': 'temperature',
                'temperature': self._temperature,
            }


def compute_calibration_comparison(
    predictions: np.ndarray,
    labels: np.ndarray,
    logits: Optional[np.ndarray] = None,
    val_ratio: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Compare calibration methods.
    
    Splits data into calibration and test sets, fits each method,
    and computes ECE before and after calibration.
    
    Args:
        predictions: Predicted probabilities
        labels: True labels
        logits: Pre-sigmoid logits (optional, for temperature scaling)
        val_ratio: Fraction of data for calibrator fitting
        
    Returns:
        Dictionary with results for each method
    """
    n = len(predictions)
    indices = np.random.permutation(n)
    n_val = int(n * val_ratio)
    
    val_idx = indices[:n_val]
    test_idx = indices[n_val:]
    
    val_preds = predictions[val_idx]
    val_labels = labels[val_idx]
    test_preds = predictions[test_idx]
    test_labels = labels[test_idx]
    
    # Baseline ECE
    ece_baseline, _ = compute_ece(test_preds, test_labels)
    
    results = {
        'baseline': {
            'ece': ece_baseline,
            'method': 'none',
        }
    }
    
    # Platt scaling
    calibrator = Calibrator(method='platt')
    calibrator.fit(val_preds, val_labels)
    platt_preds = calibrator.calibrate(test_preds)
    ece_platt, _ = compute_ece(platt_preds, test_labels)
    results['platt'] = {
        'ece': ece_platt,
        'ece_reduction': (ece_baseline - ece_platt) / ece_baseline if ece_baseline > 0 else 0,
    }
    
    # Isotonic
    calibrator = Calibrator(method='isotonic')
    calibrator.fit(val_preds, val_labels)
    iso_preds = calibrator.calibrate(test_preds)
    ece_iso, _ = compute_ece(iso_preds, test_labels)
    results['isotonic'] = {
        'ece': ece_iso,
        'ece_reduction': (ece_baseline - ece_iso) / ece_baseline if ece_baseline > 0 else 0,
    }
    
    # Temperature scaling (if logits provided)
    if logits is not None:
        val_logits = logits[val_idx]
        test_logits = logits[test_idx]
        
        calibrator = Calibrator(method='temperature')
        calibrator.fit(val_preds, val_labels, logits=val_logits)
        temp_preds = calibrator.calibrate(test_preds, logits=test_logits)
        ece_temp, _ = compute_ece(temp_preds, test_labels)
        results['temperature'] = {
            'ece': ece_temp,
            'ece_reduction': (ece_baseline - ece_temp) / ece_baseline if ece_baseline > 0 else 0,
            'temperature': calibrator._temperature,
        }
    
    return results


if __name__ == "__main__":
    # Test calibration utilities
    print("Testing calibration utilities...")
    
    np.random.seed(42)
    
    # Generate synthetic poorly calibrated predictions
    n_samples = 10000
    true_probs = np.random.beta(0.5, 0.5, n_samples)  # Bimodal
    labels = (np.random.rand(n_samples) < true_probs).astype(float)
    
    # Simulate overconfident model
    predictions = np.clip(true_probs * 1.5, 0, 1)  # Scale up predictions
    logits = np.log(predictions / (1 - predictions + 1e-7))  # Convert to logits
    
    print(f"\nSynthetic data: {n_samples} samples")
    print(f"  Positive rate: {labels.mean():.3f}")
    print(f"  Mean prediction: {predictions.mean():.3f}")
    
    # Test ECE
    ece, bin_stats = compute_ece(predictions, labels)
    print(f"\nBaseline ECE: {ece:.4f}")
    
    # Test calibration curve
    mean_pred, true_rate, counts = calibration_curve(predictions, labels)
    print("\nCalibration curve (first 5 bins):")
    for i in range(5):
        print(f"  Bin {i}: pred={mean_pred[i]:.3f}, true={true_rate[i]:.3f}, count={counts[i]}")
    
    # Test calibration methods
    print("\n" + "=" * 50)
    print("Comparing calibration methods...")
    
    results = compute_calibration_comparison(predictions, labels, logits)
    
    print("\nResults:")
    for method, metrics in results.items():
        print(f"  {method}: ECE={metrics['ece']:.4f}", end="")
        if 'ece_reduction' in metrics:
            print(f" (reduction: {metrics['ece_reduction']:.1%})", end="")
        if 'temperature' in metrics:
            print(f" (T={metrics['temperature']:.3f})", end="")
        print()
    
    # Test Calibrator class
    print("\n" + "=" * 50)
    print("Testing Calibrator class...")
    
    calibrator = Calibrator(method='isotonic')
    calibrator.fit(predictions[:5000], labels[:5000])
    calibrated = calibrator.calibrate(predictions[5000:])
    
    ece_calibrated, _ = compute_ece(calibrated, labels[5000:])
    print(f"  Isotonic calibrated ECE: {ece_calibrated:.4f}")
    print(f"  Calibrator params: {calibrator.get_params()['method']}")
    
    print("\nCalibration test passed!")

