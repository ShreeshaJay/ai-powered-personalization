"""
Chapter 9: Utility functions for Adtech modeling
"""

from .calibration import (
    compute_ece,
    platt_scaling,
    isotonic_calibration,
    temperature_scaling,
    calibration_curve,
    Calibrator,
)

from .metrics import (
    compute_adtech_metrics,
    compute_lift_curve,
    compute_gauc,
)

from .delayed_feedback import (
    compute_observation_weights,
    estimate_delay_distribution,
    apply_attribution_window_cutoff,
    DelayedFeedbackCorrector,
    simulate_delayed_feedback,
    analyze_delayed_feedback_bias,
)

__all__ = [
    # Calibration
    'compute_ece',
    'platt_scaling',
    'isotonic_calibration',
    'temperature_scaling',
    'calibration_curve',
    'Calibrator',
    # Metrics
    'compute_adtech_metrics',
    'compute_lift_curve',
    'compute_gauc',
    # Delayed Feedback
    'compute_observation_weights',
    'estimate_delay_distribution',
    'apply_attribution_window_cutoff',
    'DelayedFeedbackCorrector',
    'simulate_delayed_feedback',
    'analyze_delayed_feedback_bias',
]

