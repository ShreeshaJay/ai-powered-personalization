"""Known-answer checks for the harness metrics and the sealed split."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from harness.evaluate import expected_calibration_error, score_joined
from harness.leakage import LeakageError, check_coverage, check_metadata, check_reproduction, enforce
from harness.noise import build_noise_band, sample_std
from harness.splits import boundaries_from_end
from report import expected_maximum


def test_auc_matches_a_hand_count():
    frame = pl.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "prediction": [0.1, 0.4, 0.35, 0.8],
        }
    )
    metrics = score_joined(frame, split="validation")
    # Three of the four positive-negative pairs are in the right order.
    assert metrics["auc_roc"] == pytest.approx(0.75)
    assert metrics["split"] == "validation"


def test_ece_of_one_bin_is_the_absolute_gap():
    labels = np.array([0, 0], dtype=int)
    scores = np.array([0.25, 0.75], dtype=float)
    assert expected_calibration_error(labels, scores, n_bins=1) == pytest.approx(0.5)


def test_split_has_two_gaps_and_one_day_windows():
    bounds = boundaries_from_end(
        2_000_000,
        train_days=7,
        validation_days=1,
        test_days=1,
        gap_seconds=1800,
    )
    assert bounds["validation_start"] - bounds["train_end"] == 1800
    assert bounds["test_start"] - bounds["validation_end"] == 1800
    assert bounds["validation_end"] - bounds["validation_start"] == 86400
    assert bounds["test_end"] - bounds["test_start"] == 86400
    assert bounds["train_end"] - bounds["train_start"] == 7 * 86400


def test_leakage_and_reproduction_checks():
    assert check_metadata(
        {
            "early_stopping_split": "train_tail",
            "model_selection_split": "validation",
            "optimization_split": "validation",
            "preprocessor_fit_split": "train",
            "primary_metric": "auc_roc",
            "primary_split": "validation",
        }
    ) == []
    assert check_metadata({"early_stopping_split": "test"})
    labels = pl.DataFrame({"row_id": [1, 2], "uid": [1, 1], "label": [0, 1]})
    dropped = pl.DataFrame({"row_id": [1], "prediction": [0.2]})
    assert check_coverage(dropped, labels)
    assert check_reproduction({"auc_roc": 0.5}, {"auc_roc": 0.9})
    with pytest.raises(LeakageError):
        enforce(["forged score"])


def test_noise_band_is_the_larger_spread():
    band = build_noise_band([0.50, 0.52, 0.51, 0.53, 0.49], bootstrap_std=0.02)
    assert band["method"] == "baseline_seeds_plus_user_bootstrap"
    assert band["noise_band"] == pytest.approx(max(sample_std([0.50, 0.52, 0.51, 0.53, 0.49]), 0.02))


def test_expected_maximum_of_two_scores():
    rows = expected_maximum([0.2, 0.5])
    assert rows[0]["expected_maximum_validation_auc"] == pytest.approx(0.35)
    assert rows[1]["expected_maximum_validation_auc"] == pytest.approx(0.5)
