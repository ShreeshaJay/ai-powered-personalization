"""Tests for assignment, effect injection, and the synthetic panel in ab_test_simulation.py."""

import numpy as np
import polars as pl
import pytest

import ab_test_simulation as ab
import stats_utils as su
from common import mode_rng


def _write_tiny_listens(path):
    """Three users over days 0 to 59, so the window is days 32-59 and the pre-period days 4-31."""
    day = ab.SECONDS_PER_DAY
    rows = [  # (uid, day, seconds into the day, is_organic, played_ratio_pct)
        (1, 2, 10, 1, 100),     # before the pre-period: ignored
        (1, 10, 10, 1, 100),    # pre-period
        (1, 32, 10, 1, 90),     # window day 0, completed
        (1, 32, 500, 1, 20),    # window day 0, not completed
        (1, 59, 100, 0, 100),   # window day 27, recommendation-driven, completed
        (2, 20, 10, 1, 100),    # pre-period only: never exposed, so excluded
        (3, 40, 10, 0, 50),     # window day 8, no pre-period
    ]
    pl.DataFrame(
        {
            "uid": [r[0] for r in rows],
            "timestamp": [r[1] * day + r[2] for r in rows],
            "item_id": list(range(len(rows))),
            "is_organic": [r[3] for r in rows],
            "played_ratio_pct": [r[4] for r in rows],
        },
        schema={"uid": pl.UInt32, "timestamp": pl.UInt32, "item_id": pl.UInt32,
                "is_organic": pl.UInt8, "played_ratio_pct": pl.UInt16},
    ).write_parquet(path / "listens.parquet")


def test_load_yambda_panel_windows_and_aggregates(tmp_path):
    _write_tiny_listens(tmp_path)
    panel = ab.load_yambda_panel(tmp_path, completion_threshold=80, metric="listens")
    assert panel.n_users == 2 and panel.daily.shape == (2, ab.WINDOW_DAYS)
    np.testing.assert_array_equal(panel.pre, [1, 0])
    expected = np.zeros((2, ab.WINDOW_DAYS), dtype=int)
    expected[0, 0], expected[0, 27], expected[1, 8] = 2, 1, 1
    np.testing.assert_array_equal(panel.daily, expected)
    np.testing.assert_array_equal(panel.daily_listens, expected)
    completed = np.zeros_like(expected)
    completed[0, 0], completed[0, 27] = 1, 1
    np.testing.assert_array_equal(panel.daily_completed, completed)
    np.testing.assert_array_equal(panel.first_day, [0, 8])

    rec = ab.load_yambda_panel(tmp_path, completion_threshold=80, metric="rec_listens")
    rec_expected = np.zeros_like(expected)
    rec_expected[0, 27], rec_expected[1, 8] = 1, 1
    np.testing.assert_array_equal(rec.daily, rec_expected)
    np.testing.assert_array_equal(rec.pre, [0, 0])
    np.testing.assert_array_equal(rec.daily_listens, expected)


def test_mode_streams_are_reproducible_and_distinct():
    a = mode_rng(16, "cuped").random(5)
    np.testing.assert_array_equal(a, mode_rng(16, "cuped").random(5))
    assert not np.allclose(a, mode_rng(16, "power").random(5))
    assert not np.allclose(a, mode_rng(17, "cuped").random(5))


def test_hash_assignment_is_deterministic_balanced_and_salt_independent():
    ids = np.arange(200_000)
    z1 = ab.hash_assign(ids, "exp-1")
    np.testing.assert_array_equal(z1, ab.hash_assign(ids, "exp-1"))
    assert z1.mean() == pytest.approx(0.5, abs=0.005)
    z2 = ab.hash_assign(ids, "exp-2")
    assert np.mean(z1 == z2) == pytest.approx(0.5, abs=0.005)   # independent layers
    assert ab.hash_assign(ids, "exp-3", treat_fraction=0.1).mean() == pytest.approx(0.1, abs=0.003)


def test_count_injection_has_the_stated_mean():
    rng = np.random.default_rng(0)
    y = rng.poisson(20, 200_000)
    for eff in (0.05, -0.05):
        out = ab.inject_count_effect(y, eff, rng)
        assert out.mean() / y.mean() == pytest.approx(1 + eff, abs=0.003)
        assert np.all(out >= 0)
    np.testing.assert_array_equal(ab.inject_count_effect(y, 0.0, rng), y)


def test_ratio_injection_stays_in_range():
    rng = np.random.default_rng(1)
    listens = rng.poisson(30, 50_000)
    completed = rng.binomial(listens, 0.6)
    out = ab.inject_ratio_effect(completed, listens, 0.1, rng)
    assert np.all(out <= listens) and np.all(out >= completed)
    rate0, rate1 = completed.sum() / listens.sum(), out.sum() / listens.sum()
    assert rate1 == pytest.approx(rate0 + 0.1 * (1 - rate0), abs=0.003)


def test_novelty_curve_endpoints():
    assert ab.novelty_curve(0, 0.15, 0.03, 4.0) == pytest.approx(0.15)
    assert ab.novelty_curve(1000, 0.15, 0.03, 4.0) == pytest.approx(0.03)
    assert ab.novelty_curve(4.0, -0.1, 0.0, 4.0) == pytest.approx(-0.1 / np.e)


def test_synthetic_panel_shape_and_aa_calibration():
    rng = np.random.default_rng(2)
    panel = ab.synthetic_panel(3000, rng)
    assert panel.daily.shape[1] == ab.WINDOW_DAYS
    assert np.all(panel.total > 0)
    assert np.all(panel.daily_completed <= panel.daily_listens)
    ids = np.arange(panel.n_users)
    p = [su.welch_test(panel.total[z], panel.total[~z]).p_value
         for z in (ab.hash_assign(ids, f"t-{s}") for s in range(400))]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.03)
