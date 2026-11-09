"""End-to-end run on synthetic Yambda-shaped data through the vendored Chapter 6 pipeline."""

import json

import numpy as np
import polars as pl
import pytest

import ranker as rk
import run_exercise as rx

N_USERS = 150

# item_id is frequency-encoded (item_id_freq) above Chapter 6's 10,000-value threshold, as on
# Yambda, and label-encoded (item_id_encoded) below it, as on the small synthetic data.
EXPECTED_FEATURES = {
    "uid_encoded", "is_organic_encoded", "track_length_seconds", "hour_of_day",
    "day_of_week", "user_total_listens", "user_avg_completion", "user_std_completion",
    "user_median_completion", "user_unique_items", "user_organic_ratio", "user_active_span",
    "user_listen_rate", "item_total_plays", "item_avg_completion", "item_std_completion",
    "item_unique_listeners", "item_organic_ratio", "item_repeat_ratio", "item_like_ratio",
    "has_listened_before", "previous_listen_count",
}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("ch15")
    cfg = rx.ExerciseConfig(
        ranker=rk.RankerConfig(train_days=12, n_estimators=30, n_jobs=2),
        synthetic=True, synthetic_n_users=N_USERS, output_dir=str(out), run_name="test")
    return rx.run_experiment(cfg)


@pytest.fixture(scope="module")
def listens(result, tmp_path_factory):
    # The same synthetic listens the run used (it deletes its own copy).
    data_dir = rx.write_synthetic_yambda(tmp_path_factory.mktemp("listens"), n_users=N_USERS,
                                         seed=result.config.ranker.seed)
    return pl.read_parquet(data_dir / "listens.parquet")


def _status(res, scenario, slc, signal, feature=None):
    s = res.signals.filter((pl.col("scenario") == scenario) & (pl.col("slice") == slc)
                           & (pl.col("signal") == signal))
    if feature is not None:
        s = s.filter(pl.col("feature") == feature)
    return s["status"].to_list()


def test_chapter6_contract(result):
    # The Chapter 6 PolarsPipeline still produces the columns this exercise relies on.
    cols = set(result.prepared.feature_columns)
    assert EXPECTED_FEATURES <= cols
    assert len(cols & {"item_id_freq", "item_id_encoded"}) == 1
    frames = result.prepared.frames
    b = result.prepared.bounds
    assert frames["train"]["timestamp"].max() < b["val_start"]
    assert frames["val"]["timestamp"].max() < b["reference_start"]
    assert frames["reference"]["timestamp"].max() < b["test_start"]
    assert frames["test"]["timestamp"].max() - frames["test"]["timestamp"].min() <= rk.TS_UNITS_PER_DAY
    assert set(frames["test"]["label"].unique().to_list()) <= {0, 1}


def test_training_aggregates_exclude_own_labels(result, listens):
    # Each training row's user_avg_completion is the user's mean over the window before the
    # training rows. Aggregates fit on the training rows themselves (the old fit_transform)
    # would include the row's own played_ratio_pct and fail this.
    b = result.prepared.bounds
    assert b["train_history_start"] < b["train_rows_start"]
    ts = pl.col("timestamp")
    expected = (listens.filter((ts >= b["train_history_start"]) & (ts < b["train_rows_start"]))
                .group_by("uid").agg(pl.col("played_ratio_pct").mean().alias("expected")))
    train = result.prepared.frames["train"]
    assert train["timestamp"].min() >= b["train_rows_start"]
    got = train.select("uid", "user_avg_completion").join(expected, on="uid", how="inner")
    assert got.height > 0.5 * train.height
    np.testing.assert_allclose(got["user_avg_completion"].to_numpy(), got["expected"].to_numpy(),
                               rtol=1e-5)
    in_sample = (listens.filter((ts >= b["train_rows_start"]) & (ts < b["val_start"]))
                 .group_by("uid").agg(pl.col("played_ratio_pct").mean().alias("in_sample")))
    leaked = got.join(in_sample, on="uid")
    assert not np.allclose(leaked["user_avg_completion"].to_numpy(), leaked["in_sample"].to_numpy(),
                           rtol=1e-5)


def test_calendar_features_from_seconds(result):
    for f in result.prepared.frames.values():
        assert (f["hour_of_day"] == (f["timestamp"] // 3600) % 24).all()
        assert (f["day_of_week"] == (f["timestamp"] // rk.SECONDS_PER_DAY) % 7).all()


def test_served_days_use_refreshed_lookups(result):
    p = result.prepared
    for w in rk.SERVED_WINDOWS:
        served = p.served_pipelines[w]
        assert served.encoder is p.pipeline.encoder
        assert served.hist_builder is not p.pipeline.hist_builder
    fit_users = p.pipeline.hist_builder.user_features.sort("uid")
    test_users = p.served_pipelines["test"].hist_builder.user_features.sort("uid")
    assert not test_users.equals(fit_users)


def test_outputs_written(result):
    for name in ("run_manifest.json", "signals_long.csv", "signal_summary.csv", "signal_summary.md",
                 "quality_metrics.csv", "reliability_bins.csv", "feature_importance.csv",
                 "clean_false_alarms.csv", "model/model.json", "model/ch15_model_metadata.json"):
        assert (result.out_dir / name).is_file(), name
    manifest = json.loads((result.out_dir / "run_manifest.json").read_text())
    assert manifest["provenance"]["data"] == {"synthetic": True}
    assert manifest["targets"] == result.targets


def test_targets_are_eligible(result):
    assert len(result.targets) == 2
    for t in result.targets:
        assert not t.startswith(rk.IDENTIFIER_PREFIXES) and not t.endswith("_encoded")


def test_pipeline_monitors_stay_green(result):
    pipe = result.signals.filter((pl.col("layer") == "pipeline") & (pl.col("scenario") != "clean"))
    assert set(pipe["status"].to_list()) <= {"green", "clean also over"}
    assert set(pipe.filter(pl.col("signal") == "schema_pass")["status"].to_list()) == {"green"}


def test_mean_fill_evades_null_zero_but_not_ks_psi(result):
    for t in result.targets:
        assert _status(result, "mean_fill_all", "aggregate", "null_rate", t) == ["green"]
        assert _status(result, "mean_fill_all", "aggregate", "zero_rate", t) == ["green"]
        for sig in ("ks", "psi"):
            assert _status(result, "mean_fill_all", "aggregate", sig, t)[0].startswith("FIRES"), sig


def test_zero_fill_fires_zero_rate(result):
    for t in result.targets:
        assert _status(result, "zero_fill_all", "aggregate", "zero_rate", t) == ["FIRES"]
        assert _status(result, "zero_fill_all", "aggregate", "null_rate", t) == ["green"]


def test_segment_corruption_leaves_rest_untouched(result):
    s = result.signals.filter((pl.col("scenario") == "zero_fill_segment") & (pl.col("slice") == "rest"))
    np.testing.assert_allclose(s["statistic"].to_numpy(), s["clean_statistic"].to_numpy(),
                               rtol=0, atol=1e-12, equal_nan=True)
    seg = _status(result, "zero_fill_segment", "segment", "zero_rate", result.targets[0])
    assert seg == ["FIRES"]
