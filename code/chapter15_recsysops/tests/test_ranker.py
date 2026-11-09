"""Tests for window carving, point-in-time counts, target selection, and the contract."""

import polars as pl
import pytest

import ranker as rk
import run_exercise as rx

DAY = rk.TS_UNITS_PER_DAY


def test_pit_counts_only_earlier_plays():
    frames = {
        "train": pl.DataFrame({"uid": [1, 1, 2, 1], "item_id": [10, 10, 10, 20],
                               "timestamp": [1, 5, 3, 2]}),
        "val": pl.DataFrame({"uid": [1], "item_id": [10], "timestamp": [5]}),
        "test": pl.DataFrame({"uid": [1, 2, 3], "item_id": [10, 10, 10], "timestamp": [9, 9, 9]}),
    }
    out = rk.add_pit_user_item_features(frames)
    assert out["train"]["previous_listen_count"].to_list() == [0, 1, 0, 0]
    assert out["val"]["previous_listen_count"].to_list() == [2]       # tie at ts 5: train row first
    assert out["test"]["previous_listen_count"].to_list() == [3, 1, 0]
    assert out["test"]["has_listened_before"].to_list() == [1, 1, 0]
    assert out["train"]["uid"].to_list() == [1, 1, 2, 1]              # row order preserved


def test_pit_counts_within_one_frame():
    # Three plays of one pair, then one of another: a whole-frame count would give 2, 2, 2.
    frames = {"train": pl.DataFrame({"uid": [7, 7, 7, 7], "item_id": [3, 3, 3, 4],
                                     "timestamp": [30, 10, 20, 15]})}
    out = rk.add_pit_user_item_features(frames)["train"]
    assert out["previous_listen_count"].to_list() == [2, 0, 1, 0]


def test_pit_counts_history_plays():
    # Plays in the training rows' history window count for the training rows.
    frames = {
        "history": pl.DataFrame({"uid": [1, 1], "item_id": [10, 10], "timestamp": [1, 2]}),
        "train": pl.DataFrame({"uid": [1, 2], "item_id": [10, 10], "timestamp": [5, 5]}),
    }
    out = rk.add_pit_user_item_features(frames, lookback_units=10)
    assert out["train"]["previous_listen_count"].to_list() == [2, 0]


def test_total_order_is_input_order_free():
    df = pl.DataFrame({"timestamp": [2, 1, 1, 1], "uid": [1, 2, 1, 1], "item_id": [5, 5, 5, 5],
                       "x": [0.0, 1.0, 3.0, 2.0]})
    a = rk.total_order(df)
    assert a.equals(rk.total_order(df.reverse()))
    assert a["x"].to_list() == [2.0, 3.0, 1.0, 0.0]


def test_load_global_split_true_days(tmp_path):
    ts = list(range(0, 10 * DAY + 1, 600))
    pl.DataFrame({"timestamp": ts, "uid": [1] * len(ts)}).write_parquet(tmp_path / "listens.parquet")
    train, test = rk.load_global_split(tmp_path, train_days=3, test_days=1, dtypes={"timestamp": pl.Int32})
    hi = 10 * DAY
    assert test["timestamp"].max() == hi and test["timestamp"].min() == hi - DAY
    assert train["timestamp"].max() < hi - DAY - rk.GAP_SECONDS
    assert train["timestamp"].min() == hi - DAY - rk.GAP_SECONDS - 3 * DAY
    assert train.schema["timestamp"] == pl.Int32


def test_carve_windows_order_and_coverage():
    ts = list(range(0, 5 * DAY, 50))
    df = pl.DataFrame({"timestamp": ts, "uid": [1] * len(ts)})
    fit, val, ref, b = rk.carve_windows(df, reference_days=1, val_ratio=0.2)
    assert fit.height + val.height + ref.height == df.height
    assert fit["timestamp"].max() < b["val_start"] <= val["timestamp"].min()
    assert val["timestamp"].max() < b["reference_start"] <= ref["timestamp"].min()
    assert ref["timestamp"].max() - ref["timestamp"].min() < DAY
    assert val.height == pytest.approx(0.2 * (fit.height + val.height), abs=2)


def test_carve_windows_rejects_bad_arguments():
    df = pl.DataFrame({"timestamp": list(range(0, 3 * DAY, 500))})
    with pytest.raises(ValueError):
        rk.carve_windows(df, reference_days=0, val_ratio=0.1)
    with pytest.raises(ValueError):
        rk.carve_windows(df, reference_days=1, val_ratio=1.0)
    with pytest.raises(ValueError):
        rk.carve_windows(df, reference_days=3, val_ratio=0.1)


def test_target_selection_skips_identifiers_codes_and_binaries():
    importance = pl.DataFrame({
        "feature": ["uid_encoded", "item_id_freq", "is_organic_encoded", "has_listened_before",
                    "item_avg_completion", "user_total_listens", "unused"],
        "gain": [50.0, 40.0, 30.0, 25.0, 20.0, 10.0, 0.0],
    })
    train = pl.DataFrame({
        "uid_encoded": [0, 1, 2], "item_id_freq": [0.1, 0.2, 0.3], "is_organic_encoded": [0, 1, 0],
        "has_listened_before": [0, 1, 1], "item_avg_completion": [10.0, 20.0, 30.0],
        "user_total_listens": [1, 5, 9], "unused": [1, 2, 3],
    })
    targets, table = rk.select_target_features(importance, train, n=2)
    assert targets == ["item_avg_completion", "user_total_listens"]
    reasons = dict(zip(table["feature"], table["eligibility"]))
    assert reasons["uid_encoded"] == "identifier encoding"
    assert reasons["item_id_freq"] == "identifier encoding"
    assert reasons["is_organic_encoded"] == "categorical code"
    assert reasons["has_listened_before"] == "binary or constant"
    with pytest.raises(ValueError):
        rk.select_target_features(importance, train, n=3)


def test_target_scope_restricts_to_user_or_item_features():
    importance = pl.DataFrame({
        "feature": ["user_median_completion", "user_avg_completion", "item_avg_completion",
                    "track_length_seconds", "item_total_plays"],
        "gain": [50.0, 40.0, 30.0, 20.0, 10.0],
    })
    train = pl.DataFrame({f: [1.0, 2.0, 3.0] for f in importance["feature"]})
    assert rk.select_target_features(importance, train, n=2)[0] == ["user_median_completion",
                                                                     "user_avg_completion"]
    targets, table = rk.select_target_features(importance, train, n=2, scope="item")
    assert targets == ["item_avg_completion", "item_total_plays"]
    reasons = dict(zip(table["feature"], table["eligibility"]))
    assert reasons["track_length_seconds"] == "outside target scope 'item'"
    assert rk.select_target_features(importance, train, n=1, scope="user")[0] == ["user_median_completion"]
    with pytest.raises(ValueError):
        rk.select_target_features(importance, train, n=2, scope="session")


def test_feature_contract_bounds():
    df = pl.DataFrame({"item_id_freq": [0.1], "uid_encoded": [3], "hour_of_day": [5],
                       "day_of_week": [2], "user_organic_ratio": [0.5], "user_total_listens": [7]})
    c = {x.name: x for x in rk.feature_contract(df, df.columns)}
    assert (c["item_id_freq"].lower, c["item_id_freq"].upper) == (0.0, 1.0)
    assert c["uid_encoded"].lower == -1.0
    assert c["hour_of_day"].upper == 23.0 and c["day_of_week"].upper == 6.0
    assert c["user_organic_ratio"].upper == 1.0
    assert (c["user_total_listens"].lower, c["user_total_listens"].upper) == (0.0, None)
    assert c["uid_encoded"].dtype == "Int64"


def test_monitor_exemptions():
    ex = rx.monitor_exemptions(["uid_encoded", "item_id_encoded", "item_id_freq",
                                "is_organic_encoded", "day_of_week", "hour_of_day"],
                               binary_features=["is_organic_encoded"])
    assert set(ex) == {"uid_encoded", "item_id_encoded", "day_of_week", "is_organic_encoded:quantiles"}
    assert rx._exemption(ex, "uid_encoded", "ks") is not None
    assert rx._exemption(ex, "is_organic_encoded", "quantiles") is not None
    assert rx._exemption(ex, "is_organic_encoded", "ks") is None


def test_pit_counts_with_lookback():
    frames = {"train": pl.DataFrame({"uid": [1, 1, 1], "item_id": [10, 10, 10], "timestamp": [0, 100, 250]}),
              "test": pl.DataFrame({"uid": [1, 1], "item_id": [10, 20], "timestamp": [300, 300]})}
    out = rk.add_pit_user_item_features(frames, lookback_units=200)
    # Window [t - 200, t): row 250 sees 100; test row 300 sees 100 and 250 (100 is on the edge).
    assert out["train"]["previous_listen_count"].to_list() == [0, 1, 1]
    assert out["test"]["previous_listen_count"].to_list() == [2, 0]
    cumulative = rk.add_pit_user_item_features(frames)
    assert cumulative["train"]["previous_listen_count"].to_list() == [0, 1, 2]
    assert cumulative["test"]["previous_listen_count"].to_list() == [3, 0]


@pytest.mark.parametrize("scenario,stat,clean,expected", [
    ("clean", 0.5, 0.5, "false alarm"),
    ("clean", 0.0, 0.0, "green"),
    ("zero_fill_all", 0.5, 0.0, "FIRES"),
    ("zero_fill_all", 0.5, 0.2, "FIRES, clean also over"),
    ("zero_fill_all", 0.25, 0.2, "clean also over"),
    ("zero_fill_all", 0.05, 0.2, "green"),
    ("zero_fill_all", float("nan"), 0.2, "n/a"),
])
def test_status_labels(scenario, stat, clean, expected):
    assert rx._status(scenario, stat, clean, threshold=0.1) == expected


def test_status_exempt():
    assert rx._status("zero_fill_all", 0.9, 0.0, 0.1, exempt=True) == "exempt"
