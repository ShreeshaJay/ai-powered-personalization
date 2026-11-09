"""Locating the Yambda files in either download layout, and labelling training rows."""

import pandas as pd
import pytest

from pipeline_build.data import load_yambda, yambda_flat_dir


@pytest.mark.parametrize("layout", ["flat/50m", "flat"])
def test_both_yambda_layouts_are_found(tmp_path, layout):
    (tmp_path / layout).mkdir(parents=True)
    (tmp_path / layout / "listens.parquet").write_bytes(b"")
    assert yambda_flat_dir(tmp_path) == tmp_path / layout


def test_missing_yambda_files_name_the_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="flat/50m/listens.parquet"):
        load_yambda(tmp_path)


def test_training_labels_ignore_events_after_the_label_end():
    from pipeline_build.train_rankers import add_labels

    rows = pd.DataFrame({"uid": [1, 1], "timestamp": [1000, 1010], "item_id": [7, 8], "is_organic": [0, 0],
                         "played_ratio_pct": [100, 10], "track_length_seconds": [200, 200]})
    likes = pd.DataFrame({"uid": [1, 1], "timestamp": [1005, 1100], "item_id": [7, 8]})
    dislikes = likes.iloc[:0]
    unbounded = add_labels(rows, likes, dislikes)
    assert unbounded["label_liked"].tolist() == [1, 1]
    bounded = add_labels(rows, likes, dislikes, label_end_ts=1050)      # the like of item 8 comes after the end
    assert bounded["label_liked"].tolist() == [1, 0]
    assert bounded[["label", "label_engagement"]].values.tolist() == [[1, 1], [0, 1]]


def test_labelled_rows_come_out_in_the_same_order_whatever_the_input_order():
    from pipeline_build.train_rankers import add_labels

    # 2 users x 2 items x 2 times, 10 listens each that differ only in play ratio and organic flag.
    rows = pd.DataFrame([{"uid": uid, "timestamp": ts, "item_id": item, "is_organic": n % 2,
                          "played_ratio_pct": 10 * n, "track_length_seconds": 200}
                         for uid in (1, 2) for item in (7, 8) for ts in (1000, 1010) for n in range(10)])
    likes = pd.DataFrame({"uid": [1, 2], "timestamp": [1020, 1020], "item_id": [7, 8]})
    dislikes = pd.DataFrame({"uid": [2], "timestamp": [1015], "item_id": [7]})
    expected = add_labels(rows, likes, dislikes)
    assert expected["label_liked"].sum() == 40 and expected["label_disliked"].sum() == 20
    for seed in range(5):
        shuffled = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        pd.testing.assert_frame_equal(add_labels(shuffled, likes, dislikes), expected)
