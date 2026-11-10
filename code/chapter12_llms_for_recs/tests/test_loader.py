import io

import polars as pl
import pytest

from data.amazon_session_loader import PreparedData, assign_splits, item_text, parse_prev_items, prepare_sessions_and_items


def test_parse_prev_items_handles_newlines_and_quotes():
    df = pl.read_csv(io.StringIO("prev_items\n\"['B0A' 'B0B'\n 'B0C']\"\n"), infer_schema_length=0)
    out = df.select(parse_prev_items().alias("h"))["h"].to_list()
    assert out == [["B0A", "B0B", "B0C"]]


def test_item_text_skips_empty_parts():
    assert item_text("Mug", "", "Blue mug", 400) == "Mug [SEP] Blue mug"
    assert item_text("", "", "", 400) == "unknown product"
    assert item_text("T", "B", "x" * 10, 3) == "T [SEP] B [SEP] xxx"


def test_splits_are_disjoint_and_sized():
    import numpy as np

    labels = assign_splits(np.arange(1000), (0.8, 0.1, 0.1), seed=1)
    assert (labels == "train").sum() == 800 and (labels == "val").sum() == 100 and (labels == "test").sum() == 100
    with pytest.raises(ValueError):
        assign_splits(np.arange(10), (0.5, 0.5, 0.5), seed=1)


def frames():
    sessions = pl.DataFrame({
        "session_id": [10, 11, 12, 13, 14],
        "history": [["a", "b", "c"], ["a", "b"], ["a", "x", "c"], ["c", "b", "a"], ["b", "c", "a"]],
        "target": ["d", "c", "d", "q", "e"],
    })
    products = pl.DataFrame({"item_id": ["a", "b", "c", "d", "e", "f", "g"], "title": list("ABCDEFG"),
                             "brand": [""] * 7, "desc": [""] * 7})
    return sessions, products


def test_filters_keep_raw_session_ids():
    sessions, products = frames()
    out, items, meta = prepare_sessions_and_items(
        sessions, products, min_prev_items=3, max_sessions=None, split_fractions=(0.5, 0.25, 0.25),
        catalog="referenced", extra_random_items=1, desc_max_chars=10, seed=0)
    # 11 is too short; 12 has an unknown history item; 13 has an unknown target
    assert out["session_id"].to_list() == [10, 14]
    assert meta["sessions_with_min_history"] == 4 and meta["sessions_with_catalog_coverage"] == 2
    referenced = {"a", "b", "c", "d", "e"}
    ids = set(items["item_id"].to_list())
    assert referenced <= ids and len(ids) == len(referenced) + 1


def test_train_counts_ignore_validation_and_test(tmp_path):
    sessions = pl.DataFrame({
        "session_id": [1, 2, 3], "history": [["a", "b", "c"]] * 3, "target": ["d", "e", "f"],
        "split": ["train", "val", "test"],
    })
    items = pl.DataFrame({"item_id": list("abcdef"), "title": list("abcdef"), "brand": [""] * 6,
                          "desc": [""] * 6, "text": list("abcdef")})
    sessions.write_parquet(tmp_path / "s.parquet")
    items.write_parquet(tmp_path / "i.parquet")
    data = PreparedData(tmp_path / "s.parquet", tmp_path / "i.parquet")
    counts = dict(zip(data.item_ids, data.train_item_counts().tolist()))
    assert counts == {"a": 1, "b": 1, "c": 1, "d": 1, "e": 0, "f": 0}
    assert data.train_target_counts().tolist() == [0, 0, 0, 1, 0, 0]
