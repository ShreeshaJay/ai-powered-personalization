"""Tests for the hashed segment and the serving-side corruption."""

import numpy as np
import polars as pl
import pytest

import corruption as cx
import monitoring_signals as ms

M64 = (1 << 64) - 1


def splitmix64_reference(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def test_splitmix64_matches_pure_python():
    xs = [0, 1, 2, 12345, 2 ** 32 + 7, 2 ** 63, M64]
    got = cx.splitmix64(np.array(xs, dtype=np.uint64))
    assert [int(v) for v in got] == [splitmix64_reference(x) for x in xs]


def test_splitmix64_first_output_seed_zero():
    assert int(cx.splitmix64(0)) == 0xE220A8397B1DCDAF


def test_hash_in_unit_interval_and_salt_changes_it():
    ids = np.arange(10_000)
    h = cx.hash_to_unit_interval(ids, salt=15)
    assert h.min() >= 0.0 and h.max() < 1.0
    assert not np.array_equal(h, cx.hash_to_unit_interval(ids, salt=16))
    np.testing.assert_array_equal(h, cx.hash_to_unit_interval(ids, salt=15))


def test_segment_fraction_and_edges():
    ids = np.arange(100_000)
    assert cx.segment_mask(ids, 0.3, salt=15).mean() == pytest.approx(0.3, abs=0.01)
    assert not cx.segment_mask(ids, 0.0).any()
    assert cx.segment_mask(ids, 1.0).all()
    with pytest.raises(ValueError):
        cx.segment_mask(ids, 1.5)


def test_segment_is_per_user_not_per_row():
    uids = np.array([5, 9, 5, 5, 9, 11])
    m = cx.segment_mask(uids, 0.5, salt=3)
    for u in np.unique(uids):
        assert len(set(m[uids == u].tolist())) == 1


def _frame():
    return pl.DataFrame({
        "uid": pl.Series([1, 2, 3, 4], dtype=pl.UInt32),
        "count": pl.Series([1, 2, 2, 9], dtype=pl.Int32),
        "rate": pl.Series([0.2, 0.4, 0.6, 0.8], dtype=pl.Float32),
    })


def test_fill_values_round_integers_and_keep_floats():
    df = _frame()
    assert cx.compute_fill_values(df, ["count", "rate"], "zero") == {"count": 0, "rate": 0.0}
    mean = cx.compute_fill_values(df, ["count", "rate"], "mean")
    assert mean["count"] == 4 and isinstance(mean["count"], int)        # 3.5 rounds to even
    assert mean["rate"] == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError):
        cx.compute_fill_values(df, ["count"], "median")


def test_apply_corruption_masks_rows_and_keeps_dtypes():
    df = _frame()
    out = cx.apply_corruption(df, {"count": 0, "rate": 0.0}, row_mask=[True, False, True, False])
    assert out.schema == df.schema
    assert out["count"].to_list() == [0, 2, 0, 9]
    assert out["rate"].to_list()[1] == pytest.approx(0.4)
    everything = cx.apply_corruption(df, {"rate": 0.5})
    assert everything["rate"].to_list() == [0.5] * 4 and everything["count"].equals(df["count"])


def test_corrupt_scenarios():
    df = _frame()
    fills = {"zero": {"rate": 0.0}, "mean": {"rate": 0.5}}
    seg = np.array([True, False, False, True])
    by_name = {s.name: s for s in cx.SCENARIOS}
    assert cx.corrupt(df, by_name["clean"], fills, seg).equals(df)
    assert cx.corrupt(df, by_name["zero_fill_all"], fills, seg)["rate"].to_list() == [0.0] * 4
    seg_out = cx.corrupt(df, by_name["zero_fill_segment"], fills, seg)["rate"].to_list()
    assert seg_out[0] == 0.0 and seg_out[3] == 0.0 and seg_out[1] == pytest.approx(0.4)
    assert cx.corrupt(df, by_name["mean_fill_all"], fills, seg)["rate"].to_list() == [0.5] * 4


def test_corrupted_frame_still_passes_contract():
    df = _frame()
    contract = [ms.ColumnContract("count", "Int32", lower=0.0),
                ms.ColumnContract("rate", "Float32", 0.0, 1.0)]
    for fills in ({"count": 0, "rate": 0.0}, cx.compute_fill_values(df, ["count", "rate"], "mean")):
        out = cx.apply_corruption(df, fills)
        assert ms.schema_pass_rate(out, contract) == {"pass_rate": 1.0, "dtype_ok": True}
