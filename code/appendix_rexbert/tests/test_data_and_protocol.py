"""Split integrity, pair provenance, and the Chapter 10 query sample."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from config import (
    CHAPTER10_QUERY_COUNT,
    CHAPTER10_QUERY_SEED,
    GPU_BATCH_SIZE,
    GPU_TEST_QUERIES,
    GPU_TRAIN_PAIR_CAP,
    LEARNING_RATES,
    MNR_SCALE,
    PROFILES,
    SPLIT_SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TEXT_TEMPLATE,
    TRAIN_SEEDS,
    bucket_to_split,
    split_bucket,
)
from data import (
    chapter10_query_ids,
    parse_prev_items,
    prepare_dataset,
    render_text,
    sample_like_chapter10,
)


def test_split_bucket_matches_sha256_and_is_disjoint():
    digest = hashlib.sha256(b"42:0").digest()
    assert split_bucket(0, 42) == int.from_bytes(digest[:8], "big") % 10
    splits = [bucket_to_split(split_bucket(index, SPLIT_SEED)) for index in range(2000)]
    assert set(splits) <= {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}
    assert len(splits) == 2000
    train = splits.count(SPLIT_TRAIN) / 2000
    val = splits.count(SPLIT_VAL) / 2000
    test = splits.count(SPLIT_TEST) / 2000
    assert 0.75 < train < 0.85
    assert 0.07 < val < 0.13
    assert 0.07 < test < 0.13


def test_parse_and_template():
    assert parse_prev_items("['B09W9FND7K' 'B09JSPLN1M']") == ["B09W9FND7K", "B09JSPLN1M"]
    assert parse_prev_items(None) == []
    assert render_text("Alpha  title", "Brand", "A  description") == "Alpha title [SEP] Brand [SEP] A description"
    assert TEXT_TEMPLATE == "{title} [SEP] {brand} [SEP] {description}"


def _write_toy(tmp_path):
    products = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "locale": ["UK", "UK", "UK", "UK", "DE", "UK"],
            "title": ["Alpha", "Beta", "Gamma", "Delta", "Other", "Short"],
            "brand": ["A", "B", "C", "D", "E", "F"],
            "desc": ["one two", "three", "", "four five six", "skip", ""],
        }
    )
    rows = []
    # 30 UK sessions so the hash split is exercised, plus one degenerate and one short session.
    for index in range(30):
        left = ["a", "b", "c", "d"][index % 4]
        mid = ["b", "c", "d", "a"][index % 4]
        nxt = ["c", "d", "a", "b"][index % 4]
        rows.append({"prev_items": f"['{left}' '{mid}']", "next_item": nxt, "locale": "UK"})
    rows.append({"prev_items": "['a']", "next_item": "a", "locale": "UK"})
    rows.append({"prev_items": "['f']", "next_item": "a", "locale": "UK"})
    rows.append({"prev_items": "['a' 'b']", "next_item": "c", "locale": "DE"})
    sessions = pd.DataFrame(rows)
    data_dir = tmp_path / "kdd"
    data_dir.mkdir()
    products.to_csv(data_dir / "products_train.csv", index=False)
    sessions.to_csv(data_dir / "sessions_train.csv", index=False)
    return data_dir


def test_pairs_come_only_from_training_sessions(tmp_path):
    prepared = prepare_dataset(_write_toy(tmp_path), pair_cap=100)
    assert prepared.card["n_products"] == 5
    assert "e" not in prepared.texts
    for session_index in prepared.pair_session.tolist():
        assert int(prepared.split[session_index]) == SPLIT_TRAIN
        assert not bool(prepared.degenerate[session_index])
    for query, positive in zip(prepared.pair_query, prepared.pair_positive):
        assert (query, positive) in prepared.pair_keys_before_cap
        assert query != positive
    # A one-item history is below the Chapter 10 minimum of two previous items.
    assert "f" not in prepared.chapter10_map
    assert ("a", "a") not in prepared.pair_keys_before_cap
    assert prepared.card["n_degenerate_sessions"] >= 1
    assert len(prepared.pair_query) >= 1
    val_mask = prepared.session_frame_mask(SPLIT_VAL)
    test_mask = prepared.session_frame_mask(SPLIT_TEST)
    assert np.all(prepared.split[val_mask] == SPLIT_VAL)
    assert not np.any(prepared.degenerate[val_mask])
    assert set(np.flatnonzero(test_mask).tolist()).isdisjoint(set(prepared.pair_session.tolist()))
    train_rows = set(np.flatnonzero(prepared.split == SPLIT_TRAIN).tolist())
    val_rows = set(np.flatnonzero(prepared.split == SPLIT_VAL).tolist())
    test_rows = set(np.flatnonzero(prepared.split == SPLIT_TEST).tolist())
    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


def test_chapter10_sample_matches_numpy_choice():
    ids = [f"p{index}" for index in range(100)]
    got = sample_like_chapter10(ids, 10, 42)
    rng = np.random.RandomState(42)
    expected = [str(item) for item in rng.choice(ids, size=10, replace=False)]
    assert got == expected
    mapping = {item: {f"t{item}"} for item in ids}
    queries = chapter10_query_ids(ids, mapping, 7, CHAPTER10_QUERY_SEED)
    assert len(queries) == 7
    assert len(set(queries)) == 7


def test_profile_constants_match_preregistration():
    gpu = PROFILES["gpu_full"]
    cpu = PROFILES["cpu_small"]
    assert gpu.reported and not cpu.reported
    assert gpu.batch_size == GPU_BATCH_SIZE == 128
    assert cpu.batch_size == 64
    assert gpu.n_train_pairs == GPU_TRAIN_PAIR_CAP == 500_000
    assert cpu.n_train_pairs == 50_000
    assert cpu.n_catalog == 50_000
    assert cpu.n_test_queries == 2_000
    assert gpu.n_test_queries == GPU_TEST_QUERIES == 20_000
    assert gpu.epochs == 3 and cpu.epochs == 1
    assert gpu.seeds == TRAIN_SEEDS == (0, 1, 2)
    assert cpu.seeds == (0,)
    assert gpu.learning_rates == LEARNING_RATES
    assert gpu.run_lr_sweep and not cpu.run_lr_sweep
    assert "ettin_17m" in gpu.finetune_models
    assert "rexbert_base" in gpu.zero_shot_models
    assert "rexbert_base" not in cpu.zero_shot_models
    assert MNR_SCALE == 20.0
    assert CHAPTER10_QUERY_COUNT == 5000
    assert PROFILES["gpu_full"].finetune_models == ("rexbert_micro", "ettin_17m", "minilm")


def test_output_root_rejects_paths_outside_outputs():
    import run_all

    with pytest.raises(SystemExit):
        run_all.resolve_output_root("outputs/../../outside_the_appendix")
    kept = run_all.resolve_output_root("outputs")
    assert kept.name == "outputs"


def test_preregistration_flag_stays_unset_until_a_boolean_is_stored():
    import run_all

    assert run_all.carried_commitment(None) is None
    assert run_all.carried_commitment({}) is None
    assert run_all.carried_commitment({"git_commit": "abc123", "preregistration_committed_before_r3": None}) is None
    assert run_all.carried_commitment({"preregistration_committed_before_r3": True}) is True
