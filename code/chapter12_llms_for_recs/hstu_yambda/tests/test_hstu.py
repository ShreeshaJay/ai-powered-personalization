import math

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from data_loader import (build_leave_one_out_splits, load_hstu_data, load_yambda_events, repeat_rate,
                         split_user, training_windows)
from dataset import build_datasets, collate_eval, collate_train, sample_negatives
from evaluate import evaluate_full_catalog
from model import HSTURecommender
from movielens_loader import load_movielens_ratings
from synthetic_data import load_synthetic_data
from train import negatives_for, shift_batch, train_one_epoch
from training_utils import compute_hstu_loss, init_hstu_weights, sampled_softmax_loss
from utils import compute_rank, full_catalog_ranks, hit_rate_at_k, ndcg_at_k, summarize_ranks


@pytest.mark.parametrize("n_events", range(2, 25))
@pytest.mark.parametrize("max_seq_len", [1, 3, 7])
def test_training_windows_predict_every_transition_once(n_events, max_seq_len):
    windows = training_windows(n_events, max_seq_len)
    targets = [t for start, end in windows for t in range(start + 1, end)]
    assert sorted(targets) == list(range(1, n_events))
    assert all(end - start <= max_seq_len + 1 for start, end in windows)


def test_split_user_holds_out_last_two_events():
    n = 12
    items = list(range(1, n + 1))
    train, val, test = split_user(7, items, [1] * n, list(range(n)), [0.5] * n, max_seq_len=4)
    assert test.target_item == n and test.items[-1] == n - 1 and len(test.items) == 4
    assert val.target_item == n - 1 and val.items[-1] == n - 2
    seen_in_train = {i for s in train for i in s.items + [s.target_item]}
    assert seen_in_train == set(range(1, n - 1))


def test_history_rate_uses_full_history_and_window_rate_uses_model_window():
    # user 1: target 10 appears only before the 3-event window; user 2: target is in the window;
    # user 3: no repeat
    rows = [(1, [10, 11, 12, 13, 14, 10]), (2, [20, 21, 22, 21]), (3, [30, 31, 32, 33])]
    df = pl.DataFrame({
        "uid": [u for u, items in rows for _ in items],
        "item_id": [i for _, items in rows for i in items],
        "action_id": [1 for _, items in rows for _ in items],
        "seconds": [t for _, items in rows for t in range(len(items))],
        "value": [0.5 for _, items in rows for _ in items],
    })
    stats = build_leave_one_out_splits(df, max_seq_len=3).stats
    assert stats["test_target_in_history_rate"] == pytest.approx(2 / 3)
    assert stats["test_target_in_model_window_rate"] == pytest.approx(1 / 3)


def test_val_history_rate_uses_full_history():
    # val target = penultimate event. user 1: val target 10 appears only before the 3-event
    # window [11, 12, 13]; user 2: val target 22 is new
    rows = [(1, [10, 11, 12, 13, 10, 99]), (2, [20, 21, 22, 23])]
    df = pl.DataFrame({
        "uid": [u for u, items in rows for _ in items],
        "item_id": [i for _, items in rows for i in items],
        "action_id": [1 for _, items in rows for _ in items],
        "seconds": [t for _, items in rows for t in range(len(items))],
        "value": [0.5 for _, items in rows for _ in items],
    })
    bundle = build_leave_one_out_splits(df, max_seq_len=3)
    assert bundle.stats["val_target_in_history_rate"] == pytest.approx(1 / 2)
    assert repeat_rate(bundle.val_samples) == 0.0


def test_values_keep_missing_as_nan():
    df = pl.DataFrame({
        "uid": [1, 1, 1, 1],
        "item_id": [10, 11, 12, 13],
        "action_id": [1, 2, 1, 1],
        "seconds": [0, 5, 10, 15],
        "value": [0.5, None, 1.0, 0.25],
    })
    bundle = build_leave_one_out_splits(df, max_seq_len=10)
    assert math.isnan(bundle.test_samples[0].values[1])
    assert bundle.test_samples[0].target_value == 0.25


DAY = 86_400


def _write_yambda(tmp_path, days):
    n = len(days)
    pl.DataFrame({
        "uid": [1] * n,
        "item_id": list(range(100, 100 + n)),
        "timestamp": [int(d * DAY) for d in days],
        "played_ratio_pct": [50.0] * n,
        "event_type": ["listen"] * n,
    }).write_parquet(tmp_path / "multi_event.parquet")


def test_yambda_timestamps_are_seconds_and_window_is_in_days(tmp_path):
    _write_yambda(tmp_path, [0, 1, 5.5, 7, 29.9, 30, 30.5, 60])
    df = load_yambda_events(tmp_path, max_days=30)
    assert df["item_id"].to_list() == [100, 101, 102, 103, 104, 105]
    assert df["seconds"].to_list() == [0, DAY, int(5.5 * DAY), 7 * DAY, int(29.9 * DAY), 30 * DAY]
    last = load_yambda_events(tmp_path, max_days=30, use_last_window=True)
    assert last["item_id"].to_list() == [105, 106, 107]
    assert load_yambda_events(tmp_path, max_days=None).height == 8


def test_yambda_stats_record_the_window(tmp_path):
    _write_yambda(tmp_path, [0, 1, 7, 29.5, 45])
    stats = load_hstu_data(tmp_path, max_days=30, max_seq_len=10).stats
    assert stats["time_unit"] == "seconds"
    assert stats["max_days"] == 30 and stats["window"] == "first"
    assert stats["span_days"] == pytest.approx(29.5)
    assert stats["num_rows"] == 4


def test_movielens_ratings_are_rescaled(tmp_path):
    (tmp_path / "ratings.dat").write_text("1::5::5::100\n1::6::1::50\n1::7::3::200\n", encoding="latin-1")
    df = load_movielens_ratings(tmp_path)
    assert df["item_id"].to_list() == [6, 5, 7]
    assert df["value"].to_list() == [0.0, 1.0, 0.5]
    assert df["action_id"].to_list() == [3, 1, 2]


def test_sample_negatives_rejects_the_positive():
    torch.manual_seed(0)
    targets = torch.randint(1, 4, (5000,))
    negatives = sample_negatives(targets, num_items=3, num_negatives=4, max_rounds=20)
    assert not negatives.eq(targets[:, None]).any()


def test_sampled_softmax_masks_negatives_equal_to_positive():
    h = torch.randn(3, 8)
    emb = torch.randn(3, 8)
    pos = torch.tensor([1, 2, 3])
    loss = sampled_softmax_loss(h, emb, emb[:, None, :].expand(3, 5, 8), pos, pos[:, None].expand(3, 5), 0.05)
    assert torch.allclose(loss, torch.zeros(3), atol=1e-6)


def _toy_model(num_items=20, max_seq_len=8):
    torch.manual_seed(0)
    model = HSTURecommender(num_items=num_items, num_actions=5, dim=16, num_layers=2, max_seq_len=max_seq_len,
                            dropout=0.0)
    init_hstu_weights(model)
    return model.eval()


def test_encoding_does_not_depend_on_batch_padding():
    model = _toy_model()
    short = {"items": torch.tensor([[3, 4, 5]]), "actions": torch.tensor([[1, 1, 2]]),
             "times": torch.tensor([[0, 10, 4000]]), "mask": torch.ones(1, 3, dtype=torch.bool)}
    alone = model.encode_sequence(short["items"], short["actions"], short["times"], short["mask"])
    items = torch.tensor([[3, 4, 5, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
    actions = torch.tensor([[1, 1, 2, 0, 0, 0], [1] * 6])
    times = torch.tensor([[0, 10, 4000, 0, 0, 0], list(range(6))])
    mask = items > 0
    batched = model.encode_sequence(items, actions, times, mask)
    assert torch.allclose(alone[0], batched[0], atol=1e-5)


def test_hstu_is_causal():
    model = _toy_model()
    items = torch.tensor([[3, 4, 5, 6]])
    actions = torch.ones_like(items)
    times = torch.tensor([[0, 10, 20, 30]])
    mask = torch.ones_like(items, dtype=torch.bool)
    h1 = model.encoder(items, actions, times, mask)
    h2 = model.encoder(torch.tensor([[3, 4, 5, 9]]), actions, torch.tensor([[0, 10, 20, 99999]]), mask)
    assert torch.allclose(h1[0, :3], h2[0, :3], atol=1e-6)
    assert not torch.allclose(h1[0, 3], h2[0, 3])


def test_hstu_rejects_sequences_longer_than_max_seq_len():
    model = _toy_model(max_seq_len=4)
    items = torch.ones(1, 5, dtype=torch.long)
    with pytest.raises(ValueError):
        model.encoder(items, items, items, items.bool())


def test_loss_ignores_padding_targets_and_missing_values():
    model = _toy_model()
    torch.manual_seed(1)
    items = torch.tensor([[3, 4, 5, 0], [6, 7, 8, 9]])
    mask = items > 0
    batch = {"items": items, "actions": mask.long(), "times": torch.arange(4).expand(2, 4) * 60,
             "values": torch.tensor([[0.1, 0.2, math.nan, math.nan], [0.3, 0.4, 0.5, 0.6]]), "mask": mask}
    b = shift_batch(batch, torch.device("cpu"))
    neg = negatives_for(b["target_items"], b["target_mask"], 20, 4)
    loss_a, metrics = compute_hstu_loss(model, neg_items=neg, **b)
    b["target_values"] = b["target_values"].clone()
    b["target_values"][0, 2] = 123.0
    b["target_items"] = b["target_items"].clone()
    b["target_items"][0, 2] = 17
    loss_b, _ = compute_hstu_loss(model, neg_items=neg, **b)
    assert torch.isfinite(loss_a)
    assert torch.allclose(loss_a, loss_b)
    assert metrics["rated_fraction"] == pytest.approx(4 / 5)


def test_full_catalog_ranks_known_answer():
    scores = torch.tensor([[9.0, 0.1, 0.9, 0.5, 0.9, 0.2]])
    hist = torch.tensor([[4, 0]])
    r = full_catalog_ranks(scores, torch.tensor([2]), hist)
    assert r["unmasked"].item() == 1
    assert r["masked"].item() == 0
    r = full_catalog_ranks(scores, torch.tensor([4]), hist)
    assert r["target_in_window"].item()
    assert r["masked"].item() == 5
    assert r["unmasked"].item() == 1
    assert compute_rank(torch.tensor([0.1, 0.9, 0.9]), 2) == 1


def test_rank_summaries():
    assert hit_rate_at_k(iter([0, 5, 20]), k=10) == pytest.approx(2 / 3)
    assert ndcg_at_k([0, 1], k=10) == pytest.approx((1 + 1 / math.log2(3)) / 2)
    s = summarize_ranks([0, 50, 3], [0, 1, 3], [False, True, False], k=10)
    assert s["hr@10"] == pytest.approx(2 / 3)
    assert s["unmasked_hr@10"] == 1.0
    assert s["novel_hr@10"] == 1.0 and s["num_novel_samples"] == 2
    assert s["target_in_window_rate"] == pytest.approx(1 / 3)


def test_synthetic_training_step_and_evaluation():
    bundle = load_synthetic_data(max_seq_len=16, num_users=40, num_items=60, mean_len=12)
    assert bundle.stats["train_transitions"] > 0
    train_ds, val_ds, _ = build_datasets(bundle, max_seq_len=16)
    model = HSTURecommender(num_items=bundle.num_items, num_actions=bundle.num_actions, dim=16, num_layers=1,
                            max_seq_len=16, dropout=0.0)
    init_hstu_weights(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_train,
                        generator=torch.Generator().manual_seed(0))
    first = train_one_epoch(model, loader, opt, torch.device("cpu"), bundle.num_items, 8, use_amp=False)
    for _ in range(4):
        last = train_one_epoch(model, loader, opt, torch.device("cpu"), bundle.num_items, 8, use_amp=False)
    assert last["retrieval"] < first["retrieval"]
    metrics = evaluate_full_catalog(model, DataLoader(val_ds, batch_size=8, collate_fn=collate_eval),
                                    torch.device("cpu"))
    assert metrics["num_samples"] == len(bundle.val_samples)
    for key in ("hr@10", "ndcg@10", "unmasked_hr@10", "novel_hr@10", "target_in_window_rate"):
        assert 0.0 <= metrics[key] <= 1.0
