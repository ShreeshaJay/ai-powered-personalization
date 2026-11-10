import numpy as np
import pytest
import torch

from data.amazon_session_loader import PreparedData
from models.augmented_ranker import FEATURE_SETS, AugmentedRanker
from utils.encoders import save_embeddings
from utils.ranker_data import load_feature_tables, pad_histories, to_tensors, training_examples


def test_feature_dimensions_and_known_values():
    d = 4
    content = torch.eye(6, d)
    llm = torch.eye(6, d)
    present = torch.ones(6)
    intent = torch.zeros(3, d)
    intent[0, 2] = 1.0
    hist, mask = pad_histories([[0, 1, 2]], 5)
    args = (torch.tensor([2]), torch.from_numpy(hist), torch.from_numpy(mask), torch.tensor([0]))
    expected = {"content": d + 1, "content_history": 3 * d + 2, "content_history_llm_item": 4 * d + 5, "full": 5 * d + 8}
    for name in FEATURE_SETS:
        model = AugmentedRanker(name, content, llm, present, intent, torch.ones(3))
        feats = model.features(*args)
        assert feats.shape == (1, expected[name]) == (1, model.in_dim)
        assert feats[0, d] == pytest.approx(1.0)  # candidate 2 is the last history item
    full = AugmentedRanker("full", content, llm, present, intent, torch.ones(3)).features(*args)[0]
    assert full[-3] == pytest.approx(1.0)  # intent row 0 points at item 2's direction


def test_missing_tables_raise():
    with pytest.raises(ValueError):
        AugmentedRanker("full", torch.eye(3))


def write_llm(paths, data, encoder, item_fraction=1.0, session_order=None):
    d = np.load(paths.item_embeddings)["embeddings"].shape[1]
    n = int(len(data.item_ids) * item_fraction)
    save_embeddings(paths.llm_item_embeddings, data.item_ids[:n], np.ones((n, d), np.float32), encoder,
                    extra={"llm": np.asarray("test")})
    sids = data.sessions["session_id"].to_list()
    order = session_order if session_order is not None else list(range(len(sids)))
    vecs = np.zeros((len(sids), d), np.float32)
    for row, j in enumerate(order):
        vecs[row, j % d] = 1.0 + j  # unique per session
    save_embeddings(paths.llm_intent_embeddings, [sids[j] for j in order], vecs, encoder, key_name="session_ids",
                    extra={"llm": np.asarray("test")})
    return order, vecs


def test_encoder_mismatch_is_rejected(synthetic_work_dir):
    data = PreparedData.from_paths(synthetic_work_dir)
    write_llm(synthetic_work_dir, data, "some-other-encoder")
    with pytest.raises(ValueError, match="embedded with"):
        load_feature_tables(synthetic_work_dir, data, True, True, 0.99, False)


def test_partial_coverage_needs_explicit_permission(synthetic_work_dir):
    data = PreparedData.from_paths(synthetic_work_dir)
    write_llm(synthetic_work_dir, data, "tfidf-svd", item_fraction=0.5)
    with pytest.raises(ValueError, match="cover"):
        load_feature_tables(synthetic_work_dir, data, True, False, 0.99, False)
    tables = load_feature_tables(synthetic_work_dir, data, True, False, 0.99, True)
    assert tables["coverage"]["LLM item-knowledge vectors"] == pytest.approx(0.5, abs=0.01)
    assert tables["llm_item_present"].sum() == len(data.item_ids) // 2


def test_session_intents_align_by_session_id(synthetic_work_dir):
    data = PreparedData.from_paths(synthetic_work_dir)
    n = data.sessions.height
    order = list(np.random.default_rng(3).permutation(n))
    _, vecs = write_llm(synthetic_work_dir, data, "tfidf-svd", session_order=order)
    tables = load_feature_tables(synthetic_work_dir, data, True, True, 0.99, False)
    sids = data.sessions["session_id"].to_list()
    for row, j in enumerate(order[:50]):
        assert np.array_equal(tables["intent"][tables["session_row"][sids[j]]], vecs[row])
    assert tables["intent_present"][:-1].all() and tables["intent_present"][-1] == 0


def test_training_examples_are_grouped_by_session(synthetic_work_dir):
    data = PreparedData.from_paths(synthetic_work_dir)
    tables = load_feature_tables(synthetic_work_dir, data, False, False, 0.99, False)
    ex = training_examples(data, "train", tables["content"], tables["session_row"], max_history=10,
                           num_hard=2, num_random=2, hard_pool=20, seed=0, max_sessions=30)
    assert len(ex["cand"]) == 30 * 5 and ex["label"].sum() == 30
    for r in range(30):
        rows = np.flatnonzero(ex["hist_row"] == r)
        assert len(rows) == 5 and ex["label"][rows].sum() == 1
        history = set(ex["hist"][r][ex["hist_mask"][r] > 0].tolist())
        negatives = ex["cand"][rows][ex["label"][rows] == 0]
        assert not history & set(negatives.tolist())  # the positive may repeat a history item
    model = AugmentedRanker("content_history", **to_tensors(tables, 1))
    out = model(torch.from_numpy(ex["cand"][:5]), torch.from_numpy(ex["hist"][ex["hist_row"][:5]]),
                torch.from_numpy(ex["hist_mask"][ex["hist_row"][:5]]), torch.from_numpy(ex["session_row"][:5]))
    assert out.shape == (5,)
