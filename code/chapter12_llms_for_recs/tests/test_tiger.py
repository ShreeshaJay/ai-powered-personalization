import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.semantic_id_dataset import (
    BOS_TOKEN,
    PAD_TOKEN,
    SemanticIDSessions,
    SemanticIDTrie,
    collate,
    sids_to_tokens,
    tokens_to_sids,
    vocab_size,
)
from models.tiger import TIGER

LEVEL_SIZES = [4, 4, 2]
SIDS = np.array([[0, 1, 0], [0, 1, 1], [0, 3, 0], [2, 0, 0], [3, 3, 0], [3, 2, 0]])


def test_token_round_trip_and_offsets():
    tokens = sids_to_tokens(SIDS, LEVEL_SIZES)
    assert tokens[0].tolist() == [2 + 0, 2 + 4 + 1, 2 + 8 + 0]
    assert vocab_size(LEVEL_SIZES) == 2 + 10
    assert np.array_equal(tokens_to_sids(tokens, LEVEL_SIZES), SIDS)


def test_invalid_codes_raise_instead_of_clamping():
    with pytest.raises(ValueError):
        sids_to_tokens(np.array([[0, 4, 0]]), LEVEL_SIZES)
    with pytest.raises(ValueError):
        tokens_to_sids(np.array([[2, 2, 10]]), LEVEL_SIZES)


def test_trie_children_and_duplicates():
    tokens = sids_to_tokens(SIDS, LEVEL_SIZES)
    trie = SemanticIDTrie(tokens)
    assert trie.allowed(()) == [2, 4, 5]
    assert trie.allowed((2,)) == [7, 9]
    assert trie.allowed((2, 7)) == [10, 11]
    assert trie.allowed((3,)) == []
    assert trie.item(tuple(tokens[4].tolist())) == 4
    with pytest.raises(ValueError):
        SemanticIDTrie(np.vstack([tokens, tokens[:1]]))


def test_dataset_truncates_by_whole_items_and_left_pads():
    tokens = sids_to_tokens(SIDS, LEVEL_SIZES)
    ds = SemanticIDSessions([[0, 1, 2, 3, 4], [5]], [1, 2], tokens, max_history_items=2)
    first = ds[0]
    assert len(first["encoder_input"]) == 2 * 3
    assert first["encoder_input"].tolist() == tokens[[3, 4]].reshape(-1).tolist()
    assert first["decoder_input"].tolist() == [BOS_TOKEN] + tokens[1, :-1].tolist()
    batch = collate([ds[0], ds[1]])
    assert batch["encoder_input"][1, :3].tolist() == [PAD_TOKEN] * 3
    assert batch["encoder_mask"][1].tolist() == [False] * 3 + [True] * 3


def small_model():
    torch.manual_seed(0)
    return TIGER(LEVEL_SIZES, hidden_dim=16, num_heads=2, num_encoder_layers=1, num_decoder_layers=1,
                 ffn_dim=32, dropout=0.0, max_history_items=3)


def test_padding_row_is_zero():
    assert torch.count_nonzero(small_model().token_embedding.weight[PAD_TOKEN]) == 0


def test_constrained_beam_search_returns_valid_unique_items_with_exact_scores():
    model = small_model().eval()
    tokens = sids_to_tokens(SIDS, LEVEL_SIZES)
    trie = SemanticIDTrie(tokens)
    ds = SemanticIDSessions([[0, 3], [5, 4, 1]], [1, 2], tokens, max_history_items=3)
    batch = collate([ds[0], ds[1]])
    results = model.constrained_beam_search(batch["encoder_input"], batch["encoder_mask"], trie, beam_width=10)
    for b, row in enumerate(results):
        items = [i for i, _ in row]
        assert sorted(items) == list(range(len(SIDS)))  # wide beam: every catalog item, once
        scores = [s for _, s in row]
        assert scores == sorted(scores, reverse=True)
        for item, score in row:
            dec_in = torch.tensor([[BOS_TOKEN] + tokens[item, :-1].tolist()])
            with torch.no_grad():
                logits = model(batch["encoder_input"][b:b + 1], batch["encoder_mask"][b:b + 1], dec_in)[0]
            exact = sum(float(F.log_softmax(model.level_logits(logits[l], l), -1)[tokens[item, l] - model.offsets[l]])
                        for l in range(len(LEVEL_SIZES)))
            assert score == pytest.approx(exact, abs=1e-4)


def test_narrow_beam_returns_at_most_width():
    model = small_model().eval()
    tokens = sids_to_tokens(SIDS, LEVEL_SIZES)
    ds = SemanticIDSessions([[0, 3]], [1], tokens, max_history_items=3)
    batch = collate([ds[0]])
    row = model.constrained_beam_search(batch["encoder_input"], batch["encoder_mask"], SemanticIDTrie(tokens), 2)[0]
    assert len(row) == 2 and len({i for i, _ in row}) == 2


def test_loss_is_finite_and_decreases():
    model = small_model()
    tokens = sids_to_tokens(SIDS, LEVEL_SIZES)
    ds = SemanticIDSessions([[0, 3], [5, 4], [1, 2]], [1, 2, 3], tokens, max_history_items=3)
    batch = collate([ds[i] for i in range(3)])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        loss, _ = model.loss(batch, [1.0, 1.0, 1.0], label_smoothing=0.1)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.isfinite(loss.item()) and loss.item() < first
