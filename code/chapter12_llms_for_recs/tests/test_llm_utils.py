"""LLM helpers checked with a tiny randomly initialised Qwen2 model and a toy tokenizer (no download)."""

import numpy as np
import pytest
import torch

from train_tallrec import IGNORE_INDEX, build_example, pad_right
from utils.llm import yes_probability


class ToyBatch(dict):
    def to(self, device):
        return ToyBatch({k: v.to(device) for k, v in self.items()})


class ToyTokenizer:
    """Whitespace tokenizer with left padding, enough for the helpers under test."""

    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.replace("<eos>", " <eos>").split()]

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        ids = [self.encode(t) for t in texts]
        width = max(map(len, ids))
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for i, seq in enumerate(ids):
            input_ids[i, width - len(seq):] = torch.tensor(seq)
            mask[i, width - len(seq):] = 1
        return ToyBatch(input_ids=input_ids, attention_mask=mask)


def test_answer_tokens_are_the_only_labels():
    tok = ToyTokenizer()
    ex = build_example(tok, lambda h: "history " + " ".join(h) + " question", ["a", "b"], "Yes", 64,
                       lambda text: text + " assistant:")
    n_answer = len(tok.encode("Yes<eos>"))
    assert sum(t != IGNORE_INDEX for t in ex["labels"]) == n_answer == 2
    assert ex["labels"][-n_answer:] == ex["input_ids"][-n_answer:]


def test_long_prompts_drop_history_not_the_answer():
    tok = ToyTokenizer()
    history = [f"item{i}" for i in range(10)]
    template = lambda text: text  # noqa: E731
    ex = build_example(tok, lambda h: " ".join(h), history, "No", 6, template)
    assert ex is not None and len(ex["input_ids"]) <= 6
    assert ex["labels"][-2:] == ex["input_ids"][-2:]
    assert tok.encode("item9")[0] in ex["input_ids"]  # the most recent item survives
    assert build_example(tok, lambda h: "a b c d e f g " + " ".join(h), history, "No", 6, template) is None


def test_pad_right_masks_padding():
    batch = pad_right([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [8], "labels": [8]}], 0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def tiny_qwen(vocab=64):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.mark.slow
def test_yes_probability_is_unaffected_by_left_padding():
    model, tok = tiny_qwen(), ToyTokenizer()
    short = "a b c"
    long = "d e f g h i j k l m"
    for w in (short + " " + long).split() + ["Yes", "No"]:
        tok._id(w)
    yes_id, no_id = tok.encode("Yes")[0], tok.encode("No")[0]
    alone = yes_probability(model, tok, [short], batch_size=4, device="cpu", yes_id=yes_id, no_id=no_id)
    padded = yes_probability(model, tok, [short, long], batch_size=4, device="cpu", yes_id=yes_id, no_id=no_id)
    assert padded[0] == pytest.approx(alone[0], abs=1e-5)
    assert np.all((padded > 0) & (padded < 1))
