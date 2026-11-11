"""The trainer arguments use the duplicate-free sampler and MRR@100."""

import pytest

pytestmark = pytest.mark.slow


def test_training_arguments_select_mrr_and_skip_duplicate_batches(tmp_path):
    from sentence_transformers.base.sampler import BatchSamplers

    from train import build_training_arguments, select_learning_rate

    args = build_training_arguments(
        str(tmp_path / "trainer"),
        epochs=1,
        batch_size=8,
        learning_rate=5e-5,
        seed=0,
        bf16=False,
    )
    assert args.batch_sampler == BatchSamplers.NO_DUPLICATES
    assert args.metric_for_best_model == "eval_mrr_at_100"
    assert args.greater_is_better
    assert args.warmup_steps == 0.1
    assert args.weight_decay == 0.01
    assert select_learning_rate({2e-5: 0.2, 5e-5: 0.4, 1e-4: 0.4}) == 5e-5


def test_trainer_selects_on_validation_mrr(tmp_path):
    """One epoch on a handful of pairs, to confirm the trainer API on CPU."""
    from config import MODELS
    from encoders import build_sentence_encoder
    from train import build_selection_evaluator, train_encoder

    spec = MODELS["minilm"]
    model = build_sentence_encoder(spec, device="cpu")
    anchors = [f"anchor product {index}" for index in range(16)]
    positives = [f"next product {index}" for index in range(16)]
    queries = {f"q{index}": anchors[index] for index in range(4)}
    corpus = {f"d{index}": positives[index] for index in range(4)}
    corpus.update({f"x{index}": f"distractor {index}" for index in range(4)})
    relevant = {f"q{index}": {f"d{index}"} for index in range(4)}
    query_items = {f"q{index}": f"d{index}" for index in range(4)}
    evaluator = build_selection_evaluator(queries, corpus, relevant, query_items, batch_size=4)
    result = train_encoder(
        model,
        anchors,
        positives,
        evaluator,
        tmp_path / "mini",
        epochs=1,
        batch_size=4,
        learning_rate=5e-5,
        seed=0,
        bf16=False,
    )
    assert (tmp_path / "mini" / "best").is_dir()
    assert isinstance(result["best_validation_mrr_at_100"], float)
