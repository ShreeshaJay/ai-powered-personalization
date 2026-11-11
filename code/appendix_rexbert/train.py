"""Contrastive fine-tuning with in-batch negatives and a retrieval validation metric.

Checkpoint selection uses MRR@100 from an information-retrieval evaluator on
validation sessions. The evaluator masks the query item so a product cannot
count as a neighbor of itself. Batches use the duplicate-free sampler.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from config import MNR_SCALE, Profile, WARMUP_RATIO, WEIGHT_DECAY, set_seed

logger = logging.getLogger("appendix_rexbert.train")


def build_training_arguments(
    output_dir: str,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    bf16: bool,
):
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.base.sampler import BatchSamplers

    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        # Transformers 5 stores a fractional warmup as warmup_steps, not warmup_ratio.
        warmup_steps=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=bf16,
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mrr_at_100",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        logging_steps=50,
        save_total_limit=1,
        save_only_model=True,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


def build_selection_evaluator(
    queries: dict[str, str],
    corpus: dict[str, str],
    relevant_docs: dict[str, set[str]],
    query_item_ids: dict[str, str],
    batch_size: int,
):
    """IR evaluator on validation sessions. MRR@100 is the selection metric."""
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    class _MRREvaluator(InformationRetrievalEvaluator):
        def __init__(self, *args, query_items: dict[str, str], **kwargs):
            super().__init__(*args, **kwargs)
            self._query_items = [query_items[qid] for qid in self.queries_ids]

        def compute_all_metrics(self, model, corpus_model=None, corpus_embeddings=None, output_path=None):
            if corpus_model is None:
                corpus_model = model
            query_embeddings = self.embed_inputs(model, self.queries, encode_fn_name="query")
            doc_embeddings = self.embed_inputs(corpus_model, self.corpus, encode_fn_name="document")
            query_tensor = torch.as_tensor(np.asarray(query_embeddings), dtype=torch.float32)
            doc_tensor = torch.as_tensor(np.asarray(doc_embeddings), dtype=torch.float32)
            name = sorted(self.score_functions)[0]
            score_fn = self.score_functions[name]
            id_to_col = {corpus_id: index for index, corpus_id in enumerate(self.corpus_ids)}
            max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.precision_recall_at_k), max(self.map_at_k))
            hits = []
            step = 256
            for start in range(0, query_tensor.shape[0], step):
                stop = min(start + step, query_tensor.shape[0])
                scores = score_fn(query_tensor[start:stop], doc_tensor)
                if not torch.is_tensor(scores):
                    scores = torch.as_tensor(scores)
                for local, item_id in enumerate(self._query_items[start:stop]):
                    column = id_to_col.get(item_id)
                    if column is not None:
                        scores[local, column] = -1.0e9
                k = min(max_k, scores.shape[1])
                values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
                values = values.detach().cpu().tolist()
                indices = indices.detach().cpu().tolist()
                for local in range(stop - start):
                    row_hits = []
                    for score, column in zip(values[local], indices[local]):
                        row_hits.append(
                            {"corpus_id": self.corpus_ids[column], "score": float(score)}
                        )
                    hits.append(row_hits)
            return {name: self.compute_metrics(hits)}

        def __call__(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):
            metrics = super().__call__(model, output_path=output_path, epoch=epoch, steps=steps, *args, **kwargs)
            mrr_key = next(key for key in metrics if str(key).endswith("mrr@100"))
            metrics["mrr_at_100"] = float(metrics[mrr_key])
            self.primary_metric = "mrr_at_100"
            return metrics

    return _MRREvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        query_items=query_item_ids,
        mrr_at_k=[100],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[10, 50],
        map_at_k=[100],
        show_progress_bar=False,
        batch_size=batch_size,
        name="",
        write_csv=False,
        write_predictions=False,
    )


def select_learning_rate(scores: dict[float, float]) -> float:
    """Pick the learning rate with the highest validation MRR@100. Ties take the smaller rate."""
    if not scores:
        raise ValueError("Learning-rate scores are empty")
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _loss(model):
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    return MultipleNegativesRankingLoss(model, scale=MNR_SCALE)


def _dataset(anchors: list[str], positives: list[str]):
    from datasets import Dataset

    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def _drop_model_card_callback(trainer) -> None:
    from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardCallback

    trainer.remove_callback(SentenceTransformerModelCardCallback)


def train_encoder(
    model,
    anchors: list[str],
    positives: list[str],
    evaluator,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    bf16: bool,
) -> dict:
    """Fine-tune and save the best checkpoint under output_dir / 'best'."""
    from sentence_transformers import SentenceTransformerTrainer

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_training_arguments(
        str(output_dir / "trainer"),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        bf16=bf16,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=_dataset(anchors, positives),
        loss=_loss(model),
        evaluator=evaluator,
    )
    _drop_model_card_callback(trainer)
    trainer.train()
    best = output_dir / "best"
    model.save(str(best))
    history = []
    for entry in trainer.state.log_history:
        numeric = {
            key: float(value)
            for key, value in entry.items()
            if isinstance(value, (int, float)) and key != "epoch"
        }
        if "eval_mrr_at_100" in numeric or "loss" in numeric:
            history.append(numeric)
    best_mrr = None
    for entry in history:
        if "eval_mrr_at_100" in entry:
            value = entry["eval_mrr_at_100"]
            best_mrr = value if best_mrr is None else max(best_mrr, value)
    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "n_pairs": len(anchors),
        "best_validation_mrr_at_100": best_mrr,
        "mnr_scale": MNR_SCALE,
        "relative_model_dir": None,
    }


def use_bf16(profile: Profile) -> bool:
    if not profile.request_bf16:
        return False
    if not torch.cuda.is_available():
        return False
    return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
