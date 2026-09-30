"""A priori majority-class baseline. Does not peek at evaluation labels."""

from __future__ import annotations

from typing import Any, Iterable

from adapters.base import FieldPrediction, ItemPrediction
from adapters.schemas import MAJORITY_PRIORS, TASK_FIELDS, field_label_space


class MajorityAdapter:
    name = "majority"
    model_id = "majority_prior_v1"

    def predict_batch(
        self,
        task: str,
        examples: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[ItemPrediction]:
        del batch_size
        priors = MAJORITY_PRIORS[task]
        predictions = []
        for example in examples:
            fields = {}
            for field in TASK_FIELDS[task]:
                predicted = priors[field]
                labels = field_label_space(task, field)
                probabilities = {
                    _probability_key(label): 1.0 if label == predicted else 0.0
                    for label in labels
                }
                fields[field] = FieldPrediction(
                    field=field,
                    predicted=predicted,
                    probabilities=probabilities,
                    confidence=1.0,
                )
            predictions.append(
                ItemPrediction(
                    item_id=example["item_id"],
                    task=task,
                    fields=fields,
                )
            )
        return predictions


def _probability_key(label: Any) -> str:
    if isinstance(label, bool):
        return "true" if label else "false"
    return str(label)
