"""Local Laya typed-decision adapter."""

from __future__ import annotations

import os
import time
from typing import Any

os.environ.setdefault("USE_TF", "0")

from adapters.base import FieldPrediction, ItemPrediction
from adapters.schemas import TASK_FIELDS, compact_state, laya_questions


class LayaAdapter:
    name = "laya"

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        subfolder: str | None = "typed-decisions",
        device: str = "cuda",
        max_len: int = 512,
    ) -> None:
        self.model_id = model_id
        self.subfolder = subfolder
        self.device = device
        self.max_len = max_len
        self._agent = None

    @property
    def cache_model_id(self) -> str:
        suffix = self.subfolder or "default"
        return f"{self.model_id}#{suffix}"

    def load(self) -> None:
        if self._agent is not None:
            return
        import laya
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self._agent = laya.load(
            self.model_id,
            device=self.device,
            subfolder=self.subfolder,
        )

    def predict_batch(
        self,
        task: str,
        examples: list[dict[str, Any]],
        batch_size: int = 8,
    ) -> list[ItemPrediction]:
        if not examples:
            return []
        self.load()
        import torch

        questions = laya_questions(task)
        states = [compact_state(task, example["adjudication_input"]) for example in examples]
        if self.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        raw_results = self._agent.predict_batch(
            states,
            questions,
            batch_size=batch_size,
            max_len=self.max_len,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_item_ms = elapsed_ms / len(examples)
        return [
            decode_laya_result(
                example["item_id"],
                task,
                raw,
                elapsed_ms=per_item_ms,
            )
            for example, raw in zip(examples, raw_results, strict=True)
        ]


def decode_laya_result(
    item_id: str,
    task: str,
    raw: dict[str, Any],
    elapsed_ms: float = 0.0,
) -> ItemPrediction:
    answers = raw.get("answers", {})
    fields: dict[str, FieldPrediction] = {}
    for field in TASK_FIELDS[task]:
        answer = answers[field]
        predicted, probabilities, confidence = _normalize_answer(field, answer)
        fields[field] = FieldPrediction(
            field=field,
            predicted=predicted,
            probabilities=probabilities,
            confidence=float(confidence),
        )
    usage = raw.get("usage") or {}
    return ItemPrediction(
        item_id=item_id,
        task=task,
        fields=fields,
        input_tokens=int(usage.get("input_tokens") or 0),
        elapsed_ms=float(elapsed_ms),
        raw=raw,
    )


def _normalize_answer(field: str, answer: dict[str, Any]) -> tuple[Any, dict[str, float], float]:
    answer_type = answer.get("type")
    if answer_type == "noul" or field == "multi_target":
        probability = float(answer.get("noul", 0.0))
        predicted = probability >= 0.5
        probabilities = {
            "false": round(1.0 - probability, 6),
            "true": round(probability, 6),
        }
        return predicted, probabilities, float(answer.get("confidence", max(probability, 1.0 - probability)))
    if answer_type == "score":
        probabilities = {
            str(key): float(value)
            for key, value in (answer.get("probabilities") or {}).items()
        }
        if probabilities:
            predicted = max(probabilities, key=probabilities.get)
        else:
            predicted = str(int(round(float(answer.get("score", 0.0)))))
        return predicted, probabilities, float(answer.get("confidence", 0.0))
    probabilities = {
        str(key): float(value)
        for key, value in (answer.get("probabilities") or {}).items()
    }
    predicted = answer.get("choice")
    if predicted is None and probabilities:
        predicted = max(probabilities, key=probabilities.get)
    return predicted, probabilities, float(answer.get("confidence", 0.0))
