"""Shared prediction records for every model adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldPrediction:
    field: str
    predicted: Any
    probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ItemPrediction:
    item_id: str
    task: str
    fields: dict[str, FieldPrediction]
    input_tokens: int = 0
    elapsed_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "task": self.task,
            "fields": {
                name: prediction.to_dict() for name, prediction in self.fields.items()
            },
            "input_tokens": self.input_tokens,
            "elapsed_ms": self.elapsed_ms,
        }
