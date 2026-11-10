"""Deterministic keep-or-discard proposer.

Each call starts from the best completed validation score in its own history
and changes one pre-registered parameter. A worse trial simply stops being
the best, so the next call discards it.
"""

from __future__ import annotations

from typing import Any

from proposers.base import nudge_parameter


class GreedyProposer:
    name = "greedy"

    def __init__(self, space: dict[str, Any], baseline: dict[str, Any]):
        self.space = space
        self.baseline = dict(baseline)
        self.names = list(space["parameters"])

    def propose(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [
            row for row in history
            if row.get("status") == "completed" and row.get("validation_metrics")
        ]
        if not completed:
            return dict(self.baseline)
        best = max(completed, key=lambda row: float(row["validation_metrics"]["auc_roc"]))
        config = dict(best["config"])
        step = len(completed)
        name = self.names[step % len(self.names)]
        current = config.get(name, self.baseline.get(name))
        config[name] = nudge_parameter(current, self.space["parameters"][name], step)
        return config
