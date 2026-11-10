"""Replay proposals from a JSONL file.

The file shipped with this folder is synthetic and says so in its first
record. This proposer reads that file and returns. It does not call a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from proposers.base import load_proposals


class ReplayProposer:
    name = "replay"

    def __init__(self, path: Path):
        self.path = path
        self.proposals = load_proposals(path)

    def propose(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        index = len(history)
        if index >= len(self.proposals):
            raise IndexError("Replay file has no further proposals")
        return dict(self.proposals[index])
