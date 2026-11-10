"""Shared proposer helpers. The three proposers live in their own modules."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from harness.contract import ROOT


def trial_seed(master: int, trial_id: str) -> int:
    digest = hashlib.sha256(f"{master}:{trial_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def build_proposer(name: str, *, contract: dict[str, Any], seed: int):
    from harness.contract import load_search_space
    from proposers.greedy import GreedyProposer
    from proposers.random_search import RandomProposer
    from proposers.replay import ReplayProposer

    space = load_search_space()
    if name == "random":
        return RandomProposer(space, seed)
    if name == "greedy":
        return GreedyProposer(space, dict(contract["baseline"]))
    if name == "replay":
        return ReplayProposer(ROOT / "proposers" / "proposals_replay.jsonl")
    raise ValueError(f"Unknown proposer: {name}")


def load_proposals(path: Path) -> list[dict[str, Any]]:
    proposals = []
    saw_provenance = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "api_key" in row or row.get("provider") or row.get("model"):
            raise ValueError("A replay file cannot name a model or a provider")
        if row.get("record_type") == "provenance":
            if row.get("source") != "synthetic":
                raise ValueError("This replay file must be labelled synthetic")
            saw_provenance = True
            continue
        proposals.append(row)
    if not saw_provenance:
        raise ValueError("Replay file is missing its synthetic provenance record")
    if not proposals:
        raise ValueError("Replay file has no proposals")
    return proposals


def mix_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def sample_parameter(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    kind = spec["type"]
    if kind == "choice":
        return spec["values"][int(rng.integers(0, len(spec["values"])))]
    if kind == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if kind == "loguniform":
        low = math.log(float(spec["low"]))
        high = math.log(float(spec["high"]))
        return float(math.exp(rng.uniform(low, high)))
    raise ValueError(f"Unknown parameter type: {kind}")


def nudge_parameter(current: Any, spec: dict[str, Any], step: int) -> Any:
    kind = spec["type"]
    if kind == "choice":
        values = list(spec["values"])
        position = values.index(current) if current in values else -1
        return values[(position + 1) % len(values)]
    if kind == "loguniform":
        factor = 1.5 if step % 2 == 0 else 1.0 / 1.5
        nudged = float(current) * factor
        return float(min(float(spec["high"]), max(float(spec["low"]), nudged)))
    if kind == "uniform":
        delta = 0.2 * (float(spec["high"]) - float(spec["low"]))
        nudged = float(current) + (delta if step % 2 == 0 else -delta)
        return float(min(float(spec["high"]), max(float(spec["low"]), nudged)))
    raise ValueError(f"Unknown parameter type: {kind}")
