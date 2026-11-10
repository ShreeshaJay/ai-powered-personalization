"""Random search baseline. Independent draws, ignoring earlier scores."""

from __future__ import annotations

from typing import Any

import numpy as np

from proposers.base import mix_seed, sample_parameter


class RandomProposer:
    name = "random"

    def __init__(self, space: dict[str, Any], seed: int):
        self.space = space
        self.seed = int(seed)

    def propose(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        rng = np.random.default_rng(mix_seed(self.seed, len(history)))
        return {
            name: sample_parameter(spec, rng)
            for name, spec in self.space["parameters"].items()
        }
