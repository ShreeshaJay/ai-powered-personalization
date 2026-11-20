"""Frozen fusion streaming and the list of systems each summary scores."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_plan
from run_pipeline import _report_names, _systems
from tune_fusion import apply_frozen, iter_frozen

SELECTION = {
    "best_splade_variant": "splade_esci",
    "tuned": {"f4_alpha": 0.35, "f5_weights": [0.2, 0.5, 0.3], "f7_k": 20},
}


def _runs(query_ids: list[str]) -> dict:
    rng = random.Random(7)
    runs = {}
    for name in _systems():
        run = {}
        for query_id in query_ids:
            products = rng.sample([f"p{index:03d}" for index in range(60)], 25)
            scores = sorted((rng.random() for _ in products), reverse=True)
            run[query_id] = list(zip(products, scores))
        runs[name] = run
    return runs


class PipelineS2Tests(unittest.TestCase):
    def test_streamed_fusion_matches_apply_frozen(self) -> None:
        query_ids = [f"q{index}" for index in range(12)]
        runs = _runs(query_ids)
        expected = apply_frozen(runs, query_ids, SELECTION, depth=10)
        streamed = list(iter_frozen(runs, query_ids, SELECTION, depth=10))
        self.assertEqual([name for name, _ in streamed], list(expected))
        self.assertEqual(dict(streamed), expected)
        kept = dict(iter_frozen(runs, query_ids, SELECTION, depth=10, skip=lambda name: name.startswith("f2_")))
        self.assertEqual(kept, {name: run for name, run in expected.items() if not name.startswith("f2_")})

    def test_report_names_score_each_system_once(self) -> None:
        pools = load_plan()["fusion"]["selection"]
        hybrids = [name for name in _report_names({"best_hybrid": "f1_rrf_bm25_dense", "best_single": "bm25"}) if name.startswith("f")]
        for best_single in pools["single_pool"]:
            for best_hybrid in hybrids:
                names = _report_names({"best_hybrid": best_hybrid, "best_single": best_single})
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(names[: len(_systems())], _systems())
                self.assertIn(best_hybrid, names)
                self.assertIn(best_single, names)
        self.assertEqual(len(hybrids), 11)


if __name__ == "__main__":
    unittest.main()
