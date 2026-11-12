"""Worked examples for RRF, convex combination, DBSF, and tie breaks."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion import convex, dbsf, ranking_from_scores, read_run, rrf, simplex_weights, write_run, zscore_norm


class FusionTests(unittest.TestCase):
    def test_rrf_rank_origin_is_one(self) -> None:
        # Leg 1: d1, d2. Leg 2: d3, d1. k = 60.
        leg_a = [("d1", 5.0), ("d2", 1.0)]
        leg_b = [("d3", 9.0), ("d1", 2.0)]
        fused = dict(rrf([leg_a, leg_b], k=60))
        self.assertAlmostEqual(fused["d1"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused["d2"], 1 / 62)
        self.assertAlmostEqual(fused["d3"], 1 / 61)
        self.assertEqual([pid for pid, _ in rrf([leg_a, leg_b], k=60)], ["d1", "d3", "d2"])

    def test_rrf_missing_leg_adds_nothing(self) -> None:
        leg_a = [("d1", 1.0)]
        leg_b = [("d2", 1.0)]
        fused = dict(rrf([leg_a, leg_b], k=60))
        self.assertAlmostEqual(fused["d1"], 1 / 61)
        self.assertAlmostEqual(fused["d2"], 1 / 61)
        self.assertNotIn("d3", fused)

    def test_weighted_rrf(self) -> None:
        leg = [("d1", 1.0)]
        fused = dict(rrf([leg, []], k=60, weights=[2.0, 1.0]))
        self.assertAlmostEqual(fused["d1"], 2.0 / 61)

    def test_convex_minmax_and_missing_leg(self) -> None:
        bm25 = [("d1", 10.0), ("d2", 0.0), ("d3", 10.0)]
        dense = [("d1", 0.0), ("d2", 1.0)]
        fused = dict(convex([bm25, dense], weights=[0.5, 0.5]))
        self.assertAlmostEqual(fused["d1"], 0.5)
        self.assertAlmostEqual(fused["d2"], 0.5)
        # d3 is absent from dense, so that leg contributes 0 after normalization.
        self.assertAlmostEqual(fused["d3"], 0.5)

    def test_zero_range_normalizes_to_zero(self) -> None:
        fused = dict(convex([[("d1", 4.0), ("d2", 4.0)]], weights=[1.0]))
        self.assertEqual(fused["d1"], 0.0)
        self.assertEqual(fused["d2"], 0.0)

    def test_dbsf_formula(self) -> None:
        scores = {"d1": 1.0, "d2": 2.0, "d3": 3.0}
        values = [1.0, 2.0, 3.0]
        mean = sum(values) / 3
        var = sum((v - mean) ** 2 for v in values) / 2
        sigma = math.sqrt(var)
        normed = dict(dbsf([[("d1", 1.0), ("d2", 2.0), ("d3", 3.0)]]))
        for key, raw in scores.items():
            expected = (raw - (mean - 3 * sigma)) / (6 * sigma)
            self.assertAlmostEqual(normed[key], expected)

    def test_dbsf_singleton_is_one_half(self) -> None:
        self.assertEqual(dict(dbsf([[("d1", 4.0)]])), {"d1": 0.5})

    def test_zscore_zero_spread(self) -> None:
        self.assertEqual(zscore_norm({"d1": 2.0, "d2": 2.0}), {"d1": 0.0, "d2": 0.0})

    def test_ties_break_on_product_id(self) -> None:
        hits = ranking_from_scores(["b", "a", "c"], [1.0, 1.0, 0.5])
        self.assertEqual([pid for pid, _ in hits], ["a", "b", "c"])

    def test_run_file_round_trip_is_byte_stable(self) -> None:
        run = {"q2": [("b", 1.0), ("a", 1.0)], "q1": [("c", 0.25)]}
        # Caller is responsible for tie-broken order. Write what it is given.
        first = Path(self.id().replace(".", "_") + "_a.txt")
        second = Path(self.id().replace(".", "_") + "_b.txt")
        try:
            write_run(first, run, "unit")
            write_run(second, run, "unit")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = read_run(first)
            self.assertEqual(loaded["q1"][0][0], "c")
            self.assertEqual(loaded["q2"][0][1], 1.0)
            text = first.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("q1 Q0 c 1 "))
        finally:
            first.unlink(missing_ok=True)
            second.unlink(missing_ok=True)

    def test_simplex_step(self) -> None:
        grid = simplex_weights(0.1)
        self.assertIn((1.0, 0.0, 0.0), grid)
        self.assertIn((0.0, 0.0, 1.0), grid)
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in grid))
        self.assertEqual(len(grid), 66)


if __name__ == "__main__":
    unittest.main()
