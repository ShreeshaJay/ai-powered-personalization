"""Bootstrap, Holm, empty-run accounting, and the test-split guard."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import QueryJudgments
from evaluate import (
    hajek_mean,
    holm,
    interpretable,
    paired_bootstrap,
    quadratic_weighted_kappa,
    record_test_scoring,
    score_p1,
    score_p2,
)


def _query(qid: str, labels: dict[str, str]) -> QueryJudgments:
    return QueryJudgments(qid, qid, "test", labels, list(labels))


class StatsTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_covers_a_shift(self) -> None:
        diff = np.full(40, 0.2)
        first = paired_bootstrap(diff, 2000, 20261110)
        second = paired_bootstrap(diff, 2000, 20261110)
        self.assertEqual(first, second)
        self.assertGreater(first["low"], 0.0)
        self.assertLess(first["p"], 0.05)

    def test_holm_textbook_case(self) -> None:
        decisions = holm([0.01, 0.04, 0.03], alpha=0.05)
        self.assertTrue(decisions[0]["reject"])
        self.assertFalse(decisions[1]["reject"])
        self.assertFalse(decisions[2]["reject"])
        self.assertAlmostEqual(decisions[0]["threshold"], 0.05 / 3)

    def test_small_relative_difference_is_not_interpreted(self) -> None:
        self.assertFalse(interpretable(1.005, 1.0, 0.01))
        self.assertTrue(interpretable(1.02, 1.0, 0.01))

    def test_hajek_and_kappa(self) -> None:
        self.assertAlmostEqual(hajek_mean([1.0, 2.0, 3.0], [1.0, 0.5, 0.5]), 2.2)
        self.assertAlmostEqual(quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)

    def test_ten_empty_queries_score_zero_and_are_counted(self) -> None:
        gains = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
        queries = {f"q{i}": _query(f"q{i}", {"e": "E"}) for i in range(10)}
        run = {qid: [] for qid in queries}
        per_p1 = score_p1(run, queries, gains)
        per_p2 = score_p2(run, queries, gains)
        self.assertEqual(sum(row["empty"] for row in per_p1.values()), 10)
        self.assertTrue(all(row["ndcg"] == 0.0 for row in per_p1.values()))
        self.assertTrue(all(row["recall_at_100_exact"] == 0.0 for row in per_p2.values()))
        self.assertEqual(len(per_p2), 10)

    def test_second_test_scoring_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            record_test_scoring(output, "demo", "abc")
            with self.assertRaises(RuntimeError):
                record_test_scoring(output, "demo", "abc")
            log = (output / "eval_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(json.loads(log[0])["split"], "test")


if __name__ == "__main__":
    unittest.main()
