"""Hand-computed nDCG, recall, condensed lists, and an ir-measures cross-check."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import QueryJudgments
from evaluate import (
    condensed_ndcg,
    dcg,
    judged_at,
    macro,
    ndcg_for_ids,
    recall_at,
    score_p1,
    score_p2,
)


def _query(qid: str, labels: dict[str, str]) -> QueryJudgments:
    return QueryJudgments(qid, f"text {qid}", "test", labels, list(labels))


class MetricTests(unittest.TestCase):
    def test_hand_ndcg(self) -> None:
        # Ranking Irrelevant then Exact. Ideal is Exact then Irrelevant.
        gains = {"e": 1.0, "i": 0.0}
        value = ndcg_for_ids(["i", "e"], gains, k=None)
        expected = (0.0 / np.log2(2) + 1.0 / np.log2(3)) / (1.0 / np.log2(2))
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(dcg([1.0, 0.0]), 1.0)

    def test_undefined_when_all_irrelevant(self) -> None:
        self.assertIsNone(ndcg_for_ids(["a", "b"], {"a": 0.0, "b": 0.0}))

    def test_empty_list_scores_zero_when_defined(self) -> None:
        queries = {"q": _query("q", {"e": "E", "i": "I"})}
        per = score_p1({"q": []}, queries, {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0})
        self.assertEqual(per["q"]["ndcg"], 0.0)
        self.assertTrue(per["q"]["empty"])
        summary = macro(per, "ndcg")
        self.assertEqual(summary["n_scored"], 1)
        self.assertEqual(summary["n_empty"], 1)
        self.assertEqual(summary["mean"], 0.0)

    def test_all_irrelevant_is_excluded_not_zero(self) -> None:
        queries = {"q": _query("q", {"a": "I"})}
        per = score_p1({"q": [("a", 1.0)]}, queries, {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0})
        self.assertIsNone(per["q"]["ndcg"])
        summary = macro(per, "ndcg")
        self.assertEqual(summary["n_excluded"], 1)
        self.assertEqual(summary["n_scored"], 0)

    def test_recall_and_judged_denominators(self) -> None:
        self.assertEqual(recall_at(["a", "b"], {"b", "c"}, 1), 0.0)
        self.assertEqual(recall_at(["b"], {"b", "c"}, 10), 0.5)
        self.assertIsNone(recall_at(["a"], set(), 10))
        # Three of the top 10 slots are filled, one of them judged. Denominator stays 10.
        self.assertEqual(judged_at(["j", "u", "u"], {"j"}, 10), 0.1)
        self.assertEqual(judged_at([], {"j"}, 10), 0.0)

    def test_condensed_drops_unjudged_before_cutoff(self) -> None:
        gains = {"e": 1.0, "s": 0.1}
        # Unjudged documents sit in front. After they are removed, Exact is first.
        value = condensed_ndcg(["u1", "u2", "e", "s"], gains, k=10)
        self.assertAlmostEqual(value, 1.0)

    def test_p2_empty_recall_is_zero(self) -> None:
        queries = {"q": _query("q", {"e": "E"})}
        per = score_p2({}, queries, {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0})
        self.assertEqual(per["q"]["recall_at_100_exact"], 0.0)
        self.assertEqual(per["q"]["judged_at_10"], 0.0)
        self.assertEqual(per["q"]["condensed_ndcg_at_10"], 0.0)

    def test_ir_measures_on_five_hundred_queries(self) -> None:
        import ir_measures
        from ir_measures import ScoredDoc, Qrel
        from ir_measures.measures import nDCG, Recall

        rng = np.random.default_rng(20261110)
        qrels = []
        run = []
        ours_ndcg = []
        ours_recall = []
        for q in range(500):
            qid = f"q{q}"
            gains = {}
            labels = rng.integers(0, 4, size=8)
            labels[0] = max(int(labels[0]), 1)
            for i, label in enumerate(labels):
                doc = f"d{q}_{i}"
                gain = float(label)  # 0, 1, 2, 3
                gains[doc] = gain
                qrels.append(Qrel(query_id=qid, doc_id=doc, relevance=int(label)))
            order = rng.permutation(list(gains))
            for rank, doc in enumerate(order):
                score = float(len(order) - rank)
                run.append(ScoredDoc(query_id=qid, doc_id=doc, score=score))
            ours_ndcg.append(ndcg_for_ids(list(order), gains, 10))
            relevant = {doc for doc, gain in gains.items() if gain > 0}
            ours_recall.append(recall_at(list(order), relevant, 5))
        their = ir_measures.calc_aggregate([nDCG @ 10, Recall @ 5], qrels, run)
        self.assertAlmostEqual(float(np.mean(ours_ndcg)), float(their[nDCG @ 10]), places=6)
        self.assertAlmostEqual(float(np.mean(ours_recall)), float(their[Recall @ 5]), places=6)


if __name__ == "__main__":
    unittest.main()
