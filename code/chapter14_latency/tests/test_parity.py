import unittest

import numpy as np

from ranker.parity import (Tolerances, kendall_tau, parity_report, rank_parity, rbo, score_parity,
                           topk_overlap)


class RankingMetricTests(unittest.TestCase):
    def test_topk_overlap(self):
        s = np.arange(20, dtype=float)
        self.assertEqual(topk_overlap(s, s, 10), 1.0)
        self.assertEqual(topk_overlap(s, -s, 10), 0.0)
        shifted = s.copy()
        shifted[[19, 0]] = shifted[[0, 19]]  # swap the best and the worst item
        self.assertAlmostEqual(topk_overlap(s, shifted, 10), 0.9)

    def test_kendall_tau(self):
        s = np.arange(10, dtype=float)
        self.assertAlmostEqual(kendall_tau(s, s), 1.0)
        self.assertAlmostEqual(kendall_tau(s, -s), -1.0)

    def test_rbo_known_values(self):
        self.assertAlmostEqual(rbo(list("abcde"), list("abcde")), 1.0)
        self.assertAlmostEqual(rbo(list("abc"), list("xyz")), 0.0)
        # A = [a, b], B = [b, a], p = 0.5: X1 = 0, X2 = 2, so
        # RBO_ext = (2/2) * 0.5^2 + (0.5 / 0.5) * (0 * 0.5 + 1 * 0.25) = 0.5
        self.assertAlmostEqual(rbo(["a", "b"], ["b", "a"], p=0.5), 0.5)
        self.assertLess(rbo(list("abcdef"), list("bacdef")), 1.0)
        self.assertGreater(rbo(list("abcdef"), list("abcdfe")), rbo(list("abcdef"), list("bacdef")))

    def test_score_parity(self):
        s = np.linspace(0.1, 0.9, 50)
        out = score_parity(s, s)
        self.assertEqual(out["max_abs_diff"], 0.0)
        self.assertAlmostEqual(out["pearson"], 1.0)
        self.assertAlmostEqual(out["spearman"], 1.0)

    def test_rank_parity_groups(self):
        rng = np.random.default_rng(0)
        scores = rng.random(300)
        groups = np.repeat(np.arange(10), 30)
        same = rank_parity(scores, scores, groups)
        self.assertEqual(same["groups"], 10)
        self.assertAlmostEqual(same["top10_overlap"], 1.0)
        self.assertAlmostEqual(same["kendall_tau"], 1.0)
        reversed_ = rank_parity(scores, -scores, groups)
        self.assertAlmostEqual(reversed_["kendall_tau"], -1.0)


class ParityReportTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 5_000
        self.ref = rng.random(n)
        self.labels = (rng.random(n) < self.ref).astype(float)
        self.groups = np.repeat(np.arange(n // 25), 25)
        self.hist = rng.integers(1, 100, n)
        self.plays = rng.integers(1, 1_000, n)

    def test_identical_variant_passes_every_slice(self):
        rows = parity_report("same", self.ref, self.ref.copy(), self.labels, self.groups, self.hist,
                             self.plays, Tolerances())
        self.assertEqual({r["slice"] for r in rows},
                         {"all", "short_history_users", "long_history_users", "popular_items", "tail_items"})
        self.assertTrue(all(r["passes"] for r in rows))

    def test_calibration_shift_fails(self):
        rows = parity_report("shifted", self.ref, np.clip(self.ref + 0.02, 0, 1), self.labels, self.groups,
                             self.hist, self.plays, Tolerances())
        overall = next(r for r in rows if r["slice"] == "all")
        self.assertFalse(overall["passes"])
        self.assertIn("calibration", overall["failed_checks"])
        self.assertAlmostEqual(overall["top10_overlap"], 1.0)  # ranks unchanged


if __name__ == "__main__":
    unittest.main()
