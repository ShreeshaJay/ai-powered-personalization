import math
import unittest

import numpy as np

from latency.timing import (RESIDUAL_STAGE, SERVICE, WAIT, Trace, attribute, bootstrap_ci,
                            critical_path, critical_span_names, paired_ratio_ci, repeat_ci,
                            repeat_ci_log)


def hand_built_trace() -> Trace:
    """root 0-100: a 0-10, two parallel branches b1 10-40 and b2 10-25, c 40-90, then 10 untraced."""
    tr = Trace(0)
    root = tr.add("request", "R", SERVICE, 0, 100)
    tr.add("a", "S2", SERVICE, 0, 10, root)
    b1 = tr.add("b1", "S3", SERVICE, 10, 40, root)
    tr.add("b1.wait", "S3", WAIT, 10, 15, b1)
    tr.add("b1.work", "S3", SERVICE, 15, 40, b1)
    tr.add("b2", "S3", SERVICE, 10, 25, root)
    tr.add("c", "S5", SERVICE, 40, 90, root)
    return tr


class CriticalPathTests(unittest.TestCase):
    def test_parallel_branch_that_finishes_first_is_off_the_path(self):
        names = critical_span_names(hand_built_trace().spans)
        self.assertIn("b1.work", names)
        self.assertIn("b1.wait", names)
        self.assertNotIn("b2", names)

    def test_segments_partition_the_root(self):
        tr = hand_built_trace()
        total = sum(ns for _, ns in critical_path(tr.spans))
        self.assertEqual(total, 100)

    def test_attribution_by_stage_kind_and_residual(self):
        att = attribute(hand_built_trace().spans)
        self.assertEqual(att[("S2", SERVICE)], 10)
        self.assertEqual(att[("S3", WAIT)], 5)
        self.assertEqual(att[("S3", SERVICE)], 25)
        self.assertEqual(att[("S5", SERVICE)], 50)
        self.assertEqual(att[(RESIDUAL_STAGE, "residual")], 10)

    def test_gap_between_children_goes_to_parent(self):
        tr = Trace(1)
        root = tr.add("request", "R", SERVICE, 0, 50)
        grp = tr.add("grp", "S4", SERVICE, 0, 50, root)
        tr.add("x", "S4", WAIT, 0, 20, grp)
        tr.add("y", "S4", SERVICE, 30, 50, grp)
        att = attribute(tr.spans)
        self.assertEqual(att[("S4", WAIT)], 20)
        self.assertEqual(att[("S4", SERVICE)], 30)  # 20 in y plus the 10 gap in grp
        self.assertNotIn((RESIDUAL_STAGE, "residual"), att)


class IntervalTests(unittest.TestCase):
    def test_repeat_ci_matches_student_t(self):
        mean, lo, hi = repeat_ci([1.0, 2.0, 3.0])
        half = 4.302652729911275 / math.sqrt(3)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(lo, 2.0 - half, places=6)
        self.assertAlmostEqual(hi, 2.0 + half, places=6)
        self.assertTrue(math.isnan(repeat_ci([5.0])[1]))

    def test_log_interval_is_positive_and_matches_t_on_logs(self):
        x = [10.0, 40.0]  # two very different repeats, as in a quick run
        _, raw_lo, _ = repeat_ci(x)
        self.assertLess(raw_lo, 0.0)  # the raw-scale interval is impossible for a latency
        gm, lo, hi = repeat_ci_log(x)
        self.assertAlmostEqual(gm, 20.0)  # geometric mean
        self.assertGreater(lo, 0.0)
        half = 12.706204736174698 * np.std(np.log(x), ddof=1) / math.sqrt(2)
        self.assertTrue(math.isclose(lo, 20.0 * math.exp(-half), rel_tol=1e-9))
        self.assertTrue(math.isclose(hi, 20.0 * math.exp(half), rel_tol=1e-9))
        self.assertTrue(math.isclose(lo * hi, gm ** 2, rel_tol=1e-9))  # symmetric on the log scale

    def test_paired_ratio_removes_shared_variation(self):
        reference = np.array([10.0, 20.0, 40.0, 15.0, 30.0])  # run-to-run noise
        faster = reference * 0.8                                 # a consistent 20% gain
        ratio, lo, hi = paired_ratio_ci(faster, reference)
        self.assertAlmostEqual(ratio, 0.8)
        self.assertAlmostEqual(lo, 0.8)
        self.assertAlmostEqual(hi, 0.8)
        # The unpaired intervals overlap heavily even though every pair shows the gain.
        _, ref_lo, ref_hi = repeat_ci_log(reference)
        _, fast_lo, fast_hi = repeat_ci_log(faster)
        self.assertTrue(fast_lo <= ref_hi and ref_lo <= fast_hi)
        with self.assertRaises(ValueError):
            paired_ratio_ci([1.0], [1.0, 2.0])

    def test_bootstrap_interval_covers_true_median(self):
        rng = np.random.default_rng(3)
        values = rng.exponential(1.0, 20_000)
        point, lo, hi = bootstrap_ci(values, np.median, n_boot=300, seed=0)
        self.assertLess(lo, math.log(2))
        self.assertGreater(hi, math.log(2))
        self.assertLess(lo, point)
        self.assertLess(point, hi)


if __name__ == "__main__":
    unittest.main()
