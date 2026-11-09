import unittest

import numpy as np

from latency.histogram import LatencyHistogram, histogram_of


class HistogramTests(unittest.TestCase):
    def test_percentiles_match_numpy_within_bucket_precision(self):
        rng = np.random.default_rng(0)
        for values in (rng.lognormal(1.0, 0.8, 100_000), rng.exponential(5.0, 100_000)):
            hist = histogram_of(values, precision=0.01)
            for q in (50, 90, 99, 99.9):
                exact = np.percentile(values, q, method="inverted_cdf")
                self.assertAlmostEqual(hist.percentile(q) / exact, 1.0, delta=0.011, msg=f"q={q}")
            self.assertAlmostEqual(hist.mean_ms, values.mean(), places=6)
            self.assertEqual(hist.max_ms, values.max())

    def test_merge_equals_histogram_of_concatenation(self):
        rng = np.random.default_rng(1)
        a, b = rng.lognormal(0, 1, 5_000), rng.lognormal(2, 0.5, 7_000)
        merged = histogram_of(a).merge(histogram_of(b))
        direct = histogram_of(np.concatenate([a, b]))
        np.testing.assert_array_equal(merged.counts, direct.counts)
        self.assertEqual(merged.percentile(99), direct.percentile(99))

    def test_averaging_percentiles_is_not_the_merged_percentile(self):
        fast = np.full(9_900, 1.0)
        slow = np.full(100, 100.0)
        host_a = histogram_of(np.concatenate([fast[:4_950], slow]))  # 2% slow
        host_b = histogram_of(fast[4_950:])                           # never slow
        averaged = (host_a.percentile(99) + host_b.percentile(99)) / 2
        merged = host_a.copy().merge(host_b).percentile(99)
        self.assertAlmostEqual(merged, 1.0, delta=0.02)
        self.assertGreater(averaged, 40.0)

    def test_record_corrected_backfills_omitted_samples(self):
        hist = LatencyHistogram()
        hist.record_corrected(100.0, expected_interval_ms=10.0)
        self.assertEqual(hist.total, 10)  # 100, 90, ..., 10
        self.assertAlmostEqual(hist.percentile(0), 10.0, delta=0.1)
        short = LatencyHistogram()
        short.record_corrected(5.0, expected_interval_ms=10.0)
        self.assertEqual(short.total, 1)

    def test_rejects_bad_values_and_mismatched_layouts(self):
        with self.assertRaises(ValueError):
            LatencyHistogram().record(-1.0)
        with self.assertRaises(ValueError):
            LatencyHistogram(precision=0.01).merge(LatencyHistogram(precision=0.02))
        self.assertTrue(np.isnan(LatencyHistogram().percentile(50)))


if __name__ == "__main__":
    unittest.main()
