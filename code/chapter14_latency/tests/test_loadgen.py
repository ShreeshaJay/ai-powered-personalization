import gc
import threading
import time
import unittest
from unittest import mock

import numpy as np

import latency.loadgen as loadgen
from latency.loadgen import dispatch_lag_report, open_loop_schedule, run_closed_loop, run_open_loop


class ScheduleTests(unittest.TestCase):
    def test_poisson_schedule_rate_and_determinism(self):
        a = open_loop_schedule(1_000.0, 20.0, seed=7)
        b = open_loop_schedule(1_000.0, 20.0, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(len(a) / 20_000.0, 1.0, delta=0.03)
        self.assertTrue(np.all(np.diff(a) > 0))
        self.assertLess(a[-1], 20.0)
        gaps = np.diff(a)
        self.assertAlmostEqual(gaps.std() / gaps.mean(), 1.0, delta=0.05)  # exponential gaps

    def test_uniform_schedule_is_evenly_spaced(self):
        s = open_loop_schedule(50.0, 1.0, process="uniform")
        self.assertEqual(len(s), 50)
        np.testing.assert_allclose(np.diff(s), 0.02)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            open_loop_schedule(0.0, 1.0)
        with self.assertRaises(ValueError):
            open_loop_schedule(10.0, 1.0, process="bursty")


class CoordinatedOmissionTests(unittest.TestCase):
    """A one-off 300 ms stall at 50 requests per second, measured both ways in real time."""

    def make_handler(self):
        lock = threading.Lock()
        state = {"calls": 0}

        def handler(i, intended_ns, submitted_ns, start_ns):
            with lock:
                state["calls"] += 1
                stall = state["calls"] == 10
            time.sleep(0.3 if stall else 0.001)
        return handler

    def test_open_loop_records_the_requests_a_stall_delayed(self):
        offsets = open_loop_schedule(50.0, 1.5, process="uniform")
        recs = run_open_loop(self.make_handler(), offsets, n_workers=1)
        slow = sum(r.latency_ms > 100 for r in recs)
        self.assertGreaterEqual(slow, 6)  # about ten requests queued behind the stall

    def test_closed_loop_records_the_stall_once(self):
        recs = run_closed_loop(self.make_handler(), n_clients=1, duration_s=1.0, think_s=0.019)
        slow = sum(r.latency_ms > 100 for r in recs)
        self.assertEqual(slow, 1)


class GeneratorLagTests(unittest.TestCase):
    """The generator itself stalls for 200 ms before request 10; the server is always fast."""

    def test_lag_is_measured_counted_in_latency_and_flagged(self):
        real_sleep_until = loadgen._sleep_until
        calls = {"n": 0}

        def late_sleep_until(target_ns, spin_ns=1_000_000):
            calls["n"] += 1
            if calls["n"] == 10:
                time.sleep(0.2)
            real_sleep_until(target_ns, spin_ns)

        offsets = open_loop_schedule(100.0, 0.5, process="uniform")
        with mock.patch.object(loadgen, "_sleep_until", late_sleep_until):
            recs = run_open_loop(lambda *a: time.sleep(0.0005), offsets, n_workers=4)
        lags = np.array([r.dispatch_lag_ms for r in recs])
        # Requests due during the generator's stall went out late, by up to 200 ms ...
        self.assertGreater(lags.max(), 150.0)
        self.assertGreaterEqual(int(np.sum(lags > 50.0)), 10)
        # ... and their latency, timed from the intended send time, includes that lateness.
        for r in recs:
            self.assertGreaterEqual(r.latency_ms, r.dispatch_lag_ms)
        report = dispatch_lag_report(lags, max_lag_ms=2.0)
        self.assertTrue(report["generator_behind"])
        self.assertGreater(report["dispatch_lag_p99_ms"], 2.0)

    def test_on_schedule_run_is_not_flagged(self):
        report = dispatch_lag_report(np.full(1_000, 0.05), max_lag_ms=2.0)
        self.assertFalse(report["generator_behind"])
        self.assertFalse(dispatch_lag_report([], max_lag_ms=2.0)["generator_behind"])


class GarbageCollectionTests(unittest.TestCase):
    def run_and_watch(self, runner):
        seen = []
        runner(lambda *a: seen.append(gc.isenabled()))
        return seen

    def test_collection_is_paused_inside_the_window_and_restored_after(self):
        offsets = open_loop_schedule(200.0, 0.1, process="uniform")
        self.assertTrue(gc.isenabled())
        seen = self.run_and_watch(lambda h: run_open_loop(h, offsets, n_workers=2))
        self.assertTrue(seen and not any(seen))
        seen = self.run_and_watch(lambda h: run_closed_loop(h, n_clients=1, duration_s=0.05))
        self.assertTrue(seen and not any(seen))
        self.assertTrue(gc.isenabled())
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_collection_can_be_left_on(self):
        offsets = open_loop_schedule(200.0, 0.1, process="uniform")
        seen = self.run_and_watch(lambda h: run_open_loop(h, offsets, n_workers=2, pause_gc=False))
        self.assertTrue(seen and all(seen))


if __name__ == "__main__":
    unittest.main()
