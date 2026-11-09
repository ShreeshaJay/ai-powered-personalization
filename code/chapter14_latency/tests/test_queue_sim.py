import math
import unittest

import numpy as np

from sim.queue_sim import (_finish_with_stall, bursty_arrivals, fanout_experiment, fcfs_queue,
                           hedging_experiment, mm1_theory, poisson_arrivals, service_sampler,
                           simulate_replicas, stall_experiment, time_average_in_system)


class SingleServerTests(unittest.TestCase):
    def test_mm1_matches_closed_form(self):
        rng = np.random.default_rng(0)
        n, rho = 300_000, 0.5
        arr = poisson_arrivals(rho, n, rng)
        resp, _ = fcfs_queue(arr, rng.exponential(1.0, n))
        resp = resp[n // 10:]
        theory = mm1_theory(rho)
        self.assertAlmostEqual(resp.mean() / theory["mean"], 1.0, delta=0.04)
        self.assertAlmostEqual(np.percentile(resp, 99) / theory["p99"], 1.0, delta=0.06)

    def test_mean_response_triples_from_40_to_80_percent(self):
        self.assertAlmostEqual(mm1_theory(0.8)["mean"] / mm1_theory(0.4)["mean"], 3.0)

    def test_littles_law(self):
        rng = np.random.default_rng(1)
        n, rho = 200_000, 0.7
        arr = poisson_arrivals(rho, n, rng)
        resp, _ = fcfs_queue(arr, rng.exponential(1.0, n))
        lam = (n - 1) / (arr[-1] - arr[0])
        sampled = time_average_in_system(arr, resp, 50_000, rng)
        self.assertAlmostEqual(sampled / (lam * resp.mean()), 1.0, delta=0.05)

    def test_two_servers_match_erlang_c(self):
        # M/M/2 with lambda = 1.2, mu = 1: P(wait) = 0.45, mean response = 1 + 0.45 / 0.8 = 1.5625.
        rng = np.random.default_rng(2)
        n = 300_000
        arr = poisson_arrivals(1.2, n, rng)
        resp, wait = fcfs_queue(arr, rng.exponential(1.0, n), n_servers=2)
        keep = slice(n // 10, None)
        self.assertAlmostEqual(resp[keep].mean() / 1.5625, 1.0, delta=0.04)
        self.assertAlmostEqual(float(np.mean(wait[keep] > 0)), 0.45, delta=0.02)

    def test_samplers_have_requested_mean(self):
        rng = np.random.default_rng(3)
        for dist in ("exponential", "lognormal", "bimodal", "deterministic"):
            draw = service_sampler(dist, 2.0, rng, sigma=1.0, p_slow=0.01, slow_factor=20)
            self.assertAlmostEqual(draw(400_000).mean() / 2.0, 1.0, delta=0.03, msg=dist)

    def test_bursty_arrivals_keep_mean_rate(self):
        rng = np.random.default_rng(4)
        arr = bursty_arrivals(2.0, 200_000, rng)
        self.assertAlmostEqual(len(arr) / arr[-1] / 2.0, 1.0, delta=0.05)
        gaps = np.diff(arr)
        self.assertGreater(gaps.std() / gaps.mean(), 1.1)  # burstier than Poisson


class StallTests(unittest.TestCase):
    def test_finish_with_stall_cases(self):
        stall = (10.0, 20.0)
        self.assertEqual(_finish_with_stall(0.0, 5.0, stall), 5.0)    # before
        self.assertEqual(_finish_with_stall(25.0, 5.0, stall), 30.0)  # after
        self.assertEqual(_finish_with_stall(12.0, 5.0, stall), 25.0)  # starts inside
        self.assertEqual(_finish_with_stall(8.0, 5.0, stall), 23.0)   # spans the start

    def test_closed_loop_hides_the_stall_and_correction_recovers_it(self):
        rows, _ = stall_experiment(duration=30_000.0, stall_start=15_000.0, seed=0)
        by = {r["generator"]: r for r in rows}
        self.assertLess(by["closed_loop_recorded"]["p99_ms"], 50.0)
        self.assertGreater(by["open_loop"]["p99_ms"], 500.0)
        ratio = by["closed_loop_corrected"]["p99_ms"] / by["open_loop"]["p99_ms"]
        self.assertAlmostEqual(ratio, 1.0, delta=0.25)


class FanoutAndHedgingTests(unittest.TestCase):
    def test_fanout_share_matches_closed_form(self):
        rows = fanout_experiment([1, 10, 100], [0.01], n_requests=40_000, seed=0)
        for r in rows:
            self.assertAlmostEqual(r["share_delayed_sim"], r["share_delayed_theory"], delta=0.012)
        self.assertAlmostEqual(rows[-1]["share_delayed_theory"], 1 - 0.99 ** 100)
        self.assertAlmostEqual(1 - 0.99 ** 100, 0.634, places=3)

    def test_replica_simulator_without_hedging_equals_fcfs(self):
        rng = np.random.default_rng(5)
        arr = poisson_arrivals(0.5, 3_000, rng)
        svc_rng = np.random.default_rng(6)
        res = simulate_replicas(arr, lambda n: svc_rng.exponential(1.0, n), 1, np.random.default_rng(7))
        svc_rng = np.random.default_rng(6)
        expected, _ = fcfs_queue(arr, svc_rng.exponential(1.0, 3_000))
        np.testing.assert_allclose(res["response"], expected)

    def test_hedge_work_accounting_by_hand(self):
        # One request, two replicas. The primary needs 10 units; a backup sent at t = 2
        # needs 1 unit and finishes at t = 3, so the primary is cancelled after 3 units.
        draws = iter([np.array([10.0]), np.array([1.0])])
        res = simulate_replicas(np.array([0.0]), lambda n: next(draws), 2, np.random.default_rng(0),
                                hedge_delay=2.0)
        self.assertAlmostEqual(res["response"][0], 3.0)
        self.assertEqual(res["hedged"], 1)
        self.assertAlmostEqual(res["backup_work"], 1.0)
        self.assertAlmostEqual(res["busy_time"], 4.0)  # 3 units of cancelled primary + 1 of backup

    def test_queued_backup_is_cancelled_without_work(self):
        # Primaries need 3 and 5 units (arriving at 0 and 0.5), backups 1 unit, hedge after 1.
        def run(seed):
            draws = iter([np.array([3.0, 5.0]), np.array([1.0, 1.0])])
            return simulate_replicas(np.array([0.0, 0.5]), lambda n: next(draws), 2,
                                     np.random.default_rng(seed), hedge_delay=1.0)
        # Seed 1 routes the primaries to different replicas, so each backup queues behind the
        # other request's primary. Request 0 finishes at 3 and its queued backup is dropped
        # without work; request 1's backup runs 3 to 4 and stops its primary after 3.5 units.
        apart = run(1)
        np.testing.assert_allclose(apart["response"], [3.0, 3.5])
        self.assertAlmostEqual(apart["busy_time"], 3.0 + 1.0 + 3.5)
        self.assertAlmostEqual(apart["backup_work"], 1.0)
        # Seed 0 routes both primaries to one replica, so both backups run on the idle one.
        together = run(0)
        np.testing.assert_allclose(together["response"], [2.0, 2.5])
        self.assertAlmostEqual(together["busy_time"], 2.0 + 1.0 + 1.0 + 1.0)
        self.assertAlmostEqual(together["backup_work"], 2.0)

    def test_hedging_cuts_the_tail_with_few_backups(self):
        rows = hedging_experiment(n_requests=40_000, seed=1, hedge_percentiles=(95.0,))
        none, hedged = rows
        self.assertLess(hedged["p99"], none["p99"])
        self.assertLess(hedged["backup_requests_per_request"], 0.08)
        self.assertGreater(hedged["backup_work_share"], 0.0)
        self.assertLess(hedged["backup_work_share"], hedged["backup_requests_per_request"] * 20)
        self.assertTrue(math.isnan(none["hedge_delay"]))

    def test_hedging_rows_per_utilization(self):
        rows = hedging_experiment(n_requests=5_000, utilizations=(0.2, 0.6), seed=2,
                                  hedge_percentiles=(95.0,))
        self.assertEqual([(r["utilization"], r["policy"]) for r in rows],
                         [(0.2, "none"), (0.2, "hedge_after_p95"), (0.6, "none"), (0.6, "hedge_after_p95")])
        for r in rows:
            if r["policy"] == "none":
                self.assertEqual(r["net_work_change"], 0.0)


if __name__ == "__main__":
    unittest.main()
