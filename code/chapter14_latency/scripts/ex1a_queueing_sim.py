"""
Exercise 1 Part A: why tails grow, in virtual time.

Four simulations, all deterministic for a given seed:

1. Coordinated omission. One server stalls once. A closed-loop generator
   records the stall as a few slow samples; an open-loop generator, timing
   from intended send times, records every request the stall delayed. The
   corrected closed-loop histogram back-fills the missing samples.
2. Utilization. Mean and p99 response time from 10% to 95% utilization, for
   Poisson and bursty arrivals and three service-time distributions,
   compared with the single-server exponential formulas. Little's law is
   checked by sampling the number of requests in the system.
3. Fan-out. The share of requests that wait on at least one slow leaf, and
   the request p99, as the number of parallel leaf calls grows.
4. Hedging. Replicated servers with a backup request after a percentile
   delay and cancellation of the loser: tail change and added load, at
   several utilizations.

Usage:
    python -m scripts.ex1a_queueing_sim
    python -m scripts.ex1a_queueing_sim --quick
"""

from __future__ import annotations

import argparse
import time

import config
from latency.report import markdown_table, write_csv, write_run_info
from sim.queue_sim import fanout_experiment, hedging_experiment, stall_experiment, utilization_sweep


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--quick", action="store_true", help="small sizes for a smoke test")
    p.add_argument("--summary-dir", default=None, help="override the summary folder")
    args = p.parse_args(argv)
    out = config.summary_dir(args.quick, args.summary_dir)
    t_start = time.perf_counter()

    print("[1/4] Coordinated omission: closed loop versus open loop through a stall")
    co_rows, _ = stall_experiment(duration=20_000.0 if args.quick else 120_000.0,
                                  stall_start=10_000.0 if args.quick else 60_000.0, seed=args.seed)
    write_csv(co_rows, out / "ex1a_coordinated_omission.csv")
    print(markdown_table(co_rows))

    print("[2/4] Utilization sweep")
    rhos = [0.2, 0.4, 0.6, 0.8, 0.9] if args.quick else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    util_rows = utilization_sweep(rhos, n_jobs=20_000 if args.quick else 400_000, seed=args.seed)
    write_csv(util_rows, out / "ex1a_utilization.csv")
    mm1 = [r for r in util_rows if r["arrivals"] == "poisson" and r["service"] == "exponential"]
    print(markdown_table(mm1, ["utilization", "mean_response", "mm1_mean_theory", "p99_response",
                               "mm1_p99_theory", "littles_law_L_from_lambda_W", "littles_law_L_sampled"]))

    print("[3/4] Fan-out amplification")
    fan_rows = fanout_experiment(widths=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                                 slow_probs=[0.01, 0.001, 0.0001],
                                 n_requests=2_000 if args.quick else 50_000, seed=args.seed)
    write_csv(fan_rows, out / "ex1a_fanout.csv")
    print(markdown_table([r for r in fan_rows if r["leaf_slow_prob"] == 0.01]))

    print("[4/4] Hedged requests with cancellation, at several utilizations")
    hedge_rows = hedging_experiment(n_requests=20_000 if args.quick else 300_000,
                                    utilizations=(0.3, 0.7) if args.quick else (0.2, 0.4, 0.6, 0.8),
                                    seed=args.seed)
    write_csv(hedge_rows, out / "ex1a_hedging.csv")
    print(markdown_table(hedge_rows))

    md = [
        "# Exercise 1 Part A: queueing simulator results\n",
        "Virtual-time simulation; times are in milliseconds for the coordinated-omission run and in",
        "multiples of the mean service time (or of the leaf median, for fan-out) elsewhere.\n",
        "## Coordinated omission\n", markdown_table(co_rows),
        "\n## Utilization (Poisson arrivals, exponential service, with M/M/1 formulas)\n",
        markdown_table(mm1, ["utilization", "mean_response", "mm1_mean_theory", "p99_response", "mm1_p99_theory"]),
        "\nOther arrival and service combinations are in `ex1a_utilization.csv`.\n",
        "\n## Fan-out (each leaf slow with probability 0.01)\n",
        markdown_table([r for r in fan_rows if r["leaf_slow_prob"] == 0.01]),
        "\n## Hedging (10 replicas, bimodal service, loser cancelled)\n",
        "`utilization` is that of the unhedged system. `backup_requests_per_request` is how many backups "
        "were sent, `backup_work_share` the server time spent on them relative to the unhedged run, and "
        "`net_work_change` the change in total server time, which can be negative because cancelling a "
        "slow primary saves the rest of its work.\n",
        markdown_table(hedge_rows),
    ]
    (out / "ex1a_summary.md").write_text("\n".join(md), encoding="utf-8")
    write_run_info(out, "scripts.ex1a_queueing_sim", args,
                   {"runtime_s": time.perf_counter() - t_start})
    print(f"Done in {time.perf_counter() - t_start:.1f} s. Summaries in {out}")


if __name__ == "__main__":
    main()
