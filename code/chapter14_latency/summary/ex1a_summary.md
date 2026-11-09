# Exercise 1 Part A: queueing simulator results

Virtual-time simulation; times are in milliseconds for the coordinated-omission run and in
multiples of the mean service time (or of the leaf median, for fan-out) elsewhere.

## Coordinated omission

| generator | samples | p50_ms | p99_ms | p99.9_ms | max_ms | samples_over_100ms |
|---|---|---|---|---|---|---|
| closed_loop_recorded | 11800 | 1.538 | 10.087 | 15.168 | 996.220 | 2 |
| closed_loop_corrected | 11896 | 1.554 | 13.869 | 896.793 | 996.220 | 90 |
| open_loop | 12000 | 1.393 | 27.556 | 914.819 | 1000.393 | 112 |


## Utilization (Poisson arrivals, exponential service, with M/M/1 formulas)

| utilization | mean_response | mm1_mean_theory | p99_response | mm1_p99_theory |
|---|---|---|---|---|
| 0.100 | 1.115 | 1.111 | 5.137 | 5.117 |
| 0.200 | 1.251 | 1.250 | 5.773 | 5.756 |
| 0.300 | 1.430 | 1.429 | 6.582 | 6.579 |
| 0.400 | 1.667 | 1.667 | 7.689 | 7.675 |
| 0.500 | 1.985 | 2.000 | 9.137 | 9.210 |
| 0.600 | 2.484 | 2.500 | 11.332 | 11.513 |
| 0.700 | 3.352 | 3.333 | 15.270 | 15.351 |
| 0.800 | 5.029 | 5.000 | 23.803 | 23.026 |
| 0.850 | 6.748 | 6.667 | 31.168 | 30.701 |
| 0.900 | 9.677 | 10.000 | 48.029 | 46.052 |
| 0.950 | 18.873 | 20.000 | 83.875 | 92.103 |


Other arrival and service combinations are in `ex1a_utilization.csv`.


## Fan-out (each leaf slow with probability 0.01)

| leaf_slow_prob | fanout | share_delayed_sim | share_delayed_theory | p50_request_latency | p99_request_latency |
|---|---|---|---|---|---|
| 0.010 | 1 | 0.010 | 0.010 | 1.003 | 6.197 |
| 0.010 | 2 | 0.019 | 0.020 | 1.151 | 9.923 |
| 0.010 | 5 | 0.048 | 0.049 | 1.339 | 12.257 |
| 0.010 | 10 | 0.095 | 0.096 | 1.483 | 13.742 |
| 0.010 | 20 | 0.184 | 0.182 | 1.640 | 15.151 |
| 0.010 | 50 | 0.395 | 0.395 | 1.951 | 16.708 |
| 0.010 | 100 | 0.632 | 0.634 | 8.777 | 17.828 |
| 0.010 | 200 | 0.865 | 0.866 | 11.047 | 19.129 |
| 0.010 | 500 | 0.993 | 0.993 | 13.098 | 20.565 |
| 0.010 | 1000 | 1.000 | 1.000 | 14.484 | 21.696 |


## Hedging (10 replicas, bimodal service, loser cancelled)

`utilization` is that of the unhedged system. `backup_requests_per_request` is how many backups were sent, `backup_work_share` the server time spent on them relative to the unhedged run, and `net_work_change` the change in total server time, which can be negative because cancelling a slow primary saves the rest of its work.

| utilization | policy | hedge_delay | p50 | p95 | p99 | p99.9 | backup_requests_per_request | backup_work_share | net_work_change |
|---|---|---|---|---|---|---|---|---|---|
| 0.200 | none |  | 0.754 | 4.603 | 27.794 | 65.361 | 0.000 | 0.000 | 0.000 |
| 0.200 | hedge_after_p95 | 4.603 | 0.723 | 3.338 | 5.103 | 6.901 | 0.021 | 0.013 | -0.112 |
| 0.200 | hedge_after_p50 | 0.754 | 0.695 | 1.842 | 2.626 | 3.824 | 0.470 | 0.197 | -0.138 |
| 0.400 | none |  | 1.028 | 15.846 | 47.029 | 85.458 | 0.000 | 0.000 | 0.000 |
| 0.400 | hedge_after_p95 | 15.846 | 0.968 | 6.436 | 15.785 | 18.469 | 0.010 | 0.006 | -0.059 |
| 0.400 | hedge_after_p50 | 1.028 | 0.843 | 2.220 | 3.070 | 4.433 | 0.423 | 0.167 | -0.137 |
| 0.600 | none |  | 1.644 | 31.833 | 65.340 | 129.912 | 0.000 | 0.000 | 0.000 |
| 0.600 | hedge_after_p95 | 31.833 | 1.543 | 20.001 | 31.402 | 36.269 | 0.009 | 0.004 | -0.025 |
| 0.600 | hedge_after_p50 | 1.644 | 1.111 | 2.977 | 3.999 | 5.510 | 0.341 | 0.124 | -0.133 |
| 0.800 | none |  | 4.441 | 68.555 | 113.333 | 177.683 | 0.000 | 0.000 | 0.000 |
| 0.800 | hedge_after_p95 | 68.555 | 4.056 | 50.602 | 68.211 | 78.855 | 0.009 | 0.004 | -0.008 |
| 0.800 | hedge_after_p50 | 4.441 | 1.887 | 5.804 | 7.563 | 10.285 | 0.166 | 0.053 | -0.119 |
