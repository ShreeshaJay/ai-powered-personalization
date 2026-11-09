# Exercise 1 Part B: tail attribution report

Catalog: coveo. Workers: 4. Arrival process: poisson. Repeats per load: 5 x 15 s. Tail: requests above p99.

Platform: Linux-6.1.0-54-cloud-amd64-x86_64-with-glibc2.36, 8 logical CPUs; 1 ms sleep median overshoot 0.06 ms.

Measured baseline capacity (closed loop): 336.6 requests/s. Every configuration below faces the same offered rates.

## Configurations

* baseline: `baseline`
* lever: popularity=precomputed: `popularity=precomputed`
* lever: hedge-feature-fetch=on: `hedge_feature_fetch=True`
* lever: feature-cache=on: `feature_cache=True`
* lever: candidates=200: `candidates=200`

## Latency against load

| config | load | offered_rps | achieved_rps | p50_ms | p95_ms | p99_ms | p99_repeat_lo | p99_repeat_hi | dispatch_lag_p99_ms | generator_behind | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.300 | 100.979 | 100.714 | 9.041 | 21.919 | 30.743 | 30.281 | 31.212 | 0.291 | False | 0 |
| baseline | 0.500 | 168.298 | 165.057 | 9.597 | 23.267 | 32.634 | 31.999 | 33.682 | 0.373 | False | 0 |
| baseline | 0.700 | 235.617 | 233.085 | 11.142 | 29.251 | 37.887 | 36.425 | 38.940 | 0.565 | False | 0 |
| baseline | 0.850 | 286.107 | 282.948 | 15.784 | 46.692 | 65.489 | 50.760 | 74.094 | 0.648 | False | 0 |
| lever: popularity=precomputed | 0.300 | 100.979 | 100.713 | 8.863 | 21.702 | 30.743 | 30.312 | 31.305 | 0.282 | False | 0 |
| lever: popularity=precomputed | 0.500 | 168.298 | 165.055 | 9.408 | 23.037 | 32.634 | 32.003 | 33.278 | 0.356 | False | 0 |
| lever: popularity=precomputed | 0.700 | 235.617 | 233.096 | 10.814 | 28.110 | 37.141 | 35.354 | 38.555 | 0.537 | False | 0 |
| lever: popularity=precomputed | 0.850 | 286.107 | 282.955 | 14.432 | 39.820 | 55.298 | 47.909 | 60.127 | 0.602 | False | 0 |
| lever: hedge-feature-fetch=on | 0.300 | 100.979 | 100.744 | 8.952 | 12.186 | 14.147 | 13.200 | 15.103 | 0.292 | False | 0 |
| lever: hedge-feature-fetch=on | 0.500 | 168.298 | 165.055 | 9.597 | 14.576 | 18.693 | 17.252 | 19.934 | 0.391 | False | 0 |
| lever: hedge-feature-fetch=on | 0.700 | 235.617 | 233.158 | 10.814 | 20.444 | 26.745 | 24.483 | 29.567 | 0.515 | False | 0 |
| lever: hedge-feature-fetch=on | 0.850 | 286.107 | 283.027 | 12.808 | 30.137 | 41.437 | 37.492 | 44.186 | 0.676 | False | 0 |
| lever: feature-cache=on | 0.300 | 100.979 | 100.708 | 8.349 | 17.092 | 27.283 | 26.396 | 27.644 | 0.265 | False | 0 |
| lever: feature-cache=on | 0.500 | 168.298 | 165.025 | 8.863 | 17.786 | 28.110 | 27.876 | 28.458 | 0.355 | False | 0 |
| lever: feature-cache=on | 0.700 | 235.617 | 233.174 | 9.888 | 22.583 | 32.634 | 30.410 | 34.468 | 0.511 | False | 0 |
| lever: feature-cache=on | 0.850 | 286.107 | 283.006 | 11.480 | 28.961 | 39.035 | 36.410 | 42.692 | 0.604 | False | 0 |
| lever: candidates=200 | 0.300 | 100.979 | 100.774 | 7.558 | 17.263 | 26.745 | 26.303 | 27.413 | 0.281 | False | 0 |
| lever: candidates=200 | 0.500 | 168.298 | 165.011 | 8.023 | 17.263 | 28.110 | 27.595 | 28.862 | 0.366 | False | 0 |
| lever: candidates=200 | 0.700 | 235.617 | 233.192 | 8.863 | 19.843 | 29.839 | 28.687 | 30.668 | 0.476 | False | 0 |
| lever: candidates=200 | 0.850 | 286.107 | 283.112 | 9.888 | 24.699 | 34.988 | 33.305 | 36.320 | 0.582 | False | 0 |


`p99_repeat_lo` and `p99_repeat_hi` bound the p99 across 5 independent repeats (Student-t interval on log latency, so it is asymmetric and never below zero).


`dispatch_lag_p99_ms` is how late the load generator sent requests. Latency is timed from the intended send time, so the lateness is counted (as `generator_lag` in S1) instead of hidden, but it is the generator's delay, not the server's, and the late requests reach the server in a burst. A level is flagged `generator_behind` when the lag p99 exceeds 2 ms; its latencies overstate the server's and it is excluded from lever comparisons.

Automatic garbage collection was paused during each measured window, after a full collection, because the harness holds every trace until the window ends.


## Where the time goes at load 0.7 (baseline)

Critical-path milliseconds per stage, split into service and waiting time, for tail requests and for requests near the median. R is end-to-end time no span covers; an S1 row of kind `generator_lag` is time the load generator sent late.

| band | stage | stage_name | kind | mean_ms | share |
|---|---|---|---|---|---|
| above p99 | R | unattributed residual | residual | 0.090 | 0.002 |
| above p99 | S1 | admission and queueing | generator_lag | 0.118 | 0.003 |
| above p99 | S1 | admission and queueing | wait | 10.774 | 0.248 |
| above p99 | S2 | orchestration and request context | service | 0.128 | 0.003 |
| above p99 | S3 | candidate retrieval | service | 0.704 | 0.016 |
| above p99 | S3 | candidate retrieval | wait | 1.698 | 0.039 |
| above p99 | S4 | feature retrieval | service | 0.230 | 0.005 |
| above p99 | S4 | feature retrieval | wait | 25.651 | 0.591 |
| above p99 | S5 | model inference | service | 3.587 | 0.083 |
| above p99 | S6 | re-ranking | service | 0.374 | 0.009 |
| above p99 | S7 | response serialization | service | 0.069 | 0.002 |
| p40 to p60 | R | unattributed residual | residual | 0.090 | 0.008 |
| p40 to p60 | S1 | admission and queueing | generator_lag | 0.079 | 0.007 |
| p40 to p60 | S1 | admission and queueing | wait | 0.587 | 0.053 |
| p40 to p60 | S2 | orchestration and request context | service | 0.122 | 0.011 |
| p40 to p60 | S3 | candidate retrieval | service | 0.616 | 0.055 |
| p40 to p60 | S3 | candidate retrieval | wait | 1.849 | 0.166 |
| p40 to p60 | S4 | feature retrieval | service | 0.195 | 0.017 |
| p40 to p60 | S4 | feature retrieval | wait | 3.738 | 0.335 |
| p40 to p60 | S5 | model inference | service | 3.462 | 0.311 |
| p40 to p60 | S6 | re-ranking | service | 0.342 | 0.031 |
| p40 to p60 | S7 | response serialization | service | 0.068 | 0.006 |


### Spans on the critical path of tail requests (baseline, load 0.7)

| span | mean_critical_ms | share | critical_frequency |
|---|---|---|---|
| features.fetch.store | 25.464 | 0.586 | 1.000 |
| queue | 10.774 | 0.248 | 1.000 |
| light_ranker | 1.799 | 0.041 | 1.000 |
| heavy_ranker | 1.788 | 0.041 | 1.000 |
| two_tower.network | 1.625 | 0.037 | 1.000 |
| retrieval.merge | 0.397 | 0.009 | 1.000 |
| rerank | 0.374 | 0.009 | 1.000 |
| two_tower.search | 0.227 | 0.005 | 1.000 |
| features.fetch.pool_wait | 0.188 | 0.004 | 1.000 |
| context | 0.128 | 0.003 | 1.000 |
| queue.generator_lag | 0.118 | 0.003 | 1.000 |
| features | 0.104 | 0.002 | 1.000 |
| (residual) | 0.090 | 0.002 | 1.000 |
| features.assemble | 0.074 | 0.002 | 1.000 |
| two_tower.pool_wait | 0.073 | 0.002 | 1.000 |


`critical_frequency` is the share of tail requests in which the span was on the critical path. A parallel branch with a frequency near zero cannot move the tail.


## Levers: bound versus measured change

`bound_ms` is the baseline's mean tail critical-path time inside the spans the lever acts on: the most the lever could remove directly from a tail request's own critical path. Write it down before reading the measured change. Under load a lever that shortens a stage holding a worker also shortens the queue (S1) for the requests behind, so the measured gain can exceed the bound; `gain_exceeds_bound` marks rows where a detected reduction is larger than the bound.

`baseline_p99_ms` and `lever_p99_ms` are geometric means of the per-repeat p99, so they differ slightly from the pooled values in the latency table. `p99_ratio` is the lever's p99 divided by the baseline's, with a paired interval across repeats (repeat r of every configuration replays the same arrivals). `change_detected` is true when that interval excludes 1. `lever_hedge_rate` is the share of feature-store calls that sent a backup, which is the extra load hedging costs. Rows with `valid` false involve a level at which the generator fell behind.

| lever | load | spans | baseline_stage_mean_ms | lever_stage_mean_ms | bound_ms | baseline_p99_ms | lever_p99_ms | measured_p99_change_ms | p99_ratio | p99_ratio_lo | p99_ratio_hi | change_detected | gain_exceeds_bound | lever_hedge_rate | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lever: popularity=precomputed | 0.300 | popularity | 0.398 | 0.125 | 0.000 | 30.743 | 30.804 | 0.061 | 1.002 | 0.996 | 1.008 | False | False | 0.000 | True |
| lever: popularity=precomputed | 0.500 | popularity | 0.450 | 0.140 | 0.000 | 32.830 | 32.634 | -0.195 | 0.994 | 0.983 | 1.005 | False | False | 0.000 | True |
| lever: popularity=precomputed | 0.700 | popularity | 0.526 | 0.158 | 0.000 | 37.662 | 36.920 | -0.742 | 0.980 | 0.952 | 1.009 | False | False | 0.000 | True |
| lever: popularity=precomputed | 0.850 | popularity | 0.581 | 0.167 | 0.000 | 61.327 | 53.671 | -7.655 | 0.875 | 0.736 | 1.040 | False | False | 0.000 | True |
| lever: hedge-feature-fetch=on | 0.300 | features | 4.962 | 3.826 | 29.264 | 30.743 | 14.119 | -16.624 | 0.459 | 0.434 | 0.486 | True | False | 0.047 | True |
| lever: hedge-feature-fetch=on | 0.500 | features | 4.999 | 3.903 | 28.983 | 32.830 | 18.545 | -14.285 | 0.565 | 0.531 | 0.601 | True | False | 0.045 | True |
| lever: hedge-feature-fetch=on | 0.700 | features | 5.169 | 4.019 | 25.882 | 37.662 | 26.905 | -10.756 | 0.714 | 0.664 | 0.768 | True | False | 0.048 | True |
| lever: hedge-feature-fetch=on | 0.850 | features | 5.226 | 4.091 | 11.032 | 61.327 | 40.701 | -20.625 | 0.664 | 0.580 | 0.759 | True | True | 0.048 | True |
| lever: feature-cache=on | 0.300 | features | 4.962 | 3.837 | 29.264 | 30.743 | 27.013 | -3.730 | 0.879 | 0.862 | 0.896 | True | False | 0.000 | True |
| lever: feature-cache=on | 0.500 | features | 4.999 | 3.817 | 28.983 | 32.830 | 28.166 | -4.664 | 0.858 | 0.831 | 0.886 | True | False | 0.000 | True |
| lever: feature-cache=on | 0.700 | features | 5.169 | 3.956 | 25.882 | 37.662 | 32.376 | -5.286 | 0.860 | 0.828 | 0.892 | True | False | 0.000 | True |
| lever: feature-cache=on | 0.850 | features | 5.226 | 4.008 | 11.032 | 61.327 | 39.426 | -21.901 | 0.643 | 0.572 | 0.722 | True | True | 0.000 | True |
| lever: candidates=200 | 0.300 | features+light_ranker+retrieval | 8.678 | 6.873 | 33.056 | 30.743 | 26.852 | -3.891 | 0.873 | 0.851 | 0.896 | True | False | 0.000 | True |
| lever: candidates=200 | 0.500 | features+light_ranker+retrieval | 8.847 | 6.940 | 32.998 | 32.830 | 28.222 | -4.608 | 0.860 | 0.844 | 0.876 | True | False | 0.000 | True |
| lever: candidates=200 | 0.700 | features+light_ranker+retrieval | 9.199 | 7.113 | 30.083 | 37.662 | 29.661 | -8.001 | 0.788 | 0.759 | 0.817 | True | False | 0.000 | True |
| lever: candidates=200 | 0.850 | features+light_ranker+retrieval | 9.386 | 7.267 | 15.409 | 61.327 | 34.780 | -26.547 | 0.567 | 0.481 | 0.668 | True | True | 0.000 | True |


Full tables: `ex1b_load_curve.csv`, `ex1b_stage_shares.csv`, `ex1b_span_shares.csv`, `ex1b_levers.csv`.
Span-level traces (one Parquet file per configuration, load, and repeat) were written to the large-output folder, outside the code folder.

Stage key: S1 admission and queueing; S2 orchestration and request context; S3 candidate retrieval; S4 feature retrieval; S5 model inference; S6 re-ranking; S7 response serialization; R unattributed residual.
