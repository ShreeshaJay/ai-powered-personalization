# Exercise 2 Part A: approximate-search operating points

Single-query latency is measured at one thread; throughput uses all threads. Recall is against exact inner-product search. Hit rate is next-item hit rate on Coveo sessions. `p99_repeat_lo` and `p99_repeat_hi` are a Student-t interval on log latency across 5 repeats.

Datasets: coveo: 31,950 items, 50 dims, 2,000 evaluation queries, 90 s; synthetic: 1,000,000 items, 64 dims, 2,000 evaluation queries, 593 s

Recall floor: 0.95

## Non-dominated settings

| dataset | index | param | value | K | recall_at_K | hit_rate_at_K | p50_ms | p99_ms | p99_repeat_lo | p99_repeat_hi | batched_qps_all_threads | memory_mb | build_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coveo | flat |  |  | 100 | 1.000 | 0.463 | 0.238 | 0.255 | 0.249 | 0.258 | 7200.883 | 6.094 | 0.001 |
| coveo | flat |  |  | 500 | 1.000 | 0.625 | 0.360 | 0.387 | 0.384 | 0.389 | 5155.284 | 6.094 | 0.001 |
| coveo | flat |  |  | 1000 | 1.000 | 0.689 | 0.498 | 0.537 | 0.528 | 0.540 | 3678.384 | 6.094 | 0.001 |
| coveo | ivf_flat | nprobe | 1 | 100 | 0.434 | 0.375 | 0.014 | 0.024 | 0.020 | 0.025 | 398182.932 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 2 | 100 | 0.644 | 0.436 | 0.019 | 0.027 | 0.025 | 0.029 | 342717.334 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 4 | 100 | 0.826 | 0.447 | 0.024 | 0.033 | 0.031 | 0.034 | 280658.622 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 8 | 100 | 0.940 | 0.464 | 0.029 | 0.039 | 0.037 | 0.040 | 233665.688 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 16 | 100 | 0.984 | 0.461 | 0.035 | 0.053 | 0.052 | 0.053 | 186185.709 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 32 | 100 | 0.997 | 0.463 | 0.046 | 0.081 | 0.080 | 0.081 | 134158.651 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 64 | 100 | 0.999 | 0.463 | 0.069 | 0.144 | 0.141 | 0.143 | 86512.068 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 128 | 100 | 0.999 | 0.463 | 0.103 | 0.248 | 0.243 | 0.248 | 53166.780 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 1 | 500 | 0.142 | 0.379 | 0.030 | 0.048 | 0.047 | 0.049 | 190742.776 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 2 | 500 | 0.263 | 0.464 | 0.040 | 0.060 | 0.058 | 0.061 | 149382.245 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 4 | 500 | 0.449 | 0.531 | 0.055 | 0.078 | 0.077 | 0.079 | 110198.703 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 8 | 500 | 0.697 | 0.603 | 0.078 | 0.095 | 0.093 | 0.097 | 79153.410 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 16 | 500 | 0.891 | 0.617 | 0.100 | 0.119 | 0.117 | 0.120 | 61301.218 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 32 | 500 | 0.978 | 0.623 | 0.120 | 0.150 | 0.148 | 0.151 | 51575.009 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 64 | 500 | 0.998 | 0.624 | 0.144 | 0.214 | 0.211 | 0.214 | 42504.596 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 128 | 500 | 1.000 | 0.625 | 0.178 | 0.315 | 0.310 | 0.321 | 32633.091 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 1 | 1000 | 0.076 | 0.379 | 0.046 | 0.068 | 0.067 | 0.069 | 116730.464 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 2 | 1000 | 0.144 | 0.464 | 0.058 | 0.085 | 0.079 | 0.085 | 95662.404 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 4 | 1000 | 0.264 | 0.531 | 0.077 | 0.110 | 0.108 | 0.113 | 73747.452 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 8 | 1000 | 0.469 | 0.610 | 0.109 | 0.137 | 0.136 | 0.137 | 53712.858 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 16 | 1000 | 0.734 | 0.663 | 0.156 | 0.181 | 0.178 | 0.183 | 38181.806 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 32 | 1000 | 0.931 | 0.681 | 0.199 | 0.238 | 0.232 | 0.240 | 29745.438 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 64 | 1000 | 0.993 | 0.686 | 0.232 | 0.295 | 0.285 | 0.297 | 25680.000 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 128 | 1000 | 0.999 | 0.689 | 0.267 | 0.395 | 0.392 | 0.397 | 21757.633 | 6.439 | 0.753 |
| synthetic | flat |  |  | 100 | 1.000 |  | 22.393 | 23.902 | 23.554 | 24.053 | 208.621 | 244.141 | 0.067 |
| synthetic | flat |  |  | 500 | 1.000 |  | 22.508 | 24.017 | 23.876 | 24.101 | 205.827 | 244.141 | 0.067 |
| synthetic | ivf_flat | nprobe | 2 | 100 | 0.738 |  | 0.053 | 0.071 | 0.065 | 0.076 | 93527.744 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 4 | 100 | 0.957 |  | 0.064 | 0.086 | 0.081 | 0.092 | 79114.976 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 8 | 100 | 1.000 |  | 0.086 | 0.119 | 0.115 | 0.120 | 59394.161 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 16 | 100 | 1.000 |  | 0.139 | 0.190 | 0.176 | 0.204 | 37584.644 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 1 | 500 | 0.430 |  | 0.076 | 0.124 | 0.119 | 0.129 | 63015.504 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 2 | 500 | 0.706 |  | 0.101 | 0.132 | 0.128 | 0.132 | 52449.473 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 4 | 500 | 0.947 |  | 0.125 | 0.150 | 0.147 | 0.151 | 44347.127 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 8 | 500 | 1.000 |  | 0.147 | 0.183 | 0.179 | 0.184 | 37976.413 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 16 | 500 | 1.000 |  | 0.200 | 0.249 | 0.243 | 0.251 | 28435.589 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 1 | 1000 | 0.400 |  | 0.102 | 0.164 | 0.161 | 0.165 | 47386.886 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 2 | 1000 | 0.673 |  | 0.133 | 0.173 | 0.169 | 0.176 | 38596.843 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 8 | 1000 | 0.993 |  | 0.191 | 0.226 | 0.218 | 0.230 | 28740.192 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 16 | 1000 | 0.998 |  | 0.244 | 0.293 | 0.290 | 0.293 | 23530.926 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 32 | 1000 | 0.999 |  | 0.350 | 0.421 | 0.409 | 0.426 | 16309.256 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 64 | 1000 | 1.000 |  | 0.543 | 0.645 | 0.632 | 0.651 | 10008.924 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 128 | 1000 | 1.000 |  | 0.918 | 1.037 | 1.018 | 1.044 | 5687.920 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 256 | 1000 | 1.000 |  | 1.672 | 1.885 | 1.837 | 1.883 | 3009.779 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 512 | 1000 | 1.000 |  | 3.179 | 3.567 | 3.537 | 3.568 | 1546.159 | 252.801 | 70.406 |
| synthetic | ivf_pq | nprobe | 1 | 100 | 0.361 |  | 0.043 | 0.056 | 0.055 | 0.056 | 114752.738 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 2 | 100 | 0.472 |  | 0.049 | 0.062 | 0.059 | 0.064 | 105926.744 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 4 | 100 | 0.530 |  | 0.056 | 0.067 | 0.065 | 0.070 | 95883.735 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 1 | 500 | 0.411 |  | 0.079 | 0.121 | 0.117 | 0.123 | 65738.351 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 2 | 500 | 0.636 |  | 0.103 | 0.127 | 0.124 | 0.128 | 54363.846 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 4 | 500 | 0.770 |  | 0.121 | 0.132 | 0.131 | 0.133 | 47137.437 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 8 | 500 | 0.788 |  | 0.129 | 0.143 | 0.140 | 0.145 | 43217.867 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 1 | 1000 | 0.400 |  | 0.103 | 0.164 | 0.161 | 0.166 | 49480.706 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 4 | 1000 | 0.923 |  | 0.167 | 0.183 | 0.179 | 0.187 | 34308.990 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 8 | 1000 | 0.985 |  | 0.177 | 0.192 | 0.191 | 0.193 | 31483.942 | 23.982 | 73.070 |
| synthetic | ivf_pq | nprobe | 16 | 1000 | 0.987 |  | 0.196 | 0.213 | 0.211 | 0.214 | 28539.298 | 23.982 | 73.070 |


## Selected operating points (lowest p99 meeting the recall floor)

Exact search (`flat`) is eligible; when it is selected, no approximate index is needed at that K.

| dataset | index | param | value | K | recall_at_K | hit_rate_at_K | p50_ms | p99_ms | p99_repeat_lo | p99_repeat_hi | batched_qps_all_threads | memory_mb | build_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coveo | ivf_flat | nprobe | 16 | 100 | 0.984 | 0.461 | 0.035 | 0.053 | 0.052 | 0.053 | 186185.709 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 32 | 500 | 0.978 | 0.623 | 0.120 | 0.150 | 0.148 | 0.151 | 51575.009 | 6.439 | 0.753 |
| coveo | ivf_flat | nprobe | 64 | 1000 | 0.993 | 0.686 | 0.232 | 0.295 | 0.285 | 0.297 | 25680.000 | 6.439 | 0.753 |
| synthetic | ivf_flat | nprobe | 4 | 100 | 0.957 |  | 0.064 | 0.086 | 0.081 | 0.092 | 79114.976 | 252.801 | 70.406 |
| synthetic | ivf_flat | nprobe | 8 | 500 | 1.000 |  | 0.147 | 0.183 | 0.179 | 0.184 | 37976.413 | 252.801 | 70.406 |
| synthetic | ivf_pq | nprobe | 8 | 1000 | 0.985 |  | 0.177 | 0.192 | 0.191 | 0.193 | 31483.942 | 23.982 | 73.070 |


## Memory and build time per index (K = 100)

Serialized index size, build time, and the recall and best p99 reached across the sweep. A compressed index can be dominated on latency and recall and still be the choice when memory is the constraint.

| dataset | index | memory_mb | build_s | min_recall | max_recall | min_p99_ms |
|---|---|---|---|---|---|---|
| coveo | flat | 6.094 | 0.001 | 1.000 | 1.000 | 0.255 |
| coveo | ivf_flat | 6.439 | 0.753 | 0.434 | 0.999 | 0.024 |
| coveo | hnsw_flat | 14.387 | 0.228 | 0.981 | 0.996 | 0.090 |
| coveo | ivf_pq | 0.699 | 1.171 | 0.429 | 0.788 | 0.034 |
| synthetic | flat | 244.141 | 0.067 | 1.000 | 1.000 | 23.902 |
| synthetic | ivf_flat | 252.801 | 70.406 | 0.467 | 1.000 | 0.063 |
| synthetic | hnsw_flat | 503.664 | 17.114 | 0.999 | 0.999 | 0.195 |
| synthetic | ivf_pq | 23.982 | 73.070 | 0.361 | 0.539 | 0.056 |


All settings, including dominated ones, are in `ex2a_pareto.csv`.
