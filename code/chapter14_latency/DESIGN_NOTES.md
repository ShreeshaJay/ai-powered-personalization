# Chapter 14 design notes

Design decisions behind the companion code, departures from the chapter plan, and
the results log. Numbers belong here and in `summary/`, never in chapter prose.

## Measurement design

**Percentiles come from a mergeable histogram.** `latency/histogram.py` stores
counts in logarithmic buckets with a fixed relative precision (1% by default), in
the style of HdrHistogram. Histograms from different threads, repeats or machines
can be added together, which is impossible with pre-computed percentiles: the p99
of two runs is not the average of their p99s. The percentile is the nearest-rank
bucket's geometric midpoint, clamped to the observed minimum and maximum.

**Latency is measured from the intended send time.** The open-loop generator
(`latency/loadgen.py`) fixes the arrival schedule before the run and measures each
request from when it should have been sent. A request that waits because every
worker is busy therefore counts its waiting time, which is the correction for
coordinated omission. The generator shares a Python process (and the GIL) with the
server, so it can itself fall behind. Its lateness is still inside each request's
latency, recorded as a separate `generator_lag` span in S1 so that it is not
confused with server queueing. The late requests then reach the server in bursts,
which a real client population would not produce, so a load level whose p99
dispatch lag exceeds `--max-dispatch-lag-ms` (default 2 ms) is flagged
`generator_behind` and its lever comparisons are marked not valid. The original
wording ("understates latency") had the direction wrong: lag inflates the measured
latency rather than hiding it.

**Garbage collection is paused inside measured windows.** Exercise 1 Part B keeps
every request record and span tree of a window in memory until the window ends, so
Python's generational collector runs full collections over a growing heap, and each
one holds the GIL. On the first cloud run (collector on) the baseline's p99 dispatch
lag was 0.35, 37, 135 and 201 ms at 0.3, 0.5, 0.7 and 0.85 of capacity, 14 of the 20
configuration and load cells were flagged, and 14 of the 16 lever comparisons were not
valid. On the laptop with default settings,
single full collections took 0.56 s and 1.6 s; with the collector paused the lag p99
was 0.8 to 1.7 ms at the same loads. A pause of that kind stalls the generator and
every stage together and shows up as a tail that belongs to the harness, not to the
pipeline. `latency/loadgen.py` therefore runs every measured window inside
`gc_paused()`: one collection, `gc.freeze()` to move the surviving objects out of the
collector's view, then `gc.disable()`, all undone when the window ends. Reference
counting still frees most objects during the window. `--gc-in-window` leaves the
collector on. On the reference cloud run, with the collector paused, the dispatch-lag
p99 stayed below 0.7 ms for every configuration and load.

**Attribution follows the critical path.** Every request is a span tree. The
critical path walks backward from the end of the root span, always following the
child that finished last; time no child covers is the parent's own time. Each
critical-path segment is labelled with a stage (S1 to S7) and a kind (service or
wait), and root time no span covers is reported as the residual R so that the
shares always add up to the end-to-end latency. Tail requests (above p99) are
compared with requests between p40 and p60, because a stage that dominates the
median is often not the one that dominates the tail.

**Lever bounds.** For each lever the report gives a bound on the p99 gain from its
direct effect: the mean critical-path time, in baseline tail requests, of the spans
the lever touches. A lever that changes an off-critical-path branch has a bound near
zero however much faster it makes that branch. The popularity lever
(`popularity=naive|precomputed`) was added for exactly this lesson. The bound is not
an upper bound on the measured change under load: a lever that shortens service time
also lowers utilization, which shortens the S1 admission queue, so the measured gain
can exceed the bound. The lever table therefore reports `gain_exceeds_bound` instead
of treating such a result as an error, and `change_detected` when the paired ratio
interval excludes 1.

**Intervals.** Bootstrap intervals resample requests within pooled runs. Across
independent repeats (Georges et al., 2007), which capture run-to-run variation and
are usually the larger effect, the code uses a Student-t interval on log latency,
transformed back: the centre is the geometric mean and the bounds are always
positive. The builder's first version used a raw-scale Student-t interval left
unclamped below zero; a negative latency bound is impossible, and latency across
repeats varies multiplicatively, so the log scale is the better model. A bootstrap
across two to five repeats was rejected because it cannot produce an interval wider
than the observed range. Lever and speed-up comparisons use paired ratio intervals
(repeat r of the treatment over repeat r of the reference, again a t-interval on the
logs), which removes the run-to-run variation the two share. With the two repeats of
`--quick` mode every interval is very wide, which honestly shows how little two
repeats say.

**Platform check.** Every run records timer granularity, CPU model, the Linux
frequency governor where readable, thread environment variables and package
versions in `*_run_info.json`. Exercise 1 Part B refuses to run if a 1 ms sleep
overshoots by more than 2 ms. On the Windows development laptop `time.sleep`
overshoots by well under a millisecond, while `asyncio.sleep` overshoots by over
ten, which is why the load generator uses threads and a spin-wait instead of
asyncio.

## Exercise-specific decisions

**Exercise 1 Part A runs in virtual time.** The simulator is an event loop over a
heap of timestamps, so its results are exact for the model and identical on every
machine. M/M/1 results are checked against the closed-form mean and percentiles and
against Little's law in the unit tests.

**Hedging can reduce total work.** The hedging experiment cancels the losing copy
when the other finishes. When the primary has drawn a slow service time, the backup
often finishes first and the slow primary is cancelled, so the work saved can exceed
the backup's own work. A single "added load" figure could then be negative, which
reads as an error. The report therefore splits it into `backup_requests_per_request`
(how many backups were sent), `backup_work_share` (the share of server time spent on
backups) and `net_work_change` (total server time against the unhedged run, which can
be negative). Work can fall only because the model ties slowness to the copy (each
copy draws its own service time); if slowness belonged to the request, every backup
would add work. The experiment runs at several utilizations because hedging's value
depends on spare capacity.

**Exercise 1 Part B baseline uses the batched heavy ranker and vectorized MMR.**
With per-item heavy-ranker calls and a Python-loop MMR, the request took over a
hundred milliseconds of pure compute, capacity fell to a few requests per second,
and every other effect was invisible. Both slow variants remain available as levers
(`heavy-batching=per-item`, `mmr=naive`) but are not in the default lever set: all
configurations face the baseline's offered rate, so a much slower variant is driven
far past its capacity and its queue grows for the whole run.

**Simulated remote calls.** Retrieval network time and the feature store are
simulated with log-normal delays plus a small probability of an eight-times slower
call, in sleeping I/O threads. This gives a controllable heavy tail without a real
network, and the hedging lever acts on the same kind of tail that Exercise 1 Part A
models.

**Threads.** PyTorch, its inter-op pool and FAISS are set to one thread inside the
pipeline so the 4 server workers do not oversubscribe the CPU. On the 16-thread
development laptop, letting each library use every thread made inference several
times slower than one thread.

**Exercise 2 Part A uses inner product.** Coveo description vectors and the synthetic
vectors are L2-normalized, so inner product equals cosine similarity. The memory
column comes from serializing each index. HNSW `efSearch` values below K are
skipped, because HNSW cannot return more results than its search beam holds. Exact
search (Flat) is eligible for selection: when no approximate setting meets the
recall floor, the honest answer is exact search, not "no operating point". The
summary also lists memory and build time per index, because a compressed index can
be dominated on latency and recall yet still be the only one that fits in memory.

**Exercise 2 Part B export.** The model is exported with the TorchScript-based
exporter (`dynamo=False`, opset 17) and a dynamic batch axis, and the file is
checked with `onnx.checker`. Linear layers export as `Gemm`. `ranker/export.py`
rewrites each `Gemm` with a constant weight as `MatMul` plus `Add` before
quantizing, so every dense layer reaches the quantizer as a `MatMul`. The builder's
reason (that the dynamic quantizer skips `Gemm`) does not hold for the pinned ONNX
Runtime 1.22, which quantizes `Gemm` as well; the rewrite is kept because it has
identical arithmetic and makes the int8 model independent of that behaviour, and a
unit test checks that every linear layer ends up as an integer matmul. The fp32 ONNX
variants run the rewritten graph too. Graph optimizations are applied when the
session is created, on the machine that runs it, because saved optimized graphs are
tied to the hardware they were optimized on. The unit tests check that the rewritten graph matches PyTorch.
Quantization prints a "consider pre-processing" warning, which is harmless here.

**Parity before speed.** A variant is reported as shippable only if it meets every
tolerance on every slice. Defaults (`Tolerances` in `ranker/parity.py`): AUC drop at
most 0.002, top-10 overlap at least 0.95, absolute mean-prediction shift at most
0.005. These are teaching defaults and can be changed on the command line. Top-10
overlap is computed only over pseudo-requests with more than 10 rows, because for a
group of 10 or fewer the top 10 is the whole group and the overlap is 1 whatever the
order. The parity table shows each slice's observed positive rate beside the
reference's mean prediction, so calibration is compared with the data and not only
between variants. A speed-up counts only when its paired interval across repeats
lies wholly above 1.

## Departures from the chapter plan

- **Environment.** A dedicated virtual environment outside OneDrive
  (`C:\dev\venvs\recsys-book-ch14`, Python 3.11) instead of the shared `recsys_venv`,
  on the supervisor's instruction.
- **FAISS version.** `faiss-cpu` is pinned to a release whose Windows wheel uses the
  same OpenMP runtime as PyTorch. The earlier version aborted with `OMP: Error #15`
  when both were loaded.
- **Extra flags.** `--lever` (repeatable) and `--catalog` on Exercise 1 Part B, the
  `--popularity` lever, `--data auto` on Exercise 2 Part B, and `--train-threads`.
  All plan entry-point names and flags are unchanged.
- **Baseline defaults.** Batched heavy ranker and vectorized MMR, as explained above.
- **Extra module.** `ranker/data.py` holds the Yambda and synthetic ranking loaders.
- **Quick summaries.** `--quick` runs write to `summary/quick/` so smoke tests never
  overwrite full results.
- **Trace layout.** One Parquet file per configuration, load level and repeat, under a
  timestamped folder, instead of one in-memory trace table.
- **Yambda features and time units.** Ten dense features, a subset of Chapter 7's.
  The split follows Chapter 7 (last day as test with a 30-minute gap, label
  `played_ratio_pct >= 50`) with two corrections to Chapter 7's code:
  - Timestamps are read as seconds, not as 5-second units. Every value in the 50M
    file is a multiple of 5 and the largest is 26,000,000, which is 301 days in
    seconds (the collection period) but 4.1 years in 5-second units. Under the
    5-second reading the "last day" was the last 4.8 hours, the 30-minute gap was 6
    minutes, and hour of day and day of week cycled every 4.8 hours and 1.4 days.
  - History aggregates are point-in-time. Aggregates over the training window include
    each training row's own label, so each row's user and item features come instead
    from the `train_days` window that ends where its own window starts. Training rows
    therefore see history up to `train_days` old and test rows at most a day and 30
    minutes old, like a batch feature job refreshed once per window.

  On the first cloud run, with both defects, the reference DCN's test AUC was 0.478,
  below chance. In a laptop prototype with the same 7 training days, caps and 2
  epochs, reading seconds alone gave 0.665, and seconds with point-in-time features
  0.693; the reference run gave 0.692. The leak showed in the single-feature AUCs:
  item completion rate scored 0.746 on training rows and 0.578 on test rows with the
  leak, 0.597 and 0.578 without it.

## Vendored model

`ranker/model_dcn_vendored.py` is a copy of
`code/chapter7_advanced_ranking/models/dcn.py` from the public repository, below a
header that records the source path, its SHA-256 and the source commit. The copy is
byte-identical to the file as stored in git. The hash is computed with LF line
endings, because a Windows checkout with `core.autocrlf` has CRLF endings and a
Linux or macOS checkout has LF; the first version recorded the CRLF hash, so the
test would have failed for every reader not on Windows.
`tests/test_pipeline_and_export.py` normalizes line endings, recomputes the hash of
everything below the header and fails if the copy drifts.

The private working copy of Chapter 7 has a fix that the public file lacks:
after Xavier initialization it sets the embedding rows for padding index 0 back to
zero. Chapter 14 follows the public file, so unseen Yambda IDs (mapped to 0) get a
small random embedding that training never updates. This affects the reference
model's accuracy slightly, not latency or parity, because every variant is exported
from the same trained weights.

## Results log

From the reference cloud run described in `CLOUD_RUN.md`. Local laptop runs are
smoke tests only and are not recorded here.

The reference run is `ch14-20261109-0019`. An earlier run on the same machine type and
CPU (`ch14-20261108-2307`) is superseded: its Exercise 1 Part B was stalled by garbage
collection and its Exercise 2 Part B used the defective Yambda split (both described
above). Its Exercise 1 Part A results were identical, as virtual time guarantees, and
its Exercise 2 Part A selected the same operating points.

### Run details

| Field | Value |
|---|---|
| Date | 2026-11-09 (UTC), 08:20 to 09:05 |
| Machine type and CPU model | Google Cloud `c3-standard-8`, on demand, us-central1-a (8 vCPU = 4 cores x 2 threads, 32 GB); Intel Xeon Platinum 8481C @ 2.70 GHz (Sapphire Rapids); Debian 12, kernel 6.1.0-54-cloud-amd64; Python 3.11.2 |
| Data (Coveo, Yambda or synthetic) | Coveo sampled files (`sku_to_content.csv`, first 3 million rows of `browsing_train_sampled.csv`); Yambda flat 50M `listens.parquet` (SHA-256 `eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1`); 1-million-vector synthetic catalog in Exercise 2 Part A |
| `run_all` total time | 2,437 s (40.6 min): ex1a 42 s, ex1b 1,626 s, ex2a 685 s, ex2b 85 s. Before it: setup 1.7 min, unit tests (62) 8 s, `run_all --quick` 62 s |
| Exit codes (`run_all_status.json`) | 0 for all four exercises |
| 1 ms sleep overshoot (platform check) | median 0.064 ms, p99 0.075 ms |

The four exercises ran as `python -m scripts.run_all --only <exercise> --recall-floor 0.95`,
one after another, which passes each script exactly the arguments `run_all` uses.

Span traces from Exercise 1 Part B are kept in the project bucket, not in the repository:

- `gs://recsys-book-runs-0843377880/ch14-20261109-0019/traces/20261109_082451/` (reference
  run; 100 Parquet files, one per configuration, load and repeat; 84 MB)
- `gs://recsys-book-runs-0843377880/ch14-20261108-2307/traces/20261109_071452/` (superseded
  run with the collector on; 81 MB; kept for the garbage-collection comparison)

### Exercise 1 Part A

| Finding | Value |
|---|---|
| Stall: closed-loop recorded p99 vs open-loop p99 | Recorded closed loop p99 10.1 ms, p99.9 15.2 ms; open loop p99 27.6 ms, p99.9 915 ms; corrected closed loop p99.9 897 ms. Samples above 100 ms: 2 recorded, 90 corrected, 112 open loop (of about 12,000) |
| M/M/1 at rho 0.9: simulated vs theory p99 | p99 48.0 against 46.1 (mean 9.68 against 10.0), in mean service times. Up to rho 0.85 the simulated mean is within 1.2% of the formula and the p99 within 3.5% |
| Bursty or bimodal p99 vs Poisson at the same utilization | At rho 0.7: Poisson with exponential service 15.3; bimodal service 81.6 (5.3 times); bursty arrivals 254 (17 times) |
| Fan-out width at which P(at least one slow) exceeds 50% | Between 50 leaves (0.395) and 100 (0.632; formula 0.634) for leaves slow 1 time in 100; the formula crosses 50% at 69 |
| Hedging p99 change and `net_work_change` | Hedge after the unhedged p95, at utilization 0.2 / 0.4 / 0.6 / 0.8: p99 cut by 82% / 66% / 52% / 40%; backups on 2.1% / 1.0% / 0.9% / 0.9% of requests; net server work -11.2% / -5.9% / -2.5% / -0.8%. Hedge after the p50: backups on 47% to 17% of requests, p99 cut by 91% to 94%, net work -12% to -14% |

### Exercise 1 Part B

Coveo catalog, 4 workers, Poisson arrivals, 5 repeats of 15 s per load level.

| Finding | Value |
|---|---|
| Baseline capacity (requests/s) | 336.6 (closed loop) |
| Baseline p99 at each load level | 30.7 / 32.6 / 37.9 / 65.5 ms at 0.3 / 0.5 / 0.7 / 0.85 of capacity (repeat intervals 30.3-31.2, 32.0-33.7, 36.4-38.9, 50.8-74.1) |
| Largest critical-path stage in tail requests at the attribution load | Load 0.7, above p99: S4 feature-store waiting 25.7 ms (59%), then S1 admission queueing 10.8 ms (25%) and S5 inference 3.6 ms (8%) |
| Same for median requests | p40 to p60: S4 waiting 3.7 ms (34%), S5 inference 3.5 ms (31%), S3 retrieval waiting 1.8 ms (17%), S1 queueing 0.6 ms (5%) |
| Each lever: bound vs measured p99 change | See below |
| Maximum dispatch lag | p99 0.68 ms; no level flagged. With the collector on (superseded run), up to about 290 ms |

Levers, at 0.3 / 0.5 / 0.7 / 0.85 of capacity (measured change is the geometric mean of
per-repeat p99s; ratio is lever over baseline with a paired 95% interval):

- `popularity=precomputed`: its stage ran about three times faster, but the bound is 0 at
  every load because the popularity branch never finished last in a tail request. No change
  detected at any load (at 0.85 the ratio is 0.875, interval 0.736 to 1.040).
- `hedge-feature-fetch=on`: bound 29.3 / 29.0 / 25.9 / 11.0 ms; measured -16.6 / -14.3 /
  -10.8 / -20.6 ms (ratios 0.46 / 0.57 / 0.71 / 0.66); backups on 4.5% to 4.8% of store calls.
- `feature-cache=on`: same bounds; measured -3.7 / -4.7 / -5.3 / -21.9 ms (ratios 0.88 / 0.86
  / 0.86 / 0.64).
- `candidates=200`: bound 33.1 / 33.0 / 30.1 / 15.4 ms; measured -3.9 / -4.6 / -8.0 / -26.5 ms
  (ratios 0.87 / 0.86 / 0.79 / 0.57).

At 0.85 the hedge, cache and candidate levers each removed more than their bound
(`gain_exceeds_bound`), because at that load queueing in S1 is a large part of the tail
and every faster request shortens the queue for the requests behind it. At the lower
loads every measured gain is inside its bound.

### Exercise 2 Part A

| Finding | Value |
|---|---|
| Selected operating point per dataset and K | Coveo: IVF-Flat `nprobe` 16 (K = 100), 32 (K = 500), 64 (K = 1,000). Synthetic: IVF-Flat `nprobe` 4 (K = 100), 8 (K = 500); IVF-PQ `nprobe` 8 (K = 1,000) |
| Recall and p99 at that point | Coveo 0.984 at 0.053 ms, 0.978 at 0.150 ms, 0.993 at 0.295 ms. Synthetic 0.957 at 0.086 ms, 1.000 at 0.183 ms, 0.985 at 0.192 ms. Exact search p99: Coveo 0.26 to 0.54 ms, synthetic 23.9 to 24.0 ms |
| Memory and build time of the selected index | Coveo IVF-Flat 6.4 MB, 0.75 s. Synthetic IVF-Flat 253 MB, 70 s; IVF-PQ 24 MB, 73 s (HNSW 504 MB, 17 s) |

HNSW was never non-dominated. IVF-PQ's recall stays below 0.54 at K = 100 but reaches
0.985 at K = 1,000, where its frontier crosses IVF-Flat's. On Coveo at K = 100 the
next-item hit rate is 0.464 at `nprobe` 8 (recall 0.94), equal to exact search's 0.463,
and still 0.447 at `nprobe` 4 (recall 0.83): hit rate saturates well before recall.

### Exercise 2 Part B

| Finding | Value |
|---|---|
| Test AUC of the PyTorch reference | 0.692 (1,456,150 training rows over 7 days, 2 epochs, 18 s of training; 232,049 test rows). Mean prediction 0.630 against an observed positive rate of 0.680 overall; 0.613 against 0.708 for long-history users |
| Fastest variant at batch 1 and at batch 1,024, with speed-up | Batch 1: ONNX Runtime fp32 (basic level), p50 0.025 ms, 8.2 times PyTorch at 1 thread and 10.1 times at 8. Batch 1,024: PyTorch eager at 8 threads, p50 1.23 ms; ONNX Runtime fp32 runs at 0.41 to 0.67 times PyTorch's speed there. ONNX Runtime is faster up to batch 128 and slower from 256 at both thread counts, and gains little from 8 threads |
| Variants that failed parity, and which check and slice | `ort_int8_all`: top-10 overlap 0.87 to 0.91 on all five slices (floor 0.95); its AUC change (-0.0008) and mean shift (-0.0009) are within tolerance. The three fp32 ONNX variants match PyTorch exactly |
| Variant recommended for serving | ONNX Runtime fp32 for batches up to 128 (the three optimization levels are indistinguishable for this model); PyTorch eager with 8 threads for batches of 256 and more on this machine. int8 is rejected on rank parity, although it was faster than fp32 ONNX Runtime from batch 16 up at 1 thread |

## References

Georges, A., Buytaert, D., and Eeckhout, L. (2007). Statistically rigorous Java
performance evaluation. *Proceedings of OOPSLA 2007*.
