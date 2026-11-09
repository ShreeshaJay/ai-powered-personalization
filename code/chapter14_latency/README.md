# Chapter 14: Latency Optimization

Companion code for Chapter 14 of *AI-Powered Personalization*. The chapter treats
latency as a distribution with a tail, and the code gives the reader four
measurements that make the tail visible, attributable, and fixable on a laptop
or a single cloud CPU machine. No GPU is needed.

## What each exercise demonstrates

### Exercise 1 Part A: queueing in virtual time (`scripts/ex1a_queueing_sim.py`)
A discrete-event simulator that runs in virtual time, so results do not depend on
the machine. It shows four things:

- **Coordinated omission.** A closed-loop load generator that waits for each reply
  stops sending during a server stall and records almost none of the delay. The
  same stall measured open loop, or corrected with `record_corrected`, shows the
  real tail.
- **Utilization and the tail.** M/M/1 simulation checked against the closed-form
  mean, p99 and Little's law, then repeated with bursty arrivals (a Markov-modulated
  Poisson process) and bimodal service times to show how much faster the tail grows
  than the mean.
- **Fan-out.** The chance that a request touching N backends hits at least one slow
  one, simulated and compared with 1 - (1 - q)^N.
- **Hedged requests.** Sending a backup after the p95 delay (and, for contrast, the
  p50 delay), with cancellation of the loser, at several utilizations, and the work
  it adds or saves (`backup_requests_per_request`, `backup_work_share`,
  `net_work_change`).

### Exercise 1 Part B: tail attribution in a real pipeline (`scripts/ex1b_pipeline_attribution.py`)
A seven-stage recommendation request runs in real time on a thread pool:
admission (S1), request context (S2), two-tower retrieval with FAISS plus a popularity
branch (S3), feature fetch from a simulated remote store (S4), an XGBoost light ranker
and the vendored Chapter 7 DCN-V2 heavy ranker (S5), MMR re-ranking (S6), and
serialization (S7). An open-loop Poisson load generator drives it at fractions of
its measured capacity. Every request is traced as a span tree, and the critical
path of each request is attributed to stage and to service time versus waiting
time. The report compares tail requests (above p99) with requests near the median,
then tests levers. Each lever's measured p99 change, with a paired interval across
repeats, is set beside the bound implied by its spans' share of the critical path in
tail requests, which shows why speeding up a branch that is off the critical path
does little. The bound covers only the lever's direct effect: under load, a faster
stage also shortens the admission queue (S1), so the measured gain can exceed it.
If the load generator falls behind its schedule at a load level, the report flags
that level.

### Exercise 2 Part A: approximate-search operating points (`scripts/ex2a_ann_operating_points.py`)
FAISS Flat, IVF-Flat, HNSW and IVF-PQ indexes are swept over `nprobe` or `efSearch` at
candidate pool sizes K. Each setting reports recall against exact search, next-item
hit rate (Coveo), single-query p50 and p99 with intervals, batched throughput, memory
and build time. Dominated settings are marked, and with `--recall-floor` the script
picks the fastest setting that meets the floor; exact search (Flat) is eligible, so a
floor that no approximate setting meets selects exact search.

### Exercise 2 Part B: ranker inference and parity (`scripts/ex2b_ranker_inference_parity.py`)
Trains the Chapter 7 DCN-V2 on Yambda (or synthetic data), with user and item
history features taken from the window before each row's own window so that no row
sees its own label. It then exports the model to ONNX,
rewrites `Gemm` nodes to `MatMul`, quantizes to int8, and benchmarks PyTorch eager
against ONNX Runtime at three graph-optimization levels over batch sizes and thread
counts. Every variant is then checked against the PyTorch reference for score parity,
rank parity (top-K overlap, Kendall's tau, rank-biased overlap), AUC change and
calibration shift, overall and on four slices (short and long history users, popular
and tail items). A variant ships only if its AUC, top-10 overlap and calibration
tolerances hold on every slice; the other parity measures are reported for diagnosis.

`scripts/run_all.py` runs all four exercises in sequence, each in its own process so
thread-pool settings do not leak between them.

## Setup

Use Python 3.11 or 3.12. Create the virtual environment **outside** any folder that a
sync client (OneDrive, Dropbox, iCloud) watches, because syncing thousands of small
files slows installs and can corrupt environments.

Windows (PowerShell):

```powershell
py -3.11 -m venv C:\dev\venvs\recsys-book-ch14
C:\dev\venvs\recsys-book-ch14\Scripts\python.exe -m pip install --upgrade pip
C:\dev\venvs\recsys-book-ch14\Scripts\python.exe -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.11 -m venv ~/venvs/recsys-book-ch14
~/venvs/recsys-book-ch14/bin/python -m pip install --upgrade pip
~/venvs/recsys-book-ch14/bin/python -m pip install -r requirements.txt
```

`requirements.txt` adds the PyTorch CPU wheel index, so Linux installs the small
CPU-only build of PyTorch. On macOS pip falls back to the standard wheels.

**Windows OpenMP note.** PyTorch and some FAISS builds ship different OpenMP
runtimes; loading both in one process aborts with `OMP: Error #15`. The pinned
`faiss-cpu` version avoids the clash. If you change versions and see the error, keep
the pins in `requirements.txt`.

All commands below run from this folder:

```bash
python -m unittest discover -s tests -t .     # 62 tests, about a minute
python -m scripts.run_all --quick             # smoke test of every exercise, a few minutes
```

## Data

Every script runs on synthetic data alone. Real data is optional and is read from
two environment variables. Dataset files are never copied into this folder.

| Variable | Dataset | Files read | Used by |
|---|---|---|---|
| `SIGIR_DATA_DIR` | Coveo SIGIR eCom 2021 data challenge | `sku_to_content.csv` and `browsing_train_sampled.csv` or `browsing_train.csv`, in `sampled/`, `train/` or the folder itself | ex1b, ex2a |
| `YAMBDA_DATA_DIR` | Yandex Yambda, flat format | `listens.parquet` | ex2b |

- Coveo: request access at https://github.com/coveooss/SIGIR-ecom-data-challenge.
  Only the first 3 million rows of the browsing file are read.
- Yambda: https://huggingface.co/datasets/yandex/yambda (the `flat/50m/listens.parquet`
  file is enough).

If a variable is unset, the scripts look for the book's `Dataset/` folder next to the
code folder, and otherwise fall back to synthetic data. `--catalog synthetic`
(ex1b), `--datasets synthetic` (ex2a), `--data synthetic` (ex2b) or
`run_all --data synthetic` force synthetic data even when real data is present.

## Outputs

Small summaries (CSV, Markdown and a `*_run_info.json` with arguments, package
versions, thread settings and the platform check) go to `summary/`, or
`summary/quick/` for `--quick` runs. Large files (span traces, trained model and ONNX
files) go to a per-user cache folder outside the code folder:

| OS | Default large-output folder |
|---|---|
| Windows | `%LOCALAPPDATA%\recsys_book\ch14` |
| macOS | `~/Library/Caches/recsys_book/ch14` |
| Linux | `$XDG_CACHE_HOME/recsys_book/ch14`, else `~/.cache/recsys_book/ch14` |

Set `RECSYS_CH14_OUT` to use another folder. Each run writes under `quick/` or
`full/` inside it. Exercise 1 Part B writes a new timestamped trace folder on every
run and never deletes old ones; remove them yourself when you no longer need them.

| Script | Summary files |
|---|---|
| ex1a | `ex1a_coordinated_omission.csv`, `ex1a_utilization.csv`, `ex1a_fanout.csv`, `ex1a_hedging.csv`, `ex1a_summary.md` |
| ex1b | `ex1b_load_curve.csv`, `ex1b_stage_shares.csv`, `ex1b_span_shares.csv`, `ex1b_levers.csv`, `ex1b_attribution.md` |
| ex2a | `ex2a_pareto.csv`, `ex2a_summary.md` |
| ex2b | `ex2b_latency.csv`, `ex2b_parity.csv`, `ex2b_summary.md` |

## Main flags

```bash
# Exercise 1 Part B: choose load levels and test levers one at a time
python -m scripts.ex1b_pipeline_attribution --load 0.3,0.5,0.7,0.85 \
    --lever popularity=precomputed --lever hedge-feature-fetch=on \
    --lever feature-cache=on --lever candidates=200 --max-dispatch-lag-ms 2

# Exercise 2 Part A: pick operating points that meet a recall floor
python -m scripts.ex2a_ann_operating_points --recall-floor 0.95 --ks 100,500,1000

# Exercise 2 Part B: tighten or loosen the shipping tolerances
python -m scripts.ex2b_ranker_inference_parity --max-auc-drop 0.002 \
    --min-top10-overlap 0.95 --max-abs-mean-shift 0.005
```

Levers accepted by `--lever`: `candidates=<int>`, `feature-cache=on|off`,
`heavy-batching=per-item|batched`, `mmr=naive|vectorized`,
`popularity=naive|precomputed`, `hedge-feature-fetch=on|off`.

**Warning about the slow variants.** `heavy-batching=per-item` and `mmr=naive` are
much slower than the batched and vectorized defaults. Every configuration is offered
the same request rate, set from the baseline's capacity, so a slow variant is driven
far past its own capacity. Its queue then grows for the whole run and the run takes
much longer than the nominal duration. Use them only with short `--duration` values.

## Reading the results honestly

- **Timer granularity.** `latency/platform_check.py` measures how far a 1 ms sleep
  overshoots. Exercise 1 Part B refuses to run if it overshoots by more than 2 ms
  (override with `--force`). Run it on its own with `python -m latency.platform_check`.
- **Dispatch lag.** The load generator runs in the same Python process as the server,
  so it can fall behind its schedule. Latency is still measured from the intended send
  time, so the lag is counted rather than hidden (it appears as `generator_lag` time
  in S1), but the late requests arrive in bursts, and the measured tail then mixes the
  server's behaviour with the generator's. A load level whose `dispatch_lag_p99_ms`
  exceeds `--max-dispatch-lag-ms` (default 2 ms) is marked `generator_behind`, and
  lever comparisons at that level are marked not `valid`.
- **Garbage collection.** The harness keeps every record and trace of a window in
  memory, and a full Python collection over them holds the interpreter lock for
  hundreds of milliseconds, stalling the generator and every stage at once. Measured
  windows therefore run with the collector paused (`gc.freeze()`, then
  `gc.disable()`, restored afterwards). `--gc-in-window` leaves it running, to see
  the effect; expect `generator_behind` flags at higher loads.
- **Repeat intervals.** The `*_repeat_lo` and `*_repeat_hi` columns are Student-t
  intervals across independent repeats, computed on log latency and transformed back,
  so the centre is the geometric mean and the interval never falls below zero. Lever
  and speed-up comparisons use paired intervals for the ratio of treatment to
  reference, repeat by repeat. With the two repeats of a `--quick` run the intervals
  are very wide; they are meaningful only in full runs.
- **Quiet machine.** Laptop numbers move with background load, power plans and
  thermal throttling. `CLOUD_RUN.md` describes the reference run on a dedicated
  cloud VM.

## Code structure

```
chapter14_latency/
├── config.py                     # seed, output folders, dataset locations
├── requirements.txt
├── latency/
│   ├── histogram.py              # mergeable log-bucketed latency histogram
│   ├── timing.py                 # spans, traces, critical path, intervals
│   ├── loadgen.py                # open-loop and closed-loop load generators
│   ├── platform_check.py         # timer granularity, CPU and thread settings
│   └── report.py                 # CSV, Markdown and run-info writers
├── sim/queue_sim.py              # virtual-time queueing, fan-out and hedging
├── ann/
│   ├── data.py                   # Coveo vectors and sessions, synthetic catalog
│   └── sweep.py                  # FAISS index builds, sweeps, Pareto front
├── pipeline/
│   ├── build_artifacts.py        # index, features, light and heavy rankers
│   └── stages.py                 # the seven-stage request and its levers
├── ranker/
│   ├── model_dcn_vendored.py     # Chapter 7 DCN-V2, vendored with source hash
│   ├── data.py                   # Yambda and synthetic ranking data
│   ├── train.py                  # DCN-V2 training
│   ├── export.py                 # ONNX export, Gemm rewrite, int8 quantization
│   ├── bench.py                  # inference benchmarks
│   └── parity.py                 # score, rank, quality and slice parity
├── scripts/                      # ex1a, ex1b, ex2a, ex2b, run_all
├── tests/                        # python -m unittest discover -s tests -t .
├── summary/                      # small result files
├── CLOUD_RUN.md                  # reference run on one cloud VM
└── DESIGN_NOTES.md               # design decisions and results log
```

## References

- Coveo SIGIR eCom 2021 data challenge: https://github.com/coveooss/SIGIR-ecom-data-challenge
- Yandex Yambda: https://huggingface.co/datasets/yandex/yambda
- FAISS: https://github.com/facebookresearch/faiss
- ONNX Runtime: https://onnxruntime.ai
- HdrHistogram: https://hdrhistogram.org
