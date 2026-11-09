# Chapter 14 reference run on one Google Cloud VM

This is the single run that produces the numbers for the chapter's results log.
Everything happens on one CPU VM; no GPU is needed. The run uses paid compute, so
it needs the author's approval before the VM is created.

## Summary

| Item | Recommendation |
|---|---|
| Machine | `c3-standard-8` (compute-optimized, 8 vCPU, 32 GB). `c2d-standard-8` is an equivalent fallback if C3 is unavailable in the zone. |
| Image | Debian 12 (`debian-12` image family); its system Python is 3.11 |
| Disk | 50 GB balanced persistent disk |
| Data | Works synthetic-only. Real data is recommended for the book (see below). |
| Measured runtime | 41 minutes for the full `run_all` on `c3-standard-8`, plus about 3 minutes of setup (packages, data download from a bucket, unit tests) and 1 minute for `run_all --quick`. Uploading Coveo from a laptop adds time that depends on the connection. Reserve 1.5 hours. |
| Bring back | `summary/` (under 1 MB), `run_all_full.log`, optionally the Exercise 1 Part B trace folder (about 85 MB) |

The runtimes below were measured on the reference run (`c3-standard-8`, Intel Xeon
Platinum 8481C, 2026-11-09) with Coveo and Yambda data. That run drove the same
commands from a VM startup script, ran the four exercises as
`run_all --only <exercise> --recall-floor 0.95` one after another, and kept the
span traces in a storage bucket.

**Why this machine.** Eight vCPUs are the smallest size that keeps the measurement
honest. Exercise 1 Part B runs 4 server worker threads, a pool of I/O threads for
the simulated remote calls, and the load generator at the same time. Exercise 2
compares one-thread and all-thread inference and search. On 4 vCPUs the load
generator would compete with the server it measures. 32 GB leaves ample room for the
1-million-vector FAISS indexes and the Yambda training data. A compute-optimized
family is recommended because shared-core and E2 machines run on varying CPU
platforms with variable CPU allocation, which adds noise to tail timings and makes
repeat runs hard to compare.

**Real or synthetic data.** `python -m scripts.run_all --data synthetic` needs no
uploads and exercises every code path. Real data changes three things: Exercise 1
Part B and Exercise 2 Part A use Coveo product vectors and real session queries (so
Exercise 2 Part A can report next-item hit rate), and Exercise 2 Part B trains on
Yambda listens and computes parity on real user and item slices. Exercise 1 Part A
never reads data. Use real data for the chapter's tables if the upload is acceptable.

## 1. Create the VM (from the author's machine)

```bash
gcloud compute instances create recsys-ch14 \
    --zone=us-central1-a --machine-type=c3-standard-8 \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=50GB --boot-disk-type=pd-balanced
```

## 2. Copy the code and data

From PowerShell in the `RecSys book code` folder:

```powershell
gcloud compute scp --recurse chapter14_latency recsys-ch14:~ --zone=us-central1-a
```

Coveo (optional, about 2.1 GB). Only `sku_to_content.csv` and the browsing file are
read, and only the first 3 million rows of the browsing file:

```powershell
gcloud compute ssh recsys-ch14 --zone=us-central1-a --command "mkdir -p ~/data/coveo ~/data/yambda"
gcloud compute scp "..\Dataset\SIGIR-ecom-data-challenge\sampled\sku_to_content.csv" "..\Dataset\SIGIR-ecom-data-challenge\sampled\browsing_train_sampled.csv" recsys-ch14:~/data/coveo/ --zone=us-central1-a
```

Yambda (optional, about 370 MB). Download it on the VM directly from Hugging Face,
which is faster than uploading:

```bash
wget -O ~/data/yambda/listens.parquet \
    https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/listens.parquet
```

If the download fails, upload the local copy instead:
`gcloud compute scp "..\Dataset\Yandex\flat\listens.parquet" recsys-ch14:~/data/yambda/ --zone=us-central1-a`.

## 3. Set up the environment (on the VM)

```bash
gcloud compute ssh recsys-ch14 --zone=us-central1-a
sudo apt-get update && sudo apt-get install -y python3-venv tmux
python3 -m venv ~/venvs/recsys-book-ch14
~/venvs/recsys-book-ch14/bin/python -m pip install --upgrade pip
~/venvs/recsys-book-ch14/bin/python -m pip install -r ~/chapter14_latency/requirements.txt
```

Start a `tmux` session so a dropped SSH connection does not stop the run, then set
the paths. Leave out a variable to use synthetic data for that dataset.

```bash
tmux new -s ch14
cd ~/chapter14_latency
source ~/venvs/recsys-book-ch14/bin/activate
export SIGIR_DATA_DIR=~/data/coveo
export YAMBDA_DATA_DIR=~/data/yambda
```

Large outputs go to `~/.cache/recsys_book/ch14/full/` unless `RECSYS_CH14_OUT` is set.

## 4. Check, then run

```bash
python -m latency.platform_check                  # timer granularity, CPU model, threads
python -m unittest discover -s tests -t .         # about a minute
python -m scripts.run_all --quick                 # a few minutes; confirms data paths
python -m scripts.run_all --recall-floor 0.95 2>&1 | tee ~/run_all_full.log
```

Keep the VM otherwise idle during the full run: no other jobs, no package updates,
and no browser or IDE sessions attached to it.

## 5. Each entry point

All commands run from `~/chapter14_latency` on the VM described above. `run_all` runs
them in this order with exactly these arguments; each can also be run alone.

### Exercise 1 Part A: queueing simulation

- **Command:** `python -m scripts.ex1a_queueing_sim`
- **Hardware:** any CPU; results are in virtual time and do not depend on the machine.
- **Measured runtime:** 42 seconds.
- **Input data:** none.
- **Bring back:** `summary/ex1a_*.csv`, `summary/ex1a_summary.md`, `summary/ex1a_queueing_sim_run_info.json`.

### Exercise 1 Part B: pipeline tail attribution

- **Command:** `python -m scripts.ex1b_pipeline_attribution --catalog auto --lever popularity=precomputed --lever hedge-feature-fetch=on --lever feature-cache=on --lever candidates=200`
- **Hardware:** the 8-vCPU VM. Timings are real wall-clock time and depend on the machine.
- **Measured runtime:** 27 minutes: 5 configurations x 4 load levels x 5 repeats x 15
  seconds of measured load (25 minutes), plus warm-up, capacity measurement and trace
  analysis.
- **Input data:** Coveo through `SIGIR_DATA_DIR`, or a 200,000-item synthetic catalog.
- **Bring back:** `summary/ex1b_*.csv`, `summary/ex1b_attribution.md`,
  `summary/ex1b_pipeline_attribution_run_info.json`. Optional: the newest folder in
  `~/.cache/recsys_book/ch14/full/ex1b/traces/` (span-level Parquet, 100 files and about
  85 MB on the reference run) if the chapter needs per-request trace figures.

### Exercise 2 Part A: approximate-search operating points

- **Command:** `python -m scripts.ex2a_ann_operating_points --datasets auto --recall-floor 0.95`
- **Hardware:** the 8-vCPU VM. Single-query latency uses one thread; throughput uses all eight.
- **Measured runtime:** 11.5 minutes: 1.5 for Coveo and 10 for the 1-million-vector
  synthetic catalog (index builds and the sweep).
- **Input data:** Coveo through `SIGIR_DATA_DIR` (about 32,000 products with description
  vectors) plus a 1-million-item synthetic catalog, which always runs.
- **Bring back:** `summary/ex2a_pareto.csv`, `summary/ex2a_summary.md`, `summary/ex2a_ann_operating_points_run_info.json`.

### Exercise 2 Part B: ranker inference and parity

- **Command:** `python -m scripts.ex2b_ranker_inference_parity --data auto`
- **Hardware:** the 8-vCPU VM. Training uses 4 threads by default (`--train-threads`); the
  benchmark compares 1 thread with all 8.
- **Measured runtime:** 1.5 minutes, of which training took 18 seconds (the cap,
  `--max-train-seconds 900`, was never reached); the rest is data loading, export and
  benchmarks over batch sizes 1 to 1,024.
- **Input data:** Yambda through `YAMBDA_DATA_DIR` (7 training days, about 1.46 million
  rows, under the cap of 2 million; the last day, about 232,000 rows, is the test set),
  or synthetic data of a similar size.
- **Bring back:** `summary/ex2b_latency.csv`, `summary/ex2b_parity.csv`, `summary/ex2b_summary.md`,
  `summary/ex2b_ranker_inference_parity_run_info.json`. The model and ONNX files stay on the VM.

## 6. Bring the results back

On the VM:

```bash
tar czf ~/ch14_results.tgz -C ~/chapter14_latency --exclude=summary/quick summary -C ~ run_all_full.log
```

From PowerShell in the `chapter14_latency` folder (Windows 10 and later include `tar`):

```powershell
gcloud compute scp recsys-ch14:~/ch14_results.tgz $env:TEMP\ --zone=us-central1-a
tar xzf $env:TEMP\ch14_results.tgz -C .
```

This places the full-run summaries in `summary/` beside the local `summary/quick/` and
puts `run_all_full.log` in the code folder. Then record the numbers in
`DESIGN_NOTES.md`.

## 7. Delete the VM

```bash
gcloud compute instances delete recsys-ch14 --zone=us-central1-a
```

## If something goes wrong

- **`run_all_status.json` shows a non-zero exit code.** Re-run only that exercise with
  `python -m scripts.run_all --only ex2b` (or whichever failed).
- **Exercise 1 Part B exits with code 2.** The timer check failed, which is not
  expected on Linux. Read the printed guidance before using `--force-timer`.
- **A load level marked `generator_behind` in `ex1b_attribution.md`.** The load
  generator fell behind its schedule at that level, so its late requests reached the
  server in bursts and the measured tail mixes the server's behaviour with the
  generator's. Check that nothing else is running on the VM, that the run did not use
  `--gc-in-window` (with the collector running inside the measured windows, the first
  cloud run flagged 14 of its 20 configuration and load cells), and re-run Exercise 1
  Part B. If the highest load level is still flagged, read the lever results at the
  unflagged levels only.
