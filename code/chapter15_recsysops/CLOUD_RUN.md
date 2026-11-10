# Chapter 15: Full Run on a Cloud CPU Machine

The full exercise (30-day Yambda window, Chapter 6 model settings) needs no GPU and about 5.5 GB of memory, so a laptop with 8 GB or more is enough. This file gives the commands for running it on a small Google Cloud VM, where it was measured, and lists the files worth keeping. The same commands work on any Linux or macOS machine with Python 3.11.

## What the Run Does

1. Loads the last 30 days of Yambda `flat/50m` listens before a 30-minute gap and the final day. The row counts per window are in `run_manifest.json` under `windows.rows`.
2. Builds features with the Chapter 6 Polars pipeline, trains XGBoost (100 trees, depth 6, early stopping on a validation window carved from the training data), and scores the reference day and four test-day scenarios.
3. Writes the result files to `outputs/yambda_30d/`.

The notebook then repeats the same experiment and saves an executed copy with every plot.

## Machine and Measured Cost

Measured on Google Compute Engine `e2-standard-4` (4 vCPU, 16 GB RAM, on demand), Debian 12 with Python 3.11, 30 GB boot disk, no GPU:

| Step | Wall clock | Peak memory |
|---|---|---|
| Tests (`pytest`) | 5 s | 0.4 GiB |
| Full run (`run_exercise.py`) | 2 min 7 s | 5.2 GiB |
| Notebook (`jupyter execute`) | 2 min 8 s | 5.4 GiB |

Peak memory is the maximum resident set size reported by `/usr/bin/time -v`. The whole job, with one extra repeat of the full run, took about 20 minutes from creating the VM to its shutdown, which costs a few US cents at on-demand prices. Of the 124 seconds the full run reports for itself, about 24 go to loading data and building features, 47 to training, and 53 to the monitors. Installing packages and downloading the 369 MB data file took under three minutes on the same machine. Each further `run_exercise.py` variant (step 7) took between 2 and 2.5 minutes.

## Option A: Google Compute Engine VM

### 0. The VM

Create a VM that deletes itself after three hours even if you forget it, and connect:

```bash
gcloud compute instances create ch15-run --project=YOUR_PROJECT --zone=us-central1-a \
    --machine-type=e2-standard-4 --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=30GB --max-run-duration=3h --instance-termination-action=DELETE
gcloud compute ssh ch15-run --project=YOUR_PROJECT --zone=us-central1-a
```

### 1. Code

```bash
sudo apt-get update && sudo apt-get install -y git python3-venv
git clone https://github.com/ShreeshaJay/ai-powered-personalization.git
cd ai-powered-personalization
git rev-parse HEAD                      # record this commit
cd code/chapter15_recsysops
```

The exercise uses its vendored copy of the Chapter 6 pipeline, `ch6_polars_pipeline_vendored.py` (commit `8e8e71ab0a9ad9da387446b89f8432a23ad2d7fd`, SHA-256 `5eeb0a86dab5094d6c6b6dcffd3f4db858abebb04c4ba9150179f1e64d33e877`), and records the copy's source commit and SHA-256, and whether it still matches them, in `run_manifest.json`.

### 2. Environment

```bash
python3 -m venv ~/venv-ch15
source ~/venv-ch15/bin/activate
pip install -r requirements.txt "huggingface_hub>=0.34"
```

### 3. Data

```bash
hf download yandex/yambda flat/50m/listens.parquet --repo-type dataset --local-dir ~/yambda
export YAMBDA_DATA_DIR=~/yambda/flat/50m
```

The file is 369,110,332 bytes. The run checks its SHA-256 (`eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1`) and records `matches_published_yambda_flat_50m` in the manifest.

### 4. Tests

```bash
python -m pytest tests -q
```

All tests should pass; the end-to-end test uses synthetic data and does not need `YAMBDA_DATA_DIR`.

### 5. The Run

```bash
mkdir -p outputs
python run_exercise.py 2>&1 | tee outputs/yambda_30d.log
```

### 6. The Notebook

```bash
mkdir -p outputs/yambda_30d_notebook
jupyter execute silent_degradation.ipynb --timeout=3600 \
    --output=outputs/yambda_30d_notebook/silent_degradation.executed.ipynb
```

`jupyter execute` refuses to start unless the output folder already exists, hence the `mkdir`.

The notebook runs the experiment again and writes `outputs/yambda_30d_notebook/`. Its tables should match `outputs/yambda_30d/`.

### 7. Optional Robustness Runs

```bash
python run_exercise.py --segment_salt 16 --run_name yambda_30d_salt16 --no_verify_data_hash
python run_exercise.py --segment_salt 17 --run_name yambda_30d_salt17 --no_verify_data_hash
python run_exercise.py --seed 7 --run_name yambda_30d_seed7 --no_verify_data_hash
python run_exercise.py --target_scope item --run_name yambda_30d_item_targets --no_verify_data_hash
```

Different salts pick a different 30 percent of users; a different seed changes XGBoost's row and column sampling, and so can change which features are the targets. `--target_scope item` corrupts the two most important item features instead of whatever ranks highest overall.

### 8. Keep the Results, Then Delete the VM

```bash
tar czf ch15_results.tgz --exclude='*/model' outputs
```

This keeps every result file and log and leaves out the model folders, which can be regenerated. The archive is well under 1 MB. Copy it off the VM (for example with `gcloud compute scp`), then delete the VM:

```bash
gcloud compute instances delete ch15-run --project=YOUR_PROJECT --zone=us-central1-a
```

## Option B: Colab

A Colab CPU runtime has enough memory for the measured peak. These cells have not been run on Colab:

```python
!git clone https://github.com/ShreeshaJay/ai-powered-personalization.git
!cd ai-powered-personalization && git rev-parse HEAD
%cd ai-powered-personalization/code/chapter15_recsysops
!pip install -q -r requirements.txt "huggingface_hub>=0.34"
!hf download yandex/yambda flat/50m/listens.parquet --repo-type dataset --local-dir /content/yambda
%env YAMBDA_DATA_DIR=/content/yambda/flat/50m
!python -m pytest tests -q
!mkdir -p outputs/yambda_30d_notebook && python run_exercise.py 2>&1 | tee outputs/yambda_30d.log
!jupyter execute silent_degradation.ipynb --timeout=3600 --output=outputs/yambda_30d_notebook/silent_degradation.executed.ipynb
!tar czf ch15_results.tgz --exclude='*/model' outputs
```

Download `ch15_results.tgz` from the Files pane. If `pip` reports a conflict with Colab's preinstalled packages, note the message and the versions it installed.

## Checks After the Run

1. `outputs/yambda_30d/run_manifest.json`: `provenance.data.matches_published_yambda_flat_50m` is `true`; `provenance.chapter6.body_matches_source` is `true`; `windows.rows` and `timings_s` are filled in.
2. `outputs/yambda_30d/signal_summary.md` exists and has the status table.
3. The executed notebook has no error outputs.
4. `design_notes_ch15_exercise.md` (section 9.4) records the results of the run measured above, for comparison.

The reference results are those of run `ch15-20261109-1008` on a Linux `e2-standard-4`, and a repeat on the same machine matched them exactly. Other machines give slightly different numbers: a Windows laptop run of the same code changed the status of 6 of the 204 summary cells and moved quality metrics by up to 0.027. Alert statuses whose statistic lies within about 0.01 of its threshold can flip from one machine to another.
