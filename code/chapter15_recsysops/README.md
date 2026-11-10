# Chapter 15: Monitoring, Reliability, and Retraining

Companion code for the silent-degradation exercise in Section 15.11. It retrains the Chapter 6 XGBoost ranker on Yandex Yambda, corrupts the frame the ranker scores in ways that every schema check accepts, and records which monitors fire and which stay green, for all traffic and for a hashed 30 percent user segment.

## What the Exercise Shows

| Scenario | What arrives at the ranker |
|---|---|
| `clean` | The test day as logged (the control) |
| `zero_fill_all` | The two most important eligible features are 0 for every request |
| `zero_fill_segment` | The same, for a 30 percent user segment chosen by hashing the user ID |
| `mean_fill_all` | The same two features are their training-set mean for every request |

Training is never touched. Zeros and means keep each column's dtype and range, so the schema monitor passes in every scenario. Every monitor compares the scored test day with a reference day (the last day before it, which the model never trained or early-stopped on), and the clean control shows what each monitor reports with no corruption at all.

Monitors, grouped by layer:

| Layer | Monitors |
|---|---|
| Pipeline | Schema validation pass rate, rows per day, user and item join match rate |
| Feature | Null rate, zero rate, standardized mean shift, standardized quantile shift, KS statistic, PSI |
| Score | Mean, quantile shift, KS, PSI, histogram entropy |
| Quality (needs labels) | AUC, GAUC, NDCG@10, log loss, ECE, calibration gap |

Mean-fill is the instructive case. The fill value is neither null nor zero and equals the training mean, so the null and zero rates can only fall and the mean barely moves: monitors that alarm on a rise in nulls or zeros, or on a mean shift, stay quiet. The column's spread collapses to a single value, though, so the quantile, KS, and PSI monitors see it at once: the KS statistic against a continuous reference is at least 0.5 by construction.

## Files

| File | Purpose |
|---|---|
| `run_exercise.py` | Entry script: trains, corrupts, monitors, and writes every result file |
| `silent_degradation.ipynb` | The narrative walkthrough, with plots; runs the same experiment |
| `monitoring_signals.py` | Reusable estimators: null and zero rate, quantiles, KS, PSI, entropy, AUC, GAUC, NDCG, log loss, ECE, schema and join checks, thresholds |
| `corruption.py` | Zero-fill and mean-fill, applied to all rows or to a SplitMix64-hashed user segment |
| `ranker.py` | The Chapter 6 ranker, retrained: true-day split, window carving, aggregates from earlier windows only, daily lookup refresh, point-in-time counts |
| `ch6_polars_pipeline_vendored.py` | Verbatim copy of the Chapter 6 feature pipeline (`chapter6_ranking/utils/polars_pipeline.py`) |
| `synthetic_yambda.py` | Small Yambda-shaped data for the smoke run and tests (its numbers say nothing about Yambda) |
| `tests/` | Known-answer tests for the estimators, plus an end-to-end synthetic run |
| `design_notes_ch15_exercise.md` | Design decisions, Chapter 6 dependency, run log |
| `CLOUD_RUN.md` | Commands for the full run on a cloud CPU machine |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The Chapter 6 feature pipeline is vendored as `ch6_polars_pipeline_vendored.py`, a verbatim copy of `code/chapter6_ranking/utils/polars_pipeline.py` at commit `8e8e71ab0a9ad9da387446b89f8432a23ad2d7fd`, SHA-256 `5eeb0a86dab5094d6c6b6dcffd3f4db858abebb04c4ba9150179f1e64d33e877`; a test fails if the copy drifts. To try another copy of Chapter 6, pass `--ch6_dir` with its `chapter6_ranking` folder.

### 2. Smoke Run (No Dataset Needed)

```bash
python run_exercise.py --synthetic
```

This generates Yambda-shaped data in a temporary folder, runs the full pipeline, and writes `outputs/smoke/`. It checks the code path only.

### 3. Get the Data

Download `flat/50m/listens.parquet` from [yandex/yambda on Hugging Face](https://huggingface.co/datasets/yandex/yambda):

```bash
pip install "huggingface_hub>=0.34"
hf download yandex/yambda flat/50m/listens.parquet --repo-type dataset --local-dir ./yambda
```

Then point the exercise at the folder that holds `listens.parquet`:

```bash
export YAMBDA_DATA_DIR=./yambda/flat/50m          # Linux, macOS
$env:YAMBDA_DATA_DIR = ".\yambda\flat\50m"        # PowerShell
```

The run checks the file's SHA-256 against the published `flat/50m` file and records the result in `run_manifest.json` (`--no_verify_data_hash` skips the check).

### 4. Full Run

```bash
python run_exercise.py                 # 30-day train window, writes outputs/yambda_30d/
```

CPU only. `CLOUD_RUN.md` gives the measured runtime and peak memory of the full run on a 4-vCPU cloud machine; on Linux and macOS, `run_manifest.json` records each run's peak memory.

### 5. Notebook

```bash
jupyter notebook silent_degradation.ipynb
```

Start Jupyter from this folder. Set `CH15_SYNTHETIC=1` to run the notebook on synthetic data, or `CH15_TRAIN_DAYS` to change the window. The notebook writes to `outputs/<run_name>_notebook/`.

### 6. Tests

```bash
python -m pytest tests -q
```

## Command-Line Options

| Option | Default | Meaning |
|---|---|---|
| `--data_dir` | `$YAMBDA_DATA_DIR` | Folder holding `listens.parquet` |
| `--ch6_dir` | the vendored copy | A `chapter6_ranking` folder to use instead |
| `--synthetic` | off | Use generated data instead of Yambda |
| `--train_days` | 30 | Train window before the test day, in days |
| `--test_days`, `--reference_days` | 1, 1 | Test window and reference window, in days. Each window is scored with lookups rebuilt once, at its start, so in a longer window the later days are served staler lookups |
| `--val_ratio` | 0.1 | Share of the pre-reference rows held out for early stopping |
| `--n_estimators`, `--max_depth`, `--learning_rate`, `--early_stopping_rounds` | 100, 6, 0.1, 10 | XGBoost settings (the Chapter 6 defaults) |
| `--n_jobs` | -1 | XGBoost threads (-1 uses every core) |
| `--n_target_features` | 2 | How many features to corrupt |
| `--target_scope` | `any` | `user` or `item` restricts the corrupted features to the `user_*` or `item_*` lookup features |
| `--segment_fraction`, `--segment_salt` | 0.3, 15 | Size and hash salt of the corrupted segment |
| `--ndcg_k` | 10 | Cutoff for NDCG |
| `--seed` | 42 | XGBoost seed; also seeds the ranking tiebreak |
| `--run_name`, `--output_dir` | `yambda_<train_days>d`, `outputs/` | Where results go |
| `--no_verify_data_hash`, `--no_save_model` | off | Skip the SHA-256 check of `listens.parquet`; skip writing `model/` |

## Outputs

Each run writes `outputs/<run_name>/`:

| File | Contents |
|---|---|
| `signal_summary.md` | The headline tables: status by signal family, target-feature statistics, clean-day false alarms, quality by slice |
| `signal_summary.csv` | One status per scenario, slice, and signal family |
| `signals_long.csv` | Every monitor, scenario, slice, and feature: reference, clean, and scenario values, statistics, thresholds, status |
| `quality_metrics.csv` | Quality metrics per scenario and slice, with deltas against the reference day and the clean control |
| `reliability_bins.csv` | Reliability-diagram bins per scenario and slice |
| `feature_importance.csv` | Gain importance and target eligibility per feature |
| `clean_false_alarms.csv` | Monitors over threshold on the uncorrupted test day |
| `run_manifest.json` | Configuration, thresholds, windows, data and Chapter 6 provenance, package versions, timings |
| `model/` | `model.json` (XGBoost), `pipeline/` (Chapter 6 encoders and the fit-window lookups used for validation rows), `ch15_model_metadata.json` |

## Notes

- **Results depend on the run.** Nothing in the notebook or the summary is typed in by hand. The design notes record the runs made so far.
- **Reference run.** The reference results come from run `ch15-20261109-1008` on a Linux `e2-standard-4` (`CLOUD_RUN.md`). Other machines give slightly different numbers, and alert statuses within about 0.01 of their threshold can flip.
- **Thresholds are illustrative.** They are fixed in `monitoring_signals.Thresholds` before any run. The PSI threshold of 0.25 is the common credit-scoring rule of thumb, which carries no stated error rate. The clean control reports which thresholds are already exceeded without corruption.
- **The clean day is not quiet.** On the full Yambda data the test day contains one account with far more completed plays than a person could listen to, so several monitors are over threshold before any corruption. `clean_false_alarms.csv` lists them, and the design notes trace them to that account.
- **Service metrics are out of scope.** No serving code runs, so throughput, latency, and error rate are unaffected by construction and are not measured.
- **Departures from the Chapter 6 code** (true-day windows, daily lookup refresh, point-in-time re-listen counts across windows) are explained in `design_notes_ch15_exercise.md`.
- **Training rows never see their own labels.** Their user and item aggregates come from the window before the fit window, as in the Chapter 6 trainers.
- **Polars stays on 1.x.** Each transformed frame is also sorted into a total order, so training does not depend on the row order Polars returns.
