# Appendix C: Agentic Autoresearch

This exercise needs no LLM and no API key. Deterministic proposers play the agent's role behind the same `propose(history)` interface an LLM proposer would use. The optional replay proposer reads `proposers/proposals_replay.jsonl` and does not call a model. The file shipped here is synthetic, and its first record says so.

The exercise runs a guarded search on the Chapter 6 pointwise ranker. The Chapter 6 feature pipeline is imported from `code/chapter6_ranking`. It is not copied into this folder.

## What the run demonstrates

- A frozen split: training window, 30-minute gap, one-day validation window, 30-minute gap, one-day test window. The full run uses a 7-day training lookback, fixed in `contract.yaml`.
- Early stopping on the last 10 percent of the training window. The validation day is scored by the harness. The test labels stay in `sealed/`.
- An append-only ledger that keeps failed and timed-out trials.
- A random-search arm and a deterministic greedy arm at the same trial budget. Replay is included in the synthetic run.
- A promotion gate with a test budget of 3: the pre-registered baseline, the random-arm winner, and the greedy-arm winner.
- A noise band from five baseline seeds plus a user-level bootstrap on validation. A search winner is tested only when its validation gain clears that band and the guardrails.

No outcome is guaranteed. A winner rejected because its gain sits inside the noise band is a valid result. The optimism gap, for a candidate that is tested, is validation AUC minus sealed-test AUC. Shift-adjusted optimism subtracts the baseline's gap.

## Setup

CPU only. No GPU. Install the Chapter 6 scientific stack:

```text
pip install -r requirements.txt
```

The synthetic checks were run with polars 1.44.2, numpy 2.4.6, xgboost 3.2.0, and scikit-learn 1.9.1. PyYAML is listed for readers; the loader in this folder also accepts these two YAML files without it.

Run the commands below from `code/appendix_agentic_autoresearch`. On Windows, set `PYTHONIOENCODING=utf-8`.

## Synthetic run

This mode builds a small listens table with a drop in completion rate after the training window. It does not need Yambda, a network connection, or an API key.

```text
python build_harness.py --synthetic --run-dir runs/synthetic
python loop.py --synthetic --run-dir runs/synthetic
python promote_and_test.py --run-dir runs/synthetic --candidate baseline
python promote_and_test.py --run-dir runs/synthetic --candidate random
python promote_and_test.py --run-dir runs/synthetic --candidate greedy
python report.py --run-dir runs/synthetic
```

Measured wall clock for that sequence on CPU: **22 seconds**. The loop itself reported 2.4 seconds; the rest is interpreter startup across the six commands. `runs/` and `sealed/` are gitignored.

## Figure C.2

```text
python simulate_best_of_n.py
```

NumPy only. Seed `20261109`, 5,000 repetitions, candidate counts 1, 2, 4, 8, 16, 32, 64, 128, and 256, quality drawn from Normal(0, 0.3 squared). The script writes `outputs/best_of_n.csv` with the three plotted series: validation score of the selected candidate, sealed-test score of that candidate, and the average candidate. It does not redraw the figure.

## Yambda run

Download the Yandex Yambda flat files as described in `code/chapter6_ranking/README.md` (Hugging Face dataset `yandex/yambda`). Point the harness at the directory that contains `listens.parquet`:

```text
python build_harness.py --data-dir path/to/Yandex/flat --run-dir runs/yambda
python loop.py --run-dir runs/yambda
python promote_and_test.py --run-dir runs/yambda --candidate baseline
python promote_and_test.py --run-dir runs/yambda --candidate random
python promote_and_test.py --run-dir runs/yambda --candidate greedy
python report.py --run-dir runs/yambda
```

`YAMBDA_DATA_DIR` is accepted in place of `--data-dir`. The contract fixes 3 trials per arm, 5 baseline seeds, the baseline configuration, the noise-band method, and the promotion rule before any trial. `loop.py` on this path starts each trial in its own process and stops it at `per_trial_timeout_seconds`. Hardware and the files to bring back are in `CLOUD_RUN.md`.

Add `--replay` to `loop.py` to consume `proposals_replay.jsonl` on a Yambda run as well. Replay is not one of the three budgeted test candidates.

## How to read the outputs

`report.md` in the run directory has four tables:

- **Denominator.** Trials attempted, completed, failed, and timed out.
- **Best so far.** Best validation AUC after each trial.
- **Expected maximum.** For the random arm, the average maximum validation AUC over subsets of each budget, following Dodge et al. (2019).
- **Optimism.** Present only for candidates the gate actually tested.

`ledger.jsonl` is the research record. It stores validation metrics. Test metrics live only in `test_ledger.jsonl`.

## Tests

```text
python -m pytest
```

## Layout

| File | Role |
| --- | --- |
| `contract.yaml` | Pre-registered budget, seeds, baseline, noise band, promotion rule |
| `search_space.yaml` | XGBoost space, objective set to validation AUC |
| `build_harness.py` | Writes the split and the sealed test labels |
| `run_trial.py` | One configuration: Chapter 6 features, train-tail early stopping |
| `loop.py` | Runs the arms and appends every trial |
| `promote_and_test.py` | Gate, hashes, test budget of 3 |
| `report.py` | Curves, optimism, denominator |
| `simulate_best_of_n.py` | Figure C.2 series |
| `proposers/` | Random, greedy, and replay |
| `harness/` | Split, metrics, leakage checks, ledger, seal |
| `tests/` | Seal, budget, objective path, synthetic end-to-end, Figure C.2 |
