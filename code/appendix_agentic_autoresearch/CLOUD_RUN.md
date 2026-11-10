# Cloud run: Appendix C, full Yambda path

The synthetic commands in the README are the local check. They need a CPU and about 22 seconds. This file is the full run, which was not executed in the verification session because it needs the Yambda snapshot.

## Shared inputs

- Dataset: Yandex Yambda flat files, the same download Chapter 6 documents (`code/chapter6_ranking/README.md`, Hugging Face `yandex/yambda`).
- Required file: `listens.parquet` with `uid`, `item_id`, `timestamp`, `played_ratio_pct`, `track_length_seconds`, and `is_organic`.
- Pass the directory with `--data-dir` or `YAMBDA_DATA_DIR`. Do not commit the data or the sealed labels.
- Hardware: CPU, 4 cores, 16 GB RAM, no GPU. `contract.yaml` pins `n_jobs` at 1 so a seed repeats. The machine still wants the Chapter 6 tutorial's memory headroom, because each trial refits that feature pipeline on the 7-day window.
- Packages: `requirements.txt`. No LLM SDK and no API key.

The sealed windows are taken from the end of `listens.parquet`: 7 training days, a 30-minute gap, one validation day, a 30-minute gap, one test day.

## Commands

Run from `code/appendix_agentic_autoresearch`.

| Entry point | Command | What comes back |
| --- | --- | --- |
| Build the harness | `python build_harness.py --data-dir "$YAMBDA_DATA_DIR" --run-dir runs/yambda` | `runs/yambda/artifacts/split_manifest.json` and `seal_record.json`. Leave `sealed/` on the machine that will score the test. |
| Search | `python loop.py --run-dir runs/yambda` | `ledger.jsonl`, `noise_band.json`, `winners.json`, `loop_summary.json` |
| Promote baseline | `python promote_and_test.py --run-dir runs/yambda --candidate baseline` | one row in `test_ledger.jsonl` if the gate passes |
| Promote random winner | `python promote_and_test.py --run-dir runs/yambda --candidate random` | same ledger |
| Promote greedy winner | `python promote_and_test.py --run-dir runs/yambda --candidate greedy` | same ledger |
| Report | `python report.py --run-dir runs/yambda` | `report.md`, `best_so_far.csv`, `expected_maximum_random.csv`, `optimism.csv`, `denominator.csv` |
| Figure C.2 | `python simulate_best_of_n.py` | `outputs/best_of_n.csv`. CPU, seconds, no data download. |

`loop.py` without `--synthetic` runs each trial in a child process and stops it when `per_trial_timeout_seconds` (600) elapses. A stopped trial is logged as timed out and does not get a validation score.

## Runtime

The contract pre-registers 3 trials per arm and 5 baseline seeds: 11 model fits, plus up to 3 sealed test evaluations that reuse the saved models. Each fit calls the Chapter 6 feature pipeline on the 7-day window and then trains XGBoost.

This session did not time a Yambda trial. Keep the wall clock under about 30 minutes. If one 7-day trial is slower than about 2 minutes, stop and lower `trial_budget_per_arm` only by writing a new harness version before any further trial. Do not edit the budget after looking at scores.

## Files to bring back

Bring back the report, both ledgers, `noise_band.json`, `winners.json`, and `loop_summary.json`.

Leave behind `sealed/`, `*.parquet` predictions, `model.json`, and the saved pipeline directories. Those are gitignored and they contain labels or model dumps.
