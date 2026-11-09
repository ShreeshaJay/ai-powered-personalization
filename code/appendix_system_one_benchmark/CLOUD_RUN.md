# Structured-Output Benchmark Cloud Run

The CPU replay of the structured-output benchmark. It recomputes every committed metric from the text-free replay pack in `replay/` (the per-category ORCAS-I accuracies in aggregate); wall time, memory and cache counts are recorded, not recomputed. No GPU, no API keys, and no network after installation; seeds are fixed, so runs with the same package versions write identical files.

## Hardware

**Tested:** one Windows 11 machine, CPU only, Python 3.11.9, in a fresh virtual environment created for the run. A Google Colab CPU runtime (the standard runtime, no accelerator) or any machine with Python 3.11 or later should also work; neither was timed.

**Requirements:** at least 1 vCPU and 1 GB of free RAM. Measured peak memory stayed under 500 MB in every step: tests 475 MB (they draw the figures once), `report` 459 MB, `figures` 428 MB, `verify` 366 MB. Disk: 275 MB for the virtual environment, 3.6 MB for the replay pack, about 1 MB for outputs.

## Expected Runtime

Measured on the tested machine from clean copies of the folder across three runs with identical outputs, as wall-clock time per command:

| Step | Command | Measured runtime |
|------|---------|------------------|
| Setup | `python -m venv` | about 20 seconds |
| Setup | `pip install -r requirements.txt` | about 3.5 minutes; depends on the connection |
| 1 | Tests | 17 to 40 seconds (the first run in a new environment is the slowest) |
| 2 | `replay_scores.py verify` | 7 to 8 seconds |
| 3 | `replay_scores.py report` (2,000 bootstrap resamples) | 19 to 25 seconds |
| 4 | `replay_scores.py figures` | 5 to 10 seconds |

Steps 1 to 4 take about a minute, and up to 75 seconds on the first run in a new environment, because Python compiles bytecode and matplotlib builds its font cache. Installation dominates the session.

## Inputs

1. **Code and data.** The `code/appendix_system_one_benchmark/` folder of the public repository. The replay pack (`replay/`, 3.6 MB) and the committed summaries (`outputs/benchmark/`) are part of it, so nothing else is downloaded or uploaded.
2. **Secrets.** None. Do not create a `.env` file for this run.

## Commands

Run from inside the `appendix_system_one_benchmark` folder. Leave `--resamples`, `--seed` and `--bins` at their defaults (2,000, 42 and 15) so the outputs match the committed files.

```bash
# Setup (Colab: prefix shell lines with "!", or use a %%bash cell)
python --version                       # expect 3.11 or later
python -m pip install -q -r requirements.txt

# 1. Tests
python -m pytest tests -q

# 2. Recompute every committed metric and the results table; must exit 0
python replay_scores.py verify

# 3. Intervals, paired differences, label-source sensitivity, strata
python replay_scores.py report

# 4. Figures
python replay_scores.py figures

# Package outputs
python -m zipfile -c appendix_benchmark_replay_outputs.zip outputs/replay
```

`verify` exits 0 when every recomputed value matches, 1 when a metric or `ZERO_SHOT_RESULTS.md` differs, and 2 when the pack or a summary does not match `replay/MANIFEST.json`. `requirements.txt` sets lower bounds; the tested run installed numpy 2.4.6, pandas 3.1.1, matplotlib 3.11.2 and pytest 9.2.0. Other numpy or matplotlib versions may change the figure bytes and the bootstrap intervals; `verify` compares at stored precision.

## Outputs to Bring Back

Bring back `appendix_benchmark_replay_outputs.zip`, which contains the following.

| Folder | Files |
|--------|-------|
| `outputs/replay/` | `verify_report.json`, `REPLAY_RESULTS.md`, `replay_metrics.json` |
| `outputs/replay/figures/` | `confidence_meanings.png`, `calibration_two_numbers.png`, `judge_agreement_by_stratum.png` |

Also save the console log; each step prints its runtime.

## After the Run

- Confirm `verify` exited 0 and that the regenerated files are byte-identical to the committed ones (with the tested versions, `git status` shows no change under `outputs/`).
- Inspect every PNG before it goes into the appendix.
- Quote numbers only from `outputs/replay/REPLAY_RESULTS.md` and `outputs/benchmark/ZERO_SHOT_RESULTS.md`. `REPLAY_RESULTS.md` records the pack's build date, builder commit and prompt version.

## Optional: GPU Rerun of the Open Models

[`colab/open_models_colab.ipynb`](colab/open_models_colab.ipynb) reruns Kev-4B and JevLite on a Colab GPU runtime. It needs the reference manifests with item text (`pack_colab_bundle.py` packs them), which are not in the public repository, so it is a maintainer path rather than a reader path. The committed runs did not record the GPU type or the checkpoint revisions; a rerun's latencies are not comparable with the committed ones unless both are recorded.

## Paid Keys

None of the steps above needs a key. Three optional paths do:

- **Rebuilding the judge labels:** `adjudicate_pilots.py --execute` calls Anthropic and OpenAI. Dry-run first (no `--execute`), use `estimate_reference_costs.py` for an estimate, and set `--budget-usd`. Aggregate token counts and cost of the committed labels are in `replay/label_provenance.json`.
- **Rerunning hosted Jev:** `run_benchmark.py --model jev` needs `TYPESAFE_API_KEY` in `.env`. `estimate_jev_costs.py` gives an estimate; set `--budget-usd`.
- **Gated checkpoints:** a Hugging Face token, only if a checkpoint requires one.
