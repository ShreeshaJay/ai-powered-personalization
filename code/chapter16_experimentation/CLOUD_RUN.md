# Chapter 16 Cloud Run

Full runs of the Chapter 16 companion code. Everything is CPU-only, single-process NumPy, so a GPU adds nothing.

## Hardware

**Tested:** a Google Compute Engine `e2-standard-4` VM (4 vCPU, 16 GB RAM, Debian 12, Python 3.11.2), on demand, no GPU. A Google Colab CPU runtime (the standard runtime, no accelerator) or any similar machine also works.

**Requirements:** at least 2 vCPU and 4 GB RAM. Measured peak memory stayed under 600 MB in every step, the highest while Polars aggregates the Yambda listens file. Disk: about 400 MB for the Yambda file, under 10 MB for outputs.

## Expected Runtime

Measured on the tested `e2-standard-4` VM across three full runs with identical results, as wall-clock time per command:

| Step | Command | Measured runtime |
|------|---------|------------------|
| Setup | `pip install -q -r requirements.txt` | 35 to 40 seconds |
| Setup | Yambda download | under 5 seconds on Google Cloud |
| 1 | Tests | about 6 seconds |
| 2 | A/B simulation on Yambda | 1.7 to 2.4 minutes |
| 3 | A/B simulation, synthetic | 1.6 to 2.1 minutes |
| 4 | Interleaving simulation | 2.1 to 2.5 minutes |
| 5 | Bandit and OPE simulation | 2.8 to 3.1 minutes |
| 6 | Schematic figures | under 5 seconds |

Steps 1 to 6 take 8.5 to 10 minutes in total. The spread comes from the shared physical cores of `e2` machines. A whole run, including VM boot, installation, upload, and shutdown, took 12.5 to 14.5 minutes, so a 30-minute session is ample. A Colab CPU runtime was not timed, and download speed there will vary. With `--quick`, each script finishes in 10 to 35 seconds on a laptop CPU.

## Inputs

1. **Code.** Upload the `chapter16_experimentation` folder (zip it, upload it, unzip it). After publication it can instead be taken from the public repository under `code/chapter16_experimentation/`.
2. **Data.** Only step 2 needs data: Yambda `flat/50m/listens.parquet` (369 MB). It is public, so download it in the runtime rather than uploading it:

```bash
mkdir -p /content/yambda/flat
wget -q -O /content/yambda/flat/listens.parquet \
  https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/listens.parquet
export YAMBDA_DATA_DIR=/content/yambda/flat
```

If the download is unavailable, skip step 2. Step 3 covers the same modes on synthetic data, and the run can proceed without any dataset.

## Commands

Run from inside the `chapter16_experimentation` folder. All seeds default to 16; leave them at the default so figures match the design notes.

```bash
# Setup (Colab: prefix shell lines with "!", or use a %%bash cell)
python --version                       # expect 3.11 or later
pip install -q -r requirements.txt

# 1. Tests
python -m pytest tests -q

# 2. A/B simulation on Yambda (Figures 16.2, 16.3, 16.5)
python ab_test_simulation.py --mode all --data-dir "$YAMBDA_DATA_DIR" --out-dir outputs/ab_test

# 3. A/B simulation on synthetic data (fallback and comparison)
python ab_test_simulation.py --mode all --synthetic --out-dir outputs/ab_test_synthetic

# 4. Interleaving (Figure 16.7 and the Figure 16.6 example JSON)
python interleaving_simulation.py --mode all --out-dir outputs/interleaving

# 5. Bandits and off-policy evaluation (Figures 16.8, 16.9)
python bandit_ope_simulation.py --mode all --out-dir outputs/bandit_ope

# 6. Schematics (Figures 16.1, 16.4, 16.6); must run after step 4
python make_schematics.py --example-json outputs/interleaving/interleaving_example.json --out-dir outputs/figures

# Package outputs
zip -r chapter16_outputs.zip outputs
```

Colab installs a recent NumPy, SciPy, Polars, and matplotlib by default. Installing the pinned `requirements.txt` keeps the run identical to the tested environment.

## Outputs to Bring Back

Bring back `chapter16_outputs.zip`, which contains the following.

| Folder | Files |
|--------|-------|
| `outputs/ab_test/` | `summary.json`, `fig16_2_cuped.png`, `fig16_3_peeking.png`, `fig16_5_novelty_primacy.png`, `aa_pvalues.csv`, `aa_pvalue_histogram.png`, `power_curves.csv`, `power_curves.png`, `cuped_estimates.csv`, `ratio_ci_coverage.png`, `peeking_curves.csv`, `novelty_curves.csv` |
| `outputs/ab_test_synthetic/` | the same file set, from synthetic data |
| `outputs/interleaving/` | `summary.json`, `fig16_7_interleaving_sensitivity.png`, `sensitivity.csv`, `interleaving_example.json` |
| `outputs/bandit_ope/` | `summary.json`, `fig16_8_bandit_regret.png`, `fig16_9_ope.png`, `regret_mab.csv`, `regret_contextual.csv`, `ope_estimates.csv`, `support_sweep.csv` |
| `outputs/figures/` | `fig16_1_evaluation_ladder.png`, `fig16_4_interference_designs.png`, `fig16_6_team_draft_example.png`, `summary_schematics.json` |

Also save the console log of each step; every `summary.json` records the arguments, runtime, and library versions.

## After the Run

- Inspect every PNG before it goes into the chapter.
- Record the numbers in the run log in `docs/Chapter16_Design_Notes.md`; the chapter describes the results qualitatively.
