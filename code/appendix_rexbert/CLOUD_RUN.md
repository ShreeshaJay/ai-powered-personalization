# Cloud and local runs

Commands are run from this directory. Data arrive as `products_train.csv` and `sessions_train.csv` in a folder named by `--data_dir` or `AMAZON_KDD_DATA_DIR`. The download page is the Amazon KDD Cup 2023 challenge on AIcrowd, linked from the README. No data are stored in this folder.

Bring back `outputs/run_manifest.json`, `outputs/metrics/<profile>/`, and `outputs/figures/<profile>/`. Leave `outputs/cache/` on the machine that trained. It holds checkpoints and is not part of the published tree.

`--sync_cmd` can copy `outputs/metrics/` after each completed run. The template is supplied on the command line. This file does not name a destination.

## Entry points

### `python -m pytest`

Unit tests for the split, the metrics, the baselines, the figures, and the publishable-text check. CPU only. Under a few minutes once the environment exists. No data download.

### `python -m pytest -m slow`

Imports the trainer and fine-tunes MiniLM for one epoch on 16 synthetic pairs, with a four-query validation evaluator. CPU. Downloads `sentence-transformers/all-MiniLM-L6-v2` on the first call. Expect several minutes, dominated by the download and the first PyTorch import.

### `python run_all.py --check_models`

Loads RexBERT-micro, RexBERT-base, and Ettin-17m on CPU and writes `outputs/metrics/model_check.json` with the embedding dimension and the loaded parameter count. Downloads the base checkpoint on the first call (on the order of 600 MB). CPU, well under an hour after the download. This is a load check, not a retrieval run.

### `python run_all.py --profile cpu_small --runs R0,R1,R2,R3,R4`

Teaching profile. An 8-core laptop CPU is enough. The planning estimate was 45 to 120 minutes. On an 8-core laptop with 16 GB of RAM the two end-to-end runs each took about 135 minutes of run time: R0 about 1 minute, R1 about 23 minutes, R2 under a second (no sweep on this profile), R3 about 99 minutes, R4 about 12 minutes. Peak process RAM was about 1.2 GB for the data load and about 3.3 GB for the whole profile. Those numbers are teaching measurements. They are not appendix results. The profile needs the two CSV files and downloads MiniLM, RexBERT-micro, and Ettin-17m. Outputs:

- `outputs/metrics/cpu_small/R0.json` through `R4.json`
- `outputs/metrics/cpu_small/diagnostics.json`, `decision.json`, `figure_a4.json`, `summary.csv`
- `outputs/figures/cpu_small/figure_A_3.png` and `figure_A_4.png`
- `outputs/run_manifest.json`

These metrics are teaching numbers. They are not reported in the appendix.

### `python run_all.py --profile gpu_full --runs R1,R2,R3,R4`

Reported profile. Colab L4: Python 3.13, the preinstalled torch build `2.14.1+cu130`, about 53 GB RAM, 12 CPUs. Do not reinstall torch. Install `requirements.txt` around the existing torch, then assert that `torch.__version__` is unchanged. The notebook `colab/appendix_rexbert_colab.ipynb` is that sequence.

R0 is prepended when its JSON is missing. A run whose metrics JSON exists is skipped, so a stopped job resumes at the next missing run.

Planning estimates, not measurements:

| Run | Planning estimate |
|---|---|
| R0 | 3 to 6 min, 6 to 8 GB RAM, CPU |
| R1 | 40 to 70 min on an L4 |
| R2 | 30 to 50 min on an L4 |
| R3 | 4 to 6 h on an L4 |
| R4 | 40 to 60 min on an L4, plus the retrieval CPU work |
| Core total | about 6 to 8 L4 GPU-hours |

Outputs mirror the CPU list under `outputs/metrics/gpu_full/` and `outputs/figures/gpu_full/`.

### `python make_figures.py --metrics_dir outputs/metrics/<profile> --figure_dir outputs/figures/<profile>`

Redraws Figures A.3 and A.4 from saved JSON. CPU, seconds. `--demo` draws from synthetic arrays and is not a result.

### `python run_all.py --compare <left_metrics_dir> <right_metrics_dir>`

Compares two metric directories and ignores wall-clock and RAM fields. Used for the two-run CPU determinism check.
