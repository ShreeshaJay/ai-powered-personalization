# Appendix A companion code: RexBERT as a sentence encoder

This folder converts RexBERT, a masked language model, into a sentence encoder and compares it with a capacity-matched general encoder and with MiniLM. The questions, the decision rule, and the protocol constants are locked in `PREREGISTRATION.md`. Reported numbers come from the `gpu_full` profile. The `cpu_small` profile is a teaching run on a smaller sample. Its metrics are not the appendix's results.

This folder reuses Chapter 10's dataset (Amazon KDD Cup 2023 / Amazon-M2) and its text template. See the Chapter 10 README for download instructions. No Chapter 10 code or artifacts are required.

## Data

Download the Amazon KDD Cup 2023 training files from the challenge page:

https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge

Place `products_train.csv` and `sessions_train.csv` in one folder. Point the code at that folder with `--data_dir` or with the `AMAZON_KDD_DATA_DIR` environment variable. The data are not redistributed with this code, and derived embedding files are not written into the published tree.

For a single target, Recall@K equals hit rate at K. The Chapter 10 protocol can have several targets for one query, so that path also reports hit rate. The two protocols are labelled separately in the metric JSON.

## Profiles

| | `cpu_small` | `gpu_full` |
|---|---|---|
| Purpose | Teaching, same code paths | Reported appendix protocol |
| Catalog | 50,000 UK products, including sampled targets | All UK products |
| Test queries | 2,000 | 20,000 seeded sample |
| Training pairs | 50,000, 1 epoch, seed 0 | 500,000, up to 3 epochs, seeds 0, 1, 2 |
| Batch size | 64 | 128 |
| Models | MiniLM, RexBERT-micro, Ettin-17m | Those, plus mpnet and RexBERT-base at zero-shot, and MiniLM fine-tuned |

## How to run

From this directory, with the pins in `requirements.txt` installed:

```bash
python run_all.py --profile cpu_small --runs R0,R1,R2,R3,R4
python run_all.py --profile gpu_full --runs R1,R2,R3,R4
```

R0 runs automatically when a later run is requested and `outputs/metrics/<profile>/R0.json` is missing. A run whose metrics JSON already exists is skipped. `--force` recomputes it. Wall-clock time for each run is written to `outputs/run_manifest.json`.

`--sync_cmd` is an optional shell template run after each completed run. Placeholders are `{run}`, `{metrics_dir}`, and `{profile}`. `{metrics_dir}` is a path relative to this folder. The default is to run nothing, and no destination is stored in the code.

```bash
python run_all.py --profile gpu_full --runs R4 --sync_cmd "echo {metrics_dir}"
```

Figures A.3 and A.4 are written by the R4 step from the metric JSON. They can be redrawn alone:

```bash
python make_figures.py --metrics_dir outputs/metrics/gpu_full --figure_dir outputs/figures/gpu_full
```

`python -m pytest` runs the unit tests. Tests marked `slow` download MiniLM and exercise the trainer.

## Runtimes

On CPU, run `python run_all.py --profile cpu_small --runs R0,R1,R2,R3,R4` from this directory. That profile is teaching only. On the 8-core laptop used for the code check, each end-to-end run took about 135 minutes: R0 about 1 minute, R1 about 23 minutes, R2 under a second, R3 about 99 minutes, and R4 about 12 minutes. The data load peaked near 1.2 GB, and the whole profile peaked near 3.3 GB. Those timings are teaching measurements. They are not appendix results.

On Colab, use an L4 runtime and open `colab/appendix_rexbert_colab.ipynb`. Run the cells in order. The notebook installs `requirements.txt` only and checks that `torch.__version__` is unchanged, then runs `python run_all.py --profile gpu_full --runs R1,R2,R3,R4`.

The `gpu_full` times below are planning estimates, not measurements. No L4 run has been timed.

| Run | Estimate |
|---|---|
| R0 | 3 to 6 min |
| R1 | 40 to 70 min on an L4 |
| R2 | 30 to 50 min on an L4 |
| R3 | 4 to 6 h on an L4 |
| R4 | 40 to 60 min on an L4 |
| Core total | about 6 to 8 L4 GPU-hours |

`CLOUD_RUN.md` records the same commands and the same labels.

## Models

Each load passes the revision below and `trust_remote_code=False`.

| Key | Hub id | Revision |
|---|---|---|
| rexbert_micro | thebajajra/RexBERT-micro | `25587117376236b769710249fdcb66d6041edcee` |
| rexbert_base | thebajajra/RexBERT-base | `55bd1c52eb0714beefcb61cae5e304352197f44c` |
| ettin_17m | jhu-clsp/ettin-encoder-17m | `987607455c61e7a5bbc85f7758e0512ea6d0ae4c` |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| mpnet | sentence-transformers/all-mpnet-base-v2 | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |

## Dependencies

`requirements.txt` pins the packages that were confirmed on PyPI for Python 3.13. Torch is omitted on purpose. The Colab L4 image already provides `torch 2.14.1+cu130`. An equality pin on `2.14.1` does not match that local build string and would replace the CUDA wheel. `sentence-transformers` accepts any torch at or above 2.2.

On Windows, install the CPU wheel first, then the requirements file:

```bash
pip install torch==2.14.1
pip install -r requirements.txt
```

On Colab, install `requirements.txt` only, and check that `torch.__version__` is unchanged. `environment_lock.txt` is the freeze from the Windows CPU check (Python 3.13.15). It is not a Colab freeze. The Colab notebook is `colab/appendix_rexbert_colab.ipynb`. Hardware, commands, and expected runtimes are in `CLOUD_RUN.md`.

Checkpoints under `outputs/cache/` are local. Derived fine-tuned weights are not part of the published tree. `outputs/` keeps metric JSON and CSV summaries. Everything else in `outputs/` is ignored.

GPU training can be non-deterministic because scaled dot-product attention is not bitwise stable. The CPU profile sets Python, NumPy, and PyTorch seeds. Two CPU runs are compared by `python run_all.py --compare <left_metrics_dir> <right_metrics_dir>`.
