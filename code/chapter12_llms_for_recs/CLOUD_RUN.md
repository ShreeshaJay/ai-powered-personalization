# Running Chapter 12 on a Cloud GPU

Every script runs on a laptop CPU at small scale (see `README.md`). This file covers the larger runs on a single cloud GPU, for example a Colab notebook. Only open-weights models are used and nothing calls a paid API.

**About the runtimes below.** CPU times were measured on the development laptop with the `tiny` profile. A time marked measured on a Colab L4 is one seed from 2026-11-09 (NVIDIA L4, 24 GB). The other GPU times are estimates from model size, token counts and step counts, and they vary with GPU, driver and library versions. Treat the estimates as planning ranges and check the first few minutes of each run.

GPU classes used below:

| Class | Memory | Typical examples |
|-------|--------|------------------|
| Small | 16 GB | Colab T4 class (fp16 only, no bf16) |
| Large | 40 GB or more | Colab A100 class (bf16) |

## Setup

```python
# Colab cell 1: storage that survives a disconnect
from google.colab import drive
drive.mount("/content/drive")

# Colab cell 2: code and packages
%cd /content/drive/MyDrive/<your copy of the repo>/code/chapter12_llms_for_recs
!pip install -r requirements-gpu.txt
```

Colab already ships a CUDA build of PyTorch. To keep it, install everything except the `torch` line: `!grep -v '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt bitsandbytes==0.50.2`.

Colab preinstalls an old `torchao`, which makes PEFT refuse to import (`ImportError: Found an incompatible version of torchao`); the chapter does not use it, so run `!pip uninstall -y torchao` after installing. pip also reports a conflict between Colab's preinstalled `cudf-polars` and `polars==2.1.0`; it is harmless because the chapter does not use `cudf-polars`.

Data: upload `sessions_train.csv` and `products_train.csv` from the Amazon KDD Cup 2023 release into a folder named `Amazon KDD 2023`, and point the code at its parent:

```python
import os
os.environ["RECSYS_DATASET_ROOT"] = "/content/drive/MyDrive/Dataset"
```

Pass `--work-dir /content/drive/MyDrive/ch12_runs/<profile>` to every command (or to `run_pipeline.py`) so outputs survive a disconnect. The LLM scripts append generations to JSONL as they go and resume from where they stopped; rerun the same command after a disconnect.

Model downloads go to the Hugging Face cache (`HF_HOME`, by default `~/.cache/huggingface`). Qwen2.5-0.5B-Instruct is about 1 GB, 1.5B about 3 GB, 7B about 15 GB.

## Profiles

```bash
python run_pipeline.py --profile ranker --work-dir /content/drive/MyDrive/ch12_runs/ranker
python run_pipeline.py --profile full   --work-dir /content/drive/MyDrive/ch12_runs/full
python run_pipeline.py --profile full   --work-dir ... --only train_tiger.py     # one step
python run_pipeline.py --profile full   --dry-run                                # print the commands
```

`ranker` (20,000 sessions) produces the Section 12.4 feature ablation with real LLM features. `full` (every eligible UK session) covers Sections 12.2 and 12.5. Steps can also be run one at a time with the commands below.

## Per-script guide

Paths are relative to `--work-dir`. "Full" means all UK sessions with at least three history items (roughly 730K sessions, about 500K items with `--catalog all`).

| Script | Hardware | Runtime | Reads | Writes |
|--------|----------|---------|-------|--------|
| `prepare_data.py` | CPU | Measured: 23 s for 3,000 sessions on the development laptop (most of it reading the CSVs). Ranker profile, 20,000 sessions: measured 4 s on a Colab L4, single seed, 2026-11-09 (the write itself was 3.0 s). Full: estimate under 10 min | KDD CSVs | `data/sessions.parquet`, `data/items.parquet`, `data/prepare_meta.json` |
| `encode_items.py` (MiniLM) | Small GPU | Measured: 71.2 s for 97,869 ranker-profile items on a Colab L4, single seed, 2026-11-09 (process wall 75 s). Full catalog: estimate 10 to 30 min. CPU is slow: tens of items per second on the development laptop | `data/items.parquet` | `embeddings/item_embeddings.npz` |
| `train_rqvae.py` | Small GPU or CPU | Measured: 93 s on CPU for 18K items, K=64, 20 epochs. Full (500K items, K=256, 30 epochs): estimate 10 to 30 min on a small GPU | `embeddings/item_embeddings.npz` | `models/rqvae.pt`, `embeddings/item_semantic_ids.npz`, `metrics/rqvae.json` |
| `evaluate_semantic_ids.py` | CPU | Measured: 13 s for 18K items. Full: estimate a few minutes; `--tsne` (on a 5,000-item sample) adds a few minutes | Semantic IDs, item embeddings, sessions | `metrics/semantic_id_analysis.json`, optional `metrics/tsne_semantic_ids.png` |
| `train_tiger.py` | Small GPU | Full: estimate 1 to 3 h for up to 20 epochs with early stopping, plus 10 to 20 min of beam search on 5,000 test sessions. CPU is only practical for the `tiny` profile: measured 616 s for 5 epochs and 300 test sessions on the 3,000-session sample (the profile runs 3 epochs) | Semantic IDs, item embeddings, sessions | `models/tiger.pt`, `metrics/tiger.json` |
| `build_slates.py` | CPU | Measured: 23 s for 300 sessions per split, all four modes, on the development laptop. Ranker profile, 2,000 sessions per split: measured 19 s on a Colab L4, single seed, 2026-11-09. 5,000 sessions: estimate under 10 min | Item embeddings, sessions | `slates/{split}_{mode}_n{size}.jsonl`, `metrics/slate_baselines.json` |
| `evaluate_zeroshot_llm.py` (1.5B) | Small GPU | 1,000 sessions, 20 candidates: estimate 20 to 60 min (prompt about 700 tokens, up to 80 generated). CPU with 0.5B: measured 44 s of generation for 4 sessions, plus model download and load | Test slates, item titles | `llm/zeroshot_*.jsonl`, `metrics/zeroshot_*.json` |
| `train_tallrec.py` (1.5B) | Small GPU; Large GPU for 7B with `--quantize-4bit` | 5,000 train sessions (20,000 examples), 1 epoch, then 1,000 test slates scored twice: estimate 1 to 2 h on a small GPU. CPU with 0.5B and the `tiny` settings (8 train sessions, 2 test slates): measured 500 s (249 s training, 168 s scoring) | Train and test slates of one mode | `models/tallrec_<model>/` (LoRA adapter), `metrics/tallrec_*.json` |
| `generate_llm_features.py` (1.5B) | Large GPU preferred | `ranker` profile (97,869 items and 20,000 session intents, Qwen2.5-1.5B, batch size 64): measured 11331 s on a Colab L4, single seed, 2026-11-09 (item knowledge 9647.5 s, session intents 1622.8 s). The full 500K-item catalog is a long batch job; run it across sessions using the resume support. CPU with 0.5B: measured 254 s for 12 item descriptions and 44 s for 8 session intents, so CPU is only for checking the code path | Items, sessions, the encoder named in `item_embeddings.npz` | `llm/item_knowledge_<model>.jsonl`, `llm/session_intent_<model>.jsonl`, `embeddings/llm_*_embeddings.npz`, `metrics/llm_features_<model>.json` |
| `train_augmented_ranker.py` | CPU or any GPU | Measured: 51 s on CPU for 2,400 train sessions (12,000 examples) and four feature sets. `ranker` profile (80,000 train examples, four feature sets): measured 35 s on a Colab L4, single seed, 2026-11-09 | Item embeddings, LLM feature embeddings, sessions | `models/ranker_<feature set>.pt`, `metrics/augmented_ranker_training.json` |
| `evaluate_augmented.py` | CPU | Measured: 9 s for 300 test slates and four feature sets. `ranker` profile: measured 4 s on a Colab L4, single seed, 2026-11-09 | Ranker checkpoints, test slates | `metrics/augmented_ranker_test_retrieved_n20.json` |

This single Colab L4 seed did not rank the full feature set first.

### Order and dependencies

```
prepare_data -> encode_items -> train_rqvae -> evaluate_semantic_ids
                             \-> train_tiger (needs train_rqvae)
                             \-> build_slates --splits train val test
                                   \-> evaluate_zeroshot_llm
                                   \-> train_tallrec (needs the train slates)
                             \-> generate_llm_features -> train_augmented_ranker -> evaluate_augmented (needs test slates)
```

### Settings worth knowing

- `--quantize-4bit` (zero-shot, TALLRec, feature generation) loads the model in 4-bit NF4 through bitsandbytes. CUDA only. Use it for 7B models on a small GPU.
- On small-class GPUs without bf16, the LLM scripts run in fp16 and TALLRec uses loss scaling.
- `--batch-size` for the LLM scripts trades memory for speed. If you hit out-of-memory, halve it; outputs already written are kept.
- `generate_llm_features.py --max-items N --max-sessions N` produces partial coverage; the ranker then stops unless given `--allow-partial-coverage`. Use it to time a run before committing to the full job.
- Seeds are fixed (default 42). Greedy decoding is used throughout, so reruns give the same generations up to GPU nondeterminism.

## HSTU (Section 12.3)

Run from `hstu_yambda/`. Data: `Dataset/Yandex/flat/multi_event.parquet` from the Yambda release (https://huggingface.co/datasets/yandex/yambda), or MovieLens-1M `ratings.dat` (https://grouplens.org/datasets/movielens/1m/) under `Dataset/MovieLens/ml-1m/`. `RECSYS_DATASET_ROOT` moves both.

| Command | Hardware | Runtime | Writes (under `outputs/<dataset>/`) |
|---------|----------|---------|------------------------------------|
| `python train.py --dataset synthetic --max-seq-len 50 --dim 32 --epochs 4 --num-negatives 32` | CPU | Measured: 35 s | `hstu_best.pt`, `hstu_last.pt`, `train_log.jsonl`, `hstu_run_metadata.json`, `data_stats.json` |
| `python train.py --dataset movielens --max-seq-len 50 --dim 32 --epochs 1 --num-negatives 32` | CPU | Measured: 152 s for one epoch including loading | as above |
| `python train.py --dataset movielens` (defaults: D=64, length 200, 128 negatives, up to 100 epochs) | Small GPU | Measured: 696.7 s on a Colab L4, single seed, 2026-11-09 (early stop at epoch 78) | as above |
| `python train.py --dataset yambda --max-days 30` | Small GPU | Measured: 1070.8 s on a Colab L4, single seed, 2026-11-09 (early stop at epoch 40; the first 30 days hold about 3.5M events, 5,400 users and 250K items) | as above |
| `python baselines/sasrec_baseline.py --dataset <same>` | Small GPU | Measured on a Colab L4, single seed, 2026-11-09: 741.3 s for MovieLens defaults (100 epochs) and 1018.5 s for Yambda, first 30 days (early stop at epoch 38) | `sasrec_*.pt`, `sasrec_train_log.jsonl`, `sasrec_run_metadata.json` |
| `python evaluate.py --dataset <same> --checkpoint outputs/<dataset>/hstu_best.pt` | Small GPU or CPU | Measured on a Colab L4, single seed, 2026-11-09: 7 s for MovieLens and 8 s for Yambda (process wall between the status timestamps) | `eval_metrics.json` |

The full-catalog evaluation scores every item for every user; on Yambda's 30-day window that is about 250K items per user, which is fine on a GPU and slow on a CPU. Yambda timestamps are seconds since the dataset start (on a 5-second grid), so `--max-days` is in real days.
