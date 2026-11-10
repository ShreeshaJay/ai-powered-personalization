# HSTU: Generative Sequential Recommendation (Section 12.3)

A from-scratch PyTorch implementation of **HSTU** (Hierarchical Sequential Transduction Unit), the generative sequential recommender from Zhai et al., *Actions Speak Louder than Words* ([arXiv:2402.17152](https://arxiv.org/abs/2402.17152), ICML 2024), with a SASRec baseline trained and evaluated the same way. The architecture follows [Rahul Agarwal's MLWhiz tutorial](https://www.mlwhiz.com/p/hstu-from-scratch-in-pytorch-a-complete) and was cross-checked against Meta's reference code ([generative-recommenders](https://github.com/meta-recsys/generative-recommenders), Apache 2.0).

HSTU replaces softmax attention with **SiLU-activated pointwise attention**, adds a **relative attention bias over positions and time gaps**, and fuses each **(item, action)** pair through a small MLP before the encoder stack. Retrieval is trained with **sampled softmax over cosine similarities with a temperature**, and evaluated over the **full catalog** (HR@10, NDCG@10) with no sampled negatives at evaluation time.

This README reports no results. Run the commands below to produce your own; `CLOUD_RUN.md` in the chapter folder gives GPU guidance.

## Quick Start

```bash
pip install -r requirements.txt

# CPU, no download: synthetic histories with the Yambda schema
python train.py --dataset synthetic --max-seq-len 50 --dim 32 --epochs 4 --num-negatives 32
python baselines/sasrec_baseline.py --dataset synthetic --max-seq-len 50 --dim 32 --epochs 4 --num-negatives 32

# MovieLens-1M (CPU for a one-epoch check, GPU for a full run)
python train.py --dataset movielens --max-seq-len 50 --dim 32 --epochs 1 --num-negatives 32

# Yandex Yambda, first 30 days (GPU)
python train.py --dataset yambda --max-days 30
python baselines/sasrec_baseline.py --dataset yambda --max-days 30
python evaluate.py --dataset yambda --checkpoint outputs/yambda/hstu_best.pt --split test

# Tests
python -m pytest
```

Outputs go to `outputs/<dataset>/` (`--output-dir` moves them):

- `data_stats.json`: split sizes and how often the target already appears in the history
- `train_log.jsonl`: per-epoch training losses and validation metrics
- `hstu_run_metadata.json` / `sasrec_run_metadata.json`: configuration, hardware, dataset statistics, test metrics, completed epochs
- `hstu_best.pt`, `hstu_last.pt` (and `sasrec_*`): checkpoints; `--epoch-snapshots` also keeps one per epoch

Resume an interrupted run with `--resume` (from `*_last.pt`) or `--resume-from <file>`; `--fresh` ignores saved checkpoints.

## Data

| Dataset | Path (under `RECSYS_DATASET_ROOT`, default `<project root>/Dataset`) | Actions | Auxiliary value |
|---------|--------------------------------------------------------------------|---------|-----------------|
| Yandex Yambda | `Yandex/flat/multi_event.parquet` | listen, like, dislike, unlike, undislike | `played_ratio_pct / 100`, clipped to [0, 1]; missing for non-listen events |
| MovieLens-1M | `MovieLens/ml-1m/ratings.dat` | positive (4 to 5 stars), neutral (3), negative (1 to 2) | `(rating - 1) / 4` |
| Synthetic | generated (`synthetic_data.py`) | as Yambda | random in [0, 1] for listens |

Yambda timestamps are seconds since the first event in the dataset, rounded down to a multiple of 5 (the 50M release spans about 301 days), and feed the time buckets unchanged. `--max-days 30` keeps the first 30 days of the timeline (`--use-last-window` in `data_loader.py` keeps the last 30); `data_stats.json` records the time unit, the window and the span actually kept. Sources: [Yambda](https://huggingface.co/datasets/yandex/yambda), [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/).

## Split and Training Protocol

Per user, ordered by time (leave-one-out):

- **Test:** the last event, with every earlier event as history
- **Validation:** the second-to-last event (checkpoint selection and early stopping)
- **Train:** all earlier events, cut into windows of `max_seq_len + 1` events that overlap by one event. The loss is applied at every position of a window, so each next-event transition in the training range is predicted exactly once per epoch.

The loss covers only positions whose next event is real (padding is excluded). The retrieval loss is sampled softmax with 1 positive and `--num-negatives` uniform negatives; a negative that equals the positive is re-drawn, and any that survive are masked out. The auxiliary rating loss (weight 0.2) is a mean squared error on the value column and skips events that have no value.

## Evaluation

The model's representation of the history is scored against every item by cosine similarity. Three views of the same test users are reported:

| Key | Meaning |
|-----|---------|
| `hr@10`, `ndcg@10` | Items in the history window are masked; a target that repeats a history item counts as a miss |
| `unmasked_hr@10`, `unmasked_ndcg@10` | Nothing is masked; repeats can be hit |
| `novel_hr@10`, `novel_ndcg@10` | Masked, restricted to users whose target is not in the history window |

`target_in_window_rate` gives the share of targets that repeat a history item. Music listening repeats a lot, so the masked and unmasked numbers can differ widely on Yambda. MovieLens has no repeats (each movie is rated once), so the masked and novel views are identical there, and the unmasked view is lower only because history items stay in the competition. Ties count against the target.

## Model Details

1. **Fused input:** `Embedding(item) ⊕ Embedding(action) → Linear(2D→D) → SiLU → Linear(D→D)`
2. **Relative attention bias:** a learned bias per relative position (`2L−1` values) plus one per time-gap bucket (under 1 hour, 1 hour to 1 day, 1 to 7 days, 7 to 30 days, over 30 days)
3. **HSTU block:** `LayerNorm → Linear(D→4D) → SiLU`, split into U, V, Q, K; attention weights `SiLU(QKᵀ + bias)`, causal and padding masked, divided by the fixed maximum sequence length; output `X + Dropout(Linear(LayerNorm(AV) ⊙ U))`
4. **Loss:** sampled softmax (weight 1.0) plus rating MSE (weight 0.2)
5. **Training:** Adam with betas (0.9, 0.98), gradient clipping at 1.0, bf16 autocast on GPUs that support it

Compared with Meta's reference implementation, this version uses five hand-set time buckets in place of many log-spaced ones, a single attention head, and padded batches in place of jagged tensors. The fixed attention normaliser means a sequence's output does not depend on the lengths of the other sequences in its batch, which the tests check.

## Code Structure

```
hstu_yambda/
├── data_loader.py         # Yambda loading, per-user splits, training windows, statistics
├── movielens_loader.py    # MovieLens-1M loading with the same splits
├── synthetic_data.py      # Synthetic histories for tests and CPU runs
├── dataset.py             # PyTorch datasets, collate functions, negative sampling
├── model.py               # HSTU encoder and retrieval head
├── training_utils.py      # Loss and weight initialisation
├── train.py               # HSTU training loop
├── evaluate.py            # Full-catalog evaluation; standalone checkpoint evaluation
├── utils.py               # Seeds, ranking metrics, logging helpers
├── checkpointing.py       # Save, load and resume
├── baselines/
│   └── sasrec_baseline.py # SASRec trained and evaluated identically
└── tests/
    └── test_hstu.py
```

## History

This code began as a side experiment outside the companion code and moved here for Chapter 12 after a code review. The review changed how the experiment is run and scored, so results from the earlier experiment are withdrawn and are not comparable with runs of this version:

- Training built one sample per history prefix and applied the loss at every position of each sample, so early transitions were trained many times per epoch; it now uses windows that predict each transition once.
- The loss included positions whose next event was padding, and the rating head was trained toward 0 for events without a played ratio.
- The attention normaliser was the padded batch length, so a sequence's output depended on its batch.
- Sampled negatives could equal the positive.
- Evaluation reported only the masked view, which counts every repeat as a miss.

A later review found that Yambda timestamps had been read as counts of 5-second intervals. `--max-days 30` therefore kept about 6 days, and every time gap the relative time bias saw was five times too long. Timestamps are now read as seconds; MovieLens and synthetic runs are unaffected.

## References

- Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*, ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)
- Kang and McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 (SASRec)
- Agarwal, *HSTU From Scratch in PyTorch*, MLWhiz. [Tutorial](https://www.mlwhiz.com/p/hstu-from-scratch-in-pytorch-a-complete)
- Meta `generative-recommenders` (Apache 2.0): `generative_recommenders/research/modeling/sequential/hstu.py`
- Yandex Yambda dataset paper: [arXiv:2505.22238](https://arxiv.org/abs/2505.22238); data at https://huggingface.co/datasets/yandex/yambda
- Harper and Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 2015
