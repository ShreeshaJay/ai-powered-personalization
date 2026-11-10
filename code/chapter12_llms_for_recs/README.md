# Chapter 12: Generative AI for Recommender Systems

Companion code for Chapter 12. It implements the three ways the chapter places a generative model in a recommender: as the retriever itself (Semantic IDs and TIGER, plus HSTU for generative sequential ranking), as an offline feature generator for a conventional ranker, and as a ranker that reads candidates as text. Every model is open weights and every script runs without a paid API.

The scripts are sized so that a reader can run the whole pipeline on a laptop CPU in minutes (synthetic data) or tens of minutes (a small real sample), and the same commands scale to a single cloud GPU. This README reports no results: the numbers depend on data size, model and hardware, and you should produce your own. See `CLOUD_RUN.md` for the GPU runs.

## Chapter Structure

### Section 12.2: Semantic IDs and Generative Retrieval
- **RQ-VAE** turns each item's text embedding into a short tuple of discrete codes (a Semantic ID), with a final collision level that makes every tuple unique
- **TIGER** reads a session as a sequence of Semantic IDs and generates the next one with beam search constrained to IDs that exist in the catalog
- **Semantic ID analysis:** prefix coherence, behavioural agreement, collisions, cold versus warm items

### Section 12.3: Generative Ranking with HSTU
- From-scratch HSTU (pointwise SiLU attention, relative time and position bias, fused item and action tokens) with a SASRec baseline, on Yandex Yambda or MovieLens-1M
- Lives in `hstu_yambda/` with its own README and tests

### Section 12.4: LLMs as Offline Feature Generators
- **KAR-style generation:** an LLM writes a short knowledge text for every item and an intent sentence for every session (from the history only)
- **Augmented ranker:** an MLP that adds those features step by step to content and history features, with an ablation over four feature sets
- **Cold-start split:** every result is also reported on targets that are rare or absent in training

### Section 12.5: LLMs as Rankers
- **Zero-shot listwise ranking:** the model orders a lettered slate of candidates; the script checks output validity and position bias
- **TALLRec-style LoRA tuning:** a Yes/No engagement prompt scored by P(Yes), compared with the untuned base model on the same slates

### Section 12.6: Evaluating LLM-Based Recommenders
- **Shared candidate slates** with four construction modes, and the baselines every LLM result should be read against

---

## Quick Start

#### 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# CUDA machines that will use --quantize-4bit:
pip install -r requirements-gpu.txt
```

#### 2. Run the synthetic pipeline (no downloads, CPU, a few minutes)

```bash
python run_pipeline.py --profile synthetic
```

This generates a small catalog and sessions with known structure, encodes items with TF-IDF and SVD, and runs every non-LLM step. The LLM feature step uses the `copy-input` control, so no model is downloaded.

#### 3. Run on a small real sample (CPU)

Point the code at the Amazon KDD Cup 2023 files (see Dataset below), then:

```bash
python run_pipeline.py --profile tiny --dry-run    # print the commands
python run_pipeline.py --profile tiny
```

The `tiny` profile keeps 3,000 sessions, downloads `sentence-transformers/all-MiniLM-L6-v2` and `Qwen/Qwen2.5-0.5B-Instruct`, and runs the LLM steps on a handful of sessions so they finish on a CPU. Its numbers show that the code works, nothing more.

#### 4. Larger runs

The `ranker` and `full` profiles are GPU jobs. `CLOUD_RUN.md` gives the GPU class, runtime estimate, inputs and outputs for each step.

Each profile writes to its own work directory (default `outputs/<profile>/`); pass `--work-dir` to put it elsewhere, for example on a mounted drive in Colab. Every script also runs on its own with `--work-dir`.

---

## Data Preparation

```bash
python prepare_data.py --work-dir outputs/tiny --max-sessions 3000 --catalog referenced --extra-random-items 3000
python encode_items.py --work-dir outputs/tiny
```

`prepare_data.py` reads the UK sessions and products, keeps sessions with at least three history items whose items all exist in the product file, and splits them 80/10/10 **at random by session** with a fixed seed. The KDD Cup 2023 session file carries no user IDs and no timestamps, so a time-based or per-user split is impossible; a random session split is the honest option and is what the scripts assume. `session_id` is the row number of the session in `sessions_train.csv`, so it is stable across runs.

`--catalog referenced` keeps only the items that the kept sessions mention, plus `--extra-random-items` distractors, so small runs do not encode the full 500K-item catalog.

Every "cold" or "warm" label in this chapter is computed from **train-split counts only**.

`encode_items.py` writes `embeddings/item_embeddings.npz`. Two encoders are supported:

| Encoder | Use |
|---------|-----|
| `sentence-transformers/all-MiniLM-L6-v2` (default) | Real runs |
| `tfidf-svd` | No download; synthetic runs and slow machines |

**Why not the Chapter 10 fine-tuned encoder?** Chapter 10 fine-tunes MiniLM on next-item pairs drawn from all sessions, which includes the sessions this chapter uses for validation and testing. Using it here would leak test behaviour into the item vectors. The default is the base model; if you want a fine-tuned encoder, fine-tune it on train-split sessions only and pass its name with `--encoder` (the script warns about any non-default encoder). The encoder name is stored in every embedding file, and later scripts refuse to mix encoders.

---

## Section 12.2: Semantic IDs and Generative Retrieval

### Quick Start

```bash
python train_rqvae.py --work-dir outputs/tiny --codebook-size 64
python evaluate_semantic_ids.py --work-dir outputs/tiny
python train_tiger.py --work-dir outputs/tiny --epochs 3 --max-eval-sessions 300
```

### How It Works

**RQ-VAE.** An MLP encoder maps each item embedding to a latent vector. Three residual quantization levels each pick the nearest codeword from a codebook of K entries and pass the remainder on, so the first code is coarse and later codes refine it. Codebooks are updated by exponential moving averages, initialised with k-means on a sample of latents, and codewords whose usage falls below a fraction of the mean usage are reset to random latents from the batch. Gradients reach the encoder through one straight-through estimator on the summed quantization, and a commitment loss keeps encoder outputs near their codewords. `metrics/rqvae.json` reports codes used and perplexity per level, so codebook collapse is visible.

Different items can still receive the same three codes. Following TIGER, a fourth level numbers the items inside each collision group (0, 1, 2, ...), which makes every Semantic ID unique. `evaluate_semantic_ids.py` reports how many items share codes before that suffix, and how coherent each prefix depth is in content and in behaviour (whether a test session's last item and its target share a prefix, against random pairs).

**TIGER.** An encoder-decoder transformer. The encoder reads the most recent history items as Semantic ID tokens (each item a fixed number of tokens, so no separator is needed); the decoder generates the next item's tokens one level at a time with a separate softmax per level. Beam search is constrained by a prefix tree of the catalog's Semantic IDs, so every generated ID maps to a real item. Training stops early on validation loss; the test split is used once.

TIGER is compared on the same test sessions with two baselines, popularity and content kNN (items nearest the mean history embedding). Items already in the history are removed from every method's list, and `target_in_history_rate` reports how often that removal can cost a hit.

---

## Section 12.3: Generative Ranking with HSTU

See `hstu_yambda/README.md`. Quick check on CPU with no data download:

```bash
cd hstu_yambda
python train.py --dataset synthetic --max-seq-len 50 --dim 32 --epochs 4 --num-negatives 32
```

---

## Section 12.4: LLMs as Offline Feature Generators

### Quick Start

```bash
# Real LLM features (GPU recommended; see CLOUD_RUN.md)
python generate_llm_features.py --work-dir outputs/ranker --model Qwen/Qwen2.5-1.5B-Instruct

# CPU check of the generation path on a few items and sessions
python generate_llm_features.py --work-dir outputs/tiny --model Qwen/Qwen2.5-0.5B-Instruct --max-items 40 --max-sessions 40

# Control: embed the prompt's own text instead of an LLM generation
python generate_llm_features.py --work-dir outputs/tiny --backend copy-input

python train_augmented_ranker.py --work-dir outputs/ranker
python evaluate_augmented.py --work-dir outputs/ranker
```

### How It Works

**Generation.** For every item, the LLM writes a two-sentence knowledge text from its title, brand and description. For every session, it writes one sentence on what the shopper is after, from the history titles only; the target is never shown. Both are embedded with the same encoder as the item vectors, so the ranker compares LLM text and item content in one space. Generations are appended to JSONL files and an interrupted run resumes where it stopped.

The ranker refuses to train when fewer than 99% of items or sessions have features, because a feature that exists for a small fraction of rows teaches the model about its presence more than its content. `--allow-partial-coverage` overrides this for experiments; a presence flag for each LLM feature is always part of the input, so missing rows are visible to the model.

The `copy-input` backend embeds the prompt text itself. It checks the pipeline without a GPU, and it is the control that answers whether the LLM adds anything beyond re-encoding the product text.

**Augmented ranker.** A small MLP scores (session, candidate) pairs. Training uses train-split sessions, each with its target as the positive, hard negatives (items nearest the mean history embedding that are neither the target nor in the history) and random negatives; early stopping uses validation-split sessions. Four feature sets are trained, each adding to the previous one:

| Feature set | Adds |
|-------------|------|
| `content` | Candidate item embedding; its cosine to the last history item |
| `content_history` | Mean history embedding, its elementwise product with the candidate, and their cosine |
| `content_history_llm_item` | Candidate's LLM knowledge embedding, its cosines to the history's knowledge mean and content mean, and a presence flag |
| `full` | Session's LLM intent embedding, its cosines to the candidate's content and knowledge embeddings, and a presence flag |

The history features are a mean of item embeddings, a simple sequence summary; they are not a trained sequential model. `evaluate_augmented.py` ranks the shared test slates (Section 12.6) with every trained feature set, alongside the slate baselines, for all sessions and for cold and warm targets.

---

## Section 12.5: LLMs as Rankers

### Quick Start

```bash
python build_slates.py --work-dir outputs/tiny --splits train val test --max-sessions 300
python evaluate_zeroshot_llm.py --work-dir outputs/tiny --model Qwen/Qwen2.5-0.5B-Instruct --max-sessions 20
python train_tallrec.py --work-dir outputs/tiny --model Qwen/Qwen2.5-0.5B-Instruct --max-train-sessions 32 --max-eval-sessions 10
```

### How It Works

**Zero-shot listwise ranking.** The prompt lists the session's recent titles and the slate's candidates under letters, and asks for a line starting `Ranking:`. Generation is greedy. The parser reads only that line (or the first line if the marker is missing); candidates the model leaves out are appended in a seeded random order, so a model is never rewarded for silence. The metrics file reports how many answers were complete permutations, how many had duplicate or out-of-range letters, and how often the first choice was the first displayed candidate, against 1/N for a ranker that ignores position.

**TALLRec-style tuning.** Each training session gives one positive (its true next item, answer "Yes") and several negatives from its own train-split slate (answer "No"), so the negatives look like the candidates seen at test time. LoRA adapters are trained on the attention projections; the loss covers only the answer tokens. If a prompt is too long, the oldest history items are dropped; an example is never cut through its answer. Evaluation scores each candidate by P(Yes) / (P(Yes) + P(No)) at the first answer token, once with the adapter disabled (the base model) and once enabled, on the same slates as the zero-shot script.

### Models and Licences

| Model | Licence | Role |
|-------|---------|------|
| `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | CPU default |
| `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | GPU default |
| `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | Larger GPU option |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen Research License | Not a default; check its terms before use |

Any causal LM with a chat template can be passed with `--model`. Check the licence on its model card first. No script sets `trust_remote_code`.

---

## Section 12.6: Evaluating LLM-Based Recommenders

`build_slates.py` writes one slate file per split, mode and size, and every slate-based script reads the same file, so all methods are compared on identical candidates.

| Mode | Candidates | Contains the target? |
|------|------------|----------------------|
| `random` | Target plus uniformly sampled items | Always |
| `popular` | Target plus items sampled by training popularity | Always |
| `retrieved` | Nearest items to the session's mean history embedding | Only when retrieval finds it |
| `injected` | The retrieved slate, with the target swapped in for the last candidate when retrieval missed it | Always (diagnostic only) |

`retrieved` is the default and the closest to a real two-stage system: the target is never inserted, `target_in_slate_rate` reports how often retrieval found it, and metrics are reported both over all sessions and over sessions whose slate holds the target. `injected` reproduces a common mistake. An injected target is, by construction, less similar to the history than every other candidate, so a ranker that follows content similarity scores below random for reasons unrelated to its quality. `random` and `popular` slates are easy for the opposite reason: the target is usually the only candidate that resembles the history.

Every slate result is printed next to four baselines on the same slates: display order, a fresh random order, training popularity, and similarity to the mean history embedding. Candidates are stored in a shuffled display order, so the display-order baseline measures position effects. On `retrieved` slates the history-similarity baseline replays the retriever's own ordering, which makes it a strong reference: a ranker that cannot beat it adds nothing over the retrieval stage.

---

## Dataset

### Amazon KDD Cup 2023 (Sections 12.2, 12.4 to 12.6)

- **Domain:** E-commerce sessions, UK locale
- **Files:** `sessions_train.csv` and `products_train.csv`
- **Text fields:** Title, brand, description
- **Path:** `Dataset/Amazon KDD 2023/` under the project root, or set `RECSYS_DATASET_ROOT` to the folder that contains `Amazon KDD 2023/`
- **Source:** https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge

### Yandex Yambda and MovieLens-1M (Section 12.3)

See `hstu_yambda/README.md`.

---

## Outputs

```
<work_dir>/
├── data/          sessions.parquet, items.parquet, prepare_meta.json
├── embeddings/    item embeddings, Semantic IDs, LLM feature embeddings (.npz)
├── slates/        {split}_{mode}_n{size}.jsonl
├── llm/           raw generations (.jsonl, resumable)
├── models/        rqvae.pt, tiger.pt, ranker_*.pt, tallrec_<model>/
└── metrics/       one JSON per script, each with the arguments and environment it ran with
```

---

## Code Structure

```
chapter12_llms_for_recs/
├── config.py                    # Paths, seeds, default hyperparameters (dataclasses)
├── run_pipeline.py              # Runs the scripts in order for a profile
├── prepare_data.py              # Sessions, items, seeded session split
├── encode_items.py              # Item text embeddings
├── train_rqvae.py               # ★ 12.2: RQ-VAE -> Semantic IDs
├── evaluate_semantic_ids.py     # ★ 12.2: Semantic ID quality
├── train_tiger.py               # ★ 12.2: TIGER + baselines
├── generate_llm_features.py     # ★ 12.4: offline LLM item knowledge and session intent
├── train_augmented_ranker.py    # ★ 12.4: feature-set ablation
├── evaluate_augmented.py        # ★ 12.4: ranker on shared slates, cold and warm
├── evaluate_zeroshot_llm.py     # ★ 12.5: zero-shot listwise ranking
├── train_tallrec.py             # ★ 12.5: LoRA Yes/No tuning
├── build_slates.py              # ★ 12.6: shared candidate slates + baselines
├── data/
│   ├── amazon_session_loader.py # KDD Cup loading, filtering, splits, PreparedData
│   ├── semantic_id_dataset.py   # Semantic ID tokens, prefix tree, TIGER dataset
│   └── synthetic.py             # Synthetic catalog and sessions
├── models/
│   ├── rqvae.py                 # Residual quantizer with EMA codebooks
│   ├── tiger.py                 # Encoder-decoder with constrained beam search
│   └── augmented_ranker.py      # MLP over frozen feature tables
├── utils/                       # metrics, slates, prompts, LLM helpers, encoders, search
├── tests/                       # pytest suite (python -m pytest)
├── hstu_yambda/                 # ★ 12.3: HSTU and SASRec (own README and tests)
├── docs/
│   └── Chapter12_Design_Notes.md
└── CLOUD_RUN.md                 # GPU runs: commands, GPU class, runtime, inputs, outputs
```

---

## Configuration

Defaults live in dataclasses in `config.py` (`PrepareConfig`, `RQVAEConfig`, `TIGERConfig`, `SlateConfig`, `ZeroShotLLMConfig`, `TALLRecConfig`, `LLMFeatureConfig`, `AugmentedRankerConfig`), and every default can be overridden on the command line. Two environment variables move data and outputs:

| Variable | Default |
|----------|---------|
| `RECSYS_DATASET_ROOT` | `<project root>/Dataset` |
| `CH12_WORK_DIR` | `outputs/default` inside this folder |

All randomness is seeded (default seed 42).

---

## Evaluation Metrics

- **Recall@K:** fraction of sessions whose true next item is in the top K
- **NDCG@K:** rank-discounted hit, 1 / log2(rank + 2) when the item is in the top K
- **MRR:** mean of 1 / (rank + 1)

Ties count against the target. Each slate metrics file also gives the random-ranker expectation for its slate size.

---

## Testing

```bash
python -m pytest -m "not slow"     # fast suite
python -m pytest                   # also runs a test that builds a tiny transformers model
cd hstu_yambda && python -m pytest
```

The tests use known answers: hand-computed metrics, beam-search scores checked against teacher forcing, label masks that cover only answer tokens, slates that never contain the target unless the mode says so, and features aligned by session ID.

---

## References

### Semantic IDs and Generative Retrieval
- **TIGER:** Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023
- **RQ-VAE:** Lee et al., "Autoregressive Image Generation using Residual Quantization", CVPR 2022
- **VQ-VAE:** van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
- **HSTU:** Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations", ICML 2024

### LLMs as Feature Generators and Rankers
- **KAR:** Xi et al., "Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models", RecSys 2024
- **LLMRank:** Hou et al., "Large Language Models are Zero-Shot Rankers for Recommender Systems", ECIR 2024
- **TALLRec:** Bao et al., "TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation", RecSys 2023
- **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

### Models and Data
- **Qwen2.5:** https://huggingface.co/Qwen
- **all-MiniLM-L6-v2:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Amazon KDD Cup 2023:** https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge

---

## Questions or Issues?

See `docs/Chapter12_Design_Notes.md` for design decisions, and `CLOUD_RUN.md` for GPU runs.
