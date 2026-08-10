# Chapter 10: Item Embeddings

This chapter builds **reusable item embedding models** that can be used as features in downstream recommender systems, adtech, and personalization use cases. Instead of each model encoding item metadata from scratch, ML teams can directly leverage pre-trained item embeddings.

## Chapter Structure

### Section 10.1: Pre-trained Text Encoder Baselines ✅
- **MIND (News):** Zero-shot encoding with SBERT (MiniLM, mpnet). Evaluated via category retrieval, co-click retrieval, qualitative NN.
- **Amazon KDD 2023 (E-Commerce):** Zero-shot encoding with SBERT + RexBERT (domain-specialized). Evaluated via session co-engagement, brand coherence, qualitative NN.
- **Homework (IJCAI):** Demonstrates why zero-shot fails on hashed/anonymized metadata.

### Section 10.2: Item2Vec — Collaborative Embeddings from Sequences ✅
- Word2Vec (Skip-gram) on user sessions — learns purely from behavioral co-occurrence
- **Amazon KDD (primary):** Compare behavioral vs. content-based embeddings
- **Yandex Yambda (secondary):** Music listening sessions, compare vs. audio CNN embeddings

### Section 10.3: Contrastive Fine-Tuning ✅
- Fine-tune SBERT with MultipleNegativesRankingLoss on behavioral pairs
- **Amazon KDD (primary):** next-item pairs from 1.18M sessions
- **MIND (secondary):** co-click pairs from user reading histories
- 3-way comparison: zero-shot (10.1) vs. Item2Vec (10.2) vs. fine-tuned (10.3)

### Section 10.4: Multi-Modal Content Fusion ✅
- **Music4all:** Fuse audio (i-vectors), lyrics (Word2Vec), genre (TF-IDF) into unified embeddings
- PCA baseline (concatenate + reduce) vs. CLIP-style contrastive alignment (audio ↔ lyrics)
- Evaluated via co-listen retrieval against 109K tracks
- Key finding: Genre alone outperforms all fusion methods — more modalities ≠ better

### Appendix: Superlinked Multi-Modal Integration
- Moved to `appendix_superlinked/` — Superlinked is a startup library; core concepts are durable but API may change
- Demonstrates weighted concatenation of text + categorical spaces with query-time weight tuning
- Under the hood, Superlinked performs the same weighted concatenation you could do with numpy
- See `appendix_superlinked/Appendix_Superlinked_Design_Notes.md` for details

---

## Section 10.1: Pre-trained Text Encoder Baselines

### Quick Start

#### 1. Install Dependencies

```bash
pip install sentence-transformers transformers pandas numpy faiss-cpu tqdm
# Or faiss-gpu if you have CUDA
```

#### 2. Run the MIND Evaluation (News Domain)

```bash
# Compare SBERT models on MIND news articles
python evaluate_mind.py --compare_models

# Use a sample for faster experimentation
python evaluate_mind.py --compare_models --sample_size 5000
```

#### 3. Run the Amazon KDD Evaluation (E-Commerce Domain)

```bash
# Compare SBERT + RexBERT on Amazon product data
python evaluate_amazon.py --compare_models

# Use a smaller sample for faster experimentation
python evaluate_amazon.py --compare_models --sample_size 5000
```

#### 4. Generate and Save Embeddings

```bash
python encode_items.py --dataset mind --model sentence-transformers/all-MiniLM-L6-v2
```

#### 5. Homework: IJCAI Bi-Encoder Evaluation

```bash
# See why zero-shot fails on hashed/anonymized metadata
python evaluate_biencoder.py --compare_models --sample_size 10000
```

---

## Section 10.2: Item2Vec — Collaborative Embeddings

### Quick Start

#### 1. Install Additional Dependencies

```bash
pip install gensim>=4.3.0 pyarrow>=12.0.0
```

#### 2. Train Item2Vec on Amazon KDD Sessions

```bash
# Train Word2Vec (Skip-gram) on shopping sessions
python item2vec.py --dataset amazon

# Custom hyperparameters
python item2vec.py --dataset amazon --dim 256 --window 10 --epochs 20
```

#### 3. Train on Yandex Music Sessions

```bash
python item2vec.py --dataset yandex
```

#### 4. Evaluate: Zero-Shot vs Item2Vec on Amazon KDD

```bash
# Compare zero-shot (10.1) vs Item2Vec (10.2)
python evaluate_item2vec.py

# Item2Vec alone
python evaluate_item2vec.py --item2vec_only

# Yandex evaluation (standalone)
python evaluate_item2vec.py --dataset yandex
```

### How It Works

**Item2Vec** applies Word2Vec (Skip-gram with negative sampling) to item
sequences. Each user session is treated as a "sentence" and each item as
a "word". The skip-gram objective learns to predict context items given a
target item.

**Key properties:**
- Purely collaborative — learns from co-occurrence patterns, not metadata
- Covers only items that appear frequently enough in sessions
- `min_count` derived from the 5th percentile of the item frequency distribution
- Captures browsing/purchasing patterns that text similarity misses
- Complementary to content-based embeddings (explored in Chapter 11)

**Evaluation reports two comparison modes:**
1. **Full catalog:** Each model evaluated on its own item set (text models
   cover 100% of 500K products; Item2Vec covers frequent items only)
2. **Common subset:** All models restricted to the Item2Vec vocabulary
   for an apples-to-apples head-to-head comparison

---

## Section 10.3: Contrastive Fine-Tuning

### Quick Start

#### 1. Fine-tune on Amazon KDD (Primary)

```bash
# Train with MultipleNegativesRankingLoss on next-item pairs
python finetune_contrastive.py --dataset amazon

# Custom hyperparameters
python finetune_contrastive.py --dataset amazon --epochs 5 --batch_size 256
```

#### 2. Fine-tune on MIND (Secondary)

```bash
python finetune_contrastive.py --dataset mind
```

#### 3. Evaluate: Zero-Shot vs Fine-Tuned

```bash
# Amazon KDD — direct comparison on next-item retrieval
python evaluate_finetuned.py --dataset amazon

# MIND — category retrieval + co-click retrieval
python evaluate_finetuned.py --dataset mind
```

#### 4. Re-run 3-Way Comparison (after fine-tuning)

```bash
# Now includes fine-tuned embeddings alongside zero-shot and Item2Vec
python evaluate_item2vec.py
```

### How It Works

**Training objective:** MultipleNegativesRankingLoss (MNR) uses in-batch
negatives — every other item in a batch of size B serves as a negative,
yielding B*(B-1) contrastive pairs per batch for free.

**Positive pairs:**
- Amazon KDD: (last viewed item, next engaged item) from session data
- MIND: (article A, article B) co-clicked by the same user

**Why this matters:** Section 10.1 showed that RexBERT (pre-trained on
2.3T e-commerce tokens) underperforms general-purpose SBERT because it
lacks a contrastive training objective. Section 10.3 demonstrates what
happens when you add that objective using task-specific behavioral data.

---

## Section 10.4: Multi-Modal Content Fusion

### Quick Start

#### 1. Install Additional Dependencies

```bash
pip install scikit-learn  # PCA baseline (likely already installed)
```

#### 2. Run PCA + CLIP Training

```bash
# Both PCA baseline and CLIP-style contrastive training
python multimodal_fusion.py --mode both

# PCA baseline only (fast, no GPU needed)
python multimodal_fusion.py --mode pca

# CLIP only (requires GPU for reasonable speed)
python multimodal_fusion.py --mode clip
```

#### 3. Evaluate All Modality Combinations

```bash
# Compare audio, lyrics, genre, PCA, and CLIP embeddings
python evaluate_multimodal.py

# Increase evaluation queries
python evaluate_multimodal.py --max-queries 10000
```

### How It Works

**PCA Fusion (Approach A):** Concatenate audio (100d) + lyrics (300d) + genre (685d) = 1,085d, then apply PCA to 128d. No learning — tests whether raw concatenation captures useful signal.

**CLIP-Style Alignment (Approach B):** Train projection heads to align audio and lyrics features in a shared 128-dim space using InfoNCE contrastive loss. For each track, (audio_i, lyrics_i) is a positive pair; other tracks in the batch are negatives.

**Key insight:** Multi-modal fusion is not a magic bullet. If one modality dominates (genre → co-listen), naive fusion dilutes it. The general principle — applicable to any retrieval system including **hybrid search** (dense + sparse) — is: keep per-modality vectors separate, L2-normalize each, and weight at query time. PCA destroys this flexibility; weighted concatenation preserves it. See Design Notes for a detailed treatment connecting this to production hybrid search systems (Vespa, Weaviate, Pinecone, etc.).

---

## Datasets

### MIND (Microsoft News Dataset)

- **Domain:** News recommendation
- **Items:** ~50K+ news articles
- **Text Fields:** Title, abstract, category, subcategory, entity annotations
- **Why Useful:** Rich free-text metadata ideal for text encoders
- **Path:** `Dataset/Microsoft News/MINDsmall_train/`

**Example Item Text:**
```
Apple announces new iPhone with improved camera [SEP] The tech giant unveiled its latest smartphone featuring enhanced photography capabilities and longer battery life. [SEP] Technology smartphones
```

### Amazon KDD Cup 2023 (Multilingual Shopping Sessions)

- **Domain:** E-commerce (Amazon)
- **Items:** ~500K English (UK) products with rich metadata
- **Text Fields:** Title, brand, description, color, material, price
- **Session Data:** 1.18M English user sessions (avg 3.9 items/session)
- **Why Useful:** Real e-commerce text for meaningful RexBERT comparison + session co-engagement for behavioral evaluation
- **Path:** `Dataset/Amazon KDD 2023/`

**Example Item Text:**
```
SOCHOW Sherpa Fleece Throw Blanket, Double-Sided Super Soft Luxurious Plush Blanket [SEP] SOCHOW [SEP] The sherpa throw blanket is available in a variety of colors...
```

### Music4all-Onion (Multi-Modal Music)

- **Domain:** Music recommendation
- **Items:** ~109K tracks with multi-modal pre-extracted features
- **Modalities:** Audio i-vectors (100d), Lyrics Word2Vec (300d), Genre TF-IDF (685d)
- **Interactions:** 50M listening records from 119K users
- **Why Useful:** True multi-modal fusion — audio, text, and categorical features for the same items
- **Path:** `Dataset/music4all-onion/`

### IJCAI-18 CVR (Alibaba Sponsored Search)

- **Domain:** E-commerce / sponsored search
- **Items:** ~10K products
- **Metadata:** Structured fields (category hierarchy, properties, brand, price level, sales level, city)
- **Why Useful:** Demonstrates text serialization from structured data (most production catalogs)
- **Path:** `Dataset/Alibaba IJCAI 18 Sponsored Search CVR/`
- **Special:** Has `predict_category_property` field for query-side intent (bi-encoder evaluation)

**Example Item Text (serialized from structured fields):**
```
Category: clothing > women > dresses | Properties: cotton, summer, casual | Brand: brand_1234 | Price: medium | Sales: high | City: city_88
```

---

## Key Concepts

### 1. Zero-Shot vs. Fine-Tuned Embeddings

**Zero-shot (Section 10.1):**
- Use pre-trained encoder directly on item text
- No training required
- Fast to deploy
- Limited domain adaptation

**Fine-tuned (Section 10.3):**
- Adapt encoder to domain-specific data
- Better performance on retrieval
- Requires training data and compute

### 2. Text Serialization for Structured Metadata

Most e-commerce catalogs have **structured** metadata (categories, brands, prices), not free-text descriptions. We serialize structure into pseudo-text:

```python
template = "Category: {categories} | Properties: {properties} | Brand: {brand} | Price: {price_level}"
```

This lets us use text encoders on structured data.

### 3. Bi-Encoder Architecture

**Query Encoder:** Encodes user intent (search query, predicted categories)
**Item Encoder:** Encodes item metadata

Both use the same underlying model. At retrieval time:
1. Encode query → query_embedding
2. Find nearest items via similarity(query_embedding, item_embeddings)
3. Evaluate with Precision@K, NDCG@K against ground truth (conversions)

---

## Code Structure

```
chapter10_item_embeddings/
├── config.py                    # Configuration for all sections
├── data/
│   ├── mind_dataset.py          # MIND news loader
│   ├── amazon_kdd_dataset.py    # Amazon KDD 2023 loader
│   ├── yandex_dataset.py        # Yandex Yambda music loader
│   ├── music4all_dataset.py     # Music4all-Onion multi-modal loader
│   └── ijcai_cvr_dataset.py     # IJCAI CVR loader
├── models/
│   └── text_encoders.py         # SBERT, RexBERT (with mean-pooling wrapper)
├── utils/
│   ├── metrics.py               # Precision@K, Recall@K, NDCG@K, MRR
│   └── faiss_index.py           # FAISS index for fast search
├── evaluate_mind.py             # ★ 10.1 evaluation — News (SBERT only)
├── evaluate_amazon.py           # ★ 10.1 evaluation — E-Commerce (SBERT + RexBERT)
├── item2vec.py                  # ★ 10.2 training — Word2Vec on sessions
├── evaluate_item2vec.py         # ★ 10.2 evaluation — behavioral vs content comparison
├── finetune_contrastive.py      # ★ 10.3 training — Contrastive fine-tuning
├── evaluate_finetuned.py        # ★ 10.3 evaluation — Zero-shot vs fine-tuned
├── encode_items.py              # Generate and save item embeddings
├── multimodal_fusion.py         # ★ 10.4 training — PCA + CLIP fusion
├── evaluate_multimodal.py       # ★ 10.4 evaluation — co-listen retrieval
├── evaluate_biencoder.py        # Homework: IJCAI bi-encoder evaluation
├── outputs/
│   ├── embeddings/              # Saved embeddings (.npz)
│   └── metrics/                 # Evaluation results (.json, .csv)
├── cache/                       # Model and data cache
└── docs/
    └── Chapter10_Design_Notes.md
```

---

## Configuration

All settings are centralized in `config.py`:

```python
from config import (
    DEFAULT_TEXT_ENCODER,      # Encoder settings
    DEFAULT_MIND_CONFIG,       # MIND dataset config
    DEFAULT_IJCAI_CONFIG,      # IJCAI dataset config
    DEFAULT_BIENCODER_EVAL     # Evaluation config
)
```

**Key Parameters:**

- `model_name`: HuggingFace model identifier (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- `max_seq_length`: Maximum sequence length (default: 128)
- `batch_size`: Encoding batch size (default: 64)
- `normalize_embeddings`: L2-normalize output (default: True)
- `device`: "cuda" or "cpu" (auto-detect if None)

---

## Evaluation Metrics

### Precision@K
```
Precision@K = (# relevant items in top-K) / K
```
Measures the fraction of retrieved items that are relevant.

### Recall@K
```
Recall@K = (# relevant items in top-K) / (total # relevant items)
```
Measures the fraction of all relevant items that were retrieved.

### NDCG@K (Normalized Discounted Cumulative Gain)
```
NDCG@K = DCG@K / IDCG@K
```
Measures ranking quality with position-based discounting. Higher-ranked relevant items contribute more.

### MRR (Mean Reciprocal Rank)
```
MRR = 1 / (rank of first relevant item)
```
Measures how quickly the first relevant item appears.

---

## Results (Section 10.1)

### MIND Evaluation (News Domain — 51K articles)

**Category Retrieval** — Can nearest neighbors recover same-topic articles?

| Model | Dim | Cat P@5 | Cat P@10 | Cat MRR |
|-------|-----|---------|----------|---------|
| all-MiniLM-L6-v2 | 384 | **0.769** | **0.762** | **0.850** |
| all-mpnet-base-v2 | 768 | 0.735 | 0.724 | 0.828 |

**Co-Click Retrieval** — Are behaviorally related articles closer in embedding space?

| Model | CoClick P@5 | CoClick R@50 | CoClick MRR |
|-------|------------|-------------|-------------|
| all-MiniLM-L6-v2 | 0.0067 | 0.159 | 0.025 |
| all-mpnet-base-v2 | 0.0063 | 0.154 | 0.026 |

### Amazon KDD Evaluation (E-Commerce — 500K products, 1.18M sessions)

**Next-Item Retrieval** — Given the last viewed item, is the actual next item nearby?
Respects temporal ordering within sessions (20K queries).

| Model | Dim | Domain | P@1 | P@5 | MRR | R@50 |
|-------|-----|--------|-----|-----|-----|------|
| all-MiniLM-L6-v2 | 384 | General | **0.222** | **0.108** | **0.309** | **0.619** |
| all-mpnet-base-v2 | 768 | General | 0.223 | 0.109 | 0.310 | 0.622 |
| RexBERT-base | 768 | E-Commerce | 0.206 | 0.095 | 0.278 | 0.515 |

**Key findings:**
- MiniLM and mpnet are essentially tied despite 2x dimension difference
- RexBERT (e-commerce domain MLM) underperforms purpose-built sentence encoders by ~3 points MRR — motivates fine-tuning in Section 10.3
- P@1 of ~22% across a 500K catalog with zero training is a strong baseline

### IJCAI Evaluation (Homework)

Zero-shot encoders score near zero on IJCAI because category/property IDs are
hashed 19-digit integers. Sub-word tokenization destroys the identity of these
numbers, making all items appear equally (dis)similar. See `evaluate_biencoder.py`
for details and diagnostic output.

## Results (Section 10.2)

### Amazon KDD: Zero-Shot vs Item2Vec

> Run `python item2vec.py --dataset amazon` then `python evaluate_item2vec.py`

**Full Catalog** (each model uses all items it covers):

| Model | Dim | Coverage | P@1 | P@5 | MRR | R@50 |
|-------|-----|----------|-----|-----|-----|------|
| Zero-shot (all-MiniLM-L6-v2) | 384 | 500K | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Item2Vec (Skip-gram) | 128 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Common Subset** (items in both models' vocabularies — apples-to-apples):

| Model | P@1 | P@5 | MRR | R@50 |
|-------|-----|-----|-----|------|
| Zero-shot | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Item2Vec | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Expected findings (to verify):**
- Item2Vec should excel on the *common subset* (frequent items with rich behavioral signal)
- Text models should dominate on *full catalog* (they cover all items, including long-tail)
- The complementary nature of content vs. behavior motivates fusion in Chapter 11

## Results (Section 10.3)

### Amazon KDD: Zero-Shot vs Fine-Tuned (20K queries, 500K products)

| Model | P@1 | P@5 | P@10 | MRR | R@20 | R@50 |
|-------|-----|-----|------|-----|------|------|
| zero-shot (all-MiniLM-L6-v2) | 0.222 | 0.108 | 0.068 | 0.309 | 0.499 | 0.619 |
| **fine-tuned (MNR, 450K pairs)** | **0.224** | **0.112** | **0.073** | **0.319** | **0.527** | **0.668** |
| Delta | +0.002 | +0.004 | +0.004 | +0.010 | +0.028 | +0.049 |

**Key findings:**
- Consistent improvement across all metrics, largest at higher recall (+4.9 points R@50)
- MRR improved by 1 point — the fine-tuned model ranks the true next item ~1 position higher
- Best checkpoint was at epoch 1 (step 3500); epochs 2-3 showed mild overfitting
- Training loss decreased steadily: 1.61 → 1.47 → 1.41 over 3 epochs

**Why improvements are modest (pedagogical note):**
1. The zero-shot baseline was already strong — SBERT's general text understanding captures product similarity well
2. Next-item pairs encode *browsing* behavior, not deep semantic similarity (phone case → screen protector)
3. Each item appears in only ~1-2 training pairs on average — more interaction data would help
4. Real production systems typically see larger gains because they fine-tune on orders of magnitude more data

### 3-Way Comparison (after completing Section 10.3)

> Re-run `python evaluate_item2vec.py` to include fine-tuned embeddings

| Model | Dim | Coverage | P@1 | P@5 | MRR | R@50 |
|-------|-----|----------|-----|-----|-----|------|
| Zero-shot (all-MiniLM-L6-v2) | 384 | 500K | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Item2Vec (Skip-gram) | 128 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Fine-tuned (MNR) | 384 | 500K | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### MIND: Zero-Shot vs Fine-Tuned

> Run `python finetune_contrastive.py --dataset mind` then `python evaluate_finetuned.py --dataset mind`

| Model | Cat P@5 | Cat MRR | CoClick P@5 | CoClick MRR |
|-------|---------|---------|-------------|-------------|
| zero-shot (all-MiniLM-L6-v2) | 0.769 | 0.850 | 0.0067 | 0.025 |
| fine-tuned (MNR on co-clicks) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Results (Section 10.4)

### Music4all: Co-Listen Retrieval (5K queries, 109K tracks)

| Method | Dim | P@1 | P@10 | MRR |
|--------|-----|-----|------|-----|
| **Genre (TF-IDF)** | 685 | **0.219** | **0.197** | **0.358** |
| PCA fusion (all 3) | 128 | 0.191 | 0.140 | 0.318 |
| Audio (i-vectors) | 100 | 0.136 | 0.095 | 0.243 |
| CLIP fusion (audio ↔ lyrics) | 128 | 0.109 | 0.096 | 0.228 |
| Lyrics (Word2Vec) | 300 | 0.066 | 0.060 | 0.153 |

**Key findings:**
- Genre TF-IDF alone outperforms every fusion method — co-listening is primarily a genre-level signal
- PCA fusion *hurts* vs. genre alone (MRR 0.318 vs 0.358) — weaker modalities dilute the strongest one
- CLIP alignment learns real audio↔lyrics correspondences (loss well below random), but those correspondences don't predict co-listening
- Lyrics have near-zero predictive power for co-listen — users choose music by genre and sound, not lyrical themes
- **Lesson:** Always evaluate single modalities before fusing. Task-aware fusion (Chapter 11) is needed to weight modalities appropriately
- **Broader principle:** Multi-modal fusion, hybrid search (dense + sparse), and any combination of heterogeneous embeddings all reduce to the same operation: weighted concatenation of normalized sub-vectors. Keep them separate for query-time flexibility.

---

## Testing

Each module has a `if __name__ == "__main__"` block for standalone testing:

```bash
# Test data loaders
python data/mind_dataset.py
python data/ijcai_cvr_dataset.py
python data/amazon_kdd_dataset.py
python data/yandex_dataset.py
python data/music4all_dataset.py

# Test text encoder
python models/text_encoders.py

# Test metrics
python utils/metrics.py

# Test FAISS index
python utils/faiss_index.py
```

---

## Next Steps

1. ✅ **Section 10.1:** Pre-trained text encoder baselines (content only)
2. ✅ **Section 10.2:** Item2Vec on Amazon KDD sessions + Yandex music (behavior only)
3. ✅ **Section 10.3:** Contrastive fine-tuning (content + behavior)
4. ✅ **Section 10.4:** Multi-modal content fusion on Music4all (PCA + CLIP)
5. **Appendix:** Superlinked integration (see `appendix_superlinked/`)

---

## References

- **MIND Dataset:** https://msnews.github.io/
- **IJCAI-18 CVR:** https://tianchi.aliyun.com/dataset/147588
- **Amazon KDD Cup 2023:** https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge
- **Sentence-BERT:** Reimers & Gurevych, EMNLP 2019
- **MultipleNegativesRankingLoss:** Henderson et al., 2017
- **Bi-encoders for Retrieval:** Karpukhin et al., EMNLP 2020 (DPR)
- **Item2Vec:** Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative Filtering", MLSP 2016
- **Word2Vec:** Mikolov et al., "Distributed Representations of Words and Phrases", NeurIPS 2013
- **Yandex Yambda:** https://huggingface.co/datasets/yandex/yambda
- **Music4all-Onion:** Santana et al., "Music4all-Onion", ACM ICMR 2020
- **CLIP:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021

---

## Questions or Issues?

Check `docs/Chapter10_Design_Notes.md` for detailed design decisions and troubleshooting.
