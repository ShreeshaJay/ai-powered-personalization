# Appendix: Hybrid Search with Vector Databases

This appendix compares **three retrieval paradigms** — lexical (BM25), learned sparse (SPLADE), and dense semantic (SBERT + FAISS) — on the Amazon ESCI Shopping Queries Dataset (1.2M US products, ~97K queries). It then demonstrates how to combine them via **hybrid fusion** (Reciprocal Rank Fusion, weighted score fusion) and **BM25 re-ranking** of neural candidates.

## Why This Appendix?

Chapters 5 and 10 built retrieval and embedding systems using dense bi-encoders. But production search systems increasingly use **hybrid search** — combining lexical and semantic methods — because:

1. **BM25 excels at exact keyword matching** (brand names, model numbers, SKUs)
2. **Dense models capture semantic similarity** ("wireless headphones" → "bluetooth earbuds")
3. **SPLADE bridges both worlds** — learned sparse vectors that expand queries with semantically related vocabulary terms while retaining sparse-space efficiency

The key question: *when does each method help, and does combining them beat any single method?*

---

## Dataset: Amazon ESCI (Shopping Queries Dataset)

The [Amazon ESCI dataset](https://github.com/amazon-science/esci-data) provides real Amazon shopper queries with 4-level graded relevance annotations:

| Label | Meaning | Gain (ours) | Gain (Qdrant) |
|-------|---------|:-----------:|:-------------:|
| **E** (Exact) | Satisfies the query's core intent | 3 | 1.0 |
| **S** (Substitute) | Close alternative, not an exact match | 2 | 0.7 |
| **C** (Complement) | Related/complementary product | 1 | 0.5 |
| **I** (Irrelevant) | Not relevant to the query | 0 | 0.0 |

**Dataset statistics (US locale):**
- ~1.2M products with title, bullet points, brand, color, description
- ~97K unique queries in the training split
- ~16 labeled products per query (sparse coverage by design — ESCI only labels products that a search system surfaced)

### The Label Coverage Problem

ESCI provides only ~16 relevance judgments per query against a 1.2M product catalog. When retrieving top-50 from the full catalog, **~84% of retrieved products are unlabeled** — we don't know if they're relevant or not. This fundamentally limits full-catalog evaluation and motivated our dual evaluation strategy (see Evaluation Design below).

---

## Methods

### BM25 (Lexical Baseline)

- **Library:** `bm25s` — pre-computes BM25 scores into scipy sparse matrices at index time
- **Variant:** Lucene-compatible scoring (`k1=1.5, b=0.75`)
- **Strengths:** Exact keyword matching, no training required, extremely fast
- **Weaknesses:** No semantic understanding — "laptop" won't match "notebook computer"

### SPLADE (Learned Sparse)

SPLADE uses a BERT masked language model to produce sparse vocabulary-sized vectors. The model learns to **expand** queries and documents with semantically related terms while keeping representations sparse.

**Key insight:** For the query "wireless headphones", SPLADE activates vocabulary tokens like "bluetooth", "earbuds", "audio" — terms that aren't in the query but are semantically relevant. This gives SPLADE both exact matching AND semantic matching in sparse space.

We compare **three SPLADE variants:**

| Variant | Model | License | Backend | Notes |
|---------|-------|---------|---------|-------|
| **Prithivi** | `prithivida/Splade_PP_en_v2` | Apache 2.0 | AutoModelForMaskedLM | General-purpose, permissive license |
| **Qdrant ESCI** | `thierrydamiba/splade-ecommerce-esci` | Apache 2.0 | SparseEncoder (ST v5) | Fine-tuned on ESCI with E+S as positives |
| **Naver** | `naver/splade-cocondenser-ensembledistil` | CC-BY-NC-SA-4.0 | SparseEncoder (ST v5) | Trained on MS-MARCO, non-commercial |

**Why three variants?**
- **Prithivi** is the default because it uses the permissive Apache 2.0 license and loads with standard `AutoModelForMaskedLM` (no special dependencies)
- **Qdrant ESCI** was fine-tuned specifically on the ESCI dataset by the Qdrant team ([tutorial](https://qdrant.tech/articles/sparse-embeddings-ecommerce-part-1/)), providing a domain-specific comparison point
- **Naver** is the original SPLADE team's model trained on MS-MARCO, representing the general-domain learned sparse state-of-the-art

**Backend difference:** Prithivi uses `transformers.AutoModelForMaskedLM` directly. Qdrant and Naver models require `sentence_transformers.SparseEncoder` (introduced in sentence-transformers v5). The `SPLADEEncoder` class handles both backends transparently via the `use_sparse_encoder` config flag.

### Dense (Sentence-BERT + FAISS)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
- **Index:** FAISS `IndexFlatIP` — exact inner-product search (cosine similarity on normalized vectors)
- **Strengths:** Captures semantic similarity across paraphrases
- **Weaknesses:** May miss exact keyword matches that BM25 handles trivially

### Hybrid Fusion

Two fusion strategies combine the three retrieval methods:

1. **Reciprocal Rank Fusion (RRF):** Score-agnostic, rank-based. `RRF(d) = Σ 1/(k + rank(d))` with `k=60` (from the original Cormack et al. paper). Robust to different score scales.

2. **Weighted Score Fusion:** Min-max normalizes each method's scores to [0,1], then computes a weighted sum (default: BM25=0.3, SPLADE=0.4, Dense=0.3).

### BM25 Re-ranking

An additional hybrid strategy: retrieve candidates with a neural method (Dense or SPLADE), then **re-rank by BM25 score**. This tests whether BM25's lexical precision can improve neural recall. Implementation caches BM25 top-1000 scores per query for efficient lookup.

---

## Evaluation Design

### The Core Challenge

With only ~16 labels per query against 1.2M products, standard IR metrics become unreliable at the full-catalog scale. A method that retrieves a truly relevant but unlabeled product gets penalized. This is a known limitation of the ESCI dataset.

### Our Solution: Dual Evaluation Strategy

We run **four evaluation parts**, each addressing a different question:

#### Part A: Curated Pool Evaluation (Qdrant-style)

Creates a small product pool from ESCI-labeled products only, then retrieves within this pool.

- **Pool construction:** Select N test queries (default: 2,000 from test split). Pool = union of all products with ESCI labels for those queries (~10-30K products, deduplicated).
- **Assumption:** Unlabeled products in the pool are treated as Irrelevant (I=0.0)
- **Gains:** Qdrant-compatible graded gains (E=1.0, S=0.7, C=0.5, I=0.0) for comparability with [published SPLADE results](https://qdrant.tech/articles/sparse-embeddings-ecommerce-part-1/)
- **Metrics:** NDCG@K, MRR@K, Precision@K, Recall@K

> **Caveat:** Treating unlabeled products as Irrelevant is standard practice (used by Qdrant, BEIR, and most ESCI evaluations) but may underestimate methods that retrieve truly relevant but unlabeled products. We document this assumption explicitly in all output.

**Why this approach?** With the curated pool, label coverage is much higher (~100% for the labeled subset), making NDCG and MRR meaningful. The Qdrant-compatible gains allow direct comparison with their published SPLADE fine-tuning results.

#### Part B: Score-Based Analysis (Annotated Candidates)

For each query, scores its ~16 ESCI-labeled candidates and computes:

- **Re-ranking NDCG@K** — how well each method ranks the labeled candidates
- **Score distribution boxplots** — per-label score distributions (E vs S vs C vs I)
- **Separability AUC** — ROC AUC for distinguishing relevant (E+S+C) from irrelevant (I) using raw scores
- **Label distribution in top-K** — what fraction of top-K are E, S, C, I

This analysis is independent of the full catalog — it only asks: "given the ~16 candidates, does the method rank relevant ones higher?"

#### Part C: Full-Catalog Retrieval

Retrieves from the full 1.2M product catalog and reports:

- **Recall@K (E):** Fraction of Exact products found in top-K
- **Recall@K (E+S):** Fraction of Exact + Substitute products found
- **Label distribution** in top-K

These metrics are conservative (penalize unlabeled retrievals) but show relative method differences.

#### Part D: LLM-Augmented Evaluation (Stub)

A placeholder for future LLM-based evaluation. The idea: use an LLM (Claude, GPT) to label the top-K products retrieved by each method, providing denser coverage than ESCI's ~16 labels per query.

**Status:** We ran a pilot labeling experiment with Claude Sonnet 4 on 2,500 query-product pairs. Key findings:
- Claude achieved **71.5% agreement** with ESCI human labels on overlapping pairs
- Manual review of disagreements showed Claude was **more accurate** than human labels in most cases (e.g., human labeled charger cables as "Exact" for "vivoactive 3" query — Claude correctly labeled "Complement")
- Full-scale labeling is cost-prohibitive at naive scale ($27M+ for 75K queries x 1.2M products), but tractable when sampling: ~$300 for 10K queries x top-100 products using Haiku 3.5

The stub checks for `outputs/metrics/llm_augmented_labels.json` and will run curated-pool-style evaluation when labels are available.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `bm25s`, `transformers`, `torch`, `sentence-transformers>=5.0` (for SparseEncoder), `faiss-cpu`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

### 2. Build Indexes

Each method needs a one-time index build over the 1.2M product catalog:

```bash
# BM25 — builds sparse score matrix (~600MB cache)
python bm25_search.py --mode index

# Dense (SBERT) — encodes all products to 384-dim embeddings
python dense_search.py --mode encode --device cuda

# SPLADE (default: Prithivi) — encodes to sparse vocab-sized vectors
python splade_search.py --mode encode --device cuda

# SPLADE (all 3 variants)
python splade_search.py --mode encode --splade_model all --device cuda

# SPLADE (specific variant)
python splade_search.py --mode encode --splade_model qdrant_esci --device cuda
```

### 3. Run Evaluation

```bash
# Full evaluation (Parts A + B + C)
python hybrid_search.py --mode evaluate --max_queries 5000

# Curated pool only (Part A — fast, recommended first)
python hybrid_search.py --mode evaluate --eval_mode curated --max_queries 2000

# Full-catalog only (Parts B + C)
python hybrid_search.py --mode evaluate --eval_mode full_catalog --max_queries 5000

# With all SPLADE variants
python hybrid_search.py --mode evaluate --splade_models prithivi,qdrant_esci,naver
```

### 4. Demo Search

```bash
# Interactive demo showing all methods side-by-side
python hybrid_search.py --mode demo --demo_query "wireless bluetooth headphones"

# With multiple SPLADE variants
python hybrid_search.py --mode demo --splade_models prithivi,qdrant_esci,naver \
    --demo_query "running shoes for women"
```

### 5. SPLADE Term Expansion Visualization

```bash
# The "aha moment" — see which terms SPLADE activates beyond the query
python splade_search.py --mode visualize --demo_query "notebook for travel"
```

### 6. Individual Method Evaluation

Each method can be evaluated standalone:

```bash
python bm25_search.py --mode evaluate --max_queries 5000
python dense_search.py --mode evaluate --max_queries 5000 --device cuda
python splade_search.py --mode evaluate --max_queries 5000 --device cuda
```

---

## Code Structure

```
Appendix Hybrid Search/
├── config.py            # All configs: data, BM25, SPLADE, Dense, Hybrid, Eval
├── data_loader.py       # Amazon ESCI dataset loader (products + judgments)
├── bm25_search.py       # BM25 engine (bm25s library)
├── splade_search.py     # SPLADE encoder + index (dual backend: MLM / SparseEncoder)
├── dense_search.py      # Dense encoder + FAISS index (SBERT)
├── evaluate.py          # Metrics: NDCG, MRR, Recall, AUC, curated pool, plotting
├── hybrid_search.py     # Fusion (RRF, weighted), BM25 rerank, full eval pipeline
├── requirements.txt     # Python dependencies
├── cache/               # Cached indexes (auto-created)
│   ├── bm25/            # BM25 sparse matrix index
│   ├── embeddings/      # Dense SBERT embeddings (.npz)
│   └── splade/          # SPLADE sparse vectors (per-model subdirectories)
│       ├── prithivi/
│       ├── qdrant_esci/
│       └── naver/
└── outputs/             # Evaluation results (auto-created)
    ├── metrics/         # JSON results, comparison CSV
    └── plots/           # Score boxplots, label distributions, SPLADE expansion
```

---

## Configuration

All settings are centralized in `config.py` using dataclasses:

```python
from config import (
    DEFAULT_DATA_CONFIG,     # ESCIDataConfig: locale, split, text template
    DEFAULT_BM25_CONFIG,     # BM25Config: k1, b, method
    DEFAULT_SPLADE_CONFIG,   # SPLADEConfig: model, batch_size, backend
    DEFAULT_DENSE_CONFIG,    # DenseConfig: model, embedding_dim, FAISS type
    DEFAULT_HYBRID_CONFIG,   # HybridConfig: RRF k, score weights
    DEFAULT_EVAL_CONFIG,     # EvalConfig: k_values, gains, curated pool settings
    SPLADE_MODELS,           # Dict of 3 SPLADE variant configs
)
```

### Key Configuration Choices

**Product text template:** `"{title}. {bullet_point}. Brand: {brand}"` — omits description (often lengthy/noisy) and color (rarely useful for retrieval). This follows the text serialization pattern from Chapter 10.

**BM25 variant:** Lucene-compatible scoring (default in Elasticsearch/Solr). `k1=1.5` controls term frequency saturation; `b=0.75` controls document length normalization.

**ESCI gains (two mappings):**
- **Our gains** (`ESCI_GAINS`): E=3, S=2, C=1, I=0 — standard graded relevance
- **Qdrant gains** (`ESCI_GAINS_QDRANT`): E=1.0, S=0.7, C=0.5, I=0.0 — used for curated pool evaluation, enables comparison with Qdrant's published results

**RRF smoothing constant:** `k=60` (the standard default from the original RRF paper).

**Score fusion weights:** BM25=0.3, SPLADE=0.4, Dense=0.3. Gives SPLADE slightly more weight as it bridges lexical and semantic matching.

---

## Design Decisions

### Why bm25s instead of rank_bm25 or Elasticsearch?

`bm25s` pre-computes BM25 scores into scipy sparse matrices at index time. This makes query-time retrieval a single sparse matrix multiplication — orders of magnitude faster than traditional BM25 implementations that compute scores on-the-fly. It also avoids the operational overhead of running an Elasticsearch cluster.

### Why AutoModelForMaskedLM for Prithivi instead of SparseEncoder for all?

`SparseEncoder` (sentence-transformers v5) is the modern way to load SPLADE models, but `prithivida/Splade_PP_en_v2` was released before this API existed. It loads cleanly with `AutoModelForMaskedLM` + manual SPLADE transform (`log(1 + ReLU(logits))` → max-pool). The Qdrant and Naver models require `SparseEncoder`. Supporting both backends keeps the code compatible with all three variants.

### Why FAISS IndexFlatIP instead of approximate search?

With 1.2M products and 384-dim vectors, exact search is fast enough (sub-second per query). Approximate methods (IVF, HNSW) add complexity and recall loss that isn't justified at this scale. Production systems with 100M+ items would need approximate indexes.

### Why two ESCI gain mappings?

Our standard gains (E=3, S=2, C=1, I=0) provide more separation between relevance levels. But Qdrant's SPLADE evaluation uses (E=1.0, S=0.7, C=0.5, I=0.0). We include both so that:
1. Score-based analysis (Part B) uses our gains for maximum separability
2. Curated pool evaluation (Part A) uses Qdrant gains for direct comparability

### Why RRF over learned fusion?

RRF is score-agnostic — it only uses rank positions, making it robust to the very different score scales of BM25 (unbounded), SPLADE (sparse dot product), and Dense (cosine similarity in [0,1]). Learned fusion (e.g., training a weighted combination) would require a held-out set and risks overfitting to the specific methods used.

### Why curated pool evaluation?

With only ~16 labels per query against 1.2M products, full-catalog NDCG is dominated by the "unlabeled" category. The curated pool (~10-30K products with near-complete label coverage) makes NDCG, MRR, and Precision meaningful. This is exactly the approach used by [Qdrant's SPLADE evaluation](https://qdrant.tech/articles/sparse-embeddings-ecommerce-part-1/) and by the BEIR benchmark.

---

## Connection to Other Chapters

- **Chapter 5 (Retrieval):** Introduced dense bi-encoder retrieval. This appendix adds lexical and learned sparse retrieval, then fuses all three.
- **Chapter 10 (Item Embeddings):** Used `all-MiniLM-L6-v2` for zero-shot encoding. The dense search here reuses the same model and encoding pattern.
- **Chapter 10.4 (Multi-Modal Fusion):** Established the principle that heterogeneous embeddings should be kept separate, L2-normalized, and weighted at query time — the same principle underlying hybrid search fusion.
- **Appendix: RexBERT:** Domain-specialized SBERT for e-commerce. Could be swapped in for the dense encoder here.

---

## References

- **Amazon ESCI Dataset:** Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search", KDD 2022
- **BM25:** Robertson et al., "Okapi at TREC-3", 1995
- **bm25s:** Lù, "bm25s: Best Practices for BM25 Sparse Retrieval", 2024
- **SPLADE:** Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval", SIGIR 2022
- **SPLADE e-commerce fine-tuning:** [Qdrant tutorial](https://qdrant.tech/articles/sparse-embeddings-ecommerce-part-1/)
- **Sentence-BERT:** Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
- **Reciprocal Rank Fusion:** Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009
- **FAISS:** Johnson et al., "Billion-Scale Similarity Search with GPUs", IEEE TPDS 2021
