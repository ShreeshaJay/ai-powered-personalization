# Chapter 11: Customer/User Embeddings — Design Notes

## Table of Contents
1. [Chapter Overview](#chapter-overview)
2. [Section 11.1: Aggregation Baselines](#section-111-aggregation-baselines)
   - [Core Concept](#core-concept)
   - [Datasets & Terminology](#datasets--terminology)
   - [Item Embedding Sources](#item-embedding-sources)
   - [Aggregation Methods](#aggregation-methods)
   - [Evaluation Protocol](#evaluation-protocol)
   - [Hyperparameter Grounding](#hyperparameter-grounding)
   - [Evaluation Bias: Leave-Last-Out](#evaluation-bias-leave-last-out)
   - [Results](#results)
   - [Lessons Learned](#lessons-learned)
   - [Homework Exercises](#homework-exercises)
3. [Section 11.2: Sequence Models](#section-112-sequence-models-sasrec--gru4rec)
   - [Why Sequence Models?](#why-sequence-models)
   - [Architecture](#architecture)
   - [Training Objective](#training-objective-mnr-loss-with-multi-position-training)
   - [Data Splits](#data-splits)
   - [Hyperparameter Grounding](#hyperparameter-grounding-1)
   - [Results](#results-1)
   - [Homework Exercises](#homework-exercises-1)
   - [Student FAQ](#student-faq-common-questions-about-sequence-models)
   - [Training Results](#training-results)
4. [Section 11.3: Graph Neural Networks (LightGCN)](#section-113-graph-neural-networks-lightgcn)
   - [Why LightGCN?](#why-lightgcn)
   - [GNN Primer: Bipartite Graphs for Recommendation](#gnn-primer-bipartite-graphs-for-recommendation)
   - [What Features Does the Graph Use?](#what-features-does-the-graph-use)
   - [LightGCN Architecture](#lightgcn-architecture)
   - [Why Only Two Embedding Tables?](#why-only-two-embedding-tables-despite-multiple-layers)
   - [Loss Function: BPR vs MNR](#loss-function-bpr-vs-mnr)
   - [L2 Normalization](#l2-normalization-for-retrieval)
   - [Cold-Start Evaluation](#cold-start-evaluation)
   - [Production Inference Pattern](#production-inference-pattern)
   - [Hyperparameter Grounding](#hyperparameter-grounding-2)
   - [Results](#results-2)
   - [SBERT-Initialized LightGCN](#sbert-initialized-lightgcn-content-meets-collaborative)
   - [Homework Exercises](#homework-exercises-2)
5. [Cross-Chapter Integration](#cross-chapter-integration)
6. [Code Architecture](#code-architecture)
7. [Change Log](#change-log)

---

## Chapter Overview

**Goal**: Build user/customer embedding representations from the item embeddings produced in Chapter 10. These user embeddings capture individual preferences and can be used for downstream retrieval, ranking, and personalization.

**Key narrative arc across sections**:

| Section | Approach | Training? | Key Insight |
|---------|----------|-----------|-------------|
| 11.1 | Aggregation baselines | No | A user is the weighted sum of their interactions |
| 11.2 | Sequence models (SASRec/GRU4Rec) | Yes | Attention learns *which* items matter most |
| 11.3 | Graph Neural Networks (LightGCN) | Yes (MIND-only) | Learnable embeddings + graph propagation fix MNR mode collapse; SBERT init boosts MNR by 74% |

**Connection to Chapter 10**: Chapter 10 showed that item embeddings capture two complementary signals — content similarity ("blue fleece blanket" ≈ "grey sherpa throw") and behavioral co-occurrence ("phone case" ≈ "screen protector"). Chapter 11 asks: given these item representations, how do we build a representation of the *user* who interacted with them?

> **Note on Section 11.3**: Sections 11.1 and 11.2 directly consume Chapter 10's pre-computed item embeddings as inputs. Section 11.3 (LightGCN) takes a different approach — it learns its own item embeddings from scratch using only the behavioral graph structure, without using Chapter 10's SBERT embeddings. This demonstrates an important architectural choice: content-based (SBERT) vs. collaborative (graph-learned) item representations. The cold-start user evaluation still connects to Chapter 10's item catalog for consistency. See the [LightGCN Architecture](#lightgcn-architecture) section for a discussion of how architectures like PinSage can combine both approaches.

**The fundamental equation**:
```
user_embedding = f(item_embedding_1, item_embedding_2, ..., item_embedding_N)
```
The question is what `f` should be. Section 11.1 uses simple aggregation (mean, weighted mean). Section 11.2 uses learned sequence models. Section 11.3 uses graph propagation.

---

## Section 11.1: Aggregation Baselines

### Core Concept

The simplest user embedding is the **average of the item embeddings** the user has interacted with. This "bag of items" assumption treats a user as the centroid of their interaction history in embedding space.

**Why this works at all**: Embedding spaces learned in Chapter 10 place similar items near each other. A user who browses phone cases, screen protectors, and USB cables will have their centroid in the "mobile accessories" region of the embedding space. When we use this centroid as a query, we retrieve other items in that neighborhood — exactly the items this user is likely to want.

**Why this isn't enough**: Simple averaging treats all items equally. But a user who browsed laptops last week and phone cases today has a centroid somewhere between "laptops" and "phone cases" — neither fish nor fowl. The aggregation variants in this section progressively recover lost information:
- **Recency**: Recent items are more predictive of current intent.
- **Rarity**: Niche items are more distinctive of individual preferences.
- **Both**: The combination is the strongest signal available without training.

### Datasets & Terminology

We evaluate on two datasets that represent the two dominant patterns in production systems:

#### Amazon KDD Cup 2023: Session Embeddings (Anonymous Users)

| Property | Value |
|----------|-------|
| Identity | **No user IDs** — each row is an anonymous browsing session |
| History type | Short sequences (2–10 items per session) |
| Signal | Purchase/browse intent within a single session |
| Real-world analog | Anonymous visitors on an e-commerce site |
| Terminology | **Session embedding** (not user embedding) |

The Amazon dataset has no user IDs. Each session is an independent sequence of product interactions: `[prev_item_1, ..., prev_item_N] → next_item`. This is actually the dominant industrial pattern — most e-commerce traffic comes from anonymous or logged-out users. The system must infer intent from the current session alone.

We deliberately use the term **session embedding** for Amazon to be precise about what we're modeling: current session intent, not long-term user preferences.

#### MIND (Microsoft News): User Embeddings (Identified Users)

| Property | Value |
|----------|-------|
| Identity | **50K named users** with persistent IDs |
| History type | Long sequences (5–500+ clicks per user) |
| Signal | Reading preferences over days |
| Real-world analog | Logged-in returning users on a news site |
| Terminology | **User embedding** |

MIND provides true user IDs with timestamped impression logs. Each user has a click history accumulated over multiple sessions. This enables genuine user embeddings that capture long-term preferences.

**Pedagogical value of both**: By evaluating on both datasets, we show that the same aggregation techniques apply across the identity spectrum — from anonymous sessions (short, intent-driven) to identified users (long, preference-driven) — with different aggregation strategies excelling in each regime.

### Item Embedding Sources

We reuse cached item embeddings from Chapter 10 (no re-encoding):

| Dataset | Embedding | Source | Dim | File |
|---------|-----------|--------|-----|------|
| Amazon KDD | Fine-tuned SBERT | Ch 10.3 (MNR loss) | 384 | `amazon_kdd_embeddings_finetuned_*.npz` |
| Amazon KDD | Item2Vec | Ch 10.2 (Word2Vec) | 128 | `amazon_item2vec_embeddings.npz` |
| MIND | Zero-shot SBERT | Ch 10.1 (all-MiniLM) | 384 | `mind_embeddings_*.npz` |

**Primary embeddings**: Fine-tuned SBERT for Amazon (best Chapter 10 result, MRR=0.319) and zero-shot SBERT for MIND.

**Secondary comparison**: We include one row with Item2Vec simple-mean on Amazon to demonstrate that **user embedding quality is bounded by item embedding quality**. If simple-mean over fine-tuned SBERT beats TF-IDF-weighted over Item2Vec, the lesson is clear: invest in better item embeddings before investing in fancier aggregation.

### Aggregation Methods

Five methods, each building on the previous. The progression is designed to teach which signals matter and why:

#### Method 1: Simple Mean

```
user_emb = (1/L) * Σ item_emb[i]    for i = 0..L-1
```

- **Weights**: Uniform. Every item contributes equally.
- **Hyperparameters**: None.
- **Strengths**: Zero complexity. Reasonable when all items are equally relevant and the history is short.
- **Weaknesses**: Old interests dilute current intent. Popular items (browsed by everyone) dominate the centroid, drowning out distinctive preferences.

#### Method 2: Last-K Mean

```
user_emb = (1/K) * Σ item_emb[i]    for i = L-K..L-1
```

- **Weights**: Uniform over the K most recent items; zero for older items.
- **Hyperparameters**: K (number of recent items to include).
- **Key design**: K is determined from the empirical session/history length distribution — specifically, we sweep over the 25th, 50th, and 75th percentile lengths.
- **Strengths**: Captures recent intent without historical dilution.
- **Weaknesses**: Hard cutoff — item K+1 gets weight 0, item K gets weight 1/K. Loses any long-term preference signal.

#### Method 3: Exponential Decay

```
weight[i] = exp(-λ * (L-1-i))    (position-based)
weight[i] = exp(-λ * hours_ago)  (time-based, when timestamps available)
user_emb = Σ (weight[i] / Σ weights) * item_emb[i]
```

- **Weights**: Exponentially decreasing with distance from most recent.
- **Hyperparameters**: Half-life `h` (λ = ln(2)/h). At distance `h`, weight = 0.5.
- **Key design**: Half-life is set from data:
  - Amazon: `h = median_session_length - 1` (so the oldest item in a median-length session gets weight ~0.5).
  - MIND: `h = median_history_length / 2` (so the midpoint of a typical user's history gets weight ~0.5).
- **Strengths**: Smooth transition — recent items matter more, but old items still contribute. More graceful than Last-K's hard cutoff.
- **Weaknesses**: Assumes monotonic importance decay. In practice, a user might have a persistent preference for "science fiction" (old signal) and a recent interest in "cooking" (new signal) — exponential decay would underweight the persistent preference.

**Note on position vs. time**: Amazon sessions have no timestamps, so we use position as a proxy (item at index 0 is "oldest," index L-1 is "most recent"). MIND's history field also lacks per-article timestamps — the history list is ordered but the exact click times are not provided per article. So for both datasets, we default to position-based decay. The time-based pathway exists in the code for completeness and future datasets that provide true timestamps.

#### Method 4: TF-IDF Weighted Mean

```
weight[i] = IDF(item[i])
IDF(item) = log((N + 1) / (df(item) + 1))
user_emb = Σ (weight[i] / Σ weights) * item_emb[i]
```

- **Weights**: Inverse document frequency. Rare items get higher weight.
- **Hyperparameters**: Smoothing constant (additive, default 1.0).
- **Key insight**: A user who clicks CNN's homepage article (clicked by 80% of users) shouldn't have their embedding dominated by it. That click tells us nothing distinctive about the user. But a click on a niche robotics article — that's a real signal. TF-IDF upweights exactly those distinctive items.
- **TF is binary**: Most items appear once per session/history (no repeated clicks on the same article). So TF=1 for all items, and the weight reduces to just IDF.
- **Computed on training data only**: IDF vocabulary comes from the sessions/users used for evaluation. No test leakage.
- **Strengths**: Naturally handles the popularity bias problem.
- **Weaknesses**: Ignores recency entirely. A distinctive item from 6 months ago gets the same weight as one from today.

#### Method 5: TF-IDF + Recency (Combined)

```
weight[i] = IDF(item[i]) * exp(-λ * distance[i])
user_emb = Σ (weight[i] / Σ weights) * item_emb[i]
```

- **Weights**: Product of IDF (rarity) and exponential decay (recency).
- **Hyperparameters**: Half-life (same as Method 3) + IDF vocabulary (same as Method 4).
- **Key insight**: A rare item that was interacted with recently is the strongest possible signal. This method gives such items the highest weight.
- **Establishes the ceiling**: This is the strongest non-learned aggregation baseline. It sets the bar that trained models in Sections 11.2 and 11.3 must beat.
- **Limitation**: Still cannot learn *which* items in the sequence are actually predictive of the target. Two items might have similar IDF and recency, but one is highly predictive (a phone case predicts a screen protector purchase) while the other is not (a random accessory). Learning this requires attention mechanisms (Section 11.2).

### Evaluation Protocol

#### Amazon KDD: Next-Item Retrieval

1. **Load sessions** with ≥ 3 prev_items (ensures enough history for aggregation).
2. **For each session**:
   - History = all prev_items `[i_1, i_2, ..., i_N]`.
   - Target = next_item `i_{N+1}` (the actual next click/purchase).
3. **Aggregate** history item embeddings → session embedding.
4. **Retrieve** top-K items from the full 500K product catalog (FAISS IndexFlatIP).
5. **Evaluate**: Is the target item in the top-K?
6. **Filter**: Exclude items already in the session history from retrieval results (we don't want to "recommend" items the user already interacted with).

**Comparison baseline**: Chapter 10's "last-item only" approach (query = embedding of the single last item viewed). This achieved MRR=0.319 with fine-tuned SBERT. The aggregation must beat this to prove that incorporating more context helps.

**Scale**: 20K evaluation sessions (matching Chapter 10 for comparability).

#### MIND: Next-Click Prediction

1. **Load users** with ≥ 5 clicks in their history.
2. **Temporal split per user**: The `history` field in the last impression contains all previously clicked articles. The `clicked_articles` in the last impression are the targets.
3. **Aggregate** history article embeddings → user embedding.
4. **Retrieve** top-K articles from the full 51K article catalog.
5. **Evaluate**: Are the target articles in the top-K?
6. **Filter**: Exclude articles already in the user's history.

**Comparison baseline**: "Last-click only" (query = embedding of the single most recently clicked article).

**Scale**: Up to 10K evaluation users.

#### Metrics

Same as Chapter 10 for direct comparability:
- **MRR** (Mean Reciprocal Rank): Primary metric. Measures the average rank of the first relevant item.
- **Recall@K** for K ∈ {1, 5, 10, 20, 50}: What fraction of target items appear in the top-K?
- **Precision@K**: What fraction of the top-K are relevant?
- **NDCG@K**: Discounted cumulative gain, penalizes relevant items at lower ranks.

### Hyperparameter Grounding

**Constraint**: No arbitrary hyperparameters. Every choice is grounded in the data or explicitly flagged as a pedagogical simplification.

| Hyperparameter | Source | How it's determined |
|----------------|--------|-------------------|
| Last-K values (Amazon) | Session length distribution | 25th, 50th, 75th percentile of `len(prev_items)` |
| Last-K values (MIND) | History length distribution | 25th, 50th, 75th, 90th percentile of click count |
| Decay half-life (Amazon) | Session length median | `median_session_length - 1` positions |
| Decay half-life (MIND) | History length median | `median_history_length / 2` positions |
| IDF smoothing | Standard practice | Additive smoothing = 1.0 (Laplace) |
| IDF vocabulary | Training data | Computed from sessions/users before evaluation |
| Eval sessions/users | Chapter 10 precedent | 20K sessions (Amazon), 10K users (MIND) |

The `data_analysis.py` script computes these distributions and saves them to `outputs/analysis/` before evaluation begins. This ensures all hyperparameter choices are auditable and reproducible.

### Evaluation Bias: Leave-Last-Out

**Important methodological caveat**: Our evaluation uses a leave-last-out protocol where the aggregation always sees the full session history minus the target item. This systematically **overestimates** aggregation quality because:

1. **In production**, a recommender must generate recommendations at *every* point in the user journey — not just at the end. A user who has browsed 2 items out of an eventual 8-item session needs recommendations after item 2, not after item 7.

2. **Our evaluation** always gives the aggregator the longest possible history (all N prev_items), which is the easiest case. With more context, the session centroid is more refined.

3. **The bias is larger for short sessions**: In a 3-item session, our evaluation gives the aggregator 3 items. But in production, the system would need to make recommendations after item 1 (with 0 history) and item 2 (with 1 history item). The leave-last-out evaluation never tests these sparse scenarios.

The following diagram illustrates the gap between our offline evaluation and production serving for a session with 8 items:

```
Session:   i_1 ── i_2 ── i_3 ── i_4 ── i_5 ── i_6 ── i_7 ── i_8
           ─────────────────────────────────────────────────────────►  time

╔═══════════════════════════════════════════════════════════════════════╗
║  OUR OFFLINE EVALUATION (Leave-Last-Out)                             ║
║                                                                      ║
║  We evaluate ONCE per session, always at the end:                    ║
║                                                                      ║
║  Aggregation input:  [i_1, i_2, i_3, i_4, i_5, i_6, i_7]           ║
║                       ════════════════════════════════════            ║
║                          ALL 7 items available (easiest case)        ║
║  Target to predict:  i_8                                             ║
║                                                                      ║
║  Result: We always see the maximum possible context.                 ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║  PRODUCTION SERVING (Real-time recommendations)                      ║
║                                                                      ║
║  The system must recommend at EVERY step of the journey:             ║
║                                                                      ║
║  After i_1:  Aggregate [i_1]                    → recommend next     ║
║  After i_2:  Aggregate [i_1, i_2]               → recommend next     ║
║  After i_3:  Aggregate [i_1, i_2, i_3]          → recommend next     ║
║  After i_4:  Aggregate [i_1, i_2, i_3, i_4]     → recommend next     ║
║  After i_5:  Aggregate [i_1, ..., i_5]           → recommend next     ║
║  After i_6:  Aggregate [i_1, ..., i_6]           → recommend next     ║
║  After i_7:  Aggregate [i_1, ..., i_7]           → recommend next     ║
║                ▲                                                      ║
║                └── Only this last case matches our evaluation!        ║
║                                                                      ║
║  Result: 6 out of 7 recommendation points have LESS context than     ║
║          what our evaluation assumes. We never test the hardest       ║
║          cases (1-2 items of context).                                ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║  HOMEWORK: Random-Truncation Evaluation (Closes the gap)             ║
║                                                                      ║
║  Sample a random truncation point t ∈ [2, L-1] for each session:    ║
║                                                                      ║
║  Example with t=4:                                                   ║
║    Aggregation input:  [i_1, i_2, i_3, i_4]                         ║
║                         ════════════════════                          ║
║    Target to predict:  i_5                                           ║
║                                                                      ║
║  Example with t=2:                                                   ║
║    Aggregation input:  [i_1, i_2]                                    ║
║                         ══════════                                    ║
║    Target to predict:  i_3                                           ║
║                                                                      ║
║  This simulates the production distribution of context lengths.      ║
╚═══════════════════════════════════════════════════════════════════════╝
```

**A more realistic protocol** (see Homework Exercise 1 below) would:
- For each session/history of length L (where L ≥ 3):
  - Sample a random truncation point `t` uniformly from `[2, L-1]`.
  - Aggregation input = items `[1, ..., t]`.
  - Target = item at position `t+1`.
- This simulates the production scenario where the recommender must work at arbitrary points in the session, not just at the end.

### Results

*Evaluation run: 2026-03-01. Amazon KDD: 20K sessions, 500K product catalog, fine-tuned SBERT 384d. MIND: 10K users, 51K article catalog, zero-shot SBERT 384d.*

#### Amazon KDD Results (Session Embeddings)

Data-driven hyperparameters: Median session length = 4, Last-K = {3, 4, 6}, Decay half-life = 3.0 positions.

| Method | MRR | R@1 | R@5 | R@10 | R@20 | R@50 |
|--------|-----|-----|-----|------|------|------|
| **Last-Item Only (Ch10 Baseline)** | **0.2021** | **0.1222** | **0.2917** | **0.3610** | **0.4309** | **0.5214** |
| TF-IDF + Recency (hl=3.0) | 0.1585 | 0.0893 | 0.2316 | 0.3049 | 0.3787 | 0.4782 |
| Exp Decay (hl=3.0, position) | 0.1571 | 0.0878 | 0.2301 | 0.3038 | 0.3775 | 0.4768 |
| Last-3 Mean | 0.1532 | 0.0858 | 0.2226 | 0.2983 | 0.3736 | 0.4732 |
| Last-4 Mean | 0.1408 | 0.0756 | 0.2086 | 0.2803 | 0.3552 | 0.4572 |
| Last-6 Mean | 0.1308 | 0.0689 | 0.1938 | 0.2658 | 0.3390 | 0.4405 |
| TF-IDF Weighted | 0.1262 | 0.0664 | 0.1852 | 0.2564 | 0.3282 | 0.4292 |
| Simple Mean | 0.1253 | 0.0658 | 0.1850 | 0.2554 | 0.3285 | 0.4278 |
| Simple Mean (Item2Vec 128d) | 0.1000 | 0.0570 | 0.1414 | 0.1908 | 0.2468 | 0.3348 |

#### MIND Results (User Embeddings)

Data-driven hyperparameters: Median history length = 11, Last-K = {5, 11, 22, 42}, Decay half-life = 5.5 positions.

| Method | MRR | R@1 | R@5 | R@10 | R@20 | R@50 |
|--------|-----|-----|-----|------|------|------|
| Last-11 Mean | 0.0031 | 0.0014 | 0.0038 | 0.0064 | 0.0100 | 0.0219 |
| Last-5 Mean | 0.0030 | 0.0011 | 0.0036 | 0.0063 | 0.0124 | 0.0234 |
| TF-IDF + Recency (hl=5.5) | 0.0030 | 0.0009 | 0.0036 | 0.0070 | 0.0118 | 0.0235 |
| Last-42 Mean | 0.0029 | 0.0009 | 0.0037 | 0.0066 | 0.0106 | 0.0222 |
| Simple Mean | 0.0028 | 0.0008 | 0.0037 | 0.0069 | 0.0115 | 0.0234 |
| Exp Decay (hl=5.5, position) | 0.0028 | 0.0009 | 0.0039 | 0.0059 | 0.0109 | 0.0216 |
| Last-22 Mean | 0.0027 | 0.0009 | 0.0034 | 0.0060 | 0.0105 | 0.0225 |
| TF-IDF Weighted | 0.0026 | 0.0005 | 0.0037 | 0.0066 | 0.0129 | 0.0235 |
| Last-Click Only Baseline | 0.0022 | 0.0006 | 0.0031 | 0.0048 | 0.0089 | 0.0147 |

### Lessons Learned

#### Hypothesis vs. Reality

**Hypothesis 1: "Recency is the strongest single signal"** -- **CONFIRMED for Amazon.**

On Amazon KDD, the ranking of aggregation methods is remarkably clear: the more recent items dominate, the better. The Last-Item Only baseline (MRR=0.202) crushes all aggregation methods. Among aggregation methods, TF-IDF + Recency (0.159) and Exp Decay (0.157) -- both recency-weighted -- outperform simple mean (0.125) by 25%. The lesson: for short e-commerce sessions, the most recent item is the strongest predictor of immediate purchase intent.

**Hypothesis 2: "Aggregation should beat single-item for short sessions"** -- **REFUTED for Amazon, CONFIRMED for MIND.**

This is the most important finding. On Amazon KDD, the single last item (MRR=0.202) beats every aggregation method, including TF-IDF + Recency (0.159) -- a 21% gap. Why? Amazon sessions are short (median 4 items) and intent-driven. The last item viewed is the sharpest signal of what the user wants right now. Averaging in earlier items actually *dilutes* this signal, pulling the session embedding away from the precise region of interest.

On MIND, the pattern reverses: aggregation methods (MRR ~0.003) consistently beat the last-click-only baseline (MRR=0.002) by 30-50%. With longer histories (median 11 articles), there's more useful context to aggregate, and no single click captures the full breadth of a user's interests.

**Hypothesis 3: "IDF helps with long histories"** -- **NOT CONFIRMED (but explainable).**

TF-IDF Weighted actually performs worst or near-worst on both datasets. This is because: (a) on Amazon with short sessions, rarity is noise -- all items in a 3-item session are potentially relevant, regardless of popularity; (b) on MIND, the IDF signal exists but is dominated by the evaluation challenge (see below). TF-IDF may show more value in production scenarios with very long histories (100+ items) where popular-item dilution is a real problem.

**Hypothesis 4: "Item embedding quality bounds user embedding quality"** -- **CONFIRMED.**

Simple Mean over fine-tuned SBERT (MRR=0.125) beats Simple Mean over Item2Vec (MRR=0.100) by 25%. Same aggregation method, different underlying item embeddings. The lesson: invest in better item embeddings first, fancier aggregation second.

#### Why MIND Metrics Are So Low

The MIND results (MRR ~0.003) appear extremely low compared to Amazon (MRR ~0.15). This is **not** a bug -- it reflects a fundamental difference in task difficulty:

1. **Task asymmetry**: On Amazon, each session has exactly 1 target (the next_item). On MIND, the target is "which article did the user click from the impression" -- but the user's general reading preferences (sports, politics, tech) produce a user embedding near those broad topic clusters. The specific article clicked depends on what was shown in the impression, headline appeal, timing -- factors the embedding cannot capture.

2. **Catalog density**: MIND's 51K articles are heavily clustered by topic. Many articles within the same category have very similar embeddings. The user embedding points to the right *neighborhood* but the exact article is hard to pinpoint. This is a known limitation of retrieval-only evaluation on news datasets.

3. **Relative ordering still holds**: Despite low absolute metrics, the relative ordering of methods is informative. All aggregation methods beat Last-Click Only, confirming that historical context helps for user embeddings even when absolute performance is modest.

4. **This motivates Chapter 11.2**: Trained sequence models (SASRec/GRU4Rec) should dramatically improve MIND performance by learning which historical clicks are most predictive of the specific next click -- something fixed-weight aggregation cannot do.

#### Summary of Key Insights for the Book

1. **For short sessions (e-commerce): the last item wins.** Aggregation doesn't beat the single most recent item because the session is too short for context to help. This is counterintuitive but critically important for practitioners -- don't over-engineer when a simple lookup works.

2. **For long histories (news/media): aggregation adds clear value.** When users have 10+ interactions, combining them captures preference patterns that no single item can.

3. **Recency > Rarity for intent prediction.** What the user did *most recently* matters more than how *unusual* their behavior was. Exp Decay and TF-IDF + Recency consistently outperform TF-IDF alone.

4. **Garbage in, garbage out.** Better item embeddings produce better user embeddings, regardless of aggregation method.

5. **These baselines set the bar for Sections 11.2 and 11.3.** On Amazon, trained models must beat MRR=0.202 (last-item) or at minimum MRR=0.159 (best aggregation). On MIND, they must beat MRR=0.003 -- which should be achievable since learned attention can identify which historical clicks are truly predictive.

#### Expected Lessons (retained as intuition)

These were our pre-experiment hypotheses. Even where contradicted by results, they represent reasonable prior intuitions that are worth teaching:

1. **Session length matters**: For Amazon's short sessions (2-5 items), we expected aggregation to help -- but the last item dominates. For MIND's longer histories, aggregation does help as expected.

2. **Recency is the strongest single signal**: Confirmed. Exponential decay and recency-weighted methods outperform uniform aggregation.

3. **IDF helps with long histories**: Not confirmed in our evaluation, but the principle (upweight distinctive items, downweight popular ones) remains sound for production systems with 100+ item histories.

4. **Item embedding quality bounds user embedding quality**: Confirmed. Fine-tuned SBERT simple-mean beats Item2Vec simple-mean by 25%.

5. **Aggregation baselines are surprisingly strong**: Partially confirmed. They provide a meaningful signal and are trivial to implement, making them excellent first-pass solutions in production.

### Homework Exercises

#### Exercise 1: Random-Truncation Evaluation (Closing the Train-Serve Gap)

The leave-last-out evaluation protocol used in this chapter systematically overestimates aggregation quality because the aggregator always sees the longest possible history. In production, recommendations must be generated at *every* point in the user journey.

**Task**: Implement a random-truncation evaluation protocol:
1. For each session/history of length L ≥ 3:
   - Sample a truncation point `t` uniformly from `[2, L-1]`.
   - Aggregation input = items `[0, 1, ..., t-1]` (the first `t` items).
   - Target = item at position `t`.
2. Re-run all aggregation methods with this protocol.
3. Compare results against the leave-last-out evaluation.

**Expected findings**:
- All methods should show lower MRR/Recall compared to leave-last-out.
- The gap should be largest for methods that benefit from long histories (TF-IDF, Simple Mean) and smallest for methods that focus on recent items (Last-K with small K, Exponential Decay with short half-life).
- This exercise teaches the critical concept of **train-serve skew**: the distribution of inputs at evaluation time should match what the model will see in production.

**Bonus**: Instead of uniform sampling, weight the truncation point by the empirical distribution of session positions at which users *actually* need recommendations (e.g., proportional to session length). This more accurately simulates the production workload.

#### Exercise 2: Embedding Quality Comparison

Using the same aggregation method (Simple Mean), compare user embeddings built from:
- Fine-tuned SBERT embeddings (384-dim)
- Zero-shot SBERT embeddings (384-dim)
- Item2Vec embeddings (128-dim)

Report MRR for each on Amazon KDD. What does this tell you about the relative importance of item embedding quality vs. aggregation sophistication?

#### Exercise 3: Integrating User Embeddings into a Ranking Model

In Chapter 7, we trained a Deep Learning ranking model (MMoE) on the SIGIR e-commerce dataset. That dataset provides pre-computed item description vectors (50-dim) and image vectors (50-dim).

**Task**: Use the aggregation methods from this chapter to create user embeddings from the SIGIR dataset's item vectors, then add these as features to the Chapter 7 ranking model. Measure the lift in AUC/logloss from including user embeddings.

**Hint**: The SIGIR dataset's sessions have sequential browsing events. For each training example, the user embedding should be computed from items browsed *before* the current item (to prevent leakage).

---

## Section 11.2: Sequence Models (SASRec & GRU4Rec)

### Why Sequence Models?

Section 11.1's aggregation baselines treat all items with **fixed weights** — determined by position, frequency, or simple heuristics. They cannot learn *which* items in the history are truly predictive of the next item.

Consider a session: `[headphones, laptop stand, USB hub, mechanical keyboard]`. A simple mean gives equal weight to all four. But "mechanical keyboard" → "keycap set" is a much stronger signal than "headphones" → "keycap set". Sequence models learn these item-to-item predictive patterns from data.

SASRec and GRU4Rec are the canonical representatives of two paradigms:
- **SASRec** (Transformer / self-attention): learns *which* history items to attend to, regardless of their position.
- **GRU4Rec** (RNN / recurrence): processes items sequentially, building up a hidden state that summarizes the history.

Once students understand these two, they can explore variants: BERT4Rec (bidirectional masking), NARM (GRU + attention), BST (Behavior Sequence Transformer). The loss function is also interchangeable — MNR can be swapped for BPR, sampled softmax, or binary cross-entropy.

### Architecture

Both models follow the same pipeline:

```
╔══════════════════════════════════════════════════════════════════════╗
║  Input: (B, L, 384) frozen SBERT embeddings from Chapter 10         ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │  Linear Projection: 384 → 128 (learnable)                    │    ║
║  │  Purpose: Compress input to model's working dimension         │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                              │                                       ║
║                              ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │  Sequence Model (one of):                                     │    ║
║  │                                                                │    ║
║  │  SASRec:  2 × Transformer blocks with causal masking          │    ║
║  │           + learnable positional embeddings                    │    ║
║  │           + 2 attention heads (head_dim=64)                    │    ║
║  │           + FFN (128→512→128 with GELU)                       │    ║
║  │           + Pre-LayerNorm + Dropout(0.2)                      │    ║
║  │                                                                │    ║
║  │  GRU4Rec: 2 × GRU layers (hidden=128)                        │    ║
║  │           + packed sequences for variable lengths              │    ║
║  │           + inter-layer dropout(0.2)                           │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                              │                                       ║
║                              ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │  Output Projection: 128 → 384 (learnable)                    │    ║
║  │  Purpose: Map back to item embedding space for FAISS search   │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
║                              │                                       ║
║                              ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │  L2-normalize → User Embedding (384-dim)                      │    ║
║  │  Ready for FAISS IndexFlatIP retrieval                        │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝
```

#### Why the Output Projection Matters

The output projection (128 → 384) maps from the model's internal space back to the **same 384-dim space** where the frozen item embeddings live. This allows direct FAISS retrieval using the existing Chapter 10 index.

But same dimensions ≠ same semantic space. The MNR training loss explicitly forces alignment:

```
During training:
  1. Model processes [i_1, i_2, i_3] → produces 384-dim user_emb
  2. Target = frozen 384-dim SBERT embedding of i_4 (from Chapter 10)
  3. MNR loss = cross_entropy(cosine_sim(user_emb, all_targets_in_batch))
  4. Gradients push user_emb TOWARD the correct target's SBERT embedding
     and AWAY from all other items' SBERT embeddings
```

After training, the output lives in the same semantic space as the item embeddings because the loss function optimized for cosine similarity alignment.

#### Parameter Counts

| Model | Components | Total Params | Memory |
|-------|-----------|-------------|---------|
| SASRec | input proj (49K) + pos embed (6K) + 2×transformer (397K) + output proj (49K) | ~500K | ~2 MB |
| GRU4Rec | input proj (49K) + 2×GRU (198K) + output proj (49K) | ~300K | ~1.2 MB |

Both are trivially small compared to the frozen item embedding matrix (471K × 384 × 4 = 723 MB for Amazon).

### Training Objective: MNR Loss with Multi-Position Training

#### Loss Function

Same MNR (Multiple Negatives Ranking) loss from Chapter 10's contrastive fine-tuning. For each predicted user embedding, the positive is the corresponding target item; all other targets in the batch are in-batch negatives.

```
similarity = cosine_sim(user_emb, target_emb) / temperature
loss = cross_entropy(similarity, diagonal_labels)
```

**Temperature = 0.05**: Controls softmax sharpness (equivalent to Chapter 10's `similarity * 20`). Think of it like adjusting contrast on a photo — low temperature = high contrast, forcing the model to make sharp distinctions between the correct target and all negatives.

With batch_size = 256, each batch provides 65,280 negative comparisons.

#### Multi-Position Training

Each session contributes training signal at **every** position, not just the last:

```
Session: [i_1, i_2, i_3, i_4] → next_item i_5

Training signals (all from one session):
  After i_1: model sees [i_1]               → predict i_2  (1 item of context)
  After i_2: model sees [i_1, i_2]          → predict i_3  (2 items of context)
  After i_3: model sees [i_1, i_2, i_3]     → predict i_4  (3 items of context)
  After i_4: model sees [i_1, i_2, i_3, i_4]→ predict i_5  (4 items of context)
```

This is critical for short Amazon sessions (median 4 items → only 3-4 training signals per session). It also teaches the model to produce good embeddings from partial sequences, matching the production scenario.

### Data Splits

```
╔══════════════════════════════════════════════════════════════╗
║  AMAZON KDD: 732,861 total sessions                         ║
║                                                              ║
║  ┌────────────────────────────────────────┐  ┌────────────┐ ║
║  │  Training Pool: ~712,861 sessions      │  │  Eval Set   │ ║
║  │  (sessions NOT in eval set)            │  │  20,000     │ ║
║  │                                        │  │  sessions   │ ║
║  │  ┌──────────────┐ ┌──────────────┐    │  │  (seed=42)  │ ║
║  │  │ Train: 90%   │ │ Val: 10%     │    │  │             │ ║
║  │  │ ~641,575     │ │ ~71,286      │    │  │  Same as    │ ║
║  │  │ sessions     │ │ sessions     │    │  │  Sec 11.1   │ ║
║  │  └──────────────┘ └──────────────┘    │  └────────────┘ ║
║  └────────────────────────────────────────┘                  ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  MIND: ~50,000 total users                                   ║
║                                                              ║
║  ┌────────────────────────────────────────┐  ┌────────────┐ ║
║  │  Training Pool: ~40,000 users          │  │  Eval Set   │ ║
║  │  (users NOT in eval set)               │  │  10,000     │ ║
║  │                                        │  │  users      │ ║
║  │  ┌──────────────┐ ┌──────────────┐    │  │  (seed=42)  │ ║
║  │  │ Train: 90%   │ │ Val: 10%     │    │  │             │ ║
║  │  │ ~36,000      │ │ ~4,000       │    │  │  Same as    │ ║
║  │  │ users        │ │ users        │    │  │  Sec 11.1   │ ║
║  │  └──────────────┘ └──────────────┘    │  └────────────┘ ║
║  └────────────────────────────────────────┘                  ║
║                                                              ║
║  KEY: No user appears in more than one split.                ║
║       Eval user IDs are excluded FIRST, then remaining       ║
║       users are split 90/10 into train/val.                  ║
╚══════════════════════════════════════════════════════════════╝
```

### Hyperparameter Grounding

| Hyperparameter | Value | Source / Justification |
|----------------|-------|----------------------|
| hidden_dim | 128 | 3× compression from 384; matches Ch10 Item2Vec and multimodal fusion dims |
| num_layers | 2 | Original SASRec paper default; sufficient for sequences of 4-50 items |
| num_heads (SASRec) | 2 | head_dim=64, within standard 32-64 range |
| ffn_dim (SASRec) | 512 | 4× hidden_dim, standard Transformer ratio |
| dropout | 0.2 | Original SASRec default |
| max_seq_len (Amazon) | 20 | Data-driven: P95=12, 20 covers P95+ with margin |
| max_seq_len (MIND) | 50 | Data-driven: P90=42, 50 covers P90 |
| learning_rate | 1e-3 | Standard for training-from-scratch; original SASRec |
| batch_size | 256 | Matches Ch10; provides 65K in-batch negatives |
| epochs (Amazon) | 30 | Larger dataset; early stopping with patience=5 |
| epochs (MIND) | 20 | Smaller dataset; early stopping with patience=5 |
| temperature | 0.05 | Equivalent to Ch10's similarity×20; standard for retrieval |
| warmup | 10% of steps | Matches Ch10; prevents early divergence |
| weight_decay | 0.01 | Standard AdamW regularization |
| gradient_clip | 1.0 | Matches Ch10; prevents gradient explosion |

### Results

*Evaluation run: 2026-03-03. Amazon KDD: 20K sessions, 500K product catalog, fine-tuned SBERT 384d. MIND: 10K users, 51K article catalog, zero-shot SBERT 384d.*

#### Amazon KDD Results (Section 11.1 + 11.2, sorted by MRR)

| Method | Section | MRR | R@1 | R@5 | R@10 | R@20 | R@50 |
|--------|---------|-----|-----|-----|------|------|------|
| **Last-Item Only (Ch10 Baseline)** | 10 | **0.2021** | **0.1222** | **0.2917** | **0.3611** | **0.4310** | **0.5214** |
| TF-IDF + Recency (hl=3.0) | 11.1 | 0.1585 | 0.0894 | 0.2317 | 0.3049 | 0.3788 | 0.4782 |
| Exp Decay (hl=3.0, position) | 11.1 | 0.1571 | 0.0878 | 0.2302 | 0.3038 | 0.3775 | 0.4768 |
| Last-3 Mean | 11.1 | 0.1532 | 0.0859 | 0.2227 | 0.2983 | 0.3737 | 0.4732 |
| **SASRec (2L, 128d)** | **11.2** | **0.1512** | **0.0853** | **0.2184** | **0.2916** | **0.3700** | **0.4746** |
| **GRU4Rec (2L, 128d)** | **11.2** | **0.1435** | **0.0780** | **0.2105** | **0.2855** | **0.3631** | **0.4703** |
| Last-4 Mean | 11.1 | 0.1408 | 0.0757 | 0.2086 | 0.2804 | 0.3552 | 0.4572 |
| Last-6 Mean | 11.1 | 0.1308 | 0.0690 | 0.1938 | 0.2658 | 0.3390 | 0.4405 |
| TF-IDF Weighted | 11.1 | 0.1262 | 0.0664 | 0.1852 | 0.2564 | 0.3282 | 0.4292 |
| Simple Mean | 11.1 | 0.1253 | 0.0658 | 0.1850 | 0.2554 | 0.3285 | 0.4278 |
| Simple Mean (Item2Vec 128d) | 11.1 | 0.1000 | 0.0570 | 0.1414 | 0.1908 | 0.2468 | 0.3348 |
| **SASRec-ItemID (2L, 128d)** | **11.2** | **0.0933** | **0.0397** | **0.1418** | **0.2116** | **0.2920** | **0.4178** |
| **GRU4Rec-ItemID (2L, 128d)** | **11.2** | **0.0884** | **0.0385** | **0.1299** | **0.1985** | **0.2827** | **0.4062** |

#### MIND Results (Section 11.1 + 11.2 + 11.3, sorted by MRR)

| Method | Section | MRR | R@1 | R@5 | R@10 | R@20 | R@50 |
|--------|---------|-----|-----|-----|------|------|------|
| **LightGCN-MNR (3L, 64d)** | **11.3** | **0.0100** | **0.0022** | **0.0112** | **0.0237** | **0.0475** | **0.1032** |
| **LightGCN-BPR (3L, 64d)** | **11.3** | **0.0051** | **0.0010** | **0.0046** | **0.0098** | **0.0233** | **0.0659** |
| Last-11 Mean | 11.1 | 0.0031 | 0.0014 | 0.0038 | 0.0064 | 0.0100 | 0.0219 |
| Last-5 Mean | 11.1 | 0.0030 | 0.0011 | 0.0036 | 0.0063 | 0.0124 | 0.0234 |
| TF-IDF + Recency (hl=5.5) | 11.1 | 0.0030 | 0.0009 | 0.0036 | 0.0070 | 0.0118 | 0.0235 |
| Last-42 Mean | 11.1 | 0.0029 | 0.0009 | 0.0037 | 0.0067 | 0.0107 | 0.0222 |
| Simple Mean | 11.1 | 0.0028 | 0.0008 | 0.0037 | 0.0069 | 0.0115 | 0.0234 |
| Exp Decay (hl=5.5, position) | 11.1 | 0.0028 | 0.0009 | 0.0039 | 0.0059 | 0.0109 | 0.0216 |
| Last-22 Mean | 11.1 | 0.0027 | 0.0009 | 0.0034 | 0.0061 | 0.0105 | 0.0225 |
| TF-IDF Weighted | 11.1 | 0.0026 | 0.0005 | 0.0038 | 0.0066 | 0.0129 | 0.0235 |
| Last-Click Only Baseline | 10 | 0.0022 | 0.0007 | 0.0031 | 0.0048 | 0.0089 | 0.0147 |
| **GRU4Rec (2L, 128d)** | **11.2** | **0.0016** | **0.0004** | **0.0020** | **0.0036** | **0.0069** | **0.0143** |
| **SASRec (2L, 128d)** | **11.2** | **0.0007** | **0.0001** | **0.0004** | **0.0016** | **0.0037** | **0.0101** |
| **SASRec-ItemID (2L, 128d)** | **11.2** | **0.0006** | **0.0001** | **0.0004** | **0.0018** | **0.0032** | **0.0080** |
| **GRU4Rec-ItemID (2L, 128d)** | **11.2** | **0.0005** | **0.0001** | **0.0005** | **0.0011** | **0.0027** | **0.0057** |

### Analysis of Results

#### Amazon KDD: Sequence Models Beat Most Aggregations, But Not Last-Item

**Hypothesis: "Sequence models should beat aggregation baselines (MRR > 0.159)"** — **PARTIALLY CONFIRMED.**

SASRec (MRR=0.151) beats 7 out of 9 aggregation methods but falls short of the top 3: TF-IDF+Recency (0.159), Exp Decay (0.157), and Last-3 Mean (0.153). GRU4Rec (MRR=0.144) beats 5 out of 9. Neither comes close to Last-Item Only (MRR=0.202).

**Why sequence models don't dominate on Amazon:**

1. **Sessions are too short for attention to help.** With median 4 items, there's barely any "sequence" to model. SASRec's attention mechanism has 2-3 items to attend over — not enough to learn complex dependencies. The model essentially collapses to a learned weighted average, which the handcrafted recency weights already approximate well.

2. **The training objective fights the evaluation protocol.** Multi-position training optimizes the model to predict the *next* item at every position. But evaluation uses the *full* session history — the easiest case. The model has been trained heavily on partial sequences (positions 1, 2, 3) where it has less context. This distributional mismatch means the model's parameters are a compromise across all positions rather than being optimized for the full-history case.
A more realistic evaluation task closer to how serving takes place , may indicate sequential model metrics being closer to the aggregation based approaches.

3. **Frozen embeddings limit expressiveness.** The input projection compresses 384→128, discarding information that might distinguish items within the same category. With learnable item IDs, the model could learn that "mechanical keyboard → keycap set" is a strong pattern even if their SBERT embeddings aren't particularly close. This is why Exercise 1 (Learnable Item IDs) is an important homework.

4. **Last-Item remains king for short sessions.** This reinforces Section 11.1's lesson: for short, intent-driven e-commerce sessions, the most recent item is an extremely efficient compressed representation of user intent. It's hard to improve on because there simply isn't enough additional context to extract value from.

**SASRec > GRU4Rec on Amazon** (0.151 vs 0.144): Even on short sessions, attention's ability to directly access all positions gives a small edge over GRU's sequential recurrence. This 5% gap is meaningful and consistent across all Recall@K values.

#### MIND: Sequence Models Dramatically Underperform Aggregations

**Hypothesis: "SASRec/GRU4Rec should clearly beat aggregation methods (MRR > 0.003)"** — **REFUTED.**

This is the most surprising result. Both sequence models perform *worse* than all aggregation baselines on MIND:
- SASRec: MRR = 0.0007 (4.4× worse than best aggregation at 0.0031)
- GRU4Rec: MRR = 0.0016 (1.9× worse than best aggregation at 0.0031)
- Both are worse than even Last-Click Only (MRR = 0.0022)

**Why sequence models fail on MIND — a diagnosis:**

1. **MNR loss with frozen SBERT embeddings produces mode collapse on MIND.** The MIND article catalog has dense topic clusters — many news articles about the same topic have nearly identical SBERT embeddings. When the model tries to predict the target article, it's penalized by in-batch negatives that are semantically almost identical to the correct target. The loss signal becomes: "produce an embedding near this specific article BUT away from 255 other articles, many of which are in the same topic cluster." This contradictory signal causes the model to learn a very conservative, generic representation that doesn't point strongly at any item.

2. **The target granularity problem.** Aggregation methods produce user embeddings by averaging item embeddings — they naturally land in the right topic neighborhood. Sequence models, constrained by MNR loss to discriminate between specific articles, must produce embeddings that are *precisely* near one article and far from others. On a dataset where many articles are interchangeable, this precision requirement is a disadvantage.

3. **Val loss told us this.** The MIND val_loss (~4.0) is much higher than Amazon (~0.44). With batch_size=256, perfect discrimination would give loss=0. Loss of 4.0 means the model assigns roughly exp(-4) ≈ 2% probability to the correct target — near random among the batch. The model never learned to discriminate effectively, yet this is exactly what evaluation demands.

4. **GRU4Rec > SASRec on MIND** (0.0016 vs 0.0007): GRU4Rec's inductive bias (sequential processing, last-hidden-state output) produces a representation more influenced by recent items, which by chance lands closer to the target neighborhood. SASRec's attention tries to be "smart" about weighting history items but produces a less useful average.

**Key pedagogical takeaway**: This is one of the most important findings in the chapter. Sequence models are not universally better than simple aggregations. Their advantage depends on:
- The **quality of the training signal** (MNR loss needs distinguishable targets)
- The **density of the embedding space** (dense topic clusters confuse contrastive learning)
- The **match between training and evaluation** (what the model optimizes vs. what we measure)

This naturally motivates Section 11.3's graph-based approach and the homework exercise on learnable Item IDs, both of which address the frozen embedding limitation.

#### Summary Table: Hypotheses vs. Reality

| Hypothesis | Result | Explanation |
|-----------|--------|-------------|
| Sequence models beat aggregation on Amazon | Partially confirmed | Beat 7/9 methods but not top 3 recency-weighted ones |
| Sequence models beat aggregation on MIND | **Refuted** | Both models dramatically underperform; MNR + frozen embeddings cause mode collapse |
| SASRec beats GRU4Rec on MIND (attention > recurrence for long histories) | **Refuted** | GRU4Rec (0.0016) > SASRec (0.0007); GRU's recency bias helps |
| GRU4Rec competitive on Amazon (short sequences) | Confirmed | GRU4Rec (0.144) close to SASRec (0.151), both competitive |
| Last-Item Only dominates on Amazon | **Confirmed** | MRR=0.202 still unchallenged by any method |
| ItemID beats frozen SBERT on Amazon | **Refuted** | SASRec-ItemID (0.093) << SASRec-SBERT (0.151); sparse data + 1.55M items = underfitting |
| LightGCN-BPR beats aggregation on MIND | **Confirmed** | MRR=0.0051, 1.6x better than best aggregation (0.0031) |
| LightGCN-MNR fails on MIND (like 11.2) | **Refuted** | MRR=0.0100 — learnable embeddings fix MNR's dense-cluster problem |
| MNR failure was due to loss function itself | **Refuted** | Root cause was MNR + frozen SBERT in dense clusters, not MNR inherently |
| SBERT init improves LightGCN equally for both losses | **Refuted** | MNR gained +74% (0.0100→0.0174), BPR only +14% (0.0051→0.0058); MNR exploits initial geometry more |
| SBERT-init LightGCN-MNR is best MIND method | **Confirmed** | MRR=0.0174 — 5.6x best aggregation, 10.9x best sequence model, 1.74x random-init MNR |

### Homework Exercises

#### Exercise 1: Learnable Item ID Embeddings (Completed)

Replace frozen SBERT embeddings with a learnable `nn.Embedding(num_items, hidden_dim)` lookup table. Train it jointly with the sequence model.

**Actual findings** (see Section 11.3 Results for full tables):
- ItemID variants **significantly underperform** frozen SBERT on Amazon: SASRec-ItemID MRR=0.093 vs SASRec-SBERT MRR=0.151 (38% worse). With 1.55M products and only 641K sequences, learnable embeddings cannot capture the item space. Frozen SBERT provides strong content priors that generalize better in this sparse regime.
- ItemID variants also fail on MIND (MRR=0.0005-0.0006), even worse than frozen SBERT variants.
- **Key lesson**: Learnable item embeddings need sufficient interaction density per item. Amazon has ~2.3 interactions per product on average — far too sparse for a 128-dim embedding to learn. LightGCN succeeds with learnable embeddings because (a) MIND has ~17 interactions per article, and (b) graph propagation shares information across neighborhoods, effectively amplifying the training signal per item.

#### Exercise 2: Sequence Length Ablation

Train SASRec with max_seq_len = {5, 10, 20, 50} on MIND. Plot MRR vs. max_seq_len.

**Expected finding:** MRR increases with longer contexts up to ~20-30, then plateaus or degrades (noise from very old history overwhelms the signal).

#### Exercise 3: Attention Visualization

Extract and plot SASRec attention weights for 5 example MIND users. For each user, show which history articles receive the highest attention when predicting different target articles.

**Expected finding:** Attention concentrates on same-category articles and recent items, effectively discovering the user's current intent cluster.

### Student FAQ: Common Questions About Sequence Models

These questions came up during the design review and are likely to occur to students as well. They address the "why" behind key architectural and training decisions.

#### Q1: Are SASRec and GRU4Rec the only options? Can I swap in other architectures?

**A:** SASRec and GRU4Rec are **representative architectures**, not the only options. They were chosen because they are the canonical examples of the two dominant paradigms — attention-based and recurrence-based sequential recommendation.

Once you understand these two, you can explore variants along two axes:

| Axis | Options |
|------|---------|
| **Architecture** | BERT4Rec (bidirectional masking), NARM (GRU + attention hybrid), BST (Behavior Sequence Transformer), Transformer-XL (longer context) |
| **Loss function** | BPR (Bayesian Personalized Ranking), sampled softmax, binary cross-entropy, full softmax over item catalog |

The training pipeline (data loading, evaluation, FAISS retrieval) remains identical — only the model class and loss function change. This modularity is intentional and is one of the homework exercises.

#### Q2: How does the output projection map to the item embedding space? Won't the user embeddings be in a different semantic space?

**A:** This is a crucial insight. The output projection (128 → 384) produces vectors with the same **dimensions** as the frozen SBERT item embeddings, but same dimensions ≠ same semantic space. What forces alignment is the **training objective**, not the architecture.

Here's how it works:

```
During training:
  1. Model processes history [i_1, i_2, i_3]
     → produces 384-dim user_emb (via output projection)

  2. Target = frozen 384-dim SBERT embedding of i_4
     (this is the actual Chapter 10 embedding, never modified)

  3. MNR loss = cross_entropy(
       cosine_sim(user_emb, ALL target embeddings in batch) / temperature
     )

  4. Gradients flow ONLY through the model (not the frozen targets):
     → Push user_emb TOWARD the correct target's SBERT embedding
     → Push user_emb AWAY FROM all other items' SBERT embeddings
```

After thousands of gradient updates, the output projection learns to produce vectors that live in the same semantic space as the frozen item embeddings — because that's exactly what the loss function optimizes for. The cosine similarity between model output and frozen SBERT embeddings is the direct training signal.

**Analogy**: It's like learning to speak a foreign language (the item embedding space). The model starts producing gibberish 384-dim vectors, but the loss function rewards vectors that are "understood" (cosine-similar to the correct item embedding) and penalizes vectors that are "misunderstood" (similar to wrong items). Over training, the model learns to "speak" the SBERT language.

#### Q3: What is the temperature parameter (0.05)?

**A:** Temperature controls the **sharpness of the softmax** in the MNR loss. Think of it like adjusting contrast on a photograph:

```
similarity = cosine_sim(user_emb, target_emb) / temperature

With temperature = 0.05:
  cosine_sim = 0.8  →  similarity = 16.0   (very confident)
  cosine_sim = 0.3  →  similarity = 6.0    (less confident)
  Difference: 10.0 (sharp gradient signal)

With temperature = 1.0:
  cosine_sim = 0.8  →  similarity = 0.8    (weakly confident)
  cosine_sim = 0.3  →  similarity = 0.3    (barely different)
  Difference: 0.5 (weak gradient signal)
```

Lower temperature amplifies small cosine similarity differences, forcing the model to make **sharper distinctions** between the correct target and negatives. A temperature of 0.05 is equivalent to Chapter 10's `similarity * 20` scaling factor — maintaining consistency across chapters.

**Why 0.05 specifically?** It's a well-established default in the retrieval literature (used in CLIP, SimCLR, and most contrastive learning systems). Too low (<0.01) causes training instability; too high (>0.5) produces weak gradients.

#### Q4: How do you ensure no user/session overlap between train and eval splits?

**A:** The split uses a strict exclusion protocol:

```
Step 1: Identify eval set (FIRST, using seed=42)
  → Amazon: 20,000 session IDs randomly sampled
  → MIND: 10,000 user IDs randomly sampled
  → These are the SAME eval sets used in Section 11.1

Step 2: Remove eval set from training pool
  → Amazon: 732,861 - 20,000 = ~712,861 remaining sessions
  → MIND: ~50,000 - 10,000 = ~40,000 remaining users

Step 3: Split remaining into train/val (90/10, seed=42)
  → No session/user appears in more than one split
```

For MIND specifically, this is user-level deduplication — a user's entire click history goes into exactly one split. There is no scenario where the same user's data appears in both training and evaluation.

#### Q5: How did you calculate the parameter counts and GPU memory?

**A:** Detailed parameter breakdown:

```
SASRec (~498K parameters):
  Input projection:   384 × 128 + 128 bias    = 49,280
  Positional embed:   max_seq_len × 128       =  6,400 (Amazon) / 6,400 (MIND)
  Per Transformer block (×2):
    LayerNorm (×2):   128 × 2 × 2              =    512
    MultiheadAttn:    128×128×3 (QKV) + 128×128 (out) + biases = 66,048
    FFN:              128×512 + 512 + 512×128 + 128 = 131,712
    Subtotal per block:                         = 198,272
  Output projection:  128 × 384 + 384 bias     = 49,536
  Total: 49,280 + 6,400 + 2×198,272 + 49,536   ≈ 498,432

GRU4Rec (~297K parameters):
  Input projection:   384 × 128 + 128 bias    = 49,280
  GRU layer 1:        3×(128×128 + 128×128 + 128+128) = 99,072
  GRU layer 2:        3×(128×128 + 128×128 + 128+128) = 99,072
  Output projection:  128 × 384 + 384 bias     = 49,536
  Total: 49,280 + 99,072 + 99,072 + 49,536     ≈ 297,216
```

**GPU memory** estimate:
- Model parameters: ~2 MB (float32)
- Batch of embeddings: 256 × 50 × 384 × 4 bytes = ~19 MB (worst case, MIND)
- Intermediate activations + gradients: ~2-3× model size ≈ ~6 MB
- Peak per batch: **<50 MB** (trivial for any modern GPU)
- The dominant memory cost is the frozen item embedding matrix loaded into CPU RAM (~723 MB for Amazon)

#### Q6: Why use the same in-batch negatives approach as Chapter 10?

**A:** Three reasons:

1. **Narrative continuity**: Chapter 10 introduced MNR loss for item embedding fine-tuning. Reusing the same loss function for sequence model training reinforces the concept rather than introducing yet another loss family. Students see the same objective applied in a different context.

2. **It directly optimizes what we evaluate**: Our evaluation uses FAISS cosine similarity retrieval. MNR loss optimizes cosine similarity rankings. This alignment between training objective and evaluation metric is good ML practice.

3. **Practical efficiency**: In-batch negatives are "free" — we already have a batch of target embeddings, so using them as mutual negatives requires zero additional computation. With batch_size=256, we get 65,280 negative comparisons per batch. The alternative (explicit negative sampling) adds complexity and memory overhead.

**When would you NOT use in-batch negatives?** When your batch can't provide representative negatives — e.g., if your batch contains mostly items from the same category (positional bias). Our random shuffling prevents this.

#### Q7: Will the Design Notes include ASCII diagrams and split counts?

**A:** Yes — see the Architecture, Data Splits, and Multi-Position Training sections above. All diagrams use ASCII art for maximum compatibility (renders in any text editor, terminal, or Jupyter notebook). Record counts are approximate because:
- Amazon total sessions depends on filtering (≥3 prev_items)
- MIND total users depends on filtering (≥5 clicks)
- The exact counts are printed to console during training and logged in `training_meta.json`

### Training Results

*Training completed 2026-03-03. All models trained on CUDA GPU.*

#### Convergence Summary

| Model | Dataset | Params | Epochs | Best Val Loss | Final Train Loss | Converged? |
|-------|---------|--------|--------|---------------|------------------|------------|
| SASRec | Amazon | 498,432 | 30/30 | 0.4376 | 0.4589 | Still improving (no early stop) |
| SASRec | MIND | 502,272 | 20/20 | 4.0704 | 4.0672 | Still improving (no early stop) |
| GRU4Rec | Amazon | 297,216 | 30/30 | 0.4403 | 4.0063 | Still improving (no early stop) |
| GRU4Rec | MIND | 297,216 | 20/20 | 4.0241 | 4.0063 | Still improving (no early stop) |

**Key observations:**

1. **No early stopping triggered**: All four models completed their full planned epochs with validation loss still decreasing at the final epoch. This means additional training could yield marginal improvements — but the learning curves show diminishing returns (the loss flattens significantly in the last 5-10 epochs).

2. **Amazon val losses are much lower than MIND**: Amazon achieves val_loss ~0.44 vs. MIND ~4.0. This reflects the task difficulty difference discussed in Section 11.1 — Amazon's next-item prediction from short sessions is a more tractable task than MIND's article prediction from long histories over a dense catalog.

3. **SASRec vs. GRU4Rec on Amazon**: Nearly identical validation loss (0.4376 vs. 0.4403). The architectures reach essentially the same quality level on short sessions, consistent with our hypothesis that recurrence suffices for short sequences.

4. **GRU4Rec edges out SASRec on MIND**: GRU4Rec achieves best_val_loss=4.024 vs. SASRec's 4.070, which is directionally opposite to our hypothesis that attention would win on longer sequences. This is an interesting finding worth discussing — see analysis below.

#### Training Curves

```
Amazon Validation Loss (both models superimposed):

val_loss
  1.3 ┤ ╲
  1.1 ┤  ╲          SASRec: ─────
  0.9 ┤   ╲         GRU4Rec: - - -
  0.7 ┤    ╲╲
  0.6 ┤     ╲╲
  0.5 ┤      ╲─────────────────────────────
  0.4 ┤       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→
         1  3  5  7  9  11 13 15 20 25 30  epoch

Both models converge rapidly in the first 10 epochs, then plateau.
Final gap: SASRec 0.4376 vs GRU4Rec 0.4403 (negligible difference).


MIND Validation Loss (both models superimposed):

val_loss
  4.7 ┤╲
  4.6 ┤ ╲╲
  4.5 ┤  ╲╲        SASRec: ─────
  4.4 ┤   ╲╲       GRU4Rec: - - -
  4.3 ┤    ╲╲
  4.2 ┤     ╲─ ─
  4.1 ┤      ──────────────────────
  4.0 ┤        ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──→
         1  3  5  7  9  11 13 15 18 20  epoch

GRU4Rec converges slightly faster and to a better final loss.
Final gap: SASRec 4.070 vs GRU4Rec 4.024.
```

#### Training Times

| Model | Dataset | Actual Wall-Clock Time | Expected GPU Time | Notes |
|-------|---------|----------------------|-------------------|-------|
| SASRec | Amazon | 14.8 hours | ~22 min | Likely affected by laptop sleep (see below) |
| SASRec | MIND | 8.7 min | ~2 min | Reasonable — small dataset overhead |
| GRU4Rec | Amazon | 30.8 hours | ~15 min | Likely affected by laptop sleep (see below) |
| GRU4Rec | MIND | 8.9 min | ~1 min | Reasonable — small dataset overhead |

#### Why Did Amazon Training Take So Long?

The Amazon models took 15-31 hours of wall-clock time despite running on a CUDA GPU, compared to 8-9 minutes for MIND. Three factors explain this:

**Factor 1: Dataset size (expected)**
Amazon has ~641K training sequences vs. MIND's ~36K — an **18× difference** in data volume. Combined with 30 epochs (Amazon) vs. 20 epochs (MIND), the total training iterations are ~27× higher for Amazon. This accounts for roughly a 27× longer training time in pure compute.

**Factor 2: Multi-position loss loop (expected)**
Multi-position training evaluates the loss at every valid position in each sequence. Amazon sessions have median 4 items → ~3 loss computations per sequence. MIND histories have median 11 items → ~10 loss computations per sequence. However, Amazon has 18× more sequences, so the total position evaluations are still much higher for Amazon.

**Factor 3: Laptop sleep / power management (primary cause of discrepancy)**
The wall-clock times are dramatically higher than expected GPU compute times:
- SASRec Amazon: 14.8 hrs actual vs. ~22 min expected (~40× slower)
- GRU4Rec Amazon: 30.8 hrs actual vs. ~15 min expected (~120× slower)
- MIND models: ~9 min actual vs. ~2 min expected (~4× slower, reasonable)

The MIND models trained in a plausible time range (overhead from data loading, Python/PyTorch startup, etc.). The Amazon models' extreme slowdown almost certainly includes periods where the laptop entered sleep/hibernate mode during the long-running process. The training script uses wall-clock timing (`time.time()`), so sleep time is included in the reported duration.

**For students**: In production ML training, use epoch-level timing or GPU utilization metrics rather than wall-clock time. Tools like `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 5` can log GPU utilization to detect idle periods.

## Section 11.3: Graph Neural Networks (LightGCN)

### Why LightGCN?

Section 11.2 revealed a critical failure: SASRec and GRU4Rec with MNR loss and frozen SBERT embeddings suffered **mode collapse on MIND** (SASRec MRR=0.0007, GRU4Rec MRR=0.0016 — both worse than simple TF-IDF aggregation at MRR=0.0034). The root cause is MNR loss creating dense similarity matrices across MIND's topically clustered news articles, where many articles have nearly identical SBERT embeddings.

LightGCN (He et al., SIGIR 2020) was chosen for Section 11.3 for five reasons:

1. **Simplest GNN architecture**: LightGCN removes ALL feature transformations and nonlinearities from standard GCN. The entire model is just sparse matrix multiply + layer averaging. This makes it ideal for a textbook introduction to graph-based recommendations.

2. **Directly addresses the 11.2 failure**: BPR (pairwise) loss avoids the dense similarity matrix that caused mode collapse with MNR. Learnable embeddings differentiate articles by behavioral co-occurrence rather than content similarity. This provides a clear "diagnosis then fix" narrative arc for the chapter.

3. **Foundational**: LightGCN is the most-cited GNN-for-RecSys paper (3000+ citations). Understanding it is prerequisite for more advanced methods (PinSage, NGCF, GraphSAGE, GAT, KGAT).

4. **Pure PyTorch — no external graph library**: Implementation uses only `torch.sparse.mm()`. Students do not need to install torch-geometric, keeping the dependency footprint minimal.

5. **Completes the paradigm trifecta**: Section 11.1 = aggregation, 11.2 = sequential, 11.3 = graph-based — the three dominant paradigms for user embedding construction in modern recommender systems.

**MIND-only**: Amazon KDD has no persistent user IDs (anonymous sessions), so a bipartite user-item graph cannot be constructed. LightGCN requires knowing "which user clicked which item" across sessions.

### GNN Primer: Bipartite Graphs for Recommendation

For readers new to Graph Neural Networks, here is a conceptual walkthrough of what LightGCN does.

#### What is a bipartite graph?

A bipartite graph has two types of nodes — **users** and **items** — with edges ONLY between different types (never user-to-user or item-to-item). Each edge represents "user U clicked/read article I":

```
Users          Items (Articles)
  U1 ─────────── A1 (politics)
  U1 ─────────── A2 (sports)
  U2 ─────────── A2 (sports)
  U2 ─────────── A3 (sports)
  U3 ─────────── A1 (politics)
  U3 ─────────── A4 (tech)
```

The graph is **static and unordered** — we deliberately discard when interactions happened and in what order. The graph only knows "user X clicked article Y", not "when" or "in what sequence." This is a fundamental difference from Section 11.2's sequence models, which preserve temporal ordering.

#### What does LightGCN's graph propagation do?

LightGCN performs K rounds of **neighborhood aggregation** via sparse matrix multiplication:

```
E^{(k+1)} = D^{-1/2} A D^{-1/2} E^{(k)}
```

In plain language, each round makes every node's embedding become the weighted average of its neighbors' embeddings:

- **Layer 1**: Each user absorbs information from the articles they read; each article absorbs information from the users who read it. This is direct neighbor aggregation (1-hop).
- **Layer 2**: Each user now indirectly receives information from other users who read the same articles (2-hop: user → article → other user). This is collaborative filtering — the classic "users who read X also read Y" signal.
- **Layer 3**: 3-hop extends further — users connected through chains of shared articles, capturing broader community-level patterns.

The adjacency matrix is **symmetrically normalized** (`D^{-1/2} A D^{-1/2}`), so high-degree nodes do not dominate:
- A user who reads 100 articles sends weaker signal per edge than a user who reads 10
- A popular article clicked by 1000 users contributes less per edge than a niche article clicked by 10

#### Why average all layers?

LightGCN averages embeddings from ALL K+1 layers (including the initial layer 0):

```
E_final = (1/(K+1)) * (E^{(0)} + E^{(1)} + ... + E^{(K)})
```

Each layer captures a different scale of information:
- E^{(0)} = the node's own identity (no neighbor info)
- E^{(1)} = direct neighbors (1-hop: users ↔ their articles)
- E^{(2)} = 2-hop neighborhood (collaborative filtering: users who read the same articles)
- E^{(3)} = 3-hop neighborhood (broader community patterns through chains of shared articles)

Averaging combines local identity with increasingly broad collaborative signal. Note that collaborative filtering begins at layer 2 (2-hop), where users connect to other users through shared items.

#### The adjacency matrix structure

The bipartite graph is represented as a single symmetric adjacency matrix:

```
A = | 0    R   |   where R is (num_users x num_items) binary interaction matrix
    | R^T  0   |   and A is (N x N) with N = num_users + num_items
```

- The upper-right block R encodes user-to-item edges
- The lower-left block R^T encodes item-to-user edges (same edges, reversed direction)
- The diagonal blocks are zero (no user-user or item-item edges in base LightGCN)

Node indexing in the combined matrix: users occupy indices 0 to num_users-1, items occupy indices num_users to N-1. So item with `item_idx=5` gets global index `num_users + 5`.

### What Features Does the Graph Use?

**None.** This is a critical point that distinguishes LightGCN from content-based approaches.

LightGCN uses **zero content features** — no category, no title, no abstract, no SBERT embeddings. The graph captures ONLY the behavioral signal: "who clicked what." LightGCN learns its own embeddings purely from this structural information via `nn.Embedding`.

The `valid_news` filter in `mind_graph_builder.py` is purely a catalog consistency check — it ensures articles exist in our item catalog (have an embedding from Chapter 10). It does NOT use the embedding values themselves. Similarly, `item_id_to_idx` is a simple string-to-integer mapping (e.g., `{"N12345": 0, "N67890": 1, ...}`) that gives each article a 0-based integer index for `nn.Embedding` lookup. It carries no semantic information.

**Why does this work?** Because behavioral co-occurrence already captures semantic relationships. If many users read both article A (sports) and article B (sports), LightGCN's propagation will make their embeddings similar — without ever seeing the category label "sports." The graph structure implicitly encodes content similarity through user behavior.

#### Can metadata be added to the graph?

Yes. Several GNN architectures extend LightGCN by incorporating node features:

| Architecture | How It Uses Features | Trade-off |
|---|---|---|
| **NGCF** (Wang et al., 2019) | Adds weight matrices W at each layer | More parameters, can overfit |
| **PinSage** (Ying et al., 2018) | Uses content features as initial node embeddings | Combines content + collaborative |
| **GraphSAGE** (Hamilton et al., 2017) | Learnable aggregation functions (mean, LSTM, pooling) | More flexible but more complex |
| **GAT** (Velickovic et al., 2018) | Attention weights per edge | Different neighbors contribute differently |
| **KGAT** (Wang et al., 2019) | Knowledge graph attributes on edges | Requires structured metadata |

LightGCN's contribution was showing that removing all these components (no transforms, no nonlinearities) actually **improves** recommendation performance on standard benchmarks. The simplicity is the feature.

### LightGCN Architecture

The only learnable parameters are two embedding tables:

```python
user_embedding = nn.Embedding(num_users, hidden_dim)   # ~40K x 64 = 2.6M params
item_embedding = nn.Embedding(num_items, hidden_dim)    # ~51K x 64 = 3.3M params
# Total: ~5.9M parameters (ALL in embeddings — no other parameters)
```

The forward pass:

```
Step 1: Concatenate user + item embeddings -> E^{(0)} of shape (N, 64)
Step 2: K rounds of torch.sparse.mm(adj_norm, E^{(k)}) -> E^{(k+1)}
Step 3: Average all K+1 layers -> E_final
Step 4: Split back into user_embeddings (num_users, 64) and item_embeddings (num_items, 64)
```

**No weight matrices. No activation functions. No dropout by default.** The entire model is parameter-free graph propagation applied to learnable embedding tables.

#### Why Only Two Embedding Tables Despite Multiple Layers?

This is a common point of confusion for readers familiar with standard neural networks, where each layer has its own weight matrix. In LightGCN, each propagation layer has **zero learnable parameters** — it is pure matrix multiplication with the fixed adjacency matrix. Here is a schematic comparison:

```
STANDARD NEURAL NETWORK (e.g., 3-layer MLP):
  Input → [W1, b1] → ReLU → [W2, b2] → ReLU → [W3, b3] → Output
          ^^^^^^^^           ^^^^^^^^           ^^^^^^^^
          Layer 1            Layer 2            Layer 3
          (learnable)        (learnable)        (learnable)

  Total learnable: W1 + W2 + W3 + biases = many parameter matrices


LightGCN (3-layer GCN):
  E^(0) → sparse_mm(A_norm, ·) → E^(1) → sparse_mm(A_norm, ·) → E^(2) → sparse_mm(A_norm, ·) → E^(3)
  ^^^^^   ^^^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^
  ONLY    Layer 1                        Layer 2                        Layer 3
  learnable  (NO parameters —            (NO parameters —               (NO parameters —
  part       just multiply by             just multiply by               just multiply by
             fixed graph)                 fixed graph)                   fixed graph)

  Final = average(E^(0), E^(1), E^(2), E^(3))

  Total learnable: E^(0) ONLY = user_embedding + item_embedding = 2 tables
```

The graph layers are like mirrors in a hall of mirrors — they reflect and propagate information, but the mirrors themselves have no adjustable knobs. The only thing you can adjust is what you put IN to the mirrors (the initial embeddings E^{(0)}).

**Is this specific to LightGCN, or all GNNs?** This is specific to LightGCN. Other GNN architectures DO add learnable parameters per layer:

```
NGCF (3-layer):
  E^(0) → [W1] × sparse_mm(A, ·) → E^(1) → [W2] × sparse_mm(A, ·) → E^(2) → ...
           ^^^^                               ^^^^
           Learnable weight matrix             Learnable weight matrix
           per layer                           per layer

GAT (3-layer):
  E^(0) → attention(a1) × sparse_mm(A, ·) → E^(1) → attention(a2) × sparse_mm(A, ·) → ...
           ^^^^^^^^^^^^^                               ^^^^^^^^^^^^^
           Learnable attention                         Learnable attention
           parameters per layer                        parameters per layer

GraphSAGE (3-layer):
  E^(0) → [W1] × aggregate(neighbors) → E^(1) → [W2] × aggregate(neighbors) → ...
           ^^^^                                    ^^^^
           Learnable transform                     Learnable transform
           per layer                               per layer
```

LightGCN's key finding was that removing all these per-layer parameters actually IMPROVES recommendation quality on standard benchmarks. The intuition: for bipartite user-item graphs, the graph structure already encodes enough information — per-layer transforms add parameters that overfit without adding useful expressiveness.

#### Full-batch GCN

The MIND graph is small enough (~91K nodes, ~400K edges) for full-batch processing. Each forward pass processes the ENTIRE graph — no graph sampling needed. Three sparse matrix multiplications of ~400K non-zeros times 64 dimensions takes approximately 6ms on GPU.

#### Distinct user and item embeddings

After training, LightGCN produces BOTH user embeddings and item embeddings in the same 64-dimensional space. This is different from Sections 11.1/11.2, which operated in the 384-dim SBERT space.

```
LightGCN forward pass:
  Input:  adj_norm (sparse graph)
  Output: user_embeddings (40K, 64)  <-- user representations
          item_embeddings (51K, 64)  <-- item representations

Relevance score: dot_product(user_emb[u], item_emb[i])
```

Both sets of embeddings live in the same learned space, enabling direct user-item similarity computation via dot product.

### Loss Function: BPR vs MNR

#### BPR loss (pairwise)

BPR (Bayesian Personalized Ranking) is a pairwise loss:

```
loss = -log(sigmoid(score_pos - score_neg))
```

For each training edge (user, positive_item), we sample one random negative item (an item the user has NOT clicked). The loss pushes the positive item's score above the negative item's score by a margin.

**Why BPR is conservative but stable**: BPR compares one positive against one negative at a time. It never builds a dense B-times-B similarity matrix like MNR does. This makes training very stable (ran all 100 epochs with steady improvement), but the gradient signal is weak — only 1 negative comparison per update.

Negative sampling uses rejection: randomly sample an item index, reject if it is in the user's positive set. Since each user has roughly 11 positives out of 51K items (< 0.02%), rejection terminates in approximately 1 attempt.

#### MNR loss (in-batch negatives — surprise winner)

The MNR variant uses in-batch negatives, same as Section 11.2. It was originally included for pedagogical comparison, expected to fail similarly. Instead, **MNR produced the best MIND results of any method in the entire chapter** (MRR=0.0100 vs BPR's 0.0051). See [Results](#results-2) for analysis of why learnable embeddings fix MNR's dense-cluster problem.

#### L2 regularization

LightGCN regularizes the initial (pre-GCN) embeddings E^{(0)}, not the propagated embeddings. This is because graph propagation is parameter-free — all learnable parameters live in the initial embedding tables.

```python
reg = (user_embs_0.norm(2)^2 + pos_item_embs_0.norm(2)^2 + neg_item_embs_0.norm(2)^2) / batch_size
total_loss = bpr_loss + l2_reg_weight * reg
```

#### L2 normalization (for retrieval)

L2 normalization and L2 regularization are different operations that serve different purposes:

**L2 regularization** (during training): Penalizes large embedding values to prevent overfitting. Adds `lambda * ||e||^2` to the loss. This keeps embeddings small but does NOT make them unit-length.

**L2 normalization** (during inference): Scales each embedding vector to unit length (magnitude = 1.0). Every vector is divided by its own L2 norm:

```
                                    [3, 4]
raw_embedding = [3, 4]  →  norm = sqrt(3² + 4²) = 5  →  normalized = ───── = [0.6, 0.8]
                                      5

After normalization: ||[0.6, 0.8]|| = sqrt(0.36 + 0.64) = 1.0  ✓
```

**Why normalize before FAISS retrieval?** When all vectors have unit length, dot product becomes equivalent to cosine similarity:

```
dot_product(a, b) = ||a|| × ||b|| × cos(θ)

If ||a|| = ||b|| = 1:  dot_product(a, b) = cos(θ)
```

This makes similarity comparisons fair — a user who read 50 articles would otherwise have a larger averaged embedding (larger magnitude) than a user who read 3 articles. Without normalization, FAISS would rank users with larger embeddings higher regardless of directional similarity. L2 normalization removes this magnitude bias so retrieval is based purely on directional alignment between user and item vectors.

In our code, both item embeddings (before indexing in FAISS) and user embeddings (before querying FAISS) are L2-normalized. This ensures:
1. Popular items with many graph connections don't get artificially boosted
2. Active users with many history items don't dominate similarity scores
3. Dot-product search (which FAISS optimizes for) is equivalent to cosine similarity

### Cold-Start Evaluation

Evaluation users (10K, same seed=42 as Sections 11.1/11.2) are **excluded from the training graph**. They are cold-start users — LightGCN has never seen them during training.

**How cold-start user embeddings are computed at evaluation time**:

```
eval_user_emb = mean(post_GCN_item_embs[history_items])
```

We take the user's reading history (known articles), look up each article's post-propagation LightGCN embedding, and average them. This is equivalent to one round of LightGCN propagation for a new user node connected to their history items. The resulting embedding is then L2-normalized for FAISS retrieval.

**Important**: The FAISS index for LightGCN evaluation is built from 64-dimensional LightGCN item embeddings, NOT the 384-dimensional SBERT embeddings used in Sections 11.1/11.2. Comparison across sections is on metrics (MRR, Recall@K, NDCG@K), not embedding space.

### Production Inference Pattern

In a production deployment, the inference pattern splits into offline and online components:

```
OFFLINE (batch, periodic):
  1. Run full graph forward: user_embs, item_embs = model(adj_norm)
  2. Index all item_embs in a vector database (FAISS, Milvus, Pinecone)
  3. Cache known user_embs (for users already in the graph)

ONLINE (per request, real-time):
  For a KNOWN user (exists in training graph):
    1. Look up cached user embedding
    2. Query vector DB for top-K nearest items
    3. Return recommendations

  For a COLD-START user (new user, not in graph):
    1. Collect their recent interactions (e.g., last 10 articles read)
    2. Average the cached item embeddings: user_emb = mean(item_embs[history])
    3. L2-normalize
    4. Query vector DB for top-K nearest items
    5. Return recommendations
```

This offline/online split is standard for graph-based recommendation systems. The full graph forward pass is computationally expensive (processes all nodes), so it runs periodically (hourly/daily). Item embeddings change slowly, so the vector index stays valid between updates. User embeddings for known users can be pre-computed and cached. Cold-start users get approximate embeddings on-the-fly by averaging item vectors.

### Hyperparameter Grounding

All hyperparameters are justified by either the original LightGCN paper or data-driven analysis:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| hidden_dim | 64 | LightGCN convention; smaller than sequence models because graph propagation is parameter-free — the model has fewer parameters to overfit |
| num_layers | 3 | Original LightGCN default; captures up to 3-hop neighborhood |
| dropout | 0.0 | Standard for LightGCN — no dropout because the model has no weight matrices to overfit; the only parameters are the embeddings |
| learning_rate | 1e-3 | Standard for LightGCN with AdamW |
| l2_reg_weight | 1e-4 | Standard LightGCN regularization on initial embeddings |
| batch_size | 1024 | Edges per batch (not users); balances GPU utilization with update frequency |
| epochs | 100 | With early stopping (patience=10); LightGCN converges slowly |
| patience | 10 | More patience than sequence models because BPR loss is noisier (single negative sample) |
| neg_samples | 1 per positive | Standard; more negatives add compute cost without proportional benefit at this scale |
| train_fraction | 0.9 | 90/10 edge split for train/validation |

#### Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| User embeddings | 40K x 64 x 4B = 10 MB | Learnable |
| Item embeddings | 51K x 64 x 4B = 13 MB | Learnable |
| Sparse adjacency | ~400K non-zeros x 12B = 5 MB | Symmetric edges |
| Layer stacking | 4 x 91K x 64 x 4B = 93 MB | K+1 layers during forward |
| **Total GPU** | **~121 MB** | **Well within 16GB constraint** |

### Results

#### Training Summary

| Variant | Epochs | Best Val Loss | Training Time | Early Stop? |
|---------|--------|---------------|---------------|-------------|
| LightGCN-BPR | 100 (of 100) | 0.0620 | 8,795s (~2.4 hrs) | No — ran full 100 epochs |
| LightGCN-MNR | 14 (of 100) | 6.609 | 1,314s (~22 min) | Yes — patience=10, val loss diverged after epoch 4 |
| LightGCN-BPR (SBERT-init) | 100 (of 100) | 0.0604 | 5,094s (~1.4 hrs) | No — ran full 100 epochs |
| LightGCN-MNR (SBERT-init) | 13 (of 100) | 6.298 | 671s (~11 min) | Yes — patience=10, val loss diverged after epoch 3 |

**BPR training convergence**: Steady improvement over all 100 epochs. Train loss dropped from 0.296 (epoch 1) → 0.036 (epoch 100). Validation loss from 0.188 → 0.062. No overfitting observed — the model was still improving slowly, suggesting more epochs or a learning rate schedule could help further.

**MNR training divergence**: Rapid initial improvement (epochs 1-4), then validation loss started climbing while training loss continued to decrease — classic overfitting pattern. The model memorized in-batch discriminations but failed to generalize. Early stopping triggered at epoch 14.

**SBERT-init training observations**: Both variants show faster convergence and better final validation losses compared to random initialization. BPR-SBERT reached val loss 0.0604 (vs. 0.0620 for random) in 42% less time. MNR-SBERT reached val loss 6.298 (vs. 6.609 for random) in 49% less time. The content-aware starting point gives the optimizer a head start — the PCA-projected SBERT embeddings provide a meaningful initial item geometry that the graph propagation can refine, rather than starting from scratch.

#### Retrieval Metrics (MIND, 10K cold-start eval users)

| Method | Section | MRR | R@1 | R@10 | R@20 | R@50 |
|--------|---------|-----|-----|------|------|------|
| **LightGCN-MNR (3L, 64d, SBERT-init)** | **11.3** | **0.0174** | **0.0050** | **0.0407** | **0.0696** | **0.1306** |
| LightGCN-MNR (3L, 64d) | 11.3 | 0.0100 | 0.0022 | 0.0237 | 0.0475 | 0.1032 |
| LightGCN-BPR (3L, 64d, SBERT-init) | 11.3 | 0.0058 | 0.0008 | 0.0127 | 0.0282 | 0.0739 |
| LightGCN-BPR (3L, 64d) | 11.3 | 0.0051 | 0.0010 | 0.0098 | 0.0233 | 0.0659 |
| Best Aggregation (Last-11 Mean) | 11.1 | 0.0031 | 0.0014 | 0.0064 | 0.0100 | 0.0219 |
| Best Sequence (GRU4Rec) | 11.2 | 0.0016 | 0.0004 | 0.0036 | 0.0069 | 0.0143 |

**LightGCN-MNR with SBERT initialization is the undisputed best MIND method**: MRR=0.0174 is 5.6x better than the best aggregation (0.0031) and 10.9x better than the best sequence model (0.0016). At R@50, it retrieves a relevant article in 13.1% of queries vs. 2.2% for best aggregation — a 5.9x improvement. SBERT initialization boosted MNR's MRR by 74% over random init (0.0174 vs. 0.0100).

**SBERT initialization improves both loss variants**: BPR gained +14% MRR (0.0051 → 0.0058), MNR gained +74% MRR (0.0100 → 0.0174). The asymmetric improvement is analyzed in detail below.

**LightGCN-BPR also outperforms all prior methods**: Even with random init, MRR=0.0051 is 1.6x better than best aggregation, and SBERT init pushes this to 1.9x.

#### The Surprise: MNR Outperformed BPR

This contradicts our hypothesis that MNR loss would fail on MIND (as it did in Section 11.2). The key difference: **learnable embeddings**.

| Factor | Section 11.2 (Sequence + MNR) | Section 11.3 (LightGCN + MNR) |
|--------|-------------------------------|-------------------------------|
| Item representations | Frozen SBERT (384d) | Learnable nn.Embedding (64d) |
| Similarity structure | Dense topic clusters (many near-identical vectors) | Randomly initialized (uniformly spread in space) |
| MNR in-batch negatives | Cannot discriminate — targets indistinguishable from negatives | CAN discriminate — each item has a unique learned vector |
| Result | Mode collapse (MRR=0.0007) | Best model (MRR=0.0100) |

**Why MNR works with learnable embeddings**: When each item starts with a unique random vector, the in-batch negative contrast has clear signal. The model can push "this article" away from "those 1023 other articles" because they start out distinguishable. With frozen SBERT, many news articles about the same topic have nearly identical 384-dim vectors, making the contrast signal contradictory.

**Why MNR beats BPR here**: MNR's in-batch negatives provide 1023 negative comparisons per positive (with batch_size=1024). BPR provides only 1 random negative per positive. The richer gradient signal from MNR leads to faster and more informative learning — but ONLY when the embedding space is sufficiently diverse (i.e., not frozen SBERT topic clusters).

**Key pedagogical takeaway**: The failure of MNR in Section 11.2 was NOT a property of MNR loss itself, but of the combination of MNR + frozen content embeddings in a topically clustered catalog. LightGCN's learnable embeddings resolve this by giving each item a distinct starting point, confirming that the root cause was the embedding space structure, not the loss function.

#### SBERT-Initialized LightGCN: Content Meets Collaborative

Initializing LightGCN's item embeddings with PCA-projected SBERT vectors (384d → 64d via SVD, retaining ~58% variance) instead of random initialization dramatically improves retrieval quality:

| Variant | Random Init MRR | SBERT Init MRR | Improvement | Random Init R@50 | SBERT Init R@50 | Improvement |
|---------|----------------|----------------|-------------|-----------------|----------------|-------------|
| LightGCN-BPR | 0.0051 | 0.0058 | +14% | 0.0659 | 0.0739 | +12% |
| LightGCN-MNR | 0.0100 | 0.0174 | **+74%** | 0.1032 | 0.1306 | **+27%** |

**Why MNR benefits more than BPR from SBERT initialization**:

The asymmetry is striking — MNR gains 74% while BPR gains only 14%. This comes down to how each loss function exploits the initial embedding geometry:

- **BPR** compares one positive against one random negative per update. The quality of the starting embeddings matters less because BPR's gradient signal is already weak (one comparison per step). Whether items start in meaningful positions or random positions, BPR makes slow, incremental adjustments either way.

- **MNR** compares one positive against 1,023 in-batch negatives simultaneously. The quality of the starting embeddings matters enormously because MNR's gradient is informed by the full batch structure. When SBERT places semantically similar articles near each other, MNR can immediately distinguish "this specific politics article" from "those sports, tech, and entertainment articles in the batch." With random init, MNR must first learn these coarse topic boundaries before it can make fine-grained distinctions — SBERT initialization effectively skips this phase.

**This is a "poor man's PinSAGE"**: Full PinSAGE (Hamilton et al., 2017) incorporates content features through learned message-passing aggregators at every layer — requiring graph sampling, feature transformation matrices, and significantly more complexity. Our approach achieves a simpler version of the same idea: inject content knowledge once at initialization, then let LightGCN's parameter-free graph propagation refine it. The 74% MNR improvement suggests that even this lightweight approach captures substantial value from content features.

**Training efficiency gains**: SBERT initialization also makes training faster. BPR-SBERT trained in 5,094s vs. 8,795s for random init (42% faster). MNR-SBERT trained in 671s vs. 1,314s (49% faster). The content-aware starting point reduces the number of gradient steps needed to reach a good solution.

**Connection to the chapter narrative**: This result completes the story arc connecting Chapter 10 (item embeddings) to Chapter 11 (user embeddings). Sections 11.1 and 11.2 consumed Chapter 10's SBERT embeddings directly as frozen inputs. Section 11.3 initially learned its own embeddings from scratch. The SBERT-init variant bridges these approaches — using content embeddings as a warm start for collaborative learning, demonstrating that content and collaborative signals are complementary, not competing.

#### Section 11.2 Item-ID Variant Results

The Item-ID variant of SASRec/GRU4Rec (replacing frozen SBERT with learnable `nn.Embedding`) also completed training and evaluation:

**Amazon KDD (20K queries):**

| Method | MRR | R@10 | R@50 |
|--------|-----|------|------|
| Last-Item Only (Ch10) | 0.2021 | 0.3611 | 0.5214 |
| SASRec (frozen SBERT) | 0.1512 | 0.2916 | 0.4746 |
| GRU4Rec (frozen SBERT) | 0.1435 | 0.2855 | 0.4703 |
| SASRec-ItemID (learnable) | 0.0933 | 0.2116 | 0.4178 |
| GRU4Rec-ItemID (learnable) | 0.0884 | 0.1985 | 0.4062 |

**MIND (10K queries):**

| Method | MRR | R@10 | R@50 |
|--------|-----|------|------|
| SASRec-ItemID (learnable) | 0.0006 | 0.0018 | 0.0080 |
| GRU4Rec-ItemID (learnable) | 0.0005 | 0.0011 | 0.0057 |

**Item-ID variants significantly underperform frozen SBERT on Amazon** (0.093 vs 0.151 MRR). With 1.55M products and only 641K training sequences, learnable embeddings cannot adequately represent the full item catalog. Frozen SBERT provides strong content-based priors that generalize better in this sparse regime.

**Item-ID variants also fail on MIND** — even worse than frozen SBERT variants. With MNR loss and limited training data (36K sequences, 51K items), the learnable embeddings don't converge to useful representations within the sequence model framework.

**Contrast with LightGCN**: LightGCN's learnable embeddings succeed on MIND where Item-ID sequence models fail. The difference is the training signal: LightGCN uses 862K edges with BPR/MNR loss across the full graph structure, while sequence models use only 36K sequences with MNR loss. The graph structure provides a much richer training signal per parameter.

### Homework Exercises

#### Exercise 1: Yandex Yambda Music Dataset

The **Yandex Yambda** music listening dataset (already in the codebase at `Dataset\Yandex\`, used in Chapter 10 for Item2Vec) is an excellent candidate for applying LightGCN to a second domain:

| Property | Yandex Yambda | MIND |
|----------|---------------|------|
| Users | 9,238 | ~40K |
| Items | 877,168 tracks | ~51K articles |
| Interactions | 46.5M listens | ~500K clicks |
| User IDs | Yes (`uid` column) | Yes |
| Edge weights | `played_ratio_pct` (0-100+) | Binary (clicked) |
| Domain | Music | News |

**Assignment**: Adapt `mind_graph_builder.py` to construct a bipartite graph from the Yandex Yambda dataset. Key considerations:

1. **Graph density**: Yandex has 46.5M interactions vs. MIND's ~500K — nearly 100x denser. How does this affect convergence speed, memory requirements, and whether full-batch GCN is still feasible?
2. **Weighted edges**: Yandex provides `played_ratio_pct` (how much of the track was played). Should edges be weighted by engagement, or kept binary? How does weighting change the normalized adjacency?
3. **Item catalog size**: 877K tracks vs. 51K articles — much larger item embedding table. What is the impact on parameter count and training time?
4. **Domain differences**: Music listening is repetitive (users replay favorite songs), news reading is not (users rarely re-read articles). How does this affect the graph structure and LightGCN's assumptions?

The Yandex dataset already has a `YandexDataset` loader in `chapter10_item_embeddings/data/yandex_dataset.py` and pre-computed audio CNN embeddings are available for comparison (content-based vs. collaborative).

#### Exercise 2: Layer Analysis

Modify `evaluate_lightgcn.py` to compute retrieval metrics using each individual layer's embeddings (E^{(0)}, E^{(1)}, E^{(2)}, E^{(3)}) instead of the averaged E_final. Which layer performs best? Does the averaged version outperform any individual layer?

#### Exercise 3: Embedding Dimension Sweep

Train LightGCN with hidden_dim in {16, 32, 64, 128, 256}. Plot MRR vs. embedding dimension. Is there a point of diminishing returns? How does the optimal dimension compare to the 384-dim SBERT space used in Sections 11.1/11.2?

#### Exercise 4: Number of Layers

Train with num_layers in {1, 2, 3, 4, 5}. More layers capture longer-range collaborative signal but also risk over-smoothing (all embeddings converge to the same vector). At what depth does performance plateau or degrade?

### MIND Dataset: Impressions and Timelines

For readers unfamiliar with the MIND dataset structure, some clarifications:

**Impressions = page loads**: In MIND, an "impression" corresponds to a single page load of the Microsoft News homepage. The user sees approximately 15-20 headline candidates per page load.

**Multiple clicks per impression**: Users CAN click multiple articles within a single impression.(Note that this impression definition is slightly different from other use cases where an impression is associated with exactly 1 item.) For example, the behavior line `N12345-1 N67890-0 N11111-1` means the user saw three articles, clicked articles N12345 and N11111, and did not click N67890. This may seem counterintuitive (one might expect "impression = one item shown"), but in MIND an impression is more like a "session page view" where multiple articles are displayed and the user may interact with several.

**Point-in-time accuracy**: The `history_list` field in each impression record represents the user's cumulative click history BEFORE that page load — it is point-in-time accurate. However, LightGCN's graph construction deliberately discards this temporal ordering. The graph builder walks ALL impressions for each user and unions ALL clicked articles into a single set. This is by design: LightGCN's bipartite graph is static and unordered, capturing "who clicked what" without "when" or "in what order."

**Why discard temporal information?** LightGCN's strength is capturing collaborative patterns through graph structure. Temporal ordering is the domain of sequence models (Section 11.2). The graph's power comes from connecting users through shared items, revealing patterns like "users who read article A also tend to read article B" — regardless of reading order.

---

## Cross-Chapter Integration

### From Chapter 10 (Item Embeddings) → Chapter 11 (User Embeddings)

- **Reused artifacts**: Cached .npz item embeddings (no re-encoding).
- **Reused evaluation**: Same metrics (MRR, Recall@K, NDCG@K), same scale (20K queries).
- **New dimension**: Chapter 10 asked "which item is similar to this item?" Chapter 11 asks "which item is relevant to this user?"

### From Chapter 11 → Chapter 7 (Ranking Models)

- **Homework exercise**: Add trained user embeddings as features to the SIGIR ranking model.
- **Expected lift**: User embeddings capture personalization signal that item features alone miss.

### Weighted Concatenation Principle (from Chapter 10.4)

Chapter 10.4 established that combining heterogeneous embeddings reduces to weighted concatenation of normalized sub-vectors with query-time weight control. This principle applies when combining user embeddings with other features for downstream ranking:

```
combined = [w_user * norm(user_emb), w_item * norm(item_emb), w_context * norm(context_features)]
```

The weights can be tuned per use case (e.g., higher w_user for returning users, lower for anonymous sessions).

---

## Code Architecture

```
chapter11_user_embeddings/
├── config.py                      # All configs (dataclass-based)
│                                  # - AmazonSessionConfig
│                                  # - MINDUserConfig
│                                  # - AggregationConfig
│                                  # - EvaluationConfig
│                                  # - SequenceModelConfig (11.2)
│                                  # - LightGCNConfig (11.3)
├── data/
│   ├── __init__.py
│   ├── amazon_session_loader.py   # Session loading + embedding mapping
│   ├── mind_user_loader.py        # User history loading + embedding mapping
│   └── mind_graph_builder.py      # Bipartite graph construction (11.3, MIND-only)
├── models/
│   ├── __init__.py
│   ├── aggregators.py             # 5 aggregation methods (ABC pattern)
│   ├── sequence_models.py         # SASRec + GRU4Rec (Section 11.2)
│   └── lightgcn.py                # LightGCN graph neural network (Section 11.3)
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # Ranking metrics (from Ch10)
│   └── faiss_index.py             # FAISS wrapper (from Ch10)
├── data_analysis.py               # Empirical distributions for hyperparameters
├── evaluate_aggregation.py        # Main evaluation script (Section 11.1)
├── train_sequence.py              # Training script (Section 11.2)
├── evaluate_sequence.py           # Evaluation script (Section 11.2)
├── train_lightgcn.py              # Training script (Section 11.3, BPR + MNR)
├── evaluate_lightgcn.py           # Evaluation script (Section 11.3, cold-start)
├── docs/
│   └── Chapter11_Design_Notes.md  # This file
├── outputs/
│   ├── embeddings/                # Saved user embeddings (.npz)
│   ├── metrics/                   # JSON evaluation results
│   ├── models/                    # Trained model checkpoints
│   │   ├── sequence_models/       # Section 11.2
│   │   │   ├── sasrec_amazon/     # best_model.pt + training_meta.json
│   │   │   ├── sasrec_mind/
│   │   │   ├── gru4rec_amazon/
│   │   │   └── gru4rec_mind/
│   │   └── lightgcn/              # Section 11.3
│   │       ├── lightgcn_bpr/      # best_model.pt + training_meta.json + graph_mappings.npz
│   │       ├── lightgcn_mnr/
│   │       ├── lightgcn_bpr_sbert/ # SBERT-initialized variant (--sbert_init)
│   │       └── lightgcn_mnr_sbert/
│   └── analysis/                  # Distribution stats from data_analysis.py
└── requirements.txt
```

### Design Patterns

1. **Dataclass configs** with sensible defaults and data-driven overrides.
2. **ABC aggregator hierarchy**: All methods inherit from `UserEmbeddingAggregator`, share normalize/aggregate interface.
3. **Factory function** (`create_aggregators`) builds all methods with data-driven hyperparameters.
4. **Polars for data loading** (memory-efficient alternative to pandas, per constraint).
5. **FAISS with brute-force fallback** for portability.
6. **Chunked/lazy loading** for large datasets (Amazon: 1.55M products, 3.6M sessions).

---

## Change Log

| Date | Section | Change | Author |
|------|---------|--------|--------|
| 2026-03-01 | 11.1 | Initial implementation: 5 aggregation methods, Amazon KDD + MIND evaluation, data analysis script | Claude + Author |
| 2026-03-01 | 11.2 | Added SASRec and GRU4Rec sequence models with frozen SBERT embeddings, MNR loss, multi-position training, train/evaluate scripts | Claude + Author |
| 2026-03-03 | 11.2 | Training completed (all 4 model-dataset combos on CUDA). Added Student FAQ (7 Q&A), training results, convergence analysis, training time investigation | Claude + Author |
| 2026-03-03 | 11.2 | Evaluation completed. Added actual retrieval metrics, full comparison tables (11.1+11.2), and detailed analysis of surprising MIND results (sequence models underperform aggregations) | Claude + Author |
| 2026-03-05 | 11.2 | Item-ID variant training/evaluation completed. SASRec-ItemID MRR=0.093, GRU4Rec-ItemID MRR=0.088 on Amazon (significantly worse than frozen SBERT). Updated results tables and Exercise 1 with actual findings | Claude + Author |
| 2026-03-05 | 11.3 | Added LightGCN implementation: bipartite graph builder, LightGCN model (BPR + MNR loss), train/evaluate scripts with cold-start eval. Design Notes updated with GNN primer, architecture details, production inference pattern, hyperparameter grounding, and Yandex Yambda homework exercise | Claude + Author |
| 2026-03-06 | 11.3 | Training/evaluation completed. LightGCN-MNR MRR=0.0100 (best MIND method, 3.2x over aggregation). LightGCN-BPR MRR=0.0051. Surprise: MNR outperformed BPR — learnable embeddings fix MNR's dense-cluster problem. Added full results, comparison tables, analysis of MNR surprise, updated hypotheses table | Claude + Author |
| 2026-03-07 | 11.3 | SBERT-initialized LightGCN training/evaluation completed. LightGCN-MNR-SBERT MRR=0.0174 — new best MIND method (+74% over random init). LightGCN-BPR-SBERT MRR=0.0058 (+14%). Content initialization as "poor man's PinSAGE" bridges Ch10→Ch11 narrative. Updated all results tables, hypotheses, added SBERT-init analysis section | Claude + Author |
