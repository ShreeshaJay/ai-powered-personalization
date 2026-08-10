# Chapter 8: Value Functions - Design Notes

## Overview

This document captures the design decisions for Chapter 8, which covers the **Ordering Stage** of the recommendation pipeline. After Chapter 7's multi-task model produces predictions for multiple objectives, Chapter 8 demonstrates how to blend these into a single ranking score using **Value Functions**.

---

## 1. Chapter 7 → Chapter 8 Data Flow

### Prerequisites from Chapter 7

The Chapter 7 MMoE model produces **5 task outputs** for each (user, item) pair:

| Task | Type | Column Name | Description |
|------|------|-------------|-------------|
| Engagement | Binary | `prob_engagement` | P(user starts listening) |
| Completion | Binary | `prob_completion` | P(user listens ≥50%) |
| **Play Ratio** | **Regression** | `prob_play_ratio` | E[play_ratio] in [0, 1] |
| Like | Binary | `prob_like` | P(user likes item) |
| Dislike | Binary | `prob_dislike` | P(user dislikes item) |

### Key Insight: Regression Task for Expected Listen Time

The `play_ratio` regression task is critical for Chapter 8 because it enables:

```python
# Expected listen time calculation
expected_listen_time = P(engagement) * E[play_ratio] * track_length_seconds

# This is MORE accurate than the binary approximation:
# expected_listen_time ≈ P(engagement) * P(completion) * 0.75 * track_length
```

### Artifacts Exported by Chapter 7

```
chapter7_advanced_ranking/outputs/mmoe_30d_Xexp_YYYYMMDD_HHMMSS/
├── best_model.pt                 # Model checkpoint with config
├── predictions.parquet           # Pre-computed test set predictions
├── calibration.json              # Calibration curves for each task
├── value_function_config.json    # Task stats + suggested weight schemes
├── gate_weights_sample.parquet   # Expert utilization for interpretability
└── inference/
    ├── feature_processor.pkl     # Fitted encoders/scalers
    ├── polars_pipeline/          # Feature engineering pipeline state
    └── inference_example.py      # Example inference script
```

---

## 2. Field Name Mapping

### Chapter 7 → Chapter 8 Translation

| Chapter 7 Output | Chapter 8 ValueFunction Param | Description |
|-----------------|------------------------------|-------------|
| `prob_engagement` | `p_listen` | Base engagement probability |
| `prob_play_ratio` | `e_engagement` | Expected play ratio (regression) |
| `prob_like` | `p_like` | Explicit positive feedback |
| `prob_dislike` | `p_dislike` | Explicit negative feedback |

### Why Different Naming?

- **Chapter 7** uses task-centric naming (`engagement`, `completion`, `play_ratio`, `like`, `dislike`)
- **Chapter 8** uses value-function-centric naming (`listen`, `like`, `engagement`, `dislike`)

The `e_engagement` name in Chapter 8 represents "engagement depth" which is the regression-predicted play ratio.

---

## 3. Value Function Formulation

### Core Formula

```python
Value(i) = w_listen * P(listen)
         + w_like * P(listen) * P(like)
         + w_engagement * P(listen) * E[play_ratio]
         - w_dislike * P(listen) * P(dislike)
```

### Why Conditional on P(listen)?

The formulation multiplies downstream signals by P(listen) because:

1. **P(like|listen)** only matters if the user listens first
2. **E[play_ratio]** only contributes value if engagement happens
3. **P(dislike)** penalty should scale with engagement probability

### Business-Aligned Weight Schemes

From `value_function_config.json`:

```python
# Engagement-focused: Maximize listening time
{
    'engagement': 1.0,
    'completion': 2.0,
    'play_ratio': 0.5,
    'like': 5.0,
    'dislike': -10.0,
}

# Satisfaction-focused: Prioritize long-term retention
{
    'engagement': 0.5,
    'completion': 1.0,
    'play_ratio': 1.0,
    'like': 10.0,
    'dislike': -20.0,
}

# Listen-time-focused: Uses regression prediction
{
    'engagement': 0.5,
    'completion': 0.5,
    'play_ratio': 3.0,  # High weight on predicted ratio
    'like': 2.0,
    'dislike': -5.0,
}
```

---

## 4. Calibration Considerations for Value Functions

### The Calibration Problem

Neural network outputs after sigmoid produce values in [0, 1], but they are often **not well-calibrated**. A model predicting P(like) = 0.30 should mean "30% of items with this score actually get liked" — but neural networks are often overconfident or underconfident.

### Evidence from Chapter 7

From `calibration.json`, the like task shows miscalibration:

```
mean_predicted:     [0.011, 0.143, 0.245, 0.341, 0.424, 0.503]
fraction_positives: [0.007, 0.038, 0.112, 0.083, 0.000, 0.250]
```

When the model predicts ~0.24, the actual positive rate is ~0.11 — the model is **overconfident**.

### When Does Miscalibration Matter?

| Use Case | Impact |
|----------|--------|
| **Pure ranking** | ✅ Usually fine — if miscalibration is monotonic, rank order is preserved |
| **Value function blending** | ⚠️ Problematic — weights assume probabilities are meaningful |
| **Business reporting** | ❌ Problem — "we expect 30% like rate" is wrong if overconfident |

### Impact on Value Functions

```python
value = w_listen * P(listen) + w_like * P(listen) * P(like) + ...
```

If `P(like)` is systematically 2x overconfident, then `w_like` effectively becomes `2 * w_like`. The **relative weighting between tasks gets distorted**.

### Mitigation Options

**Option 1: Post-hoc Prediction Calibration**

1. **Platt Scaling**: Fit logistic regression on validation set
   ```python
   calibrated_prob = sigmoid(a * raw_logit + b)
   ```

2. **Temperature Scaling**: Divide logits by learned temperature T
   ```python
   calibrated_prob = sigmoid(logit / T)
   ```

3. **Isotonic Regression**: Non-parametric calibration (monotonic transformation)

**Option 2: Absorb Calibration into Weights (Recommended for Ranking)**

For value functions used only for ranking, a simpler approach is to adjust the weights:

```python
# Original: value = w_like * P(like) * calibration_factor
# Equivalent: value = (w_like * calibration_factor) * P(like)
#           = effective_w * P(like)

from chapter8_value_functions import CalibrationFactors, compute_calibrated_weights

# Load calibration data from Chapter 7
calib = CalibrationFactors.from_chapter7_calibration(
    "chapter7_advanced_ranking/outputs/mmoe_.../calibration.json"
)
# Example output:
# CalibrationFactors(
#   engagement=1.84,  # under-predicting
#   like=0.058,       # over-predicting by 17x!
#   dislike=0.031,    # over-predicting by 32x!
# )

# Compute adjusted weights
base_config = ValueFunctionConfig(w_like=2.0, w_dislike_penalty=1.0, ...)
adjusted_config = compute_calibrated_weights(base_config, calib)
# Result: w_like = 2.0 * 0.058 = 0.116
#         w_dislike = 1.0 * 0.031 = 0.031
```

**Why this works for ranking:**
- Calibration factors are **global constants** per task
- Multiplying by a constant doesn't change rank order within a task
- The effect is to **change relative importance** between tasks
- This is the *intended* effect: over-predicted P(like) should contribute less

### Regression Head Calibration

For the `play_ratio` regression head, calibration means: if the model predicts 0.65 for a set of items, the average actual play_ratio should be ~0.65.

MSE training encourages this naturally, but systematic bias can still occur. A simple linear correction (`corrected = a * predicted + b`) can fix this.

### Practical Recommendation

| Use Case | Recommended Approach |
|----------|---------------------|
| **Ranking only** | Absorb calibration into weights (Option 2) |
| **Ranking + Business reporting** | Platt scaling or temperature scaling |
| **A/B test expected metrics** | Full calibration required |

For this book's teaching purposes, we demonstrate **Option 2** (absorbing calibration into weights) because:
1. It's conceptually simpler
2. It doesn't require modifying inference pipelines
3. It naturally shows how miscalibration affects task weighting

---

## 5. The Regression Task Advantage

### Binary vs Regression for Engagement Depth

| Approach | Formula | Granularity |
|----------|---------|-------------|
| Binary (completion) | P(engage) * P(complete) * 0.75 | Coarse (loses 10% vs 45%) |
| **Regression (play_ratio)** | P(engage) * E[play_ratio] | **Fine-grained** |

### Example Impact

Consider two items with same P(engage) = 0.8:

| Item | Binary Completion | Regression Play Ratio | Expected Ratio |
|------|------------------|----------------------|----------------|
| A | P(complete)=0.6 → 0.45 | E[ratio]=0.42 | 0.34 |
| B | P(complete)=0.6 → 0.45 | E[ratio]=0.85 | 0.68 |

Binary approach treats A and B equally (both complete with 60% probability), but regression captures that B has much deeper engagement.

---

## 5. MMR Diversity with Yambda Embeddings

### Embedding Source

Yambda provides 128-dimensional **audio embeddings** for 7.7M items:

```python
# Load embeddings
embeddings = load_yambda_embeddings("embeddings.parquet", use_normalized=True)
# Returns: Dict[int, np.ndarray]  # item_id → 128-dim vector
```

### MMR Algorithm

```python
MMR(i) = λ * Relevance(i) - (1-λ) * max_{j ∈ S} Similarity(i, j)
```

Where:
- `Relevance(i)` = Value function score (normalized)
- `Similarity(i, j)` = Cosine similarity of audio embeddings
- `λ` = Trade-off parameter (0.7 = 70% relevance, 30% diversity)

### Diversity vs Relevance Trade-off

| Lambda (λ) | Behavior | Use Case |
|------------|----------|----------|
| 1.0 | Pure relevance | A/B test baseline |
| 0.7-0.8 | Balanced | **Production default** |
| 0.5 | High diversity | Exploration/discovery |
| 0.3 | Very diverse | New user onboarding |

---

## 6. Business Rules: Artist Pacing

### The Problem

Without pacing, a user who loves Taylor Swift might see:

```
1. Taylor Swift - Song A (value=0.95)
2. Taylor Swift - Song B (value=0.93)
3. Taylor Swift - Song C (value=0.91)
...
10. Taylor Swift - Song J (value=0.82)
```

This is bad UX even if the model is "correct."

### Artist Pacing Solution

```python
pacer = ArtistPacer(
    artist_mapping=load_artist_mapping("artist_item_mapping.parquet"),
    config=PacingConfig(max_per_artist=2, max_per_album=1)
)

paced_list = pacer.apply_pacing(ranked_items)
```

Result:
```
1. Taylor Swift - Song A
2. Taylor Swift - Song B
3. Ed Sheeran - Song X      # Inserted due to pacing
4. Beyoncé - Song Y         # Inserted due to pacing
5. Taylor Swift - Song C    # Deferred, now appears later
```

### Pacing Config Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_per_artist` | 2 | Max tracks from same artist |
| `max_per_album` | 1 | Max tracks from same album |
| `defer_violations` | True | Move violations to end (vs drop) |

---

## 7. Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Chapter 8: Ordering Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Candidate items from Retrieval (Chapter 5)                       │
│         ~500-1000 candidates per user                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 1: Multi-Task Scoring (Chapter 7 Model)                   │    │
│  │                                                                  │    │
│  │  For each candidate:                                            │    │
│  │    → P(engagement), P(completion), E[play_ratio],               │    │
│  │      P(like), P(dislike)                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 2: Value Function (This Chapter)                          │    │
│  │                                                                  │    │
│  │  value = w1*P(engage) + w2*P(engage)*P(like)                    │    │
│  │        + w3*P(engage)*E[ratio] - w4*P(engage)*P(dislike)        │    │
│  │                                                                  │    │
│  │  Output: Single score per item                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3: MMR Diversity Reranking                                │    │
│  │                                                                  │    │
│  │  Using Yambda audio embeddings:                                 │    │
│  │    MMR(i) = λ*value(i) - (1-λ)*max_sim(i, selected)             │    │
│  │                                                                  │    │
│  │  Output: ~40 items (2x final list for pacing buffer)            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 4: Business Rules (Artist Pacing)                         │    │
│  │                                                                  │    │
│  │  Enforce: max 2 tracks per artist, max 1 per album              │    │
│  │  Defer violations to later positions                            │    │
│  │                                                                  │    │
│  │  Output: Final 20 items                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                           │
│  Output: Final ranked list for page construction                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Evaluation Metrics

### Ranking Quality

| Metric | Description | Use Case |
|--------|-------------|----------|
| NDCG@k | Graded relevance, position-weighted | Primary offline metric |
| MRR | First relevant item position | "Quick win" scenarios |
| Hit Rate@k | Any relevant in top-k | Binary relevance |

### Diversity Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| ILD | Intra-list diversity (avg pairwise dissimilarity) | Higher = more diverse |
| Coverage | Catalog coverage across users | Higher = less popularity bias |
| Artist Entropy | Distribution of artists in list | Higher = more balanced |

### Business Metrics (for A/B Testing)

| Metric | Description |
|--------|-------------|
| Listen Starts | Did user start any recommended track? |
| Listen Time | Total minutes listened from recs |
| Like Rate | Proportion of recs that were liked |
| Skip Rate | Proportion skipped within 30 seconds |
| Session Length | Total session time (indirect) |

---

## 9. Integration Testing

### Run Integration Test

After Chapter 7 training completes:

```bash
cd chapter8_value_functions

python test_chapter7_integration.py \
    --model_dir ../chapter7_advanced_ranking/outputs/mmoe_30d_4exp_YYYYMMDD_HHMMSS \
    --data_dir "C:/Users/.../Dataset/Yandex"
```

### Expected Output

```
============================================================
Chapter 8 ← Chapter 7 Integration Test
============================================================

1. Checking predictions.parquet schema
   Found 39,338 rows, 15 columns
   ✅ All 5 task columns present
   ✅ play_ratio regression task found

2. Checking value_function_config.json
   Tasks in config: ['engagement', 'completion', 'play_ratio', 'like', 'dislike']
   ✅ play_ratio regression task configured

3. Checking calibration.json
   ✅ engagement: 9 bins, positive_rate=0.9198
   ✅ completion: 10 bins, positive_rate=0.6090
   ✅ play_ratio: calibration data present
   ✅ like: 6 bins, positive_rate=0.0072
   ✅ dislike: 1 bins, positive_rate=0.0020

4. Checking Yambda embeddings for MMR diversity
   Loaded 7,748,623 item embeddings
   ✅ normalized_embed available (dim=128)

5. Checking artist mapping for business rules
   Loaded 7,748,623 artist-item mappings
   Unique artists: 1,234,567

6. Testing value function computation
   ✅ Value function computed successfully

============================================================
✅ ALL CHECKS PASSED
============================================================
```

---

## 10. Code Organization

```
chapter8_value_functions/
├── __init__.py
├── value_function.py        # ValueFunction class + configs
├── mmr_diversity.py         # MMRReranker + embedding loading
├── business_rules.py        # ArtistPacer, SlotAllocator
├── pipeline.py              # OrderingPipeline (end-to-end)
├── model_interface.py       # Abstract base for Chapter 7 model
├── evaluate.py              # NDCG, MRR, ILD metrics
├── test_chapter7_integration.py  # Integration test script
├── requirements.txt
├── README.md
└── docs/
    └── Chapter8_Design_Notes.md  # This file
```

---

## 11. Optional Concepts Not Implemented

The following concepts are common in production ordering systems but were **not implemented** in this chapter. They are documented here for readers who want to extend the pipeline for real-world applications.

### 11.1 Explore-Exploit Strategies

**What it is**: Balancing showing items with known high value (exploit) vs. items with uncertain value (explore) to improve long-term recommendations.

**Techniques**:
- Thompson Sampling: Sample from posterior distribution of item value
- Upper Confidence Bound (UCB): Add uncertainty bonus to value scores
- ε-greedy: Random exploration with probability ε

**Why we deferred it**: Explore-Exploit is covered in **Chapter 12: Experimentation and Bandits**, where it receives full treatment alongside A/B testing and online learning.

### 11.2 Freshness/Recency Boosting

**What it is**: Time-decay functions that boost newer content and demote stale content.

```python
# Example freshness boost
hours_since_release = (now - release_timestamp).total_seconds() / 3600
freshness_boost = np.exp(-hours_since_release / half_life_hours)
final_value = base_value * (1 + freshness_weight * freshness_boost)
```

**Why we skipped it**: The Yambda dataset doesn't have strong temporal signals (release dates, trending indicators). For news, video, or e-commerce platforms, freshness is critical and should be added to the value function.

### 11.3 Source-Level Score Fusion

**What it is**: Instead of slot-based allocation, blend scores from multiple retrieval sources before ranking.

```python
# Current approach: Slot-based (pick source per position)
template = ['organic', 'organic', 'new_release', 'organic', ...]

# Alternative: Score fusion (blend then rank)
final_score = (
    0.7 * personalized_value +     # User-specific predictions
    0.2 * popularity_score +        # Fallback for cold users
    0.1 * editorial_boost           # Curated content promotion
)
```

**When to use**: When you have multiple retrieval systems (collaborative filtering, content-based, knowledge-graph) and want their scores to compete directly rather than being allocated to fixed slots.

### 11.4 Position Bias Correction

**What it is**: Adjusting for the fact that users are more likely to interact with items shown in higher positions, independent of item quality.

**Techniques**:
- Inverse Propensity Weighting (IPW) during training
- Position-aware models that jointly predict CTR and position bias
- Randomization experiments to estimate position effects

**Why we skipped it**: Position bias correction is primarily a **training-time** concern (Chapter 7) or an **online learning** problem (Chapter 12). At ordering time, you assume your model has already been debiased.

### 11.5 Ads/Sponsored Content Blending

**What it is**: Integrating auction-based sponsored items with organic recommendations while maintaining user experience.

**Considerations**:
- Auction mechanisms (second-price, VCG)
- Relevance-gated ads (only show if quality score > threshold)
- Ad load management (max ads per page)
- Native vs. display format decisions

**Why we skipped it**: Yambda is a music dataset without advertising. For ad-supported platforms, this is a critical extension that combines value functions with auction theory.

### 11.6 Real-Time Feature Injection

**What it is**: Incorporating signals that are only available at serving time (not in batch predictions).

**Examples**:
- Current session context (recently played items)
- Real-time inventory/availability
- Live trending signals
- User's current device/network conditions

**Implementation pattern**:
```python
# Batch predictions from Chapter 7
batch_value = model.predict(user_features, item_features)

# Real-time adjustments at serving time
real_time_boost = compute_session_relevance(current_session, item)
final_value = batch_value * (1 + real_time_boost)
```

**Why we skipped it**: Requires infrastructure for real-time feature serving, which is covered conceptually in **Chapter 9: Online Deployment**.

---

## 12. Key Takeaways for the Book

1. **Value functions encode business priorities** — There's no "correct" value function, only trade-offs between objectives.

2. **Regression tasks enable finer optimization** — `E[play_ratio]` is more informative than `P(completion)` for expected listen time.

3. **Diversity is a post-scoring concern** — MMR operates on value scores, not raw model outputs.

4. **Business rules are non-negotiable** — Even perfect ML predictions must respect artist pacing for good UX.

5. **Calibration matters for value functions** — If P(like) predictions are miscalibrated, the weight `w_like` loses its intended meaning.

6. **The pipeline is modular** — Each stage (scoring → value → diversity → rules) can be A/B tested independently.

---

## 13. Chapter 8 Narrative Arc

### Opening Hook

> "In Chapter 7, we built a multi-task model that predicts five different aspects of user-item interactions. But here's the challenge: a recommendation system can only show items in *one* order. How do we combine engagement probability, like probability, expected play ratio, and dislike risk into a single ranking that balances immediate engagement with long-term user satisfaction? This is the problem of **value functions**."

### The Regression Task Payoff

> "Remember the `play_ratio` regression head we added in Chapter 7? Here's where it pays off. Instead of the crude approximation `expected_time = P(complete) × 0.75 × track_length`, we can compute the precise `expected_time = P(engage) × E[ratio] × track_length`. The regression prediction lets us optimize directly for minutes listened."

### The Diversity-Relevance Trade-off

> "MMR forces us to answer a fundamental question: How much relevance are we willing to sacrifice for diversity? The answer varies by context—new users benefit from exploration (λ=0.5), while loyal users prefer familiar content (λ=0.9). The beauty of MMR is that this trade-off is a single, tunable parameter."

### Closing Insight

> "Value functions are where ML meets business strategy. The weights you choose reflect your platform's priorities—are you optimizing for engagement today, or retention next month? The technical implementation is straightforward; the hard part is deciding what 'value' means for your users."

---

---

## 14. Offline A/B Test Simulation

### The Counterfactual Problem

A key challenge in recommender systems is evaluating new ranking strategies without deploying them. **You only observe outcomes for items that were actually shown.** If your new value function would surface different items, you have no ground truth for those items.

```
What you have:     User saw items [A, B, C] → User clicked B
What you want:     If we had shown [A, D, E], would user have clicked?
                   ← Unknown! D and E weren't shown in logged data.
```

### When Offline Simulation Works Well

For **value function weight tuning**, offline evaluation is reasonable because:

1. **Same candidates** — You're reordering items that were already scored by the model
2. **Same predictions** — Model outputs (p_listen, p_like, etc.) are fixed
3. **Just different combinations** — Only the weight formula changes

```python
# Feasible offline experiment for value function weights:
for weights in [weights_A, weights_B, weights_C]:
    ranked_items = sort_by(value_function(predictions, weights))
    ndcg = compute_ndcg(ranked_items, actual_engagements)
    print(f"Weights {weights} → NDCG: {ndcg}")
```

### Approaches for More General Offline Evaluation

| Method | Technique | Works When |
|--------|-----------|------------|
| **Direct Comparison** | Compare NDCG/MRR of different rankings on same test set | Reordering same candidates |
| **Inverse Propensity Scoring (IPS)** | Reweight outcomes by logging propensity | You have propensity scores |
| **Replay/Rejection Sampling** | Only use samples where new policy matches logged policy | Logging had random exploration |
| **Doubly Robust** | Combine model + IPS for variance reduction | Model is reasonably accurate |

### Key Caveat

Offline metrics measure **ranking quality on logged data**, not **user behavior under the new ranking**. Users might engage differently when shown a different order! The gold standard remains **online A/B testing** (covered in Chapter 12).

| Scenario | Trust Offline? | Reasoning |
|----------|---------------|-----------|
| Reordering same items | ✅ High | Same candidates, different order |
| Surfacing rare items | ⚠️ Medium | Limited outcome data for rare items |
| Major policy change | ❌ Low | No counterfactual data |
| Weight tuning | ✅ High | Same model outputs, different combination |

---

## 15. Script Execution Order

### State Transition Diagram

The following diagram shows the dependencies between scripts and when each should be executed:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHAPTER 7 (Prerequisites)                           │
│                                                                             │
│   ┌───────────────────┐      ┌───────────────────┐      ┌──────────────┐   │
│   │ prepare_data.py   │─────▶│  train_mmoe.py    │─────▶│ Export       │   │
│   │                   │      │                   │      │ Artifacts    │   │
│   │ • Split data      │      │ • 5-task MMoE     │      │              │   │
│   │ • Feature eng.    │      │ • Binary + Regr.  │      │ • model.pt   │   │
│   │ • Save parquets   │      │ • Calibration     │      │ • preds.pq   │   │
│   └───────────────────┘      └───────────────────┘      │ • calib.json │   │
│                                                          └──────┬───────┘   │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CHAPTER 8 (This Chapter)                         │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  STEP 1: Integration Validation (One-time)                         │    │
│   │  ─────────────────────────────────────────                         │    │
│   │  python test_chapter7_integration.py \                             │    │
│   │      --predictions "../chapter7_.../predictions.parquet" \         │    │
│   │      --data "../data/Yandex"                                       │    │
│   │                                                                    │    │
│   │  ✓ Validates column names match expected format                    │    │
│   │  ✓ Tests value function computation                                │    │
│   │  ✓ Tests MMR diversity with embeddings                             │    │
│   │  ✓ Tests artist pacing rules                                       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  STEP 2: Use Chapter 8 Modules (Import in Notebooks/Scripts)       │    │
│   │  ───────────────────────────────────────────────────────────       │    │
│   │                                                                    │    │
│   │  from chapter8_value_functions import (                            │    │
│   │      ValueFunction, ValueFunctionConfig,                           │    │
│   │      MMRReranker, DiversityConfig,                                 │    │
│   │      ArtistPacer, PacingConfig,                                    │    │
│   │      OrderingPipeline, PipelineConfig,                             │    │
│   │  )                                                                 │    │
│   │                                                                    │    │
│   │  # See pipeline.py for full end-to-end usage                       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
│                                      ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  STEP 3: Evaluation & Analysis (Optional)                          │    │
│   │  ─────────────────────────────────────────                         │    │
│   │                                                                    │    │
│   │  from chapter8_value_functions.evaluate import (                   │    │
│   │      compute_ndcg,                                                 │    │
│   │      compute_intra_list_diversity,                                 │    │
│   │      analyze_relevance_diversity_tradeoff,                         │    │
│   │  )                                                                 │    │
│   │                                                                    │    │
│   │  # Compare different weight configurations                         │    │
│   │  # Analyze λ (lambda) parameter for diversity trade-off            │    │
│   └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
                         ┌──────────────────┐
                         │  pipeline.py     │  (Orchestrator)
                         │  OrderingPipeline│
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │value_function.py│ │mmr_diversity.py │ │business_rules.py│
    │                 │ │                 │ │                 │
    │• ValueFunction  │ │• MMRReranker    │ │• ArtistPacer    │
    │• Config         │ │• load_embeddings│ │• SlotAllocator  │
    │• Calibration    │ │• DiversityConfig│ │• PacingConfig   │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    model_interface.py   │
                    │  (Contract for Ch7 model)│
                    └─────────────────────────┘
```

### File Descriptions

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_chapter7_integration.py` | Validates Chapter 7 → 8 compatibility | Run once after Chapter 7 model training |
| `value_function.py` | Multi-objective scoring | Import when computing item values |
| `mmr_diversity.py` | Diversity-aware re-ranking | Import when diversity matters |
| `business_rules.py` | Artist pacing, slot allocation | Import for final list construction |
| `pipeline.py` | End-to-end orchestration | Import for production-like flow |
| `model_interface.py` | Model contract/specification | Reference for Chapter 7 compatibility |
| `evaluate.py` | Metrics (NDCG, ILD, coverage) | Import for offline evaluation |

### Quick Start Commands

```bash
# 1. Ensure Chapter 7 artifacts exist
ls chapter7_advanced_ranking/outputs/mmoe_*/predictions.parquet

# 2. Run integration test
cd chapter8_value_functions
python test_chapter7_integration.py \
    --predictions "../chapter7_advanced_ranking/outputs/mmoe_30d_4exp_YYYYMMDD_HHMMSS/predictions.parquet" \
    --data "../data/Yandex"

# 3. Interactive exploration (in Python/Jupyter)
from chapter8_value_functions import ValueFunction, ValueFunctionConfig
# ... see README.md for full examples
```

---

*Last updated: February 8, 2026*

> **Note**: For detailed calibration analysis and model improvement strategies (increased experts, BPR pairwise loss), see **Chapter 7 Design Notes, Section 17**.

> **Note**: Section 11 documents optional extensions for production systems that were not implemented in this teaching chapter.

> **Note**: Section 14 covers the theory and limitations of offline A/B test simulation for value function tuning.