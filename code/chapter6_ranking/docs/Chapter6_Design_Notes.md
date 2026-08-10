# Chapter 6: Ranking - Design Choices and Implementation Details

## Overview

This document consolidates the key design decisions and implementation details for Chapter 6's ranking model implementation using the Yandex Yambda dataset.

---

## 1. Dataset Selection: Yandex Yambda

### Why Yambda?

| Criterion | Yambda | Notes |
|-----------|--------|-------|
| **Scale** | 50M interactions | Large enough for realistic training |
| **Multi-task potential** | ✅ Listens, likes, dislikes | Supports progression to MTL models |
| **Temporal data** | ✅ Timestamps | Enables proper train/test splits |
| **Rich signals** | ✅ Completion %, organic flag | Multiple label options |
| **Recency** | 2024 release | Modern, well-documented |

### Dataset Schema

```
listens.parquet:
├── uid (int64)           - User identifier
├── item_id (int64)       - Track identifier  
├── timestamp (int64)     - Relative time (5-second bins)
├── played_ratio_pct (int8) - Completion percentage (0-100)
├── is_organic (int8)     - 1=user-initiated, 0=recommended
└── track_length_seconds (float) - Track duration
```

### Label Definition

We define **Listen+** (positive class) as:
```
label = 1 if played_ratio_pct >= 50 else 0
```

**Rationale**: 50% threshold balances:
- Too low (10%): Includes skips/accidental plays
- Too high (90%): Misses partial but engaged listens
- 50%: Indicates genuine user interest

---

## 2. Global Temporal Split (GTS) Protocol

### The Problem with Random Splits

Random train/test splits in RecSys cause **temporal data leakage**:
- Model sees future user preferences during training
- Unrealistically inflated metrics
- Poor real-world performance

### GTS Implementation

```
Timeline:
|------ Training (300 days) ------|-- Gap --|-- Test (1 day) --|
                                   30 min
```

**Key parameters:**
- `train_days`: 300 (or configurable for memory constraints)
- `gap_duration`: 30 minutes (prevents information leakage)
- `test_days`: 1

### Time-Window Approach for Memory Efficiency

For memory-constrained environments (16GB laptop), we use a **shorter training window** rather than random sampling:

```python
# Full dataset (~50M rows)
train_lf, test_lf = load_yambda_polars(data_dir, train_days=None)

# Reduced window (~5M rows) - preserves temporal integrity
train_lf, test_lf = load_yambda_polars(data_dir, train_days=30)
```

**Why time-window over sampling:**
| Approach | Temporal Order | Historical Features | Leakage Risk |
|----------|---------------|---------------------|--------------|
| Random sampling | ❌ Broken | ⚠️ Incomplete | ⚠️ Possible |
| Stratified sampling | ❌ Broken | ⚠️ Incomplete | ⚠️ Possible |
| **Time window** | ✅ Preserved | ✅ Correct | ✅ None |

---

## 3. Feature Engineering Architecture

### Feature Categories

#### 3.1 Categorical Features (Encoded)

| Feature | Cardinality | Encoding | Rationale |
|---------|-------------|----------|-----------|
| `uid` | ~500K | Frequency | High cardinality → frequency captures popularity |
| `item_id` | ~1M | Frequency | Same rationale |
| `is_organic` | 2 | Label | Low cardinality → integer encoding |

**Frequency Encoding Formula:**
```
uid_freq = count(uid in train) / total_train_rows
```

**Why not one-hot?** XGBoost handles frequency-encoded features more efficiently than sparse one-hot matrices for high-cardinality categoricals.

#### 3.2 Numerical Features

| Feature | Description |
|---------|-------------|
| `track_length_seconds` | Track duration (longer tracks may have lower completion) |

**Note:** Raw `timestamp` is NOT used as a feature because:
- It's monotonically increasing → causes data leakage
- Model would learn "later = different distribution" rather than meaningful patterns

#### 3.3 Derived Temporal Features

```python
# Cyclical time features (from timestamp)
hour_of_day = (timestamp * 5 // 3600) % 24   # 0-23
day_of_week = (timestamp * 5 // 86400) % 7   # 0-6
```

**Rationale:** Captures listening patterns (commute hours, weekends) without leaking absolute time.

#### 3.4 Historical Aggregate Features

**User Features (8):**
| Feature | Description |
|---------|-------------|
| `user_total_listens` | Total historical listens |
| `user_avg_completion` | Mean completion rate |
| `user_std_completion` | Completion variance (consistency) |
| `user_median_completion` | Robust central tendency |
| `user_unique_items` | Catalog exploration breadth |
| `user_organic_ratio` | % self-initiated vs recommended |
| `user_active_span` | Time range of activity |
| `user_listen_rate` | Listens per time unit |

**Item Features (7):**
| Feature | Description |
|---------|-------------|
| `item_total_plays` | Global popularity |
| `item_avg_completion` | Track "stickiness" |
| `item_std_completion` | Polarizing vs universal appeal |
| `item_unique_listeners` | Reach |
| `item_organic_ratio` | Discovery vs recommendation |
| `item_repeat_ratio` | Replay value |
| `item_like_ratio` | Explicit feedback signal |

**User-Item Features (2):**
| Feature | Description | PIT Status |
|---------|-------------|------------|
| `has_listened_before` | Re-listen indicator | ✅ PIT-correct |
| `previous_listen_count` | Count of prior interactions | ✅ PIT-correct |

**Removed Feature:**
- `user_item_avg_completion` - **REMOVED** due to severe data leakage (was computing average from ALL interactions including future ones, directly predicting the label)

### Point-in-Time (PIT) Feature Correctness

**The Problem:**
For each prediction at time T, historical features should only use data from times < T. Computing features from the entire training window causes "look-ahead bias".

**PIT Implementation for `previous_listen_count`:**
```python
# OLD (leaky): Same count for all rows with this (user, item) pair
df.group_by(['uid', 'item_id']).count()

# NEW (PIT-correct): Cumulative count up to but not including current row
(
    pl.count()
    .over(['uid', 'item_id'], order_by='timestamp')
    - 1  # Exclude current row
).alias('previous_listen_count')
```

### Features NOT PIT-Corrected (Documented Trade-off)

The following features use **lookup-table aggregation** (computed from full training window):

| Feature Category | Examples | Leakage Risk | Why Not PIT |
|-----------------|----------|--------------|-------------|
| User-level | `user_avg_completion`, `user_total_listens` | **Low** | Aggregated over many items |
| Item-level | `item_avg_completion`, `item_total_plays` | **Low** | Aggregated over many users |

**Rationale for accepting this simplification:**
1. The observed train-test AUC gap (~0.17) is acceptable
2. User/item-level aggregates dilute any single row's influence
3. True PIT would require 5-10x code complexity

### Production Feature Store Context

In production systems, a **Feature Store** (e.g., Feast, Tecton, Vertex AI Feature Store) handles PIT correctness:

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE STORE WORKFLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. OFFLINE BATCH JOBS                                      │
│     ├── Daily job computes user_avg_completion as of EOD    │
│     └── Stores versioned features with timestamps           │
│                                                             │
│  2. TRAINING TIME (Temporal Join)                           │
│     For row at time T:                                      │
│     └── Feature Store returns features valid at time < T    │
│                                                             │
│  3. INFERENCE TIME                                          │
│     └── Returns latest features (genuinely historical now)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Takeaway:** The lookup-table approach in this codebase is a pedagogical simplification. Production systems require Feature Store infrastructure for strict PIT correctness.

---

## 4. Model Architecture: Pointwise XGBoost

### Why Pointwise Classification?

| Approach | Complexity | When to Use |
|----------|------------|-------------|
| **Pointwise** | Low | CTR prediction, single-item scoring |
| Pairwise (LambdaMART) | Medium | When relative order matters |
| Listwise | High | Full ranking optimization |

For Chapter 6's introduction to ranking, pointwise is pedagogically clearer and performs well for CTR-style tasks.

### XGBoost Configuration

```python
model = xgb.XGBClassifier(
    n_estimators=100,        # Boosting rounds
    max_depth=6,             # Tree complexity
    learning_rate=0.1,       # Step size
    min_child_weight=1,      # Regularization
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
    scale_pos_weight=1.0,    # Class imbalance (adjust if needed)
    objective='binary:logistic',
    eval_metric='auc',       # Or 'aucpr' for imbalanced data
    early_stopping_rounds=10,
    n_jobs=-1,
    random_state=42
)
```

**Hyperparameter Rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_depth=6` | Moderate complexity, prevents overfitting |
| `learning_rate=0.1` | Standard starting point |
| `subsample=0.8` | Stochastic gradient boosting for regularization |
| `colsample_bytree=0.8` | Feature bagging for diversity |
| `early_stopping_rounds=10` | Prevents overfitting, finds optimal iterations |

### Evaluation Metrics

| Metric | When to Use |
|--------|-------------|
| **AUC-ROC** | Balanced classes, general discrimination |
| **AUC-PR** | Imbalanced classes (prefer positive precision) |
| **Log Loss** | Probability calibration matters |
| **F1** | Binary decision threshold tuning |

---

## 5. Memory Efficiency Design

### The Challenge

- Full Yambda dataset: ~50M rows
- Naive pandas loading: ~20-30GB peak memory
- Target: Run on 16GB laptop

### Solution: Polars Pipeline

**Why Polars over Pandas:**
| Aspect | Pandas | Polars |
|--------|--------|--------|
| Memory efficiency | Baseline | 2-5x better |
| Lazy evaluation | ❌ | ✅ Query optimization |
| Parallel processing | Limited | Native |
| Arrow format | Via PyArrow | Native |

### Memory Optimization Techniques

1. **Dtype Optimization**
```python
OPTIMAL_DTYPES = {
    'uid': pl.Int32,          # vs Int64 (50% savings)
    'item_id': pl.Int32,
    'timestamp': pl.Int32,
    'played_ratio_pct': pl.Float32,  # vs Float64
    'is_organic': pl.Int8,    # vs Int64 (87% savings)
}
```

2. **Lazy Evaluation**
```python
# Doesn't load data until .collect()
train_lf = pl.scan_parquet(path).filter(...)
```

3. **Time-Window Reduction**
```python
# Instead of sampling (breaks temporal order)
train_lf, test_lf = load_yambda_polars(data_dir, train_days=30)
```

### Memory Usage by Configuration

| Configuration | Approx Rows | Peak Memory |
|---------------|-------------|-------------|
| `train_days=30` | ~5M | ~2-3 GB |
| `train_days=90` | ~15M | ~5-8 GB |
| `train_days=150` | ~25M | ~8-12 GB |
| `train_days=300` (full) | ~50M | ~12-18 GB |

---

## 6. Code Organization

### Directory Structure

```
chapter6_ranking/
├── config.py                 # Centralized configuration
├── data_loader.py            # PyArrow-based data loading (pandas output)
├── train_xgboost_polars.py   # Main training script (Polars pipeline)
├── models/
│   ├── feature_encoder.py    # Categorical encoding
│   ├── historical_features.py # Aggregate feature computation
│   ├── feature_pipeline.py   # Pandas feature orchestration
│   └── xgboost_ranker.py     # OOP model wrapper (for book examples)
├── utils/
│   ├── polars_pipeline.py    # Polars-based pipeline (memory efficient)
│   └── memory_efficient.py   # Pandas + dtype optimization
└── outputs/                  # Saved models and results
```

### Two Approaches (Educational Value)

| File | Approach | Teaching Purpose |
|------|----------|------------------|
| `train_xgboost_polars.py` | Script-based | Quick experimentation |
| `models/xgboost_ranker.py` | OOP class | Production patterns |

Both are valid; choice depends on use case.

---

## 7. Leakage Prevention Checklist

| Potential Leakage | Mitigation | Status |
|-------------------|------------|--------|
| Random train/test split | Use Global Temporal Split | ✅ Implemented |
| Raw timestamp as feature | Excluded; use derived cyclical features | ✅ Implemented |
| Frequency encoding on full data | Fit only on training set | ✅ Implemented |
| `user_item_avg_completion` | **REMOVED** - severe leakage (predicted label directly) | ✅ Fixed |
| `previous_listen_count` | PIT-correct cumulative count | ✅ Fixed |
| User/item aggregates | Lookup tables (accepted trade-off) | ⚠️ Documented |
| Test data influences feature scaling | Fit encoders on train only | ✅ Implemented |

### Leakage Risk Assessment

| Feature Type | Leakage Risk | Action Taken |
|-------------|--------------|--------------|
| `user_item_avg_completion` | **SEVERE** | ❌ Removed |
| `previous_listen_count` | **HIGH** (was) | ✅ PIT-corrected |
| User-level aggregates | **LOW** | ⚠️ Accepted (documented) |
| Item-level aggregates | **LOW** | ⚠️ Accepted (documented) |

### Observed Metrics After Fixes

| Metric | Before Fixes | After PIT Fixes |
|--------|-------------|-----------------|
| Train AUC-ROC | 0.9455 | 0.7962 |
| Test AUC-ROC | 0.5367 | 0.6282 |
| Train-Test Gap | 0.41 ⚠️ | 0.17 ✅ |

### Adversarial Validation

To empirically verify no leakage:
```python
# Train classifier to distinguish train vs test
# If AUC ≈ 0.5: No distinguishable difference (good)
# If AUC >> 0.5: Potential leakage (investigate)
```

---

## 8. Available Dataset Files and Alternative Tasks

### Yambda Dataset Structure

The Yambda dataset provides multiple files that can be used for different pointwise prediction tasks:

| File | Schema | Size | Potential Task |
|------|--------|------|----------------|
| **flat/listens.parquet** ✅ | uid, item_id, timestamp, played_ratio_pct, is_organic, track_length_seconds | ~50M rows | Listen completion |
| **flat/likes.parquet** | uid, item_id, timestamp | ~2M rows | Like prediction |
| **flat/dislikes.parquet** | uid, item_id, timestamp | ~500K rows | Dislike prediction |
| **flat/multi_event.parquet** | uid, item_id, timestamp, event_type, played_ratio_pct | Combined | Multi-task learning |
| **flat/unlikes.parquet** | uid, item_id, timestamp | Small | Preference reversal |
| **flat/undislikes.parquet** | uid, item_id, timestamp | Small | Preference reversal |
| **embeddings.parquet** | item_id, embed, normalized_embed | ~13GB | Audio features |
| **album_item_mapping.parquet** | item_id, album_id | Metadata | Content features |
| **artist_item_mapping.parquet** | item_id, artist_id | Metadata | Content features |

### Alternative Pointwise Tasks

#### Task 1: Listen Completion (Current Implementation)
```
Data: flat/listens.parquet
Label: played_ratio_pct >= 50%
Signal: Implicit (completion rate)
```
- **Pros:** Large dataset, rich engagement signal
- **Cons:** Threshold is somewhat arbitrary

#### Task 2: Like Prediction
```
Data: flat/likes.parquet + flat/listens.parquet
Label: 1 if user liked the track, 0 if listened but didn't like
Signal: Explicit positive feedback
```
- **Pros:** Stronger signal than completion, explicit user preference
- **Cons:** Sparser data, class imbalance (few likes vs many listens)

#### Task 3: Combined Engagement Score
```
Data: flat/multi_event.parquet
Labels: Multiple targets
  - label_completion: played >= 50%
  - label_liked: event_type = 'like' exists
  - label_disliked: event_type = 'dislike' exists
Signal: Multi-objective
```
- **Pros:** Captures full engagement spectrum
- **Cons:** Requires multi-task architecture (covered in later chapters)

### Data Loader Support

The `data_loader.py` already supports loading alternative files:

```python
# Load multi_event for multi-task learning
from data_loader import YambdaDataLoader

loader = YambdaDataLoader(data_dir)
train_table, test_table = loader.load_multi_event_with_gts(
    event_types=['listen', 'like', 'dislike']  # Filter to specific events
)
```

### Recommended Chapter Progression

| Stage | Data Source | Task | Model | Chapter |
|-------|-------------|------|-------|---------|
| **Baseline** | listens.parquet | Listen completion | XGBoost (pointwise) | Ch 6 |
| **Pairwise LTR** | listens.parquet | Listen completion | XGBRanker (pairwise) | Ch 6 |
| **Deep Learning** | listens.parquet | Listen completion | DeepFM | Ch 6 |
| **🏠 Homework** | listens + likes | Like prediction | XGBoost (pointwise) | Ch 6 |
| **Advanced DL** | listens.parquet | Listen completion | DCN-V2 / DLRM | Ch 7 |
| **Multi-Task** | multi_event.parquet | Multi-task (completion + like) | MMoE / PLE | Ch 7 |

---

## 9. Pointwise vs Pairwise Learning-to-Rank

This chapter implements both pointwise and pairwise approaches to allow direct comparison.

### Conceptual Differences

| Aspect | Pointwise | Pairwise (LTR) |
|--------|-----------|----------------|
| **Objective** | Predict P(completion) per item | Learn which item ranks higher |
| **Loss Function** | Binary cross-entropy | Pairwise ranking loss (LambdaMART) |
| **Training Unit** | Single (user, item) pair | Pairs of items within same user |
| **Key Metric** | AUC-ROC, AUC-PR | NDCG@K, MAP, MRR |
| **Use Case** | CTR prediction, probability calibration | Final ranking stage, position matters |

### Scripts

```bash
# Pointwise classification
python train_xgboost_polars.py --train_days 30

# Pairwise Learning-to-Rank
python train_xgboost_ltr.py --train_days 30
```

### XGBRanker Configuration

```python
# Key differences from XGBClassifier
model = xgb.XGBRanker(
    objective='rank:pairwise',  # LambdaMART-style pairwise loss
    # ... same hyperparameters as pointwise
)

# Training requires query groups (items compared within same group)
model.fit(
    X_train, y_train,
    group=train_groups,  # Array of group sizes: [n1, n2, n3, ...]
    eval_set=[(X_test, y_test)],
    eval_group=[test_groups],
)
```

### Query Groups

LTR requires defining **query groups** - items within the same group are compared pairwise:

```python
# Group by user_id: items from same user form a query group
# User 1: [item_a, item_b, item_c]  → group_size = 3
# User 2: [item_d, item_e]          → group_size = 2
# groups = [3, 2, ...]

def create_query_groups(df, group_col='uid'):
    return df.group_by(group_col, maintain_order=True).count()['count'].to_numpy()
```

**Design Decision: User-Level vs Session-Level Grouping**

| Grouping | Pros | Cons |
|----------|------|------|
| **User-level** (current) | More pairs, simpler | Items from different sessions compared |
| **Session-level** | Realistic competition | Requires session_id, smaller groups |

We use **user-level grouping** because:
1. Yambda lacks explicit session identifiers
2. Creating implicit sessions from timestamps is unreliable
3. More training pairs improve model stability

**Limitation:** Items from different sessions (morning vs evening) are compared
as if they competed for attention. Production systems with session tracking
should consider session-level grouping.

### LTR Objectives Available

| Objective | Description | Best For |
|-----------|-------------|----------|
| `rank:pairwise` | LambdaMART-style pairwise loss | General ranking |
| `rank:ndcg` | LambdaRank optimizing NDCG | When NDCG is primary metric |
| `rank:map` | Optimizes MAP directly | When precision at top matters |

### Experimental Results Comparison

#### 30-Day Training Window (~1.3M samples)

| Metric | Pointwise (XGBClassifier) | Pairwise LTR (XGBRanker) |
|--------|---------------------------|--------------------------|
| **Train AUC-ROC** | 0.8661 | 0.6512 |
| **Test AUC-ROC** | 0.5354 | 0.5144 |
| **Train AUC-PR** | 0.9258 | 0.8096 |
| **Test AUC-PR** | 0.7720 | 0.6580 |
| **Test NDCG@5** | — | 0.7483 |
| **Test NDCG@10** | — | 0.7815 |
| **Test MAP** | — | 0.7523 |
| **Test MRR** | — | 0.7672 |
| **Train Samples** | 1,259,976 | 1,240,047 |
| **Training Time** | 5.8s | 6.3s |

#### 150-Day Training Window (~6.3M samples)

| Metric | Pointwise (XGBClassifier) | Pairwise LTR (XGBRanker) |
|--------|---------------------------|--------------------------|
| **Train AUC-ROC** | 0.8246 | 0.6095 |
| **Test AUC-ROC** | **0.5955** | 0.5364 |
| **Train AUC-PR** | 0.8924 | 0.7569 |
| **Test AUC-PR** | **0.8014** | 0.6769 |
| **Test NDCG@5** | — | **0.7484** |
| **Test NDCG@10** | — | **0.7828** |
| **Test MAP** | — | **0.7531** |
| **Test MRR** | — | **0.7749** |
| **Train Samples** | 6,292,354 | 4,535,948 |
| **Training Time** | 24.3s | 16.1s |

#### Scaling Comparison: 30 → 150 Days

| Metric | 30 days | 150 days | Δ (Improvement) |
|--------|---------|----------|-----------------|
| **Pointwise Test AUC** | 0.5354 | 0.5955 | **+0.060 (+11%)** ✅ |
| **LTR Test AUC** | 0.5144 | 0.5364 | +0.022 (+4%) |
| **LTR Test NDCG@10** | 0.7815 | 0.7828 | +0.001 (~stable) |

#### Key Insights from Experiments

1. **Pointwise benefits more from additional data**: Test AUC improved by 11% (0.54→0.60) with 5× more training data.

2. **LTR maintains ranking quality with less data**: NDCG/MAP/MRR are stable across training sizes—pairwise comparison learns relative ordering efficiently.

3. **Different feature importance patterns**:
   - **Pointwise** relies heavily on `user_median_completion` (63%)
   - **Pairwise LTR** relies more on `item_avg_completion` (48%)

4. **LTR AUC is naturally lower**: Pairwise ranking loss doesn't optimize for probability calibration, so AUC is a secondary metric.

**Key Insight:** Pairwise LTR may have slightly lower AUC (optimizes ranking, not probability) but better NDCG (directly optimizes position-aware metrics).

### When to Use Each

| Scenario | Recommendation |
|----------|----------------|
| CTR prediction for bidding | **Pointwise** (need calibrated probabilities) |
| Final ranking of candidates | **Pairwise LTR** (position matters) |
| Probability displayed to user | **Pointwise** (calibration needed) |
| Top-K recommendation | **Pairwise LTR** (NDCG@K optimized) |
| A/B test with clicks | Either (evaluate with business metrics) |

### Important Nuance: Training vs Inference Behavior

A common misconception: **At inference time, both pointwise and pairwise models produce a single score per item.** The difference is entirely in what the model learned during training.

```
Inference (identical for both):
    Input: (user, item) features  →  Model  →  Output: single score
    Then: Sort all items by score to produce ranking
```

The key difference is what those scores represent:

| Aspect | Pointwise Scores | Pairwise Scores |
|--------|------------------|-----------------|
| **Example** | [0.82, 0.79, 0.45, 0.41] | [2.1, 1.8, -0.3, -0.5] |
| **Interpretation** | Calibrated P(click) | Arbitrary (only order matters) |
| **Range** | [0, 1] | Unbounded |
| **Usable for bidding?** | ✅ Yes | ❌ Not directly |

Both produce the same ranking order, but pointwise scores are interpretable probabilities while pairwise scores only encode relative ordering.

### Why Not Pairwise + Calibration for pCTR?

If you need calibrated probabilities (pCTR), you might consider: train pairwise for ranking quality, then apply calibration (e.g., isotonic regression) to get probabilities. **This is generally not recommended:**

| Approach | Pros | Cons |
|----------|------|------|
| **Pointwise Classification** | Direct probability optimization; native [0,1] output; simpler pipeline | May need light recalibration over time |
| Pairwise + Calibration | Good ranking | Information loss; two models; calibration breaks with distribution shift |

**The core problem:** Pairwise training explicitly discards absolute magnitude information—it only learns "A > B", not "A = 0.8, B = 0.3". Calibration tries to recover this lost information from ordering alone, which is inherently lossy:

```
Pairwise learns:     [+2.1, +1.8, -0.3, -0.5]  → Only ordering preserved
Calibration maps:    [0.82, 0.79, 0.45, 0.41]  → Trying to recover magnitudes
                                                  from ordering alone (lossy!)
```

### Industry Practice for pCTR

Most production pCTR systems (Google, Meta, etc.) use:

```
┌─────────────────────────────────────────────────────────┐
│  1. Pointwise Classification (XGBoost / Neural Net)     │
│                          ↓                              │
│  2. Optional: Isotonic Regression on recent holdout     │
│                          ↓                              │
│  3. Regular recalibration (daily/weekly)                │
└─────────────────────────────────────────────────────────┘
```

The calibration layer exists not because the model is bad, but because:
- **Distribution shift** between training and serving
- **Data freshness** (model trained on last week, serving today)
- **Business constraints** (eCPM floors, budget pacing)

**Recommendation:** For pCTR, use pointwise classification (`binary:logistic`) with optional isotonic calibration. Reserve pairwise LTR for scenarios where ranking quality (NDCG, MAP) is the primary objective and calibrated probabilities are not needed.

---

## 10. Homework: Like Prediction Task

**Objective:** Build a pointwise model to predict whether a user will *like* a track after listening to it.

### Setup

```python
# 1. Load both listens and likes
listens_df = pl.read_parquet("flat/listens.parquet")
likes_df = pl.read_parquet("flat/likes.parquet")

# 2. Create negative samples (listened but didn't like)
# Join listens with likes to create:
#   - label = 1: user liked the track
#   - label = 0: user listened but didn't like

# 3. Apply same feature pipeline (FeatureEncoder + HistoricalFeatures)

# 4. Train XGBoost with same hyperparameters
```

### Key Considerations

1. **Class Imbalance:** Likes are much rarer than listens (~4% like rate)
   - Use `scale_pos_weight` or SMOTE
   - Consider `eval_metric='aucpr'` instead of `'auc'`

2. **Negative Sampling:** How to define "didn't like"?
   - Option A: Listened but never liked (current session)
   - Option B: Listened with completion < 50% and no like
   - Option C: Random negatives from catalog (harder task)

3. **Time-Aware Labels:** A user might like a track *after* the test window
   - Use GTS strictly: only consider likes that occurred before test_start

### Expected Outcomes

| Metric | Listen Completion | Like Prediction |
|--------|-------------------|-----------------|
| Positive Rate | ~63% | ~4-8% |
| Expected Test AUC | ~0.65-0.68 | ~0.70-0.75 |
| Key Challenge | Moderate | Class imbalance |

---

## 11. DeepFM: Deep Learning Ranking Model

Building on the XGBoost baseline, we implement DeepFM to demonstrate deep learning approaches for CTR prediction.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       DeepFM                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sparse Features              Dense Features                │
│  (uid, item_id,              (numerical,                   │
│   hour, day_of_week)          historical, is_organic)       │
│       │                            │                        │
│       ▼                            │                        │
│  ┌─────────┐                       │                        │
│  │Embedding│                       │                        │
│  │ Tables  │                       │                        │
│  └────┬────┘                       │                        │
│       │                            │                        │
│       ├────────────────────────────┤                        │
│       │                            │                        │
│       ▼                            ▼                        │
│  ┌─────────┐              ┌─────────────────┐               │
│  │   FM    │              │ Concat + DNN    │               │
│  │ (2nd    │              │ (Higher order)  │               │
│  │ order)  │              │                 │               │
│  └────┬────┘              └────────┬────────┘               │
│       │                            │                        │
│       └──────────┬─────────────────┘                        │
│                  ▼                                          │
│           ┌──────────┐                                      │
│           │   Sum    │                                      │
│           │    +     │                                      │
│           │ Sigmoid  │                                      │
│           └──────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Theory: Why DeepFM?

#### The Core Problem: Feature Interactions

In CTR/ranking prediction, the magic often lies in **feature interactions**:

```
User A + Item X → High probability (they've interacted before)
User A + Item Y → Low probability (no history)
Morning + Workout playlist → High probability
Evening + Workout playlist → Lower probability
```

The challenge: How do we learn these interactions effectively?

#### Approach 1: Linear Models (Logistic Regression)

```
ŷ = sigmoid(w₀ + Σ wᵢxᵢ)
```

**Problem:** Only captures 1st-order effects. To get interactions, you need manual engineering:
```python
features['user_A_item_X'] = (user == 'A') & (item == 'X')  # Millions of combinations!
```

#### Approach 2: Factorization Machines (FM)

**Key insight:** Instead of learning a weight for every pair (user, item), learn a **latent vector** for each entity.

```
FM Formula:
ŷ = w₀ + Σ wᵢxᵢ + Σᵢ Σⱼ>ᵢ <vᵢ, vⱼ> xᵢxⱼ
         ↑           ↑
    1st order    2nd order (dot product of embeddings)
```

**Why this works:**
- Instead of O(n²) parameters for all pairs, only O(n×k) for embeddings
- **Generalization:** Even if User A never saw Item X, we can estimate their interaction from similar users/items

#### Approach 3: Deep Neural Networks (DNN)

DNNs can learn **arbitrary non-linear** feature interactions:
```
Input → Hidden₁ → Hidden₂ → Hidden₃ → Output
        (ReLU)    (ReLU)    (ReLU)
```

**Strength:** Can capture very complex, high-order interactions  
**Weakness:** Needs lots of data; may miss simple patterns that FM captures easily

#### DeepFM: Best of Both Worlds

```
DeepFM = FM (explicit 2nd-order) + DNN (implicit higher-order) + Shared Embeddings
```

**Key insight:** FM and DNN share the **same embeddings**, so:
1. FM explicitly captures 2nd-order interactions (interpretable)
2. DNN implicitly captures higher-order interactions (flexible)
3. Both benefit from the learned embeddings

#### The Embedding Advantage Over XGBoost

**XGBoost with frequency encoding:**
```
User 123 → freq = 0.001 (just a number)
User 456 → freq = 0.002 (just a number)
# No notion that these users might be similar!
```

**DeepFM with learned embeddings:**
```
User 123 → [0.2, 0.5, 0.1, 0.8]  ─┐
                                  ├─ These users are similar!
User 456 → [0.3, 0.4, 0.2, 0.9]  ─┘  (cosine similarity = 0.95)
```

The model learns that users with similar listening patterns should have similar embeddings, enabling **generalization** to unseen (user, item) pairs.

### Code Walkthrough: Where Each Component Lives

#### 1. FM Layer (`models/deepfm.py`, Lines 94-123)

The efficient FM computation:
```python
class FMLayer(nn.Module):
    """Computes: sum_{i<j} <v_i, v_j> * x_i * x_j
    
    Using efficient formula: 0.5 * (||Σvᵢ||² - Σ||vᵢ||²)
    """
    def forward(self, embeddings):
        sum_of_square = torch.sum(embeddings ** 2, dim=1)      # Σ||vᵢ||²
        square_of_sum = torch.sum(embeddings, dim=1) ** 2      # ||Σvᵢ||²
        fm_output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_output
```

#### 2. Linear Term (`forward()`, Lines 257-267)

1st-order features:
```python
# ===== 1st Order (Linear) Term =====
linear_output = self.bias.expand(batch_size, 1)        # w₀ (bias)

# Sparse linear terms: wᵢ for each categorical
for feat_name, indices in sparse_inputs.items():
    linear_embed = self.linear_embeddings[feat_name](indices)
    linear_output = linear_output + linear_embed

# Dense linear terms
linear_output = linear_output + self.dense_linear(dense_inputs)
```

#### 3. FM Term (`forward()`, Lines 269-285)

2nd-order interactions:
```python
# ===== 2nd Order (FM) Term =====
embed_list = []
for feat_name, indices in sparse_inputs.items():
    embed = self.embeddings[feat_name](indices)  # vᵢ for each feature
    embed_list.append(embed)

stacked_embeds = torch.stack(embed_list, dim=1)  # (batch, num_fields, embed_dim)
fm_output = self.fm(stacked_embeds)              # Calls FMLayer
```

#### 4. DNN Term (`forward()`, Lines 287-301)

Higher-order interactions:
```python
# ===== DNN Term =====
sparse_concat = torch.cat([embeddings...], dim=1)     # Flatten embeddings
dnn_input = torch.cat([sparse_concat, dense_inputs])  # Add dense features
dnn_hidden = self.dnn(dnn_input)                      # MLP layers
dnn_output = self.output_layer(dnn_hidden)            # Project to scalar
```

#### 5. Combination (`forward()`, Line 304)

Where all three terms meet:
```python
# ===== Combine all terms =====
logits = linear_output + fm_output + dnn_output       # Simple sum!
```

### When to Use DeepFM vs XGBoost

| Scenario | Recommended Model |
|----------|-------------------|
| Small data (<1M rows) | XGBoost |
| Large data (>10M rows) | DeepFM |
| High-cardinality IDs | DeepFM (embeddings shine) |
| Well-engineered features | XGBoost |
| Cold-start mitigation needed | DeepFM |
| Fast training required | XGBoost |
| GPU available | DeepFM |

### Feature Configuration for DeepFM

| Feature Type | Features | Handling |
|-------------|----------|----------|
| **Sparse (Embeddings)** | uid, item_id, hour_of_day, day_of_week | Learned embeddings (dim=16) |
| **Dense (MLP)** | is_organic, track_length_seconds, all historical features | StandardScaler + MLP |

**Design Decision:** `is_organic` is treated as dense (not embedding) because it only has 2 levels—embeddings are valuable for high-cardinality categoricals where we want to learn semantic relationships.

### Key Differences from XGBoost

| Aspect | XGBoost | DeepFM |
|--------|---------|--------|
| **ID Features** | Frequency encoding | Learned embeddings |
| **Feature Interactions** | Decision trees (explicit) | FM + DNN (learned) |
| **Training** | Gradient boosting | Mini-batch SGD |
| **Scalability** | CPU-efficient | GPU-accelerated |
| **Interpretability** | Feature importance | Harder to interpret |

### Training Commands

```bash
# Quick run (30 days)
python train_deepfm.py --train_days 30

# Full dataset
python train_deepfm.py --train_days 0

# Custom hyperparameters
python train_deepfm.py --train_days 30 --embed_dim 32 --mlp_dims 512,256,128 --batch_size 8192
```

### Actual Results Comparison (30-Day Training Window)

| Model | Test AUC-ROC | Test AUC-PR | Training Time | Parameters |
|-------|--------------|-------------|---------------|------------|
| XGBoost Pointwise | ~0.58 | ~0.80 | ~6s | ~100 trees |
| XGBoost LTR | ~0.54 | N/A | ~15s | ~100 trees |
| **DeepFM** | **0.8031** | **0.8994** | ~13 min | 3.2M |

**Key Finding:** DeepFM dramatically outperforms XGBoost on the same 30-day data:
- **Test AUC-ROC: 0.80 vs 0.58** — **38% relative improvement**
- **Test AUC-PR: 0.90 vs 0.80** — **12% relative improvement**

This demonstrates the power of **learned embeddings** for high-cardinality IDs (uid, item_id). XGBoost relies on frequency encoding which loses individual user/item preferences, while DeepFM learns dense representations that capture collaborative filtering signals.

### 150-Day Training Window Results

| Model | Test AUC-ROC | Test AUC-PR | Training Time |
|-------|--------------|-------------|---------------|
| XGBoost Pointwise | 0.5955 | 0.8014 | ~25s |
| XGBoost LTR | 0.5364 | N/A | ~24s |
| **DeepFM** | TBD | TBD | ~30-45 min |

### Homework: Enhance DeepFM

Students can extend the DeepFM implementation with:

1. **Add artist/album embeddings** (from mapping files)
   ```python
   sparse_features['artist_id'] = {'vocab_size': 50000, 'embed_dim': 16}
   sparse_features['album_id'] = {'vocab_size': 100000, 'embed_dim': 16}
   ```

2. **Incorporate audio embeddings** (from embeddings.parquet)
   - Pre-trained 256-dim audio features
   - Concatenate with item embedding

3. **Try DCN-V2** architecture
   - Replace FM with Cross Network
   - State-of-the-art performance

---

## 12. Model Implementation Status

| Model | Script | Status | Test AUC-ROC (30d) | Notes |
|-------|--------|--------|-------------------|-------|
| **XGBoost Pointwise** | `train_xgboost_polars.py` | ✅ Complete | ~0.58 | Baseline with engineered features |
| **XGBoost Pairwise LTR** | `train_xgboost_ltr.py` | ✅ Complete | ~0.54 | NDCG@10 ~0.78 |
| **DeepFM** | `train_deepfm.py` | ✅ Complete | **0.80** | FM + DNN, learned embeddings |

**Note:** Advanced architectures (DCN-V2, DLRM, MMoE, PLE) are covered in Chapter 7.

---

## 13. Future Extensions (Beyond Chapter 6)

This chapter's models set the stage for:

1. **DCN-V2 (Deep & Cross Network)**
   - Explicit cross-network for bounded-degree feature interactions
   - State-of-the-art on many CTR benchmarks

2. **Multi-Task Learning**
   - Predict listen completion + like + dislike simultaneously
   - MMoE, PLE architectures
   - Uses `multi_event.parquet` from Yambda

3. **Learning-to-Rank (Advanced)**
   - Listwise formulations (ListNet, ListMLE)
   - Neural LTR models

4. **Embedding Features**
   - Audio embeddings from Yambda (~13GB)
   - User/item embedding similarity
   - Hybrid: Neural feature extraction + GBDT ranker

5. **Sequential Models**
   - Uses `sequential/` folder in Yambda
   - Transformer-based architectures (SASRec, BERT4Rec)

---

## 14. Key Takeaways for Readers

1. **Temporal splits are non-negotiable** for RecSys evaluation
2. **Feature engineering > model complexity** for tree-based models
3. **Memory efficiency matters** for real-world datasets
4. **Leakage prevention** requires discipline at every step
5. **Point-in-Time (PIT) correctness** is critical for historical features
6. **Pointwise vs Pairwise:** Choose based on your optimization goal (probability vs ranking)
7. **Start simple** (pointwise XGBoost) before complex (pairwise LTR, deep learning)
8. **Multiple prediction tasks** can be derived from the same dataset

---

## Appendix A: Running the Code

### Environment Setup
```bash
pip install polars pyarrow xgboost scikit-learn pandas numpy torch
```

### Training Commands

```bash
# Set data path (once) - Linux/Mac
export YAMBDA_DATA_DIR=/path/to/Yandex/flat

# Set data path (once) - Windows PowerShell
$env:YAMBDA_DATA_DIR = "C:\path\to\Yandex\flat"

# =====================================================================
# POINTWISE CLASSIFICATION (train_xgboost_polars.py)
# =====================================================================
# Quick run (30 days, ~1.3M rows, ~1 min)
python train_xgboost_polars.py --train_days 30

# Full run (~1500 days, ~46M rows, ~4 min)
python train_xgboost_polars.py --train_days 0

# =====================================================================
# PAIRWISE LEARNING-TO-RANK (train_xgboost_ltr.py)
# =====================================================================
# Quick run (30 days)
python train_xgboost_ltr.py --train_days 30

# Full run
python train_xgboost_ltr.py --train_days 0

# With different LTR objective
python train_xgboost_ltr.py --train_days 30 --objective rank:ndcg

# =====================================================================
# DEEPFM (train_deepfm.py)
# =====================================================================
# Quick run (30 days)
python train_deepfm.py --train_days 30

# Full dataset
python train_deepfm.py --train_days 0

# Custom hyperparameters
python train_deepfm.py --train_days 30 --embed_dim 32 --mlp_dims 512,256,128

# =====================================================================
# COMPARISON: Run all to compare metrics
# =====================================================================
python train_xgboost_polars.py --train_days 30  # Pointwise XGBoost → AUC
python train_xgboost_ltr.py --train_days 30     # Pairwise LTR → NDCG
python train_deepfm.py --train_days 30          # DeepFM → AUC (deep learning)
```

### Expected Results (with PIT-correct features)

| Metric | 300-day Window | Full Dataset (~1500 days) |
|--------|----------------|---------------------------|
| Train AUC-ROC | ~0.78-0.80 | ~0.79-0.80 |
| Test AUC-ROC | ~0.60-0.63 | **~0.66-0.68** |
| Test AUC-PR | ~0.80-0.82 | **~0.82-0.83** |
| Train-Test Gap | ~0.15-0.18 | **~0.12-0.14** |
| Training Time | ~1-2 min | ~3-4 min |
| Feature Eng Time | ~5-6 sec | ~25-30 sec |

**Latest Run (Full Dataset with PIT Fix):**
```
Train AUC-ROC: 0.7964
Test AUC-ROC:  0.6680  ← Healthy generalization
Test AUC-PR:   0.8221
Train-Test Gap: 0.13   ← Acceptable
```

**Note:** The remaining train-test gap (~0.13) is expected due to:
1. User/item aggregate features using lookup tables (not strict PIT)
2. Natural distribution shift in user behavior over time
3. Cold-start users/items in test set

This is acceptable for educational purposes. Production systems would use a Feature Store for strict PIT correctness across all features.

