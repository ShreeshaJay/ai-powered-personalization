# Chapter 7: Advanced Ranking Model Design Notes

## Overview

This document captures the key design decisions made when implementing advanced ranking models (DCN-V2, DLRM, MMoE) for the Yambda music recommendation dataset. These notes are intended to support chapter writing and serve as a reference for readers who want to understand the "why" behind implementation choices.

---

## Script Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CHAPTER 7 WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   YAMBDA DATASET    │
│  (Required Input)   │
├─────────────────────┤
│ • listens.parquet   │
│ • likes.parquet     │
│ • dislikes.parquet  │
│ • embeddings.parquet│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SINGLE-TASK MODELS (Comparison)                      │
│                                                                             │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │   train_dcn.py    │    │train_dcn_pretrained│   │  train_dlrm.py    │   │
│  │  (learned embeds) │    │   (recommended)   │    │  (for contrast)   │   │
│  │                   │    │                   │    │                   │   │
│  │ • Shows overfitting│   │ • Uses pre-trained│    │ • Dot interactions│   │
│  │ • Baseline only   │    │   item embeddings │    │ • Comparison only │   │
│  └───────────────────┘    └─────────┬─────────┘    └───────────────────┘   │
│                                     │                                       │
│  ┌───────────────────┐              │                                       │
│  │train_mmoe_single.py│             │                                       │
│  │(single-task MMoE) │◄─────────────┼── Compare architectures               │
│  │                   │              │                                       │
│  │ • Same features   │              │                                       │
│  │ • MMoE vs DCN-V2  │              │                                       │
│  └───────────────────┘              │                                       │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-TASK MODEL (Main Focus)                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         train_mmoe.py                               │   │
│  │                    (5-task multi-task learning)                     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  Tasks:                                                             │   │
│  │    • engagement (binary)  - P(any listening)                        │   │
│  │    • completion (binary)  - P(≥50% listened)                        │   │
│  │    • play_ratio (regression) - E[% listened]                        │   │
│  │    • like (binary)        - P(user likes)                           │   │
│  │    • dislike (binary)     - P(user dislikes)                        │   │
│  │                                                                     │   │
│  │  Key flags:                                                         │   │
│  │    --train_days 30        (data window)                             │   │
│  │    --num_experts 4        (best from experiments)                   │   │
│  │    --use_focal_loss       (for imbalanced tasks)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXPORTED ARTIFACTS                                │
│                        (outputs/mmoe_30d_4exp_*/)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For Chapter 8 (Value Functions):                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │  Option A: Inference│    │  Option B: Pre-computed│                     │
│  ├─────────────────────┤    ├─────────────────────┤                        │
│  │ • best_model.pt     │    │ • predictions.parquet│                       │
│  │ • feature_processor │    │ • calibration.json  │                        │
│  │ • polars_pipeline/  │    │ • value_function_   │                        │
│  │ • inference_example │    │   config.json       │                        │
│  └─────────────────────┘    │ • gate_weights_     │                        │
│                             │   sample.parquet    │                        │
│                             └─────────────────────┘                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │      CHAPTER 8        │
                          │   Value Functions     │
                          │                       │
                          │ Combines multi-task   │
                          │ predictions into      │
                          │ ranking scores        │
                          └───────────────────────┘
```

### Recommended Execution Order

```bash
# 1. Set data directory
export DATA_DIR="path/to/yambda/flat"

# 2. (Optional) Run DCN with learned embeddings to see overfitting
python train_dcn.py --train_days 30 --data_dir $DATA_DIR

# 3. Run DCN with pre-trained embeddings (recommended single-task baseline)
python train_dcn_pretrained.py --train_days 30 --data_dir $DATA_DIR

# 4. (Optional) Compare architectures with single-task MMoE
python train_mmoe_single.py --train_days 30 --data_dir $DATA_DIR

# 5. Run multi-task MMoE (main model for Chapter 8)
python train_mmoe.py --train_days 30 --data_dir $DATA_DIR

# 6. Outputs ready for Chapter 8 in outputs/mmoe_30d_4exp_*/
```

### Script Dependencies

| Script | Depends On | Produces | Purpose |
|--------|------------|----------|---------|
| `train_dcn.py` | listens.parquet | Model checkpoint | Demonstrate overfitting |
| `train_dcn_pretrained.py` | listens + embeddings.parquet | Model + metrics | Best single-task baseline |
| `train_dlrm.py` | listens + embeddings.parquet | Model + metrics | Architecture comparison |
| `train_mmoe_single.py` | listens + embeddings.parquet | Model + metrics | MMoE vs DCN comparison |
| `train_mmoe.py` | listens + likes + dislikes + embeddings | **All artifacts for Ch.8** | Main multi-task model |

---

## 1. Handling High-Cardinality Item Features

### The Problem

The `item_id` feature has **176,000+ unique values** in a 30-day sample (counted before the 2026-11-09 time-unit fix, when that sample covered about 6 days). Learning embeddings from scratch for such high-cardinality features leads to:

1. **Massive parameter count**: 176K × 16 dim = 2.8M parameters just for item embeddings
2. **Severe overfitting**: Model memorizes training items, fails to generalize
3. **Cold-start issues**: New items have random embeddings with no semantic meaning

**Evidence from the 30-day re-measurement (one seed):**
- Learned embeddings: train AUC-ROC is 0.8383 at the best epoch and climbs to 0.8983 by epoch 4, while validation peaks at epoch 1 (0.7748) and then falls. Training stopped at epoch 4. Test AUC-ROC at the selected checkpoint is 0.7242.
- Pre-trained embeddings: validation peaks at epoch 4 (0.7783) and training stops at epoch 7. Test AUC-ROC is 0.7440.
- The train-test gap stays near 0.10 for both (0.114 learned, 0.103 pre-trained).

### The Solution: Pre-trained Item Embeddings

The Yambda dataset includes `embeddings.parquet` with **128-dimensional pre-trained item embeddings** for 7.7M items. Instead of learning item embeddings, we:

1. **Remove `item_id` from sparse features** (no learned embedding)
2. **Load pre-trained embeddings** as fixed vectors
3. **Concatenate to dense features** for model input

```python
# Before: item_id as sparse feature (learned)
SPARSE_FEATURES = ['uid', 'item_id', 'hour_of_day', 'day_of_week']

# After: item_id as dense feature (pre-trained)
SPARSE_FEATURES = ['uid', 'hour_of_day', 'day_of_week']
# Item embeddings added to dense: 18 features + 128 dims = 146 total
```

### Benefits

| Aspect | Learned Embeddings | Pre-trained Embeddings |
|--------|-------------------|------------------------|
| Parameters | 5,750,202 | 262,950 |
| Item representation | 16-dim (limited) | 128-dim (richer) |
| Train-test gap | 0.114 | 0.103 |
| Cold-start | Random vectors | Zero vector fallback |
| Semantic meaning | Learned from scratch | Pre-computed from larger corpus |

### Implementation Details

The `PretrainedItemEmbeddings` class handles:

1. **Memory-efficient loading**: Reads parquet in batches (100K rows) to avoid 8GB memory spike
2. **Filtered loading**: Only loads embeddings for items present in training data
3. **Lookup table**: `item_to_idx` dict for O(1) embedding retrieval
4. **Cold-start handling**: Items without embeddings receive zero vectors

```python
def get_embeddings(self, item_ids: np.ndarray) -> np.ndarray:
    result = np.zeros((len(item_ids), self.embed_dim), dtype=np.float32)
    for i, item_id in enumerate(item_ids):
        if item_id in self.item_to_idx:
            result[i] = self.embeddings[self.item_to_idx[item_id]]
    return result
```

### Homework Extension: Alternative Approaches

Students can explore other techniques for handling high-cardinality features:

1. **Frequency bucketing**: Group rare items (< N interactions) into "RARE_ITEM" bucket
2. **Hashing trick**: Hash 176K items → 50K buckets, accepting some collisions
3. **Hierarchical features**: Add `artist_id` and `album_id` (lower cardinality)

---

## 2. Per-Feature Embedding Dimensions

### The Problem

Using a single `embed_dim=16` for all sparse features is suboptimal:

| Feature | Vocabulary | 4th Root Heuristic | Default (16) | Optimal |
|---------|------------|-------------------|--------------|---------|
| uid | ~8,000 | 9.4 | 16 ✓ | 8-16 |
| hour_of_day | 24 | 2.2 | 16 ❌ | 4-8 |
| day_of_week | 7 | 1.6 | 16 ❌ | 2-4 |

**Why it matters:**
- Wasted parameters: 7 days × 16 dims = 112 params for something 7 one-hot dims could represent
- Potential overfitting on low-cardinality features
- Unnecessary computational cost

### The Solution: Per-Feature Embedding Dimensions

The `DCNV2Config` now supports per-feature embedding dimensions:

```python
config = DCNV2Config(
    sparse_features={'uid': 8000, 'hour_of_day': 24, 'day_of_week': 7},
    embed_dim=16,  # Default
    embed_dims={    # Per-feature overrides
        'uid': 16,
        'hour_of_day': 4,
        'day_of_week': 2,
    },
)
```

### Heuristics for Choosing Embedding Dimensions

**Rule of thumb** (from embedding literature):
```
embed_dim ≈ min(50, vocab_size ** 0.25)
```

**Practical guidelines:**
- **Very small vocab (< 10)**: 2-4 dims, or consider one-hot encoding
- **Small vocab (10-100)**: 4-8 dims
- **Medium vocab (100-10K)**: 8-16 dims
- **Large vocab (10K-1M)**: 16-32 dims
- **Very large vocab (> 1M)**: 32-64 dims, or use pre-trained embeddings

### Homework Extension: Cyclical Encoding for Time Features

For periodic features like `hour_of_day` and `day_of_week`, **cyclical encoding** is often superior to embeddings:

```python
# Cyclical encoding captures that hour 23 is close to hour 0
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

# Similarly for day of week
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

**Benefits:**
- Only 4 dimensions total (vs 6 for 4+2 embeddings)
- Naturally captures periodicity
- No learned parameters
- Works well for any time-based feature

---

## 3. Input Dimension Calculation

With per-feature embeddings, the total input dimension is computed as:

```python
# Old: uniform embedding dimension
input_dim = num_sparse_features * embed_dim + num_dense_features

# New: sum of per-feature dimensions
input_dim = sum(embed_dim[f] for f in sparse_features) + num_dense_features
```

**Example calculation for our setup:**

| Component | Dimension |
|-----------|-----------|
| uid embedding | 16 |
| hour_of_day embedding | 4 |
| day_of_week embedding | 2 |
| **Total sparse** | **22** |
| Original dense features | 18 |
| Pre-trained item embedding | 128 |
| **Total dense** | **146** |
| **Total input dim** | **168** |

Compare to original (uniform 16-dim, learned item embedding):
- Sparse: 4 features × 16 = 64
- Dense: 18
- Total: 82

The pre-trained version has more information (168 vs 82 dims) but fewer learnable parameters.

---

## 4. Model Architecture Summary

### DCN-V2 with Pre-trained Embeddings

```
Input Features:
├── Sparse (learned embeddings):
│   ├── uid: 8K vocab → 16-dim embedding
│   ├── hour_of_day: 24 vocab → 4-dim embedding  
│   └── day_of_week: 7 vocab → 2-dim embedding
│   └── Total: 22 dims
│
└── Dense (fixed):
    ├── Original features: 18 dims (scaled)
    └── Pre-trained item embedding: 128 dims
    └── Total: 146 dims

Combined input: 168 dims
       ↓
┌──────────────────┐
│  Cross Network   │ (explicit feature interactions)
│  3 layers        │ x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
│  rank=32         │ W_l = U_l · V_l^T (low-rank)
└────────┬─────────┘
         │ (parallel structure)
┌────────┴─────────┐
│  Deep Network    │ (implicit interactions)
│  [256, 128, 64]  │
│  ReLU + Dropout  │
└────────┬─────────┘
         ↓
    Concatenate
         ↓
    Final Linear → Sigmoid → P(completion)
```

---

## 5. Training Configuration

### Recommended Hyperparameters

```bash
python train_dcn_pretrained.py \
    --train_days 30 \
    --embed_dim 16 \          # Default (overridden by SPARSE_EMBED_DIMS)
    --cross_layers 3 \
    --cross_rank 32 \
    --mlp_dims 256,128,64 \
    --dropout 0.1 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --batch_size 4096 \
    --patience 3
```

### Measured comparison (30-day window, one seed)

| Metric | Learned Item Embed | Pre-trained Item Embed |
|--------|-------------------|------------------------|
| Train AUC-ROC at the best epoch | 0.8383 | 0.8473 |
| Validation AUC-ROC | 0.7748 | 0.7783 |
| Test AUC-ROC | 0.7242 | 0.7440 |
| Parameters | 5,750,202 | 262,950 |
| Train-test gap | 0.114 | 0.103 |

---

## 6. Code Organization & Separation of Concerns

### File Structure

```
chapter7_advanced_ranking/
├── models/
│   └── dcn.py                    # Model architecture (dataset-agnostic)
├── train_dcn.py                  # Training script: learned item embeddings
├── train_dcn_pretrained.py       # Training script: pre-trained item embeddings
└── docs/
    └── DCN_Design_Notes.md       # This file
```

### Conceptual Distinction: `dcn.py` vs `train_dcn_pretrained.py`

| Aspect | `models/dcn.py` | `train_dcn_pretrained.py` |
|--------|-----------------|---------------------------|
| **Purpose** | Define the neural network architecture | Apply the model to Yambda dataset |
| **Scope** | Dataset-agnostic | Yambda-specific |
| **Contains** | `DCNV2Config`, `CrossNetworkV2`, `DNN`, `DCNV2` classes | Data loading, feature engineering, training loop |
| **Reusability** | Can be used with any dataset | Specific to Yambda + pre-trained embeddings |
| **Answers** | "What is DCN-V2?" | "How do I train DCN-V2 on Yambda?" |

### Why This Separation Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                    train_dcn_pretrained.py                      │
│  (Application Layer - Yambda-specific)                          │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│  │ Data Loading    │  │ Feature Eng.     │  │ Training Loop │   │
│  │ (Yambda)        │  │ (Pre-trained     │  │ (Logging,     │   │
│  │                 │  │  embeddings)     │  │  Checkpoints) │   │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬───────┘   │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    models/dcn.py                        │    │
│  │  (Model Layer - Dataset-agnostic)                       │    │
│  │                                                         │    │
│  │  DCNV2Config → DCNV2 → forward() → logits              │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits of Separation:**

| Benefit | Example |
|---------|---------|
| **Reusability** | Use `DCNV2` on MovieLens by writing a new training script |
| **Testability** | Test `dcn.py` in isolation with synthetic data |
| **Maintainability** | Change model architecture without touching data pipeline |
| **Clarity** | Readers understand "what is the model" vs "how to train it" |

### What Each File Contains

#### `models/dcn.py` — The Architecture

```python
# Defines the math and structure
class DCNV2Config:       # Hyperparameters (layers, dims, structure)
class CrossNetworkV2:    # x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
class DNN:               # Standard MLP with BN, ReLU, Dropout
class DCNV2:             # Combines Cross + Deep networks
class DCNV2Dataset:      # Generic PyTorch Dataset wrapper
```

#### `train_dcn_pretrained.py` — The Application

```python
# Defines how to use the model on Yambda
SPARSE_FEATURES = [...]          # Yambda-specific feature list
SPARSE_EMBED_DIMS = {...}        # Yambda-specific embed dims
PretrainedItemEmbeddings         # Yambda embeddings.parquet loader
DCNV2PretrainedFeatureProcessor  # Yambda feature transformation
main()                           # Training loop for Yambda
```

### Comparison: `train_dcn.py` vs `train_dcn_pretrained.py`

Both scripts use `models/dcn.py`, but differ in how they handle item_id:

| Aspect | `train_dcn.py` | `train_dcn_pretrained.py` |
|--------|----------------|---------------------------|
| **item_id handling** | Learned 16-dim embedding | Pre-trained 128-dim (frozen) |
| **Sparse features** | `uid, item_id, hour, day` | `uid, hour, day` |
| **Embedding dims** | Uniform 16 for all | Per-feature: uid=16, hour=4, day=2 |
| **Dense features** | 18 features | 18 + 128 item embed = 146 |
| **Parameters** | 5,750,202 | 262,950 |
| **Test AUC-ROC** | 0.7242 | 0.7440 |
| **Use case** | Learned item embeddings | Pre-trained item embeddings (about 22× fewer parameters) |

---

## 7. Complete Design Parameter Reference

### Feature Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `SPARSE_FEATURES` | `['uid', 'hour_of_day', 'day_of_week']` | Removed item_id (using pre-trained) |
| `uid` embed_dim | 16 | Primary personalization signal, ~8K vocab → 4th root ≈ 9.4 |
| `hour_of_day` embed_dim | 4 | Small vocab (24), captures time-of-day patterns |
| `day_of_week` embed_dim | 2 | Minimal vocab (7), weekday/weekend distinction |
| `DENSE_FEATURES` | 18 features | User/item historical + user-item interaction features |
| `ITEM_EMBED_DIM` | 128 (frozen) | Pre-trained from embeddings.parquet, prevents overfitting |

### Model Architecture

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `cross_layers` | 3 | Captures up to 4th-order interactions; diminishing returns beyond |
| `cross_rank` | 32 | 62% param reduction vs full-rank while preserving expressiveness |
| `mlp_dims` | [256, 128, 64] | Funnel shape for hierarchical abstraction |
| `structure` | 'parallel' | Cross + Deep learn complementary patterns; often outperforms stacked |
| `dropout` | 0.1 | Light regularization; pre-trained embeddings already reduce overfitting |
| `use_bn` | True | Batch normalization for training stability |

### Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `batch_size` | 4096 | GPU efficiency, stable gradients, ~307 batches/epoch |
| `lr` | 1e-3 | Standard Adam rate, combined with ReduceLROnPlateau |
| `weight_decay` | 1e-5 | Light L2 regularization on all parameters |
| `patience` | 3 | Early stopping balance: not too early, not too late |
| `epochs` | 10 | Upper bound; typically stops at epoch 4-7 with early stopping |

### Data Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `train_days` | 30 | Captures weekly patterns, manageable for iteration (the ~1.26M sample count predates the 2026-11-09 time-unit fix, when this window covered about 6 days) |
| `test_days` | 1 | Global Temporal Split; mimics production (predict tomorrow) |
| `gap_seconds` | 1800 | 30-min gap prevents label leakage across train/test boundary |
| `val_split_ratio` | 0.1 | Last 10% of the train window by time, used for all model selection |

### Validation Protocol (Model Selection)

All five training scripts (`train_dcn.py`, `train_dcn_pretrained.py`, `train_dlrm.py`, `train_mmoe.py`, `train_mmoe_single.py`) use three time-ordered splits, the same as Chapter 6:

```
[ train (fit) ][ validation ][ 30-min gap ][ test ]
```

- `split_validation_window()` in `utils/polars_pipeline.py` holds out the last `--val_split_ratio` of the train window **before** the feature pipeline is fitted, so validation rows get historical features from earlier data only, like test rows.
- Training rows get their user and item aggregates from the window just before the fit window (`split_training_history()`, same length as the fit window or `--history_days`), passed as `pipeline.transform(train_lf, history=...)`. Most aggregates are means of `played_ratio_pct`, the label column, so fitting them on the training rows and applying them to the same rows would put each row's own label in its features. Before 2026-11-09 the scripts did that. If the data does not reach far enough back, the earliest part of the fit window becomes history only and is dropped from training (a warning is logged).
- In `train_mmoe.py`, `load_multitask_splits()` cuts each split's like and dislike labels at the end of that split (see Section 14).
- Early stopping, `ReduceLROnPlateau`, and best-checkpoint selection all monitor the validation metric. The test window is scored once, with the selected checkpoint.
- In `train_mmoe.py`, classification thresholds are tuned on validation and applied to test (`f1_at_threshold`). `f1_max` on test uses a threshold tuned on test itself and is reported only as an optimistic diagnostic.

### Computed Dimensions

| Component | Calculation | Result |
|-----------|-------------|--------|
| Sparse embeddings | uid(16) + hour(4) + day(2) | 22 dims |
| Dense features | 18 original + 128 item embed | 146 dims |
| **Total input** | 22 + 146 | **168 dims** |
| Cross network output | Same as input | 168 dims |
| Deep network output | Last MLP layer | 64 dims |
| Concatenated (parallel) | 168 + 64 | 232 dims |

### Parameter Count Breakdown

| Component | Calculation | Parameters |
|-----------|-------------|------------|
| uid embedding | 8K × 16 | ~128K |
| hour embedding | 24 × 4 | 96 |
| day embedding | 7 × 2 | 14 |
| Cross network | 3 × (168 × 32 × 2 + 168) | ~32K |
| Deep network | 168×256 + 256×128 + 128×64 | ~84K |
| Output layer | 232 × 1 | 232 |
| **Total** | | **~250K sketch; the 30-day run reports 262,950** |

Compare to `train_dcn.py` with learned item embeddings: **5,750,202 parameters** in the 30-day run, about 22× this pre-trained model (262,950).

---

## 8. Key Takeaways for Chapter Writing

1. **High-cardinality features are the enemy of generalization** in neural ranking models. Pre-trained embeddings or dimensionality reduction techniques are essential.

2. **Embedding dimensions should scale with vocabulary size**, not be uniform across features. The 4th-root heuristic (`embed_dim ≈ vocab_size ** 0.25`) is a good starting point.

3. **Pre-trained embeddings trade parameter count for information density**. A 128-dim pre-trained vector carries more semantic information than a 16-dim learned vector for items with few training examples.

4. **Cold-start handling is implicit** when using pre-trained embeddings: new items get zero vectors, but the model can still make predictions based on other features.

5. **Monitor train/test AUC divergence** to detect overfitting. A growing gap is a red flag that the model is memorizing rather than learning.

6. **Separate model architecture from training logic**. The model (`dcn.py`) should be dataset-agnostic; the training script (`train_dcn_pretrained.py`) handles dataset-specific concerns.

7. **Justify every hyperparameter**. Don't use defaults blindly—understand why each value was chosen (e.g., cross_rank=32 gives 62% parameter reduction while preserving expressiveness).

8. **Provide multiple training scripts for pedagogical value**. `train_dcn.py` trains learned item embeddings. Train AUC keeps climbing after the best epoch, so early stopping is doing the work. `train_dcn_pretrained.py` trains the smaller pre-trained model. On this seed the test-day gain is 0.020 AUC, and the train-test gap stays near 0.10 for both.

---

## 9. Experimental Results: Pre-trained vs Learned Embeddings

> **Note (2026-11-13):** The result tables in this section, Section 14, and Section 17 are the re-measurement: one seed (seed 42), Colab L4, after the time-unit fix, the training-row history fix, and validation-set early stopping. AUC gaps under 0.005 are ties. Wall time and peak GPU memory for DCN, DLRM, and MMoE are not quoted, because another job shared the GPU.

> Earlier runs read one Yambda timestamp unit as 5 seconds. Timestamps are seconds, so those runs' "30-day" window covered about 6 days. The numbers below replace those runs.

### Results Comparison (30-day training window)

| Metric | Learned item embeddings | Pre-trained item embeddings |
|---|---:|---:|
| Test AUC-ROC | 0.7242 | 0.7440 |
| Validation AUC-ROC | 0.7748 | 0.7783 |
| Test AUC-PR | 0.8560 | 0.8655 |
| Test F1 | 0.7777 | 0.7859 |
| Train AUC-ROC at the best epoch | 0.8383 | 0.8473 |
| Train-test gap | 0.114 | 0.103 |
| Parameters | 5,750,202 | 262,950 |
| Best epoch (training stopped) | 1 (stopped at 4) | 4 (stopped at 7) |
| Item coverage | learned for every seen item | 97.3% |

Pre-trained embeddings are a head start, not a different regime. The test-day gain is 0.020 AUC (about 2.7% relative). On validation the two models are tied (0.7783 vs 0.7748). Both still have a train-test gap near 0.10. With learned embeddings, train AUC keeps climbing after epoch 1 (0.8383 to 0.8983 by epoch 4) while validation falls, so early stopping is doing the work. Pre-training does not remove that gap. It does cut the parameter count by about 22×, because the item table is no longer trained.

### Comparison with Other Models

| Model | Test AUC-ROC | Parameters | Notes |
|-------|-------------|------------|-------|
| XGBoost pointwise (30d) | 0.7247 | — | Validation 0.7825. Tied with learned DCN on this test day |
| XGBoost pointwise (150d) | 0.7593 | — | Longer window, same test day |
| DeepFM (30d) | 0.7088 | 6,088,859 | Below pointwise XGBoost on this window |
| DCN-V2 learned (30d) | 0.7242 | 5,750,202 | Validation 0.7748 |
| DCN-V2 pre-trained (30d) | 0.7440 | 262,950 | Test-day gain of 0.020 over learned embeddings. Validation 0.7783, tied with the learned model |

### What This Demonstrates

Pre-trained embeddings are a head start on this window. The test-day gain is 0.020 AUC. Validation does not separate the two models on this seed.

1. **A pre-trained item table is smaller.** Dropping the learned item embeddings cuts the parameter count from 5,750,202 to 262,950, about 22×, and raises test AUC from 0.7242 to 0.7440.

2. **The train-test gap stays near 0.10.** Early stopping is what holds the learned model back after epoch 1. Pre-training does not remove that gap.

3. **How you represent `item_id` still changes the result.** The change on this seed is a modest test-day gain and a much smaller model, not a different accuracy regime.

---

## 10. Connection to Later Chapters: Creating Embeddings

### The "Why Before How" Narrative

This chapter establishes *why* embedding quality matters through concrete experimental evidence:

```
Chapter 7 (This Chapter)              Later Chapter (Embeddings)
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  "Why pre-trained embeddings    │   │  "How to create these           │
│   matter for ranking"           │ → │   embeddings in the first place"│
│                                 │   │                                 │
│  Evidence:                      │   │  Methods:                       │
│  • Learned 16-dim test AUC 0.7242│  │  • Item2Vec (sequence-based)    │
│  • Pre-trained 128-dim AUC 0.7440│ │  • Matrix Factorization (ALS)   │
│  • about 22× fewer parameters   │   │  • Two-Tower Models             │
│  • train-test gap stays near 0.10│ │  • Contrastive Learning         │
│                                 │   │  • Graph Neural Networks        │
└─────────────────────────────────┘   └─────────────────────────────────┘
         Stakes & Impact                      Methods & Techniques
```

### Pedagogical Value of This Ordering

1. **Motivation first**: Readers see a concrete but modest gain from pre-trained embeddings (test AUC 0.7242 to 0.7440) before learning how to build them.

2. **Stakes are clear**: The test-day gain is 0.020 AUC (about 2.7% relative). On validation the two models are tied. The train-test gap stays near 0.10 for both, and the pre-trained model has about 22× fewer parameters.

3. **Practical grounding**: Readers understand *where* embeddings get used in the ranking stack before diving into creation methods.

4. **Connects retrieval and ranking**: The embeddings used here in ranking are the same kind created during retrieval/candidate generation, showing how the recommendation pipeline fits together.

### About the Yambda Embeddings

The `embeddings.parquet` file used in this chapter contains:

- **7.7 million items** with pre-computed embeddings
- **128 dimensions** per item (normalized)
- Likely created using **audio content features** or **collaborative filtering** at scale by Yandex

In later chapters, readers will learn to create similar embeddings for their own datasets using:

| Method | Best For | Key Idea |
|--------|----------|----------|
| **Item2Vec** | Sequential data | Treat item sequences like sentences, apply Word2Vec |
| **Matrix Factorization** | Explicit ratings | Decompose user-item matrix into latent factors |
| **Two-Tower Models** | Large-scale retrieval | Separate user and item encoders, trained with contrastive loss |
| **Graph Neural Networks** | Rich relationship data | Propagate information through user-item interaction graphs |

### Forward Reference for Readers

> "Pre-trained item embeddings raise test AUC from 0.7242 to 0.7440 on this 30-day window. That is a modest gain, validation does not separate the two models, and the train-test gap stays near 0.10. How were these embeddings created? In Chapter X, we'll explore techniques for generating high-quality item and user embeddings, including Item2Vec, matrix factorization, and two-tower models."

---

## 11. DLRM: Architectural Suitability for Music Streaming

### What is DLRM?

DLRM (Deep Learning Recommendation Model) was developed by Meta (Facebook) for **click-through rate (CTR) prediction at massive scale**. It's the reference architecture for many industrial recommendation systems.

```
┌─────────────────────────────────────────────────────────┐
│                      Top MLP                            │
│              (Final prediction layer)                   │
│                         ↑                               │
│              ┌──────────┴──────────┐                    │
│              │   Concatenation     │                    │
│              └──────────┬──────────┘                    │
│         ┌───────────────┼───────────────┐               │
│         ↓               ↓               ↓               │
│   [Bottom MLP]    [Dot Interactions]                    │
│   (Dense → d)      (All pairwise dots)                  │
│         ↑               ↑                               │
│    Dense Features  Sparse Embeddings                    │
│   (numerical)      (user, item, context)                │
└─────────────────────────────────────────────────────────┘
```

### Key Design Principles of DLRM

1. **Separate processing paths**: Dense features go through Bottom MLP; sparse features become embeddings
2. **Dot product interactions**: All pairwise dot products between embedding vectors
3. **Top MLP**: Combines transformed dense features with interaction outputs
4. **Designed for scale**: Efficient distributed training across many GPUs

### Why DLRM May Not Be Optimal for Music Streaming

While DLRM is a proven architecture, several characteristics make it suboptimal for music recommendation specifically:

#### 1. Limited Interaction Order (2nd-Order Only)

DLRM's interaction layer computes **pairwise dot products** between embeddings:

```python
# For embeddings e_user, e_item, e_hour, e_day:
interactions = [
    dot(e_user, e_item),   # user-item affinity
    dot(e_user, e_hour),   # user-time preference
    dot(e_user, e_day),    # user-weekday preference
    dot(e_item, e_hour),   # item-time popularity
    dot(e_item, e_day),    # item-weekday popularity
    dot(e_hour, e_day),    # time-day correlation
]
```

This is mathematically equivalent to **2nd-order Factorization Machines**. It captures pairwise relationships but **cannot learn higher-order patterns** like:

- "User A prefers jazz on weekday mornings" (user × genre × day_type × time)
- "Item X is popular with young users on Friday nights" (item × age_group × day × time)

**DCN-V2's Cross Network**, by contrast, explicitly models bounded-degree interactions up to the number of cross layers, enabling higher-order pattern learning.

#### 2. Music Preferences are Contextually Complex

| Domain | Interaction Complexity | DLRM Fit |
|--------|------------------------|----------|
| **Ads CTR** | User clicks ad → mostly user-ad affinity | ✅ Good |
| **E-commerce** | User buys product → user-product + price | ✅ Good |
| **Music Streaming** | User listens → user × mood × time × activity × genre × energy | ❌ Limited |

Music listening is highly contextual:
- **Temporal patterns**: Morning commute (upbeat) vs. evening relaxation (calm)
- **Sequential patterns**: What you just listened to predicts what you want next
- **Mood-based**: Same user, same item, different context → different outcome

DLRM's pairwise dot products struggle to capture these multi-dimensional interactions.

#### 3. No Sequence Modeling

DLRM treats each prediction as **independent**, with no awareness of:
- What the user listened to 5 minutes ago
- Listening session continuity
- Genre/mood momentum within a session

For music, **sequential models** (SASRec, BERT4Rec, GRU4Rec) often outperform pointwise models because "what's next" depends heavily on "what just happened."

#### 4. Same Item Embedding Challenge

Like DCN-V2, DLRM will suffer from the **same overfitting problem** with high-cardinality `item_id`:

```
Learned item embeddings → 176K items × 16 dims = 2.8M parameters
→ Severe overfitting (Train AUC high, Test AUC low)
```

The same solution applies: use pre-trained item embeddings as dense features rather than learning from scratch.

### When DLRM IS Appropriate

Despite limitations for music, DLRM makes sense when:

| Scenario | Why DLRM Works |
|----------|----------------|
| **Massive scale** (billions of examples) | Optimized for distributed training |
| **Inference latency critical** | Simpler than attention-based models |
| **2nd-order interactions sufficient** | True for many ads/e-commerce scenarios |
| **Industry benchmark needed** | Standard reference architecture |
| **Team familiarity** | Well-documented, many implementations |

### Architectural Comparison for Chapter 7

| Aspect | DLRM | DCN-V2 | Recommendation |
|--------|------|--------|----------------|
| **Interaction modeling** | 2nd-order (dot products) | Bounded-degree (cross layers) | DCN-V2 for complex patterns |
| **Feature crossing** | Implicit (pairwise) | Explicit (cross network) | DCN-V2 for interpretability |
| **Sequential awareness** | None | None | Neither; use SASRec for this |
| **Scale efficiency** | Excellent | Good | DLRM at massive scale |
| **Parameter efficiency** | Moderate | Better (low-rank) | DCN-V2 with low-rank cross |

### Pedagogical Value of Including DLRM

Even though DLRM isn't optimal for music streaming, it serves important educational purposes:

1. **Industry relevance**: Many readers will encounter DLRM variants at work (Meta, Pinterest, etc.)
2. **Architectural contrast**: Shows different philosophy vs. DCN-V2 (implicit vs. explicit interactions)
3. **Baseline comparison**: Demonstrates what pairwise interactions buy you (and don't)
4. **Same overfitting lesson**: Reinforces the pre-trained embeddings solution
5. **Design trade-offs**: Illustrates that architecture choice depends on problem characteristics

### Recommended Framing for the Chapter

> "DLRM represents Meta's approach to large-scale ranking, optimized for efficiency and pairwise feature interactions. While powerful for ads CTR prediction where user-item affinity dominates, music streaming's richer contextual patterns—spanning user, time, mood, and sequential history—benefit from DCN-V2's explicit feature crossing capabilities. Both architectures face the same fundamental challenge with high-cardinality item embeddings, which we address using pre-trained representations."

### Future Extensions (Beyond Chapter 7)

For readers who want to explore architectures better suited for music streaming:

| Architecture | Key Advantage for Music |
|--------------|------------------------|
| **SASRec** | Self-attention over listening history |
| **BERT4Rec** | Bidirectional sequence modeling |
| **GRU4Rec** | RNN-based session modeling |
| **Two-Tower** | Efficient retrieval with separate user/item encoders |
| **PLE** | Better multi-task learning than MMoE |

These could be covered in an advanced chapter or left as exercises for motivated readers.

---

## 12. MMoE: Expert Count Selection

### Default Configuration

The MMoE implementation uses **4 experts** by default, which applies to both single-task and multi-task configurations.

### Selection Heuristics

| Rule of Thumb | Reasoning |
|---------------|-----------|
| **≥ Number of Tasks** | Each task should have access to at least one expert that can specialize |
| **1.5x - 2x Tasks** | Allows for shared experts plus task-specific ones |
| **2-8 typical range** | Below 2 defeats the purpose; above 8 often leads to underutilized experts |

### Observed Expert Utilization (Single-Task, 4 Experts)

From our experiments with the completion task:

```
Expert 1: 34.9%  ← Dominant specialist
Expert 2: 21.5%  ← Supporting
Expert 3: 20.7%  ← Supporting  
Expert 4: 22.9%  ← Supporting
```

Even with a single task, the model learns to:
- **Specialize one expert** (~35%) for the dominant pattern
- **Distribute remaining work** across others for complementary patterns

### Trade-offs

| More Experts (↑) | Fewer Experts (↓) |
|------------------|-------------------|
| ✅ More capacity for specialization | ✅ Fewer parameters |
| ✅ Better for diverse tasks | ✅ Faster training |
| ❌ Risk of underutilized experts | ❌ May force over-sharing |
| ❌ More parameters | ❌ Less task specialization |

### 📝 Homework: Expert Count Experiments

**Objective**: Understand how the number of experts affects model performance and expert utilization.

**Experiments to run**:

```bash
# Conservative (forces sharing)
python train_mmoe.py --num_experts 4 --train_days 30 --data_dir <path>

# Balanced (1.5x tasks for 4 tasks)
python train_mmoe.py --num_experts 6 --train_days 30 --data_dir <path>

# Generous (2x tasks)  
python train_mmoe.py --num_experts 8 --train_days 30 --data_dir <path>
```

**Questions to answer**:

1. How do test AUC metrics change across the four tasks (engagement, completion, like, dislike)?
2. How does the gate weight distribution change? Do some experts become underutilized with more experts?
3. What is the parameter count vs. performance trade-off?
4. Does increasing experts help more for the minority tasks (like, dislike) than majority tasks (engagement, completion)?

**Expected observations**:
- With 4 experts (= 4 tasks): Experts may be forced to share across tasks
- With 6 experts: More room for task specialization while maintaining some sharing
- With 8 experts: Risk of "dead" experts with near-zero gate weights

---

## 13. References

- **DCN-V2 Paper**: Wang et al. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (WWW 2021)
- **DLRM Paper**: Naumov et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (arXiv 2019)
- **Factorization Machines**: Rendle, "Factorization Machines" (ICDM 2010) — DLRM's dot-product interactions are equivalent to 2nd-order FM
- **Embedding Dimension Heuristics**: Various industry blog posts, Google's recommendation guidelines
- **Pre-trained Embeddings in RecSys**: Item2Vec, BERT4Rec, and similar approaches
- **Low-Rank Factorization**: Standard technique for reducing parameters while preserving capacity
- **Sequential Recommendation**: SASRec (Kang & McAuley, 2018), BERT4Rec (Sun et al., 2019)

---

## 14. Multi-Task Learning: Temporal Leakage and Corrected Labels

### The Problem: Temporal Leakage in Like/Dislike Labels

When building multi-task models that predict user actions like "like" or "dislike," a common pitfall is **temporal leakage**—using future information to label past events.

#### Initial (Incorrect) Approach

```python
# WRONG: Join only on (uid, item_id), ignoring timestamps
liked_pairs = likes_lf.select(['uid', 'item_id']).unique()
listens_lf.join(liked_pairs, on=['uid', 'item_id'], how='left')
```

This approach labels a listen as "liked" if the user **ever** liked that item, even if the like happened days or weeks **after** the listen event.

#### The Evidence of Leakage

| Metric | Leaky Labels | Corrected Labels | Interpretation |
|--------|--------------|------------------|----------------|
| **Like positive rate** | 24.98% | **1.04%** | leaky join vs training window |
| **Dislike positive rate** | 0.75% | **0.20%** | leaky join vs training window |

A 25% like rate is unrealistic for music streaming. Users do not like 1 in 4 songs they listen to. Corrected like rate 1.04% on the training window (0.74% on the test day), about 1 in 100 listens. Corrected dislike rate 0.20% (0.15% on the test day). The 24.98% and 0.75% figures describe the old leaky join. That bug was not re-run.

### The Solution: Time-Windowed Causal Join

We implemented a causal join that only counts a like if it occurred **after the listen** and **within a time window**:

```python
# CORRECT: Time-windowed join (24-hour window)
# A listen is labeled "liked" only if:
#   listen_timestamp <= like_timestamp <= listen_timestamp + 24_hours

LIKE_WINDOW_HOURS = 24
# Yambda timestamps are seconds (rounded down to multiples of 5)
LIKE_WINDOW_TS = LIKE_WINDOW_HOURS * 60 * 60  # 86,400

# Join with timestamp filtering
joined = listens.join(likes, on=['uid', 'item_id'])
filtered = joined.filter(
    (col('like_ts') >= col('listen_ts')) &
    (col('like_ts') <= col('listen_ts') + LIKE_WINDOW_TS)
)
```

#### Visual Timeline

```
Timeline showing correct labeling (timestamps in seconds):
|-- Listen at ts=100,000
|   |-- Like at ts=100,300 (5 mins later)   -> labeled as liked
|   |-- Like at ts=182,800 (23 hrs later)   -> labeled as liked
|   |-- Like at ts=190,000 (25 hrs later)   -> NOT labeled (outside window)
|   `-- Like at ts=50,000 (before listen)   -> NOT labeled (before listen)
```

#### Cut at the End of Each Split

A listen near the end of its split has part of its 24-hour window in the next
split. Since 2026-11-09 the join also requires `like_ts < label_end_ts`, the end
of the listen's own split: fit rows stop at the start of the validation window,
validation rows at the end of the train window, and test rows at the end of the
data. A training or validation label therefore never depends on validation, gap
or test events (the same principle as the validation window itself). The cost is
that listens in the last 24 hours of a split have a shorter window, so their
like and dislike rates are slightly lower.

```python
fit_lf, val_lf, test_lf = load_multitask_splits(data_dir, train_days, test_days,
                                                val_split_ratio=0.1)
```

### Corrected Multi-Task MMoE Results

After fixing the temporal leakage, here are the 30-day metrics (`mmoe30`, focal loss, 4 experts). F1 is `f1_at_threshold` at the validation threshold applied to the test day, not `f1_max`.

| Task | Test AUC-ROC | Test AUC-PR | F1 at validation threshold | Threshold | Positive rate |
|------|-------------|-------------|----------------------------|-----------|---------------|
| **Engagement** | 0.7681 | 0.9778 | 0.9435 | 0.4372 | 93.41% |
| **Completion** | 0.7229 | 0.8547 | 0.7950 | 0.3459 | 67.96% |
| **Like** | 0.7012 | 0.0174 | 0.0459 | 0.1618 | 0.74% |
| **Dislike** | 0.6933 | 0.0153 | 0.0150 | 0.0999 | 0.15% |

Validation AUC-ROC on the same run: engagement 0.8094, completion 0.7766, like 0.7386, dislike 0.7007.

### Interpreting Metrics for Imbalanced Tasks

#### AUC-PR for Like Task

Test AUC-PR is 0.0174 against a test positive rate of 0.0074 (about 2.3×). Validation AUC-PR is 0.0380 against a validation positive rate of 0.0103 (about 3.7×).

| Split | Model AUC-PR | Positive rate | Ratio |
|-------|--------------|---------------|-------|
| Test | 0.0174 | 0.0074 | about 2.3× |
| Validation | 0.0380 | 0.0103 | about 3.7× |

#### Thresholds Are Critical

For highly imbalanced tasks, the default threshold of 0.5 is wrong. The thresholds below were chosen on validation and applied to the test day.

| Task | Default Threshold | Validation threshold | Why It Matters |
|------|-------------------|----------------------|----------------|
| Like | 0.5 (50%) | 0.1618 (16.2%) | A threshold of 0.5 would miss the rare likes |
| Dislike | 0.5 (50%) | 0.0999 (10.0%) | A threshold of 0.5 would miss the rare dislikes |

Choose the threshold on the validation set and apply it unchanged to the test set. A threshold tuned on the test set makes the reported test F1 optimistic.

### Key Takeaways for the Book

1. **Always check for temporal leakage** when joining event data for labels
2. **Unrealistic positive rates are a red flag** (25% like rate vs. expected 1-5%)
3. **Use time-windowed joins** for causal correctness
4. **AUC-PR is more informative than AUC-ROC** for imbalanced tasks
5. **Optimal thresholds** should be computed, not assumed to be 0.5
6. **Multi-task learning with hard auxiliary tasks** may require capacity tuning

### Window Size Considerations

The 24-hour window is a design choice with trade-offs:

| Window Size | Pros | Cons |
|-------------|------|------|
| **Shorter (1-6 hours)** | Stronger causal signal | Fewer positive labels |
| **24 hours (chosen)** | Balanced coverage | May include delayed reactions |
| **Longer (7+ days)** | More positive labels | Weaker causal relationship |

For music streaming, 24 hours captures most immediate reactions while maintaining reasonable causality.

Before 2026-11-09 the code set this window to 17,280 timestamp units on the assumption that one unit was 5 seconds. Yambda timestamps are seconds, so the window was 4.8 hours. The code now uses 86,400 seconds. The corrected rates in the table above are from the 30-day re-measurement after that change.

---

## 15. Appendix: Quick Reference Commands

### DCN-V2

```bash
# Run with learned item embeddings (train AUC climbs after the best epoch; early stopping matters)
python train_dcn.py --train_days 30

# Run with pre-trained item embeddings (recommended)
python train_dcn_pretrained.py --train_days 30

# Run with custom hyperparameters
python train_dcn_pretrained.py \
    --train_days 30 \
    --cross_layers 4 \
    --cross_rank 64 \
    --mlp_dims 512,256,128 \
    --dropout 0.2
```

### MMoE

```bash
# Single-task (completion only) - for architecture comparison with DCN-V2
python train_mmoe_single.py --train_days 30

# Multi-task (5 tasks: 4 binary + 1 regression)
python train_mmoe.py --train_days 30

# Multi-task with custom expert count (homework)
python train_mmoe.py --train_days 30 --num_experts 6
```

---

## 15. Mixed Task Types: Binary Classification + Regression

### The Challenge

Most multi-task learning tutorials focus on tasks of the same type (e.g., all binary classification). Real-world recommendation systems often need **mixed task types**:

| Task | Type | Target | Loss Function |
|------|------|--------|---------------|
| Engagement | Binary | User started listening (yes/no) | BCE |
| Completion | Binary | User listened ≥50% (yes/no) | BCE |
| **Play Ratio** | **Regression** | Percentage listened [0, 1] | **MSE** |
| Like | Binary | User liked (yes/no) | Focal Loss |
| Dislike | Binary | User disliked (yes/no) | Focal Loss |

### Why Add a Regression Task?

**Limitation of Binary Completion:**
- Binary threshold (50%) loses granular information
- Can't distinguish between 10% vs 40% listening
- Can't distinguish between 55% vs 95% listening

**Benefits of Regression Prediction:**
1. **More granular signal**: Predicts actual listening percentage
2. **Better value functions**: `E[listen_time] = P(engage) × pred_ratio × track_length`
3. **Richer training signal**: More gradient information than binary cross-entropy
4. **Direct business metric**: "Expected minutes listened" is tangible

### Implementation Details

**Loss Function:**
```python
# In MultiTaskLoss
if name in self.regression_tasks:
    # Regression: apply sigmoid, use MSE
    pred = torch.sigmoid(pred)
    loss = nn.MSELoss()(pred, target)
else:
    # Binary: use BCE (sigmoid applied internally)
    loss = nn.BCEWithLogitsLoss()(pred, target)
```

**Evaluation Metrics:**
- Binary tasks: AUC-ROC, AUC-PR, F1-max
- Regression task: MSE, MAE, R²

**Value Function with Regression:**
```python
# Old: Binary-only approach
value = P(engage) * P(complete) * track_length

# New: With regression prediction (more accurate)
value = P(engage) * pred_play_ratio * track_length
```

### Why This Matters for the Book

This demonstrates a **practical pattern** rarely covered in tutorials:
- How to handle mixed task types in a single model
- How to combine different loss functions
- How regression heads provide more nuanced predictions for value functions

---

## 16. Narrative Arc: Chapter 7 → Chapter 8 (Value Functions)

### The Story So Far

By the end of Chapter 7, readers have:

1. **Built a multi-task ranking model** (MMoE) that predicts 5 signals:
   - P(Engagement) — Will the user start listening?
   - P(Completion) — Will they listen past 50%?
   - E[Play Ratio] — How much will they listen? (regression)
   - P(Like) — Will they explicitly like it?
   - P(Dislike) — Will they explicitly dislike it?

2. **Understood the trade-offs** between architectures (DCN-V2 vs DLRM vs MMoE)

3. **Learned to handle practical challenges**:
   - High-cardinality features (pre-trained embeddings)
   - Temporal leakage (time-windowed joins)
   - Class imbalance (focal loss, AUC-PR, optimal thresholds)

### The Bridge to Chapter 8

**The key question Chapter 8 answers:**

> "We now have 5 predictions per (user, item) pair. How do we combine them into a single score that reflects business value?"

### Suggested Chapter 8 Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHAPTER 8: VALUE FUNCTIONS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  8.1 The Problem: Multiple Signals, One Decision                            │
│  ├── We have 5 predictions, but we can only rank items in one order        │
│  ├── Different stakeholders care about different metrics                    │
│  └── How do we balance short-term engagement vs. long-term satisfaction?   │
│                                                                             │
│  8.2 Linear Value Functions: The Simple Approach                            │
│  ├── value = w₁·P(engage) + w₂·P(complete) + w₃·E[ratio] + w₄·P(like)      │
│  │            - w₅·P(dislike)                                               │
│  ├── Load predictions.parquet from Chapter 7                                │
│  ├── Experiment with different weight combinations                          │
│  └── Analyze: How do rankings change with different weights?                │
│                                                                             │
│  8.3 Business-Driven Weight Selection                                       │
│  ├── "Engagement-focused": Maximize listening time                          │
│  │   └── value = P(engage) × E[ratio] × track_length                       │
│  ├── "Satisfaction-focused": Maximize long-term retention                   │
│  │   └── value = P(like) - 2×P(dislike) + 0.5×P(complete)                  │
│  ├── "Balanced": Multi-objective optimization                               │
│  └── How to align weights with business KPIs                                │
│                                                                             │
│  8.4 Calibration: Can We Trust the Probabilities?                           │
│  ├── Load calibration.json from Chapter 7                                   │
│  ├── Plot reliability diagrams for each task                                │
│  ├── Understanding over/under-confidence                                    │
│  └── Calibration methods: Platt scaling, isotonic regression               │
│                                                                             │
│  8.5 The Regression Advantage: Expected Listen Time                         │
│  ├── Why E[ratio] is more useful than P(complete)                          │
│  ├── expected_listen_time = P(engage) × E[ratio] × track_length            │
│  ├── Compare rankings: completion-based vs. regression-based               │
│  └── Business impact: "Minutes listened" as a direct metric                │
│                                                                             │
│  8.6 Expert Gating Analysis: What Did the Model Learn?                      │
│  ├── Load gate_weights_sample.parquet from Chapter 7                        │
│  ├── Visualize expert specialization per task                               │
│  ├── Do like/dislike tasks share experts? (Correlation analysis)           │
│  └── Interpretability: Which experts handle which patterns?                │
│                                                                             │
│  8.7 Counterfactual Analysis: "What If?"                                    │
│  ├── What if we only optimized for engagement?                              │
│  │   └── Simulate: Would like/dislike rates suffer?                        │
│  ├── What if we heavily penalized dislikes?                                │
│  │   └── Simulate: Does diversity increase?                                │
│  └── Trade-off frontiers: Pareto curves for multi-objective ranking        │
│                                                                             │
│  8.8 From Offline to Online: A/B Testing Considerations                     │
│  ├── Offline metrics vs. online business impact                             │
│  ├── How to design experiments for value function changes                   │
│  └── Guardrail metrics: Ensuring we don't harm user experience             │
│                                                                             │
│  8.9 Advanced: Learning Weights from Data                                   │
│  ├── Inverse reinforcement learning (brief intro)                           │
│  ├── Contextual bandits for weight optimization                             │
│  └── When to use learned vs. hand-tuned weights                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Artifacts from Chapter 7 → Chapter 8

| Artifact | Location | Use in Chapter 8 |
|----------|----------|------------------|
| `val_predictions.parquet` | `outputs/mmoe_*/` | Value function experiments and weight tuning (validation split) |
| `predictions.parquet` | `outputs/mmoe_*/` | Final reporting of the chosen weights only (test split) |
| `calibration.json` | `outputs/mmoe_*/` | Reliability analysis (validation split) |
| `gate_weights_sample.parquet` | `outputs/mmoe_*/` | Expert interpretability (validation split) |
| `value_function_config.json` | `outputs/mmoe_*/` | Starting weight suggestions (validation split) |
| `best_model.pt` | `outputs/mmoe_*/` | Full inference pipeline |
| `inference/` | `outputs/mmoe_*/inference/` | Loading model for new data |

Since 2026-11-09, `calibration.json`, `gate_weights_sample.parquet` and the statistics and suggested weights in `value_function_config.json` are computed from the validation split, and weight experiments (8.2 above) should read `val_predictions.parquet`. `predictions.parquet` holds test predictions for final reporting only.

### Narrative Hooks

**Opening for Chapter 8:**

> "In Chapter 7, we built a multi-task model that predicts five different aspects of user-item interactions: engagement, completion, listening depth, likes, and dislikes. But here's the challenge: a recommendation system can only show items in *one* order. How do we combine these five signals into a single ranking that balances immediate engagement with long-term user satisfaction? This is the problem of **value functions**."

**The Regression Task Payoff:**

> "Remember the `play_ratio` regression head we added in Chapter 7? Here's where it pays off. Instead of the crude approximation `expected_time = P(complete) × 0.75 × track_length`, we can compute the precise `expected_time = P(engage) × E[ratio] × track_length`. The regression prediction lets us optimize directly for minutes listened, not just a binary completion threshold."

**The Calibration Connection:**

> "Before we trust our probability predictions in a value function, we need to verify they're *calibrated*. A prediction of P(like) = 0.1 should mean that roughly 10% of items with that prediction actually get liked. The calibration data we exported from Chapter 7 lets us diagnose and fix miscalibrated predictions."

### Key Insights to Reinforce

1. **Multi-task predictions enable nuanced ranking** — You can't get this from single-task models
2. **Regression > Binary for quantitative objectives** — Play ratio gives expected values, not just thresholds
3. **Calibration matters for value functions** — Uncalibrated probabilities lead to suboptimal rankings
4. **Weights encode business priorities** — There's no "correct" value function, only trade-offs
5. **Expert gating reveals model structure** — Understanding specialization helps debugging and trust

### Homework Bridge

At the end of Chapter 7, suggest:

> "Before moving to Chapter 8, explore the `value_function_config.json` file in your model output directory. Try computing value scores with different weight schemes and see how the top-10 recommendations change. This will give you intuition for how value functions reshape rankings."

---

## 17. Calibration Analysis and Model Improvements

> **Note (2026-11-13):** Calibration ratios below are the 30-day focal-loss run (`mmoe30`), on the validation window unless a row says test. The expert-count table is focal loss off, and the decision split is validation.

### Observed Calibration Ratios (5-Task MMoE, 4 Experts, 30 Days)

Ratios for the binary tasks are the validation F1 threshold divided by the validation positive rate. The play-ratio ratio is mean prediction divided by mean label.

| Task | Validation rate or mean | Threshold or mean prediction | Ratio | Interpretation |
|------|-------------------------|------------------------------|-------|----------------|
| **Engagement** | 93.36% | 0.4372 | 0.47× | Threshold below the base rate |
| **Completion** | 65.91% | 0.3459 | 0.52× | Threshold below the base rate |
| **Play Ratio** | 0.672 (mean label) | 0.672 (mean pred) | 1.00× | Close to calibrated on validation. R² 0.224 |
| **Like** | 1.03% | 0.1618 | 15.7× | Over-predicting |
| **Dislike** | 0.18% | 0.0999 | 56× | Over-predicting |

On the test day the play-ratio head under-predicts (mean prediction 0.632 against mean label 0.691, ratio 0.91×) and R² is −0.086.

### Key Finding

On the validation window the play-ratio head is close to calibrated: mean prediction 0.672 against mean label 0.672, R² 0.224. On the test day it under-predicts (0.632 against 0.691) and R² is −0.086, worse than predicting the mean. Like and dislike probabilities are still far above their base rates. Do not describe the play-ratio head as well calibrated without saying that this is the validation window, not the test day.

### Root Cause Analysis

| Factor | Impact |
|--------|--------|
| **Extreme class imbalance** | Like: about 1 in 96 training listens (1.04%). Dislike: about 0.20% (about 1 in 500) |
| **Expert count** | 4, 6, and 8 experts tie on validation completion. The default stays at 4 because it is the smaller model |
| **Multi-task negative transfer** | High-volume tasks can dominate shared experts |
| **Feature insufficiency** | No explicit feedback history features |

### Expert Count Experiments

Focal loss off, 30-day window. The decision split is validation.

| Experts | Validation completion AUC-ROC | Like AUC-ROC | Like AUC-PR | Dislike AUC-PR |
|---------|-------------------------------|--------------|-------------|----------------|
| 4 | 0.7763 | 0.8067 | 0.0650 | 0.0381 |
| 6 | 0.7760 | 0.8071 | 0.0747 | 0.0969 |
| 8 | 0.7765 | 0.7905 | 0.0661 | 0.0533 |

Validation completion AUC-ROC is 0.7763, 0.7760, and 0.7765 for 4, 6, and 8 experts. That is a tie on one seed. Like AUC-ROC is 0.8067 and 0.8071 for 4 and 6 experts, and 0.7905 for 8. Four experts did not win. Eight experts are worse on likes. The default stays at 4 because it is the smaller model, not because a comparison crowned it. Revisit the default if a later multi-seed run separates these.

Test completion AUC-ROC, not used for the decision: 0.7009 (4 experts), 0.7059 (6 experts), 0.7282 (8 experts). Eight experts look better on that one test day. Do not move the default because of the test number.

### Ideas for Further Improvement (Not Implemented)

For readers who want to improve like/dislike prediction further:

| Approach | Description |
|----------|-------------|
| **Focal Loss** | Use `--use_focal_loss` flag (already available) |
| **Oversampling** | Repeat positive examples 10-50x during training |
| **Additional features** | Add user's historical like rate, item's like rate |
| **Pairwise Learning-to-Rank** | BPR loss (covered in dedicated LTR chapters) |
| **Calibration correction** | Post-hoc Platt scaling or temperature scaling |

### Key Takeaway for the Book

> "Extreme class imbalance (about a 1% like rate on the training window, 0.74% on the test day) makes probability calibration very challenging. For value functions in Chapter 8, consider using calibration factors to adjust the effective weights, or focus on rank-based blending rather than probability-based blending."

---

## 18. Homework Exercises

### Exercise 1: Conditional Like/Dislike Prediction

The current model predicts P(like) unconditionally for all listen events. An alternative is to model P(like | engaged), which would:
- Train only on samples where `label_engagement = 1`
- Result in a higher positive rate (easier to learn)
- Be more interpretable for value functions: `value = P(engage) × P(like | engage)`

**Task**: Modify `train_mmoe.py` to optionally filter training data to only engaged listens for like/dislike tasks. Compare the results.

### Exercise 2: Cyclical Time Encoding

Replace the categorical encoding of `hour_of_day` and `day_of_week` with cyclical (sin/cos) encodings:

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

This captures that hour 23 is close to hour 0. Compare embedding-based vs. cyclical encoding.

### Exercise 3: Expert Specialization Analysis

Use the exported `gate_weights_sample.parquet` to analyze which experts specialize in which tasks:
- Plot average gate weights per task
- Identify if certain experts dominate specific tasks
- Correlate expert usage with prediction accuracy

### Exercise 4: Uncertainty Weighting for Multi-Task Learning

The current implementation uses **fixed task weights** (e.g., `like_weight=2.0`). An alternative is **uncertainty weighting** (Kendall et al., 2018), which learns task-specific uncertainties automatically.

**How it works:**

Instead of:
```python
total_loss = Σ (weight_i × loss_i)
```

Uncertainty weighting uses:
```python
total_loss = Σ (loss_i / (2 × σ_i²) + log(σ_i))
```

Where `σ_i` is a **learnable parameter** per task. The model automatically:
- Down-weights high-variance (noisy) tasks
- Up-weights tasks it's confident about

**Potential advantages:**
1. **No manual weight tuning** — the model learns optimal task balancing
2. **Adapts during training** — weights adjust as tasks converge at different rates
3. **Handles scale differences** — automatically normalizes losses of different magnitudes

**Potential disadvantages:**
1. **More parameters** — adds one learnable `log(σ²)` per task
2. **May overfit** — on small datasets, learned weights might not generalize
3. **Less interpretable** — harder to reason about why certain tasks dominate

**Implementation:**

The `MultiTaskLoss` class already supports this! To enable it:

```python
# 1. Add CLI argument in train_mmoe.py
parser.add_argument('--uncertainty_weighting', action='store_true',
                   help='Use learnable uncertainty weighting (Kendall et al.)')

# 2. Pass to MultiTaskLoss
criterion = MultiTaskLoss(
    task_names=TASK_NAMES,
    task_weights={...},
    use_uncertainty_weighting=args.uncertainty_weighting,  # Add this line
    ...
)
```

**Task**: 
1. Add the `--uncertainty_weighting` flag to `train_mmoe.py`
2. Run experiments comparing fixed weights vs. uncertainty weighting
3. Analyze the learned `σ` values — which tasks have highest uncertainty?

**Reference**: Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018.

---

*Last updated: February 8, 2026*

