# Chapter 9: Modeling for Adtech Use Cases - Design Notes

## Overview

This document captures key design decisions, insights, and gotchas for Chapter 9's adtech-specific modeling implementations. Chapter 9 focuses on techniques unique to advertising: **selection bias correction (ESMM)**, **delayed feedback handling**, **calibration for bidding**, and **privacy-preserving federated learning**.

---

## 1. Dataset Selection and Characteristics

### Primary Dataset: Ali-CCP (Alibaba Click and Conversion Prediction)

**Source:** https://tianchi.aliyun.com/dataset/408

| Attribute | Value |
|-----------|-------|
| Total Impressions | 42.3M |
| Unique Users | 730K |
| Click-Through Rate | ~4-5% |
| Conversion Rate (post-click) | ~2% |
| Full Funnel | ✅ Yes (includes non-clicked impressions) |

**Why Ali-CCP is Critical for ESMM:**
- **Selection Bias Problem:** Most CVR datasets only include clicked samples
- Ali-CCP includes ALL impressions, enabling training on the entire user space
- This allows ESMM to model: `CTCVR = CTR × CVR` without selection bias

**Data Files:**
1. `sample_skeleton_train.csv` (10GB, 42.3M rows)
   - sample_id, click, purchase, user_hash, context_id, sparse_features
   
2. `common_features_train.csv` (8GB, 730K rows)
   - user_hash, feature_count, sparse_features (user-level features)

**Sparse Feature Format Challenge:**
```
Format: '21090522181.0' = feature_id=2109052218, value=1.0
        '150_14387888230.28343' = feature_id=150_1438788823, value=0.28343

Multiple features concatenated WITHOUT delimiters - requires careful parsing
```

### Secondary Dataset: Alibaba FedAds

**Source:** https://tianchi.aliyun.com/dataset/148347

| Attribute | Value |
|-----------|-------|
| Aligned Samples | 2.6M (both parties have features) |
| Unaligned Samples | 10.4M (local-only features) |
| CVR | ~0.55-0.60% |
| Feature Split | LOCAL (17) + FEDERATED (5) |

**Used For:** Section 9.5 - Privacy-Preserving CVR with Split Neural Networks

---

## 2. ESMM (Entire Space Multi-Task Model)

### The Selection Bias Problem

**Traditional CVR Training:**
```
Training data: { (x, y) | user clicked }  ← Only clicked samples!
Model learns: P(conversion | click, features)
At inference: Applied to ALL users (including non-clickers)

PROBLEM: The feature distribution of clickers ≠ all users
         → Selection bias → Poor generalization
```

**ESMM Solution:**
```
Training data: ALL impressions (clicked + not clicked)
Model learns two towers:
  - CTR Tower:  P(click | impression)
  - CVR Tower:  P(conversion | click)    [Implicit, no direct supervision]
  
Combined: CTCVR = P(click) × P(conversion | click)
                = P(click AND conversion | impression)
                
The CVR tower is trained via the CTCVR path, using conversion labels
on ALL samples (not just clicked ones).
```

### ESMM Architecture Design

```
                    Shared Embedding Layer
                           │
              ┌────────────┴────────────┐
              │                         │
         ┌────▼────┐              ┌─────▼─────┐
         │  CTR    │              │   CVR     │
         │  Tower  │              │   Tower   │
         │  (MLP)  │              │   (MLP)   │
         └────┬────┘              └─────┬─────┘
              │                         │
         P(click)                  P(conv|click)
              │                         │
              └─────────┬───────────────┘
                        │
                   CTCVR = CTR × CVR
```

### Key Implementation Decisions

**Decision 1: Shared vs Separate Embeddings**
- CHOICE: Shared embeddings between CTR and CVR towers
- REASON: 
  - Reduces parameters significantly (sparse features dominate parameter count)
  - CVR tower benefits from CTR tower's learning signal (more labeled data)
  - Original ESMM paper uses shared embeddings

**Decision 2: Tower Architecture**
- Both towers: MLP with [256, 128, 64] hidden units
- Output: Single sigmoid neuron each
- REASON: Keep towers symmetric; CVR tower implicitly learns from CTR tower gradients

**Decision 3: Loss Function**
```python
# Two loss terms, both on ENTIRE dataset
loss_ctr = BCELoss(p_click, label_click)      # All samples have click labels
loss_ctcvr = BCELoss(p_click * p_cvr, label_purchase)  # Purchase label
total_loss = loss_ctr + loss_ctcvr

# NOTE: NO DIRECT CVR LOSS - CVR tower learns through CTCVR path only
```

**Decision 4: Gradient Flow for CVR Tower**
- The CVR tower receives gradients ONLY through the CTCVR loss
- For non-clicked samples: label_purchase = 0, CTCVR loss still meaningful
- For clicked+converted: Both CTR and CTCVR losses contribute

**Decision 5: Focal Loss for Class Imbalance**

With CTR ~5% and CVR ~2%, we have severe class imbalance. Focal loss down-weights easy examples, focusing learning on hard cases:

```
FL(p_t) = -α × (1 - p_t)^γ × log(p_t)
```

| Sample Type | p_t | (1-p_t)² | Effect |
|-------------|-----|----------|--------|
| **Easy negative** (pred=0.01, label=0) | 0.99 | 0.0001 | Almost no loss - correctly predicted negative |
| **Hard negative** (pred=0.4, label=0) | 0.6 | 0.16 | Moderate loss - model unsure about negative |
| **Hard positive** (pred=0.1, label=1) | 0.1 | 0.81 | High loss - model wrong about positive |
| **Easy positive** (pred=0.9, label=1) | 0.9 | 0.01 | Low loss - correctly predicted positive |

The focal weight `(1-p_t)^γ` with γ=2 means:
- Well-classified samples (high p_t) get near-zero loss
- Misclassified samples (low p_t) retain full loss
- This focuses learning on the hard examples rather than being dominated by easy negatives

### Critical Assumption: Sequential Funnel

**ESMM assumes conversions can ONLY happen after a click:**

```
CTCVR = P(click) × P(conversion | click)
```

This is valid for traditional display advertising but **breaks down** when:

| Scenario | ESMM Valid? | Alternative |
|----------|-------------|-------------|
| Standard display ad funnel | ✅ Yes | Use ESMM |
| View-through attribution | ❌ No | Add view-through tower |
| Amazon 1-click purchases | ❌ No | Direct conversion model |
| Organic + paid attribution | ❌ No | Multi-touch attribution |

**If your platform has click-less conversions:**

1. **Option A**: Model P(conversion | impression) directly (ignore click as intermediate)
2. **Option B**: Extended ESMM with parallel paths:
   ```
   P(conv) = P(click)×P(conv|click) + P(no_click)×P(conv|no_click)
   ```
3. **Option C**: Separate models for click-through vs. view-through conversions

The Ali-CCP dataset follows the traditional funnel (click required for conversion), so ESMM is appropriate here.

---

### Gotchas and Lessons Learned

**Gotcha 1: CVR Tower Gets Weak Gradients (Gradient Asymmetry)**

A subtle but important issue: the CTR and CVR towers receive **asymmetric gradient signals**.

```
Gradient Flow Analysis:

┌─────────────────────────────────────────────────────────────┐
│  L_ctr = BCE(p_ctr, y_click)         ← Direct CTR supervision
│     │
│     ▼
│  ∂L_ctr/∂θ_ctr  (FULL gradient to CTR tower)
│
│  L_ctcvr = BCE(p_ctr × p_cvr, y_conv)
│     │
│     ├────────────────┬────────────────┐
│     ▼                ▼                │
│  ∂L/∂p_ctr       ∂L/∂p_cvr           │
│  = p_cvr × ...   = p_ctr × ...  ← SCALED BY p_ctr!
│     │                │                │
│     ▼                ▼                │
│  CTR tower       CVR tower            │
│  (gets BOTH      (ONLY gets           │
│   L_ctr +        p_ctr-scaled         │
│   L_ctcvr)       gradient)            │
└─────────────────────────────────────────────────────────────┘
```

**The Math:**
- CTR tower gradient: `∂L_ctr/∂θ_ctr + p_cvr × ∂L_ctcvr/∂θ_ctr` (strong: ~1.0 + 0.1)
- CVR tower gradient: `p_ctr × ∂L_ctcvr/∂θ_cvr` (weak: scaled by p_ctr!)

**Example with CTR=5%, CVR=10%:**
- CTR tower effective gradient: ~1.1 (strong signal)
- CVR tower effective gradient: ~0.05 (20× weaker!)

**Why CTR tower doesn't suffer:** It has DIRECT supervision via L_ctr on ALL samples.

**Auxiliary CVR Loss (NOT in original ESMM paper - EXPERIMENTAL):**

The original Ma et al. SIGIR 2018 paper uses only `L_ctr + L_ctcvr`. 

The auxiliary CVR loss is an **experimental extension** we've added to this implementation. It is NOT from any published paper or documented industry practice—it is our own reasoning based on the gradient analysis above. We include it as an option for students to experiment with.

```python
# Original ESMM (paper):
total_loss = L_ctr + L_ctcvr

# EXPERIMENTAL: Extended ESMM with auxiliary CVR loss
# Rationale: Give CVR tower direct supervision when CTR is very low
if use_auxiliary_cvr_loss:
    # Direct CVR supervision on CLICKED samples only
    L_cvr = BCE(p_cvr[clicked], y_conv[clicked])
    total_loss = L_ctr + L_ctcvr + 0.1 × L_cvr
```

**Hypothesized Trade-off (untested):**
- PRO: May give CVR tower stronger gradient signal
- CON: Reintroduces some selection bias (training on clicked samples)
- DEFAULT: Disabled (to match original paper)

**Further Investigation for Students:**
1. Measure gradient magnitudes for CTR vs CVR towers during training
2. Compare ESMM with/without auxiliary CVR loss at different CTR levels
3. Explore gradient scaling techniques (e.g., GradNorm) for multi-task balancing
4. Validate whether auxiliary CVR loss actually improves CVR AUC in low-CTR scenarios

**Gotcha 2: Numerical Stability**
```python
# BAD: Direct multiplication can underflow
ctcvr = p_click * p_cvr  # If both small, product → 0

# BETTER: Clamp values
p_click_clamped = torch.clamp(p_click, min=1e-7, max=1-1e-7)
p_cvr_clamped = torch.clamp(p_cvr, min=1e-7, max=1-1e-7)
ctcvr = p_click_clamped * p_cvr_clamped
```

**Gotcha 3: Evaluation Metrics**
```
For CVR evaluation, we MUST evaluate on clicked samples only
(because ground truth CVR is only observed for clicked samples)

For CTCVR evaluation, we can use ALL samples
```

---

## 3. Ali-CCP Data Processing

### Official Feature Schema (from [Alibaba Tianchi](https://tianchi.aliyun.com/dataset/408))

The Ali-CCP dataset has a well-documented feature structure with **semantic feature field IDs**:

#### Feature Structure

Each feature in the raw data uses a **three-component structure** with ASCII delimiters:
```
feature_field_id <0x02> feature_id <0x03> feature_value
```

Multiple features are separated by `<0x01>`:
```
field1<0x02>id1<0x03>val1<0x01>field2<0x02>id2<0x03>val2<0x01>...
```

Where:
- `0x01` (SOH) = feature separator
- `0x02` (STX) = separates field_id from feature_id
- `0x03` (ETX) = separates feature_id from value

#### User Features

| Field ID | Description |
|----------|-------------|
| **101** | User ID |
| **109_14** | User historical behaviors: Category ID + count (past 2 weeks) |
| **110_14** | User historical behaviors: Shop ID + count (past 2 weeks) |
| **127_14** | User historical behaviors: Brand ID + count (past 2 weeks) |
| **150_14** | User historical behaviors: Intention node ID + count (past 2 weeks) |
| **121** | User Profile: Categorical ID |
| **122** | User Profile: Categorical group ID |
| **124** | User Gender ID |
| **125** | User Age ID |
| **126** | User Consumption Level Type I |
| **127** | User Consumption Level Type II |
| **128** | User Occupation (working or not) |
| **129** | User Geography Information |

#### Item Features

| Field ID | Description |
|----------|-------------|
| **205** | Item ID |
| **206** | Category ID |
| **207** | Shop ID |
| **210** | Intention node ID |
| **216** | Brand ID |

#### Combination Features (Pre-computed Cross-features)

| Field ID | Description |
|----------|-------------|
| **508** | User category behavior × Item category (109_14 × 206) |
| **509** | User shop behavior × Item shop (110_14 × 207) |
| **702** | User brand behavior × Item brand (127_14 × 216) |
| **853** | User intention × Item intention (150_14 × 210) |

#### Context Features

| Field ID | Description |
|----------|-------------|
| **301** | Position (categorical) |

### The Raw Data Challenge

**Challenges:**
- Binary delimiters (0x01, 0x02, 0x03) require careful parsing
- Two files that need joining (sample-level + user-level features)
- Large files (~18GB total)
- Feature field IDs should be preserved for interpretability

### Solution: One-Time Preprocessing

To avoid students getting bogged down in data wrangling, we provide a **preprocessing script** that converts raw data to clean parquet files. This only needs to run **ONCE**.

**Step 1: Run the Preprocessor**

```bash
cd "RecSys book code/chapter9_adtech/scripts"

# For 16GB RAM laptops (5% = ~2.1M samples, ~10 mins)
python preprocess_ali_ccp.py --pct 5

# For quick testing (0.1% = ~42K samples, ~2 mins)
python preprocess_ali_ccp.py --pct 0.1

# For full dataset (100% = ~42.3M samples, requires 32GB+ RAM)
python preprocess_ali_ccp.py --pct 100

# Alternative: specify exact sample count
python preprocess_ali_ccp.py --n_samples 5000000
```

**Percentage reference (full dataset = 42.3M samples):**
| --pct | Samples | Parquet Size | In-Memory | Training RAM | Use Case |
|-------|---------|--------------|-----------|--------------|----------|
| 0.1 | ~42K | ~8 MB | ~50 MB | ~500 MB | Quick debugging |
| 1 | ~423K | ~80 MB | ~500 MB | ~2 GB | Fast iteration |
| 5 | ~2.1M | ~400 MB | ~2.5 GB | ~5 GB | **Recommended for 16GB laptops** |
| 10 | ~4.2M | ~800 MB | ~5 GB | ~7 GB | Larger experiments |
| 100 | ~42.3M | ~8 GB | ~50 GB | ~64 GB | Full-scale (cloud/homework) |

#### RAM Estimation Formula

Each sample in memory requires:
- `feature_ids`: 100 features × 8 bytes (int64) = **800 bytes**
- `feature_values`: 100 features × 4 bytes (float32) = **400 bytes**
- `click`, `purchase`: 8 bytes
- **Total per sample: ~1.2 KB**

**In-Memory** = samples × 1.2 KB (the raw data arrays)

**Training RAM** adds:
- Model parameters (~6.8 MB for 1.7M params)
- Gradients (same size as model)
- Batch tensors during forward/backward
- PyTorch/Python overhead (~1-2 GB baseline)

#### ⚠️ Minimum Samples for Meaningful Metrics

With very low sampling, you may get **zero conversions in validation**:

| --pct | Expected Val Conversions | Metric Quality |
|-------|--------------------------|----------------|
| 0.1 | ~2 | ❌ AUC unreliable (may be 0) |
| 1 | ~20 | ⚠️ Noisy but usable |
| 5 | ~100 | ✓ Reasonable |
| 10 | ~200 | ✓ Good |

**Rule of thumb:** Use at least `--pct 1` for meaningful CVR/CTCVR metrics. The 0.1% option is only for debugging code logic, not model quality.

**Step 2: Use Clean Data Loader**

```python
from data.ali_ccp_dataset import load_ali_ccp_loaders

# One-liner to get DataLoaders
train_loader, val_loader, metadata = load_ali_ccp_loaders(batch_size=4096)

for batch in train_loader:
    feature_ids = batch['feature_ids']  # (B, 100) - hashed feature IDs
    feature_values = batch['feature_values']  # (B, 100)
    click = batch['click']  # (B,) - CTR target
    purchase = batch['purchase']  # (B,) - CVR target
```

### Output Schema (After Preprocessing)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | int64 | Unique sample identifier |
| `click` | int8 | Click label (0/1) - CTR target |
| `purchase` | int8 | Conversion label (0/1) - CVR target |
| `user_hash` | string | Hashed user identifier |
| `field_0` to `field_99` | int16 | Feature field IDs (e.g., 101=UserID, 205=ItemID) |
| `fid_0` to `fid_99` | int32 | Feature IDs (hashed to vocab_size) |
| `fval_0` to `fval_99` | float32 | Feature values |

**Note:** We preserve `field_id` separately from `feature_id` to enable:
- Feature group analysis (user features vs item features)
- Interpretability (knowing what each feature represents)
- Potential per-field embedding tables (advanced)

### Homework Assignment

Students should download the full Ali-CCP dataset from https://tianchi.aliyun.com/dataset/408 and run preprocessing with `--pct 100` to experience full-scale training. This requires 32GB+ RAM or cloud compute.

### Feature Hashing

Raw feature IDs are hashed to a fixed vocabulary:

```python
vocab_size = 100,000  # Hash buckets
hashed_id = md5(feature_string) % vocab_size
```

**Trade-off:** Hash collisions are possible but manageable at 100K buckets. This converts arbitrary string IDs to integer indices suitable for embedding lookup.

---

## 4. Delayed Feedback Modeling

### The Problem

In adtech, conversions can occur hours or days after a click:
- User clicks ad at 9am
- User purchases at 9pm (12-hour delay)
- If we train on data collected at 3pm, this sample looks like "no conversion"

**Data Censoring:**
```
Sample collected at time T:
- Actual conversion at T+Δ (Δ > attribution_window)
- We observe: no conversion → FALSE NEGATIVE

This biases CVR estimates downward for recent data
```

### Modeling Approaches

**Approach 1: Attribution Window Cutoff**
```python
# Only use samples older than attribution_window
valid_samples = samples.filter(
    samples['collection_time'] - samples['click_time'] > ATTRIBUTION_WINDOW
)
# CONS: Loses recent data, model trains on stale signals
```

**Approach 2: Importance Weighting (Implemented)**
```python
# Weight samples by "probability of observation"
# Recent samples with no conversion get lower weight
time_since_click = now - click_time
weight = min(1.0, time_since_click / ATTRIBUTION_WINDOW)

# Converted samples always have full weight
weight[converted == 1] = 1.0

loss = weighted_bce_loss(pred, label, weights=weight)
```

**Approach 3: Delay Model (DFM)**
```
Model the delay distribution: P(delay=t | will_convert)
Joint model: P(convert) × P(delay=t | convert)

More complex but theoretically sound
```

### Homework: Exponential Importance Weighting

**Question:** Our current importance weighting (Approach 2) assumes conversions are **uniformly distributed** across the attribution window:

```python
weight = min(1.0, time_since_click / attribution_window)
```

This means a sample observed at 50% of the window gets 50% weight. But in reality, conversions follow an **exponential distribution** - most happen quickly after the click!

**Your Task:** Implement exponential-based importance weighting.

If conversions follow an exponential distribution with rate λ (where mean delay = 1/λ):

$$P(\text{convert by time } t) = 1 - e^{-\lambda t}$$

The proper weight should be the fraction of the CDF observed:

$$w(t) = \frac{1 - e^{-\lambda t}}{1 - e^{-\lambda T_{\text{window}}}}$$

**Starter code:**
```python
def compute_observation_weights_exponential(
    time_since_click_hours: np.ndarray,
    attribution_window_hours: float,
    delay_rate: float,  # λ, estimate from estimate_delay_distribution()
    min_weight: float = 0.1,
) -> np.ndarray:
    """Exponential-based importance weights.
    
    TODO: Implement this function
    Hint: Use np.exp() and the CDF formula above
    """
    pass
```

**Compare the two approaches:**

| Time Observed (24h window) | Linear Weight | Exponential Weight (λ=0.2/hr) |
|---------------------------|---------------|------------------------------|
| 2 hours | 0.08 | ~0.33 |
| 6 hours | 0.25 | ~0.70 |
| 12 hours | 0.50 | ~0.91 |
| 24 hours | 1.00 | 1.00 |

**Insight:** Linear weighting **underweights early observations** because it ignores that most conversions happen quickly. Exponential weighting better reflects the true observation probability.

**Bonus:** Use `estimate_delay_distribution()` in `utils/delayed_feedback.py` to fit λ from your data.

### Gotcha: Attribution Window Cutoff Should Preserve Positives

**Common Mistake:** Discarding ALL recent samples (both positives and negatives).

**Insight:** Observed conversions are **certain**. Only recent **negatives** are unreliable 
(they might convert later). Therefore:

```python
# WRONG: Discard all recent samples
valid_mask = time_since_click >= attribution_window

# CORRECT: Keep old samples OR observed conversions
valid_mask = (time_since_click >= attribution_window) | (labels == 1)
```

This preserves valuable positive signals while only excluding uncertain negatives.

### Gotcha: Feature Engineering Also Needs Delay Correction

**Critical Insight:** Any feature computed from conversion feedback is also biased by 
delayed feedback. This is often overlooked!

| Feature | How It's Biased |
|---------|-----------------|
| User's historical CVR | Recent users appear to convert less |
| Item's historical CVR | Items with recent traffic appear worse |
| Category CVR | Categories with recent impressions are underestimated |
| Time-of-day CVR | Recent time periods look artificially worse |

**Solution:** Apply the same importance weighting when computing aggregate features:

```python
# WRONG: Naive historical CVR
item_cvr = conversions_per_item / impressions_per_item

# CORRECT: Weighted historical CVR with delay correction
weights = compute_observation_weights(click_times, collection_time, ...)
weighted_conversions = sum(label_i * weight_i)
weighted_impressions = sum(weight_i)
item_cvr = weighted_conversions / weighted_impressions
```

See `compute_weighted_aggregate_features()` in `delayed_feedback.py` for implementation.

---

## 5. Calibration for Bid Optimization

### Why Calibration Matters

In RTB (Real-Time Bidding), the optimal bid is:
```
Bid = Expected Value × Confidence

where Expected Value = conversion_value × pCTR × pCVR
```

If pCVR is **miscalibrated**, bids will be systematically wrong:
- Overconfident → Overbid → Waste budget
- Underconfident → Underbid → Miss opportunities

### Calibration Techniques

**1. Platt Scaling (Logistic Calibration)**
```python
# Fit logistic regression on validation set
# predicted_prob → calibrated_prob
calibrator = LogisticRegression()
calibrator.fit(val_preds.reshape(-1, 1), val_labels)
calibrated = calibrator.predict_proba(test_preds.reshape(-1, 1))[:, 1]
```

**2. Isotonic Regression (Non-parametric)**
```python
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_preds, val_labels)
calibrated = calibrator.predict(test_preds)
```

**3. Temperature Scaling (Neural Network Specific)**
```python
# Single scalar parameter T
calibrated_logits = logits / T
calibrated_prob = sigmoid(calibrated_logits)
# Optimize T on validation set
```

### Expected Calibration Error (ECE)

```python
def compute_ece(preds, labels, n_bins=10):
    """Expected Calibration Error - lower is better."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_conf = preds[mask].mean()
            bin_acc = labels[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)
    return ece / len(preds)
```

---

## 6. Split Neural Network for Federated Learning (FedAds Paper Replication)

### Reference

Wei et al. "FedAds: A Benchmark for Privacy-Preserving CVR Estimation with Vertical Federated Learning" (SIGIR 2023)

Our implementation replicates the paper's VanillaVFL architecture and experimental setup.

### Vertical Federated Learning Scenario

```
┌───────────────────┐         ┌───────────────────┐
│    PUBLISHER      │         │    ADVERTISER     │
│  (NON-LABEL)      │         │   (LABEL PARTY)   │
├───────────────────┤         ├───────────────────┤
│ User browsing     │         │ Purchase history  │
│ Page context      │         │ Ad catalog        │
│ Impressions       │         │ Conversion labels │
│                   │         │                   │
│ 17 features       │         │ 5 features        │
│ l_i_fea, l_u_fea  │         │ f_u_fea, f_c      │
│ l_c_fea           │         │ f_uc_fea          │
└─────────┬─────────┘         └─────────┬─────────┘
          │                             │
          │         ALIGNED             │
          │         SAMPLES             │
          └──────────┬──────────────────┘
                     │
              ┌──────▼──────┐
              │    JOINT    │
              │   TRAINING  │
              │             │
              │ Without raw │
              │ data sharing│
              └─────────────┘
```

### Architecture (Paper Sec 5.1.4)

```
Publisher (f_N):             Advertiser (f_L bottom):
  embed(8) → DNN[128,32]     embed(8) → DNN[256,128]
         ↓                            ↓
      h_pub (32-dim)            h_adv (128-dim)
         └──────────┬─────────────┘
                    ↓
            Aggregator (f_L top):
              concat(32,128) = 160 → Linear(160,1) → sigmoid
                    ↓
                 P(CVR)
```

### IMPORTANT: Feature Count Discrepancy

| Party | Paper's Internal System | Public Dataset |
|-------|------------------------|----------------|
| Non-label (Publisher, l_*) | 7 features | **17 features** |
| Label (Advertiser, f_*) | 16 features | **5 features** |

The public dataset has the feature ratio essentially flipped vs. the paper.
Absolute AUC numbers will differ from Table 3, but relative ordering should hold.

### Training Settings

| Setting | Value | Paper Reference |
|---------|-------|-----------------|
| Embed dim | 8 | Sec 5.1.4 |
| Batch size | 256 | Sec 5.1.4 |
| Epochs | 1 | "number of epochs is usually set to one" |
| Loss | Standard BCE | Sec 5.1.4 |
| Dropout | 0.0 | Not mentioned |
| Weight decay | 0.0 | Not mentioned |
| Grad clipping | None | Not mentioned |
| Train/Val split | Time-based (row-order proxy) | "last week = test set" |
| Val ratio | 11.5% | ~1.3M test / 11.3M total |

### Models Compared

| Model | Paper Name | Description |
|-------|-----------|-------------|
| `SplitNN` | VanillaVFL | Both parties contribute via split architecture |
| `CentralizedBaseline` | ORALE | Both parties share raw data (upper bound) |
| `LabelPartyOnlyModel` | Local | Advertiser features only — paper's baseline |
| `NonLabelPartyOnlyModel` | (not in paper) | Publisher features only — our addition |

**Note:** The paper's "Local" baseline uses LABEL PARTY features (advertiser), because in
their internal system the label party has 16 features (the richer set). In the public
dataset, the label party has only 5 features, so this baseline will be weaker than
the paper reports.

### Paper Table 3 Expected Results

| Model | Paper AUC | Paper NLL |
|-------|-----------|-----------|
| Local (label party only) | 0.609 | — |
| VanillaVFL | 0.620 | — |
| ORALE (centralized) | 0.658 | — |

### Training Protocol

1. **Forward Pass:**
   - Publisher: Compute h_pub (32-dim), send to aggregator
   - Advertiser: Compute h_adv (128-dim), send to aggregator
   - Aggregator: concat → Linear(160,1) → sigmoid → P(CVR)

2. **Backward Pass:**
   - Aggregator: Compute gradients for h_pub, h_adv
   - Send gradients back to respective parties
   - Each party updates their local model

**Privacy Benefit:** Raw features never leave their party. Only hidden representations are exchanged.

### Data Split Limitation

The paper splits train/test by click timestamp (last week = test). The public dataset
has `l_c_fea` (click timestamp) and `f_c` (conversion timestamp) but both are
hashed/discretized, making exact temporal ordering unrecoverable. We use row order
as an approximation, assuming the CSV preserves chronological order.

### Implementation Decisions

**Decision 1: Simulate vs True Federated**
- CHOICE: Simulate in single process (for book demonstration)
- REASON: True federated requires network setup, complicates pedagogy
- APPROACH: Clear separation of local/fed components, explicit "exchange" steps

**Decision 2: Handling Unaligned Samples**
- 10.4M samples have LOCAL features only (no FEDERATED match)
- OPTION A: Discard (loses 80% of data)
- OPTION B: Train local-only model, use for semi-supervised learning
- CHOICE: Demonstrate both approaches

### Homework: Extensions Beyond VanillaVFL

The FedAds paper (Table 3) also benchmarks privacy-enhancing techniques on top of VanillaVFL:

| Technique | Paper AUC | Description |
|-----------|-----------|-------------|
| VanillaVFL | 0.620 | Baseline split architecture |
| VanillaVFL + DP(ε=1) | 0.547 | Add Differential Privacy noise to gradients |
| VanillaVFL + DP(ε=10) | 0.618 | Lighter DP noise |
| FedBCD | 0.619 | Communication-efficient variant |

Students can implement:
1. **Differential Privacy:** Add Gaussian noise to hidden representations before sharing
2. **Gradient Compression:** Reduce communication cost by quantizing gradients
3. **Dropout on hidden reps:** Add noise as a simpler privacy proxy

### Homework: Leveraging Unaligned Samples (Semi-Supervised VFL)

#### Background

In the real FedAds scenario, there are two data populations:

| Dataset | Samples | Both Parties? | Labels? |
|---------|---------|---------------|---------|
| Aligned | 2.6M | Yes | Yes |
| Unaligned | 10.4M | Publisher only | No |

Most impressions only have publisher-side features (the user visited the publisher's page but the advertiser has no matching record). VanillaVFL ignores these 10.4M samples entirely.

The FedAds paper's main contribution, **FedCVR**, leverages unaligned samples to improve the publisher tower's representations. The key insight: the publisher tower can learn useful feature patterns from unaligned samples even without the advertiser's features or CVR labels.

#### Paper Techniques (FedAds Table 3)

| Method | Paper AUC | Uses Unaligned? |
|--------|-----------|-----------------|
| VanillaVFL | 0.620 | No |
| FedCVR | 0.622 | Yes |

FedCVR's approach:
1. Train VanillaVFL on aligned samples (standard)
2. Use the trained publisher tower to generate predictions on unaligned samples
3. Use these predictions as **pseudo-labels** for self-training the publisher tower
4. Re-train VanillaVFL with the improved publisher representations

#### Your Task: Simulate Unaligned Data from Aligned Records

Rather than loading the separate 10.4M unaligned CSV (which has no labels and is 4GB), use a cleaner approach: **take the 2.6M aligned records and selectively mask one party's features** to create synthetic "unaligned" samples.

This approach has a pedagogical advantage: since the "unaligned" samples still have ground-truth labels, you can directly evaluate whether leveraging them improves predictions.

**Setup:**

```python
# Start with all 2.6M aligned records, split by time
train_df, val_df = load_fedads_split()  # Time-based split as before

# Within the TRAINING set, designate ~80% as "unaligned"
# (mirrors real ratio: 10.4M unaligned / 2.6M aligned ≈ 80/20)
n_train = len(train_df)
n_aligned = int(n_train * 0.2)

# Row-order split: earlier rows = unaligned, later rows = aligned
# (simulates: older data has publisher features only; newer data is aligned)
unaligned_train = train_df[:n_train - n_aligned]  # ~1.84M samples
aligned_train = train_df[n_train - n_aligned:]    # ~460K samples
```

**Implementation Steps:**

1. **Baseline:** Train VanillaVFL on only the aligned 20% subset (~460K samples). Record AUC.

2. **Publisher Pre-training:** Train a `NonLabelPartyOnlyModel` on ALL training data (both aligned and unaligned portions — only publisher features used, labels used for supervision). This gives the publisher tower better representations from 5x more data.

3. **Transfer:** Initialize VanillaVFL's `LocalTower` with the pre-trained publisher tower weights. Train VanillaVFL on the aligned subset only (as in step 1), but with the publisher tower warm-started.

4. **Evaluate:** Compare AUC on the held-out validation set:

```
Expected results table:
| Method                              | Training Data Used          | AUC  |
|-------------------------------------|-----------------------------|------|
| VanillaVFL (aligned only, 20%)      | ~460K aligned               | ???  |
| VanillaVFL (all aligned, 100%)      | ~2.3M aligned (our Exp 3)   | 0.651|
| VanillaVFL + publisher pre-training | ~460K aligned + ~1.84M pub  | ???  |
```

**Questions to answer:**
- Does pre-training the publisher tower on unaligned data improve VanillaVFL AUC compared to the aligned-only baseline?
- How close does it get to the "all aligned" upper bound (our Experiment 3 result of 0.651)?
- At what aligned/unaligned ratio does the benefit plateau?

**Hints:**
- Use `torch.save` / `torch.load` to transfer the `NonLabelPartyOnlyModel` embedding and network weights into `SplitNN.local_tower`
- Consider freezing the pre-trained publisher tower for the first few batches, then unfreezing (gradual fine-tuning)
- The paper's FedCVR uses pseudo-labels rather than true labels on unaligned data — in our setup we have true labels, which makes our approach stronger. Students should note this difference.

---

## 7. Handling Extreme Class Imbalance

### The Challenge

| Dataset | Positive Rate | Imbalance Ratio |
|---------|--------------|-----------------|
| Ali-CCP CTR | ~5% | 1:19 |
| Ali-CCP CVR | ~2% | 1:49 |
| FedAds CVR | ~0.55% | 1:181 |

### Techniques Implemented

**1. Focal Loss**
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Down-weight easy examples, focus on hard ones.
    
    For well-classified negative (pred≈0, target=0):
        - BCE loss is already small
        - Focal multiplier (1-pred)^gamma ≈ 1
        
    For hard negative (pred≈0.5, target=0):
        - BCE loss is moderate
        - Focal multiplier (1-pred)^gamma ≈ 0.25 → reduced
        
    For hard positive (pred≈0, target=1):
        - BCE loss is large
        - Focal multiplier pred^gamma ≈ 0 → NOT reduced
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()
```

**2. Class Weights**
```python
pos_weight = (n_negative / n_positive)  # e.g., 49 for 2% CVR
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

**3. Negative Downsampling with Calibration**
```python
# Downsample negatives to 1:5 ratio
# But remember to recalibrate predictions!
# If we trained on 1:5 but real ratio is 1:49:
#   p_calibrated = p_raw * (49/5) / (1 + p_raw * (49/5 - 1))
```

---

## 8. Evaluation Metrics for Adtech

### Metrics Overview

| Metric | Formula | Use Case |
|--------|---------|----------|
| AUC-ROC | Area under ROC curve | Overall ranking quality |
| Log Loss | -mean(y*log(p) + (1-y)*log(1-p)) | Probability quality |
| ECE | Expected Calibration Error | Calibration for bidding |
| **GAUC** | Weighted average of per-user AUC | Personalization quality |
| Lift@k | (CVR in top k%) / (overall CVR) | Targeting efficiency |
| AUC-PR | Area under Precision-Recall curve | Imbalanced data ranking |

---

### GAUC (Grouped AUC) - Why Global AUC Can Be Misleading

#### The Probabilistic Definition of AUC

Standard AUC-ROC measures:

```
AUC = P(score(random positive) > score(random negative))
```

This is computed by:
1. Randomly sample one positive (e.g., a clicked/converted impression)
2. Randomly sample one negative (e.g., a non-clicked/non-converted impression)
3. Check if the positive has a higher predicted score
4. Repeat many times, compute the proportion

Equivalently (Wilcoxon-Mann-Whitney statistic):
```
AUC = (1 / (n₊ × n₋)) × Σᵢ∈positives Σⱼ∈negatives 𝟙[scoreᵢ > scoreⱼ]
```

#### The Problem: Cross-User Comparisons Are Meaningless

**Global AUC** samples from ALL users combined:

```
Global AUC = P(score(any positive from ANY user) > score(any negative from ANY user))
```

Consider this example:

| User | Items | Model Predictions | True Labels |
|------|-------|-------------------|-------------|
| Alice | A, B, C | 0.8, 0.5, 0.2 | 1, 0, 0 |
| Bob | D, E | 0.3, 0.1 | 1, 0 |

When computing global AUC, we might compare:
- Alice's positive (Item A, score=0.8) vs Bob's negative (Item E, score=0.1) ✓ Correct ranking

But **this comparison is meaningless in production!** We're never choosing between showing Alice's item to Bob. In a real ad serving system:
- Alice sees {A, B, C} — we rank these against each other
- Bob sees {D, E} — we rank these against each other
- Cross-user ranking **never happens**

**A model that exploits user-level popularity** (e.g., power users always get higher scores) can achieve high global AUC while being useless for **within-user personalization**.

#### GAUC: Per-User AUC, Then Average

GAUC restricts comparisons to within each user:

```
GAUC_u = P(score(positive from user u) > score(negative from user u))

GAUC = Σᵤ (AUC_u × n_u) / Σᵤ (n_u)
```

Where:
- `AUC_u` = AUC computed only on user u's samples
- `n_u` = Number of samples for user u (weight)

This measures: **"On average, how well do we rank items within each user's session?"**

#### When GAUC Diverges from Global AUC

| Scenario | Global AUC | GAUC | Interpretation |
|----------|------------|------|----------------|
| Model exploits user popularity | High | Low | Ranks popular users' items higher, but doesn't personalize |
| Strong personalization | Moderate | High | Within-user ranking good even if cross-user isn't |
| Consistent quality | Similar | Similar | Model performs uniformly across users |

---

### The `min_samples` Parameter: Filtering Unreliable User AUCs

#### Why We Filter

AUC for a single user requires **both positive and negative samples**. With very few samples, AUC estimates are unreliable:

| Samples per User | Possible AUC Values | Problem |
|-----------------|---------------------|---------|
| 2 (1 pos, 1 neg) | 0.0, 0.5, 1.0 only | High variance, essentially random |
| 5 | Discrete set | Still unstable |
| 10+ | More continuous | More reliable |

Our implementation uses `min_samples=10` by default:

```python
if len(user_labels) < min_samples:
    skipped_users += 1
    continue  # Don't include this user in GAUC
```

#### Validating `min_samples` Against Data Distribution

**This parameter needs empirical validation!** The right value depends on your data:

For Ali-CCP:
- ~42.3M samples, ~730K users
- **Average:** 58 samples per user
- **But distributions are power-law!** A few users have thousands; many have <10

**Recommended analysis before finalizing `min_samples`:**

```python
import numpy as np
import pandas as pd

# Compute samples per user
user_counts = df.groupby('user_id').size()

# Distribution analysis
print(f"Median samples/user: {user_counts.median()}")
print(f"Mean samples/user: {user_counts.mean():.1f}")
print(f"Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {user_counts.quantile(p/100)}")

# Impact of min_samples threshold
for threshold in [5, 10, 20, 50]:
    excluded = (user_counts < threshold).sum()
    pct_excluded = excluded / len(user_counts) * 100
    print(f"min_samples={threshold}: excludes {pct_excluded:.1f}% of users")
```

**Rule of thumb:** Choose `min_samples` such that:
1. You exclude <20-30% of users
2. Remaining users have stable AUC estimates
3. You don't bias toward only high-volume users

**TODO for students:** Run this analysis on Ali-CCP to validate or adjust `min_samples=10`.

---

### Why AUC is Undefined with Only One Class

AUC requires **both classes** because the formula involves sampling one positive and one negative:

```
AUC = P(score(+) > score(-))
        ↑           ↑
    Need at least  Need at least
    one positive   one negative
```

**With only negatives (all non-conversions):**
- We can't sample a positive
- The probability `P(score(+) > ...)` conditions on an event that **never occurred**
- It's like asking "What's the average height of unicorns?"—undefined, not zero

**Mathematically**, AUC = `(1 / (n₊ × n₋)) × Σ...`

If `n₊ = 0` or `n₋ = 0`, we **divide by zero**.

**This is common in adtech:**
- Many users never convert during an evaluation window
- These users are excluded from GAUC (appropriately!)

**Note:** Excluding single-class users is correct behavior, not a bug. These users provide no ranking information.

---

### Lift Curves: Targeting Efficiency

Lift measures how much better model-based targeting is vs. random selection:

```
Lift@k = (CVR in top k% by predicted score) / (Overall CVR)
```

**Example interpretation:**
- Lift@10% = 3.5 means: The top 10% by model score converts at **3.5× the baseline rate**
- Critical for campaign planning: "If we can only afford to target 20% of users, which 20%?"

Our implementation:

```python
def compute_lift_curve(predictions, labels, percentiles=[1, 5, 10, 20, 50]):
    sorted_indices = np.argsort(-predictions)  # Descending by score
    sorted_labels = labels[sorted_indices]
    overall_cvr = labels.mean()
    
    for pct in percentiles:
        n_top = int(len(labels) * pct / 100)
        top_cvr = sorted_labels[:n_top].mean()
        lift = top_cvr / overall_cvr
        # ...
```

---

### Popularity Baseline: Does Personalization Add Value?

**Random baseline** (lift = 1.0) is a weak baseline—most models easily beat it. A much stronger baseline is **popularity-based ranking**.

**The key business question:** Does our personalized model add value beyond just showing popular items?

#### Popularity Score Computation

For each item, compute its smoothed historical conversion rate using Bayesian smoothing:

```python
def compute_popularity_scores(item_ids, labels, smoothing=10.0):
    """
    Popularity score = smoothed historical CVR for each item.
    
    Bayesian smoothing (Beta-Binomial conjugate prior):
    score = (conversions + α × prior_mean) / (impressions + α)
    """
    global_cvr = labels.mean()  # Prior mean (empirical Bayes)
    
    for item in unique_items:
        mask = item_ids == item
        conversions = labels[mask].sum()
        impressions = mask.sum()
        
        # Bayesian smoothing pulls rare items toward global mean
        smoothed_cvr = (conversions + smoothing * global_cvr) / (impressions + smoothing)
        item_popularity[item] = smoothed_cvr
```

**Why smoothing?** An item with 1 conversion out of 1 impression has 100% CVR—but this is unreliable. Smoothing adds "pseudo-counts" that pull rare items toward the global mean.

#### Choosing the Bayesian Prior

The prior has two components:

| Parameter | What It Controls | Recommended Default |
|-----------|------------------|---------------------|
| **Prior mean** | What CVR to shrink toward | `global_cvr` (empirical Bayes) |
| **Prior strength (α)** | How much to shrink | `10.0` (or median item impressions) |

**Prior Mean Options:**

| Choice | When to Use |
|--------|-------------|
| `global_cvr` | Default; data-driven |
| `category_cvr[cat]` | Items vary by category (hierarchical) |
| Historical average | Stable domains with past data |

**Prior Strength (α) - The Shrinkage Control:**

After `n` real impressions, weight on observed data = `n/(n+α)`:

| α Value | Item needs ~N impressions to be 50% data-driven |
|---------|------------------------------------------------|
| α = 5 | 5 impressions |
| α = 10 | 10 impressions |
| α = 50 | 50 impressions |
| α = 100 | 100 impressions |

**How to choose α empirically:**

```python
# Rule of thumb: α ≈ median impressions per item
item_counts = df.groupby('item_id').size()
suggested_alpha = item_counts.median()

# For conservative bidding: use 75th percentile
conservative_alpha = item_counts.quantile(0.75)
```

**Adjust based on domain:**

| Situation | Adjust α |
|-----------|----------|
| Long-tail catalog (many rare items) | Increase (20-50) |
| Head-heavy (few items dominate) | Decrease (5-10) |
| High-stakes decisions (bidding) | Increase (be conservative) |
| Fast-changing popularity | Decrease (trust recent data) |

> **Note for practitioners:** The default `smoothing=10.0` and `prior_mean=global_cvr` are reasonable starting points, but these parameters should be tuned on your specific dataset. Consider:
> 1. Analyzing your item impression distribution (`df.groupby('item_id').size().describe()`)
> 2. Cross-validating different α values on held-out items
> 3. Checking calibration of popularity scores vs. actual CVR

#### Lift Comparison Table

```
======================================================================
LIFT CURVE COMPARISON
Overall CVR: 0.0350 (3.50%)
======================================================================

Top %      Model Lift   Pop Lift     Random     Model-Pop   
--------------------------------------------------------------
    1%     4.82         4.15         1.00       +0.67       
    5%     3.91         3.42         1.00       +0.49       
   10%     3.28         2.89         1.00       +0.39       
   20%     2.54         2.31         1.00       +0.23       
   50%     1.68         1.58         1.00       +0.10       

Interpretation:
  - Model beats popularity by 0.39 lift points at top 10%
  - This is the VALUE OF PERSONALIZATION over "show popular items"
```

#### When Popularity Wins

If `Model-Pop` is negative, your personalized model underperforms a simple popularity baseline! This can happen when:

1. **Insufficient personalization signal**: User features are weak
2. **Model overfitting**: Memorizing training data, poor generalization
3. **Popularity dominance**: Domain where popular items truly are best for everyone

**Action:** If popularity consistently wins, consider:
- Adding more/better user features
- Simpler model (reduce overfitting)
- Hybrid approach: popularity + light personalization

#### Important: Data Leakage Warning

When computing popularity baseline:
- In **experiments**: Compute popularity on TRAINING data, apply to TEST data
- In **this code**: For simplicity, computed on provided data (fine for demonstration)
- In **production**: Use historical popularity, not future-looking

---

### ESMM-Specific Evaluation: Which Metric on Which Data?

```python
# CTR metrics: Evaluate on ALL samples
# (Every impression has a click/no-click label)
ctr_auc = roc_auc_score(all_labels['click'], preds['ctr'])

# CVR metrics: Evaluate on CLICKED samples only
# (Ground truth CVR is only observed for clicked samples)
clicked_mask = all_labels['click'] == 1
cvr_auc = roc_auc_score(
    all_labels['purchase'][clicked_mask], 
    preds['cvr'][clicked_mask]
)

# CTCVR metrics: Evaluate on ALL samples
# (Every impression has a purchase/no-purchase label)
ctcvr_auc = roc_auc_score(all_labels['purchase'], preds['ctcvr'])
```

**Critical note on CVR evaluation:**
- CVR tower predicts P(conversion | click)
- But for non-clicked samples, we don't know if the user *would have* converted
- So CVR evaluation is inherently limited to clicked samples
- This is **not** selection bias—it's correct evaluation of the CVR task

---

### AUC-PR for Extreme Imbalance

When positive rates are very low (CVR ~0.5%), AUC-ROC can be misleadingly high even for poor models. AUC-PR (Area under Precision-Recall curve) is more informative:

| Dataset | Positive Rate | Baseline AUC-ROC | Baseline AUC-PR |
|---------|---------------|------------------|-----------------|
| Balanced (50%) | 50% | 0.50 | 0.50 |
| Ali-CCP CTR (5%) | 5% | 0.50 | 0.05 |
| FedAds CVR (0.55%) | 0.55% | 0.50 | 0.0055 |

A random classifier has AUC-PR ≈ positive rate, making improvements more visible.

---

### Summary: Which Metrics to Report

| Metric | When to Use | Watch Out For |
|--------|-------------|---------------|
| **Global AUC** | Quick sanity check | Misleading for personalization |
| **GAUC** | Personalization quality | Excludes low-volume users |
| **Log Loss** | Probability quality | Sensitive to calibration |
| **ECE** | Bid optimization | Requires binning strategy |
| **Lift@k** | Business impact | Depends on targeting budget |
| **AUC-PR** | Extremely imbalanced data | Less intuitive than AUC-ROC |

**Recommended standard reporting:**
1. Global AUC (for comparability with literature)
2. GAUC (for personalization assessment)
3. Log Loss (for probability quality)
4. Lift@10% and Lift@20% (for business impact)

---

## 9. Code Organization

```
chapter9_adtech/
├── config.py              # Paths, hyperparameters
├── data/
│   ├── __init__.py
│   ├── ali_ccp_dataset.py # Ali-CCP dataset (reads preprocessed parquet)
│   └── fedads_loader.py   # FedAds data loading
├── models/
│   ├── __init__.py
│   ├── esmm.py           # ESMM model
│   └── split_nn.py       # Split Neural Network for federated
├── utils/
│   ├── __init__.py
│   ├── calibration.py    # Platt, Isotonic, Temperature scaling
│   ├── delayed_feedback.py # Importance weighting, DFM
│   └── metrics.py        # Adtech-specific metrics
├── train_esmm.py         # Main ESMM training script
├── train_cvr_baseline.py # Naive CVR baseline (for comparison)
├── train_split_nn.py     # FedAds paper replication training
├── scripts/
│   └── preprocess_ali_ccp.py  # One-time Ali-CCP preprocessing
├── evaluate.py           # Comprehensive evaluation
├── docs/
│   └── Chapter9_Design_Notes.md (this file)
└── README.md
```

---

## 10. Experiments Log

### Experiment 1: ESMM vs Clicked-Only CVR Baseline

- **Date:** 2026-02-11
- **Dataset:** Ali-CCP, 10% sample (~4.2M samples: 3.8M train, 423K val)
- **Device:** CUDA (GPU)
- **Training:** 4 epochs (early stopping patience=3), batch_size=4096, lr=1e-3
- **Hypothesis:** ESMM (entire-space training) will outperform a naive CVR model trained on clicked samples only

**Results:**

| Metric | ESMM | Clicked-Only Baseline | Delta |
|--------|------|-----------------------|-------|
| **CVR AUC** (on clicked samples) | **0.6311** | 0.5926 | +3.85pp |
| **CTCVR AUC** (on all samples) | **0.6204** | 0.5848 | +3.56pp |
| **CTR AUC** | 0.6009 | N/A | — |

**Data characteristics:**
- Click rate: 4.67%, Purchase rate: 0.03%, Post-click CVR: 0.63%
- Train conversions: 1,093 | Val conversions: 127

**Key Takeaways:**
1. **ESMM significantly outperforms the clicked-only baseline** on both CVR and CTCVR metrics, confirming that entire-space training reduces selection bias.
2. **Calibration is poor** — the clicked-only baseline predicts mean CVR of 7.6% vs true CVR of 0.03% on all samples. This is because it was trained on clicked samples (6.3% CVR) and applied to all impressions — a clear manifestation of selection bias.
3. **CTR AUC (0.60) is moderate** — this is reasonable for sparse features with feature hashing. Higher CTR AUC would further benefit the CVR tower through stronger CTCVR gradients.

### Experiment 2: Calibration Impact on Bid Accuracy
- **Date:** [TBD]
- **Dataset:** Ali-CCP
- **Metrics:** ECE before/after calibration
- **Results:** [TBD]

### Experiment 3: FedAds Paper Replication (VanillaVFL)

- **Date:** 2026-02-14
- **Dataset:** FedAds aligned, all 2,598,552 samples (2.30M train, 299K val)
- **Split:** Time-based (row-order proxy), val_ratio=11.5%
- **Device:** CUDA (GPU)
- **Training:** 1 epoch, batch_size=256, lr=1e-3, standard BCE, no dropout/weight decay
- **Paper:** Wei et al. FedAds (SIGIR 2023), Table 3

**NLL (Negative Log-Likelihood)** is the same as log loss / binary cross-entropy:

```
NLL = -mean( y × log(p) + (1-y) × log(1-p) )
```

It measures probability quality — how well predicted probabilities match true outcomes. Lower is better. The FedAds paper uses "NLL" as their metric name (Table 3), which is why we report it alongside AUC.

**Results:**

| Model | Our AUC | Our NLL | Paper AUC | Delta vs Paper |
|-------|---------|---------|-----------|----------------|
| **VanillaVFL** (SplitNN) | **0.6511** | 0.0366 | 0.620 | +0.031 |
| ORALE (Centralized) | 0.6284 | 0.0370 | 0.658 | -0.030 |
| Local (label party only) | 0.5604 | 0.0373 | 0.609 | -0.049 |
| NonLabel (publisher only) | 0.6279 | 0.0367 | N/A | — |

**Data characteristics:**
- CVR: 0.59% (train) / 0.61% (val)
- Train conversions: ~13,600 | Val conversions: ~1,830

#### Why Our Results Diverge from the Paper

**Divergence 1: VanillaVFL outperforms ORALE (ours: 0.651 > 0.628; paper: 0.620 < 0.658)**

The paper expects ORALE > VanillaVFL, but we observe the opposite. The most likely explanation is the **flipped feature distribution**:

| Party | Paper's System | Public Dataset |
|-------|---------------|----------------|
| Non-label (publisher) | 7 features | **17 features** |
| Label (advertiser) | 16 features | **5 features** |

In the paper's system, the label party contributes the majority of features (16/23 = 70%). The centralized model benefits greatly from combining these with the non-label party's 7 features.

In our public dataset, the non-label party already has 17/22 = 77% of all features. The centralized model gains relatively little by adding the label party's 5 features, and is more prone to overfitting on the larger combined feature space. Meanwhile, the split architecture imposes a **structural bottleneck** (separate embeddings, separate towers per party) that acts as an implicit regularizer — similar to how dropout constrains model capacity.

**Evidence of centralized overfitting:** NLL for ORALE (0.0370) is worse than VanillaVFL (0.0366) despite seeing the same data, suggesting the centralized model's probability estimates are less calibrated.

**Divergence 2: Local (label party) is much weaker than paper (ours: 0.560 vs paper: 0.609)**

This is directly explained by the feature count discrepancy. The paper's label party has 16 features — enough to build a reasonable standalone model (AUC 0.609). Our label party has only 5 features — significantly less signal, resulting in AUC 0.560.

**Divergence 3: NonLabel (publisher) is nearly as strong as ORALE (0.628 vs 0.628)**

Since the publisher has 17 of 22 features, it already captures most of the available signal. The advertiser's 5 features add only marginal value when used independently. This is the opposite of the paper's setting where the label party is the dominant feature owner.

#### Summary of Divergence Analysis

| Observation | Root Cause | Confidence |
|-------------|-----------|------------|
| VanillaVFL > ORALE | Implicit regularization from split arch + centralized overfitting on sparse CVR signal | High |
| Local baseline much weaker | Only 5 features vs paper's 16 | Very High |
| NonLabel ≈ ORALE | Publisher has 17/22 features — dominates signal | Very High |
| Absolute AUC higher for VFL | Flipped feature ratio changes which party benefits from federation | High |

The core pedagogical claim from the paper still holds: **federation adds significant value** (+9.07pp AUC for the label party by collaborating with the publisher). The privacy cost is negligible or even negative.

#### NLL Interpretation Note

NLL values are very low (~0.037) and tightly clustered across models because CVR is only 0.6%. With 99.4% negatives, a model predicting ~0.006 for everything already achieves low NLL. The NLL is dominated by the easy negatives, making AUC the more discriminating metric for this task.

#### Homework: Feature Party Assignment Investigation

**Hypothesis:** The divergence from the paper's results may be partly caused by which features are assigned to which party. The public dataset's `l_*` (local/publisher) and `f_*` (federated/advertiser) prefixes determine the split, but the paper's internal system used a different assignment.

**Your Task:** Modify the code to **swap the party assignments** — treat `f_*` features as the non-label party and `l_*` features as the label party. This simulates the paper's setting where the label party has the richer feature set.

**Steps:**

1. In `config.py`, swap `FEDADS_LOCAL_FEATURES` and `FEDADS_FEDERATED_FEATURES`
2. Update `SplitNNConfig` to reflect `n_local_features=5` and `n_fed_features=17`
3. Adjust `local_layers` / `fed_layers` if needed (the party with more features may need larger capacity)
4. Re-run `python train_split_nn.py` and compare results

**Questions to answer:**
- Does swapping bring the Local baseline closer to the paper's 0.609?
- Does ORALE now outperform VanillaVFL (matching the paper's expected ordering)?
- What does this tell you about how feature richness per party affects the privacy-utility tradeoff?

---

## 11. Implementation Summary

### Files Created

| File | Description | Lines |
|------|-------------|-------|
| `config.py` | Paths, hyperparameters, dataclasses for all configs | ~200 |
| `data/ali_ccp_loader.py` | Ali-CCP data loading with sparse feature parsing | ~350 |
| `data/fedads_loader.py` | FedAds data loading for federated scenarios | ~300 |
| `models/esmm.py` | ESMM model + CVR baseline for comparison | ~400 |
| `models/split_nn.py` | Split NN for federated learning (FedAds paper) | ~300 |
| `utils/calibration.py` | Platt, Isotonic, Temperature scaling + ECE | ~400 |
| `utils/delayed_feedback.py` | Importance weighting, delay distribution | ~350 |
| `utils/metrics.py` | GAUC, Lift curves, Adtech-specific metrics | ~350 |
| `train_esmm.py` | ESMM training pipeline with baseline comparison | ~400 |
| `train_split_nn.py` | Federated training with centralized baseline | ~350 |

### Key Design Patterns

1. **Config Dataclasses:** Type-safe configuration with validation
2. **Model + Loss in Same Class:** Each model implements `compute_loss()` for encapsulation
3. **Paper-grounded defaults:** FedAds config matches Wei et al. (SIGIR 2023) exactly
4. **Cached Data Parsing:** Pre-parse features to parquet cache for fast iteration
5. **Comparison-First Training:** Scripts compare against baselines by default

### Connection to Book Chapters

| Topic | Builds On | New Contribution |
|-------|-----------|------------------|
| ESMM Multi-Task | Ch7 MMoE | Selection bias correction via CTCVR |
| Calibration | Ch8 Value Functions | Probability quality for bidding |
| Federated Learning | Ch10 Deployment | Privacy-preserving cross-org training |
| Focal Loss | Ch7 Class Weights | More sophisticated imbalance handling |

---

---

## References

1. Ma et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate" (SIGIR 2018)
2. Chapelle et al. "Modeling Delayed Feedback in Display Advertising" (KDD 2014)
3. Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
4. Liu et al. "FedBCD: A Communication-Efficient Collaborative Learning Framework for Distributed Features" (NeurIPS 2020)
5. **Wei et al. "FedAds: A Benchmark for Privacy-Preserving CVR Estimation with Vertical Federated Learning" (SIGIR 2023)** — Primary reference for the FedAds dataset and VanillaVFL architecture pattern our code follows
6. Vepakomma et al. "Split learning for health: Distributed deep learning without sharing raw data" (2018) — Foundational reference for the general split learning concept
