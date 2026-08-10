# Chapter 9: Modeling for Adtech Use Cases

This chapter covers adtech-specific modeling techniques that address challenges unique to advertising systems: selection bias, delayed feedback, privacy constraints, and extreme class imbalance.

## Key Topics

### 1. ESMM (Entire Space Multi-Task Model)
**Problem:** Traditional CVR models train on clicked samples only, causing selection bias when applied to all impressions.

**Solution:** ESMM models `CTCVR = CTR × CVR`, training on ALL impressions to eliminate selection bias.

```
┌─────────────────────────────────────────────────┐
│                    ESMM                         │
├─────────────────────────────────────────────────┤
│   Shared Embeddings                             │
│         │                                       │
│   ┌─────┴─────┐                                 │
│   │           │                                 │
│ CTR Tower   CVR Tower                           │
│   │           │                                 │
│ P(click)   P(conv|click)  ← No direct loss!     │
│   │           │                                 │
│   └─────┬─────┘                                 │
│         │                                       │
│   CTCVR = CTR × CVR                             │
│         │                                       │
│   CTCVR Loss (on ALL samples)                   │
└─────────────────────────────────────────────────┘
```

### 2. Split Neural Networks for Federated Learning
**Problem:** Publishers and advertisers have complementary data but can't share raw features due to privacy.

**Solution:** Each party trains their own tower; only hidden representations are exchanged.

### 3. Model Calibration for Bidding
**Problem:** Miscalibrated probabilities lead to systematic over/under-bidding.

**Solution:** Post-hoc calibration using Platt scaling, isotonic regression, or temperature scaling.

### 4. Handling Extreme Class Imbalance
CVR is typically 0.5-2%, requiring specialized techniques:
- Focal loss
- Class weighting
- Negative downsampling with calibration correction

## Datasets

### Ali-CCP (Primary)
- **Source:** https://tianchi.aliyun.com/dataset/408
- **Size:** 42M impressions, 730K users
- **Key Feature:** Full funnel data (clicked + non-clicked)
- **Used For:** ESMM, calibration, delayed feedback

### FedAds (Secondary)
- **Source:** https://tianchi.aliyun.com/dataset/148347
- **Size:** 2.6M aligned + 10.4M unaligned samples
- **Key Feature:** Features split between publisher/advertiser
- **Used For:** Federated learning demonstration

## Code Structure

```
chapter9_adtech/
├── config.py                 # Paths and hyperparameters
├── scripts/
│   └── preprocess_ali_ccp.py # One-time data preprocessing
├── data/
│   ├── ali_ccp_dataset.py   # Clean data loader (recommended)
│   └── fedads_loader.py     # FedAds data loading
├── models/
│   ├── esmm.py              # ESMM and CVR baseline
│   └── split_nn.py          # Split Neural Network for federated
├── utils/
│   ├── calibration.py       # Calibration methods (Platt, Isotonic)
│   ├── delayed_feedback.py  # Delayed feedback handling
│   └── metrics.py           # Adtech metrics (GAUC, Lift, Popularity)
├── train_esmm.py            # ESMM training pipeline
├── train_split_nn.py        # Federated learning training
├── docs/
│   └── Chapter9_Design_Notes.md
└── README.md
```

## Quick Start

### Step 0: Preprocess Ali-CCP Data (ONE TIME ONLY)

The raw Ali-CCP dataset has a tricky sparse feature format. Run preprocessing once to create clean parquet files:

```bash
cd chapter9_adtech/scripts

# For 16GB RAM laptops (5% = ~2.1M samples, ~10 minutes)
python preprocess_ali_ccp.py --pct 5

# For quick testing (0.1% = ~42K samples, ~2 minutes)
python preprocess_ali_ccp.py --pct 0.1

# Full dataset (100%, requires 32GB+ RAM, ~30 minutes)
python preprocess_ali_ccp.py --pct 100
```

**Homework:** Download the full dataset from https://tianchi.aliyun.com/dataset/408 and run with `--pct 100`.

This creates:
- `Dataset/Ali-CCP Entire State Model/processed/ali_ccp_train.parquet`
- `Dataset/Ali-CCP Entire State Model/processed/ali_ccp_val.parquet`
- `Dataset/Ali-CCP Entire State Model/processed/metadata.json`

### Step 1: Train ESMM
```bash
cd chapter9_adtech
python train_esmm.py --dev
```

### Step 2: Compare ESMM vs CVR Baseline
```bash
python train_esmm.py --compare_baseline
```

### Step 3: Train Split Neural Network (Federated Learning)
```bash
python train_split_nn.py --n_samples 500000
```

## Requirements

```
torch>=2.0
polars>=0.19
numpy
scikit-learn
tqdm
```

## Key Insights

1. **Selection Bias is Real:** CVR models trained on clicked samples only systematically underperform on the full impression space.

2. **Calibration Matters:** In bidding systems, miscalibrated probabilities directly translate to wasted budget or missed opportunities.

3. **Privacy-Utility Tradeoff:** Split Neural Networks achieve ~95% of centralized performance while preserving data privacy.

4. **Extreme Imbalance:** At 0.5% CVR, standard cross-entropy loss barely learns—use focal loss or careful sampling.

## References

1. Ma et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate" (SIGIR 2018)
2. Chapelle et al. "Modeling Delayed Feedback in Display Advertising" (KDD 2014)
3. Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
4. Wei et al. "FedAds: A Benchmark for Privacy-Preserving CVR Estimation with Vertical Federated Learning" (SIGIR 2023)
5. Vepakomma et al. "Split learning for health: Distributed deep learning without sharing raw data" (2018)

