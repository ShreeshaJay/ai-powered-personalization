# Chapter 6: Ranking

## Overview

This chapter demonstrates ranking models for recommender systems using the **Yandex Yambda** dataset - a large-scale music recommendation dataset with 50 million user-item interactions.

**Learning Objectives:**
- Understand the role of ranking in multi-stage recommender systems
- Implement gradient boosting baselines (XGBoost) for ranking
- Progress to neural ranking architectures (DeepFM, DCN)
- Apply multi-task learning (MMoE, PLE) for joint optimization

## Dataset: Yandex Yambda

**Source:** [HuggingFace - yandex/yambda](https://huggingface.co/datasets/yandex/yambda)  
**Paper:** [Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking And Retrieval](https://arxiv.org/abs/2505.22238)

### Key Features

| Feature | Description |
|---------|-------------|
| `uid` | User identifier |
| `item_id` | Track identifier |
| `timestamp` | Delta time (binned to 5-second intervals) |
| `is_organic` | 0 = recommendation-driven, 1 = organic discovery |
| `played_ratio_pct` | Percentage of track played (target for listen completion) |
| `track_length_seconds` | Track duration |
| `event_type` | listen, like, dislike, unlike, undislike |

### Evaluation Protocol: Global Temporal Split (GTS)

Following the paper's protocol:

```
┌─────────────────────────────────────┬──────┬─────────┐
│         TRAINING (300 days)         │ GAP  │  TEST   │
│                                     │30min │ (1 day) │
└─────────────────────────────────────┴──────┴─────────┘
```

- **Training:** First 300 days of data
- **Gap:** 30 minutes (prevents information leakage)
- **Test:** Next 1 day of data

## Code Structure

```
chapter6_ranking/
├── config.py               # Centralized configuration (paths, hyperparameters)
├── data_loader.py          # Data loading with GTS implementation
├── models/                 # Model classes
│   ├── __init__.py
│   ├── feature_encoder.py  # Reusable feature encoding
│   └── xgboost_ranker.py   # XGBoost model class
├── train_xgboost.py        # Training script
├── evaluate.py             # Evaluation script
├── predict.py              # Inference script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── outputs/                # Experiment results
    ├── models/             # Saved model checkpoints
    ├── metrics/            # Evaluation metrics
    └── predictions/        # Model predictions
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Update `config.py` to point to your Yambda data:

```python
# In config.py
DATA_DIR = Path('path/to/Dataset/Yandex/flat')
```

### 3. Train XGBoost Baseline

```bash
# Train with default settings
python train_xgboost.py

# Train with sample for faster iteration
python train_xgboost.py --sample_frac 0.01

# Train with custom hyperparameters
python train_xgboost.py --n_estimators 200 --max_depth 8 --learning_rate 0.05
```

### 4. Evaluate Model

```bash
# Evaluate trained model
python evaluate.py --model_path outputs/models/20231215_120000

# Generate evaluation plots
python evaluate.py --model_path outputs/models/20231215_120000 --plot
```

### 5. Generate Predictions

```bash
# Predict on test data
python predict.py --model_path outputs/models/20231215_120000

# Get top-k recommendations per user
python predict.py --model_path outputs/models/20231215_120000 --top_k 10
```

## Training Script Options

```
python train_xgboost.py [OPTIONS]

Data Options:
  --data_dir PATH         Path to Yambda flat data directory
  --sample_frac FLOAT     Fraction of data to sample (None for full)

GTS Options:
  --train_days INT        Number of days for training (default: 300)
  --gap_minutes INT       Gap between train/test in minutes (default: 30)
  --test_days INT         Number of days for test (default: 1)

Model Hyperparameters:
  --n_estimators INT      Number of boosting rounds (default: 100)
  --max_depth INT         Maximum tree depth (default: 6)
  --learning_rate FLOAT   Learning rate (default: 0.1)
  --subsample FLOAT       Subsample ratio (default: 0.8)
  --colsample_bytree FLOAT Column sampling ratio (default: 0.8)

Training Options:
  --val_split_ratio FLOAT Validation split from training data (default: 0.1)
  --early_stopping_rounds INT Rounds to stop after no improvement (default: 10)
  --completion_threshold INT  Played ratio threshold for positive label (default: 50)

Output Options:
  --output_dir PATH       Output directory for model and metrics
  --experiment_name STR   Experiment name for output directory
```

## Chapter Progression

### 6.1-6.2: Problem Setup & Feature Engineering
- Understanding ranking as classification/regression
- Categorical encoding for high-cardinality features
- Feature aggregations and derived features

### 6.3: XGBoost Baseline ← **Current**
- Gradient boosting for ranking (pointwise approach)
- Binary classification: P(listen_completion | user, item, context)
- Evaluation: AUC, Log Loss, Average Precision

### 6.4: DeepFM (Coming Next)
- Embedding layers for categorical features
- Factorization Machine for 2nd-order interactions
- Deep component for higher-order patterns

### 6.5: Deep Cross Network (DCN)
- Explicit cross layers for feature interactions
- Comparison with DeepFM

### 6.6-6.8: Multi-Task Learning
- **MMoE:** Multiple experts with task-specific gating
- **PLE:** Progressive separation for task conflict handling
- Tasks: Listen completion + Like prediction + Dislike prediction

## Expected Results

Baseline performance on Yambda (from paper):

| Model | NDCG@10 | MRR@10 | HR@10 |
|-------|---------|--------|-------|
| Random | 0.003 | 0.001 | 0.010 |
| Popular | 0.019 | 0.009 | 0.048 |
| ItemKNN | 0.051 | 0.027 | 0.112 |
| iALS | 0.054 | 0.028 | 0.118 |
| SANSA | 0.061 | 0.033 | 0.128 |
| SASRec | **0.069** | **0.038** | **0.141** |

Our ranking models (listen completion prediction):

| Model | AUC | Log Loss |
|-------|-----|----------|
| XGBoost | ~0.70 | ~0.55 |
| DeepFM | TBD | TBD |
| DCN | TBD | TBD |

## Key Concepts

### Why Ranking Matters

In a multi-stage recommender system:

```
Retrieval (Chapter 5)     Ranking (Chapter 6)      Re-ranking (Chapter 7)
     ↓                          ↓                        ↓
1000s of candidates  →   Score & rank 100s   →    Final ordering
(fast, recall-focused)   (accurate predictions)   (business rules, diversity)
```

### Ranking vs. Retrieval

| Aspect | Retrieval | Ranking |
|--------|-----------|---------|
| Candidates | Millions | Hundreds |
| Latency budget | ~10ms | ~50-100ms |
| Model complexity | Simple (two-tower) | Complex (cross features) |
| Optimization | Recall@K | AUC, NDCG |

### Pointwise vs. Learning-to-Rank

The current XGBoost implementation uses a **pointwise** approach:
- Each sample is scored independently
- Binary classification: "will user complete this track?"
- Simple but doesn't optimize for list-level metrics

Future work could extend to **pairwise** (LambdaMART) or **listwise** approaches.

### Multi-Task Learning Benefits

Using MMoE/PLE with multiple feedback signals:

1. **Data efficiency:** Sparse signals (likes) benefit from abundant signals (listens)
2. **Task correlation:** Shared experts capture general preferences
3. **Task conflict handling:** PLE prevents negative transfer between conflicting tasks

## API Reference

### XGBoostRanker

```python
from models import XGBoostRanker

# Initialize
model = XGBoostRanker(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# Train
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Predict
proba = model.predict_proba(X_test)
metrics = model.evaluate(X_test, y_test)

# Save/Load
model.save('outputs/models/my_model')
model = XGBoostRanker.from_pretrained('outputs/models/my_model')
```

### FeatureEncoder

```python
from models import FeatureEncoder

# Initialize with frequency encoding for high-cardinality features
encoder = FeatureEncoder(high_cardinality_threshold=10000)

# Fit on training data
encoder.fit(train_df, ['uid', 'item_id', 'is_organic'])

# Transform
train_encoded = encoder.transform(train_df)
test_encoded = encoder.transform(test_df)

# Save/Load
encoder.save('outputs/models/my_model')
encoder.load('outputs/models/my_model')
```

## References

1. [Yambda Paper](https://arxiv.org/abs/2505.22238) - Dataset and benchmarks
2. [DeepFM](https://arxiv.org/abs/1703.04247) - Deep Factorization Machines
3. [DCN](https://arxiv.org/abs/1708.05123) - Deep & Cross Network
4. [MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007) - Multi-gate Mixture-of-Experts
5. [PLE](https://dl.acm.org/doi/10.1145/3383313.3412236) - Progressive Layered Extraction
