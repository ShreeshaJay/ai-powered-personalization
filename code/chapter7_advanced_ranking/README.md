# Chapter 7: Advanced Ranking Models

## Overview

This chapter builds on Chapter 6's foundation with sophisticated neural ranking architectures for recommender systems. While Chapter 6 covers XGBoost baselines and DeepFM, this chapter introduces state-of-the-art models from Google and Facebook.

**Learning Objectives:**
- Implement DCN-V2 (Google) for explicit feature interactions with low-rank factorization
- Understand DLRM (Facebook) architecture for large-scale recommendations
- Apply multi-task learning with MMoE for joint optimization

## Models Implemented

| Model | Type | Key Innovation | Reference |
|-------|------|----------------|-----------|
| **DCN-V2** | Single-Task | Cross Network with low-rank W = U·V^T | [Wang et al., WWW 2021](https://arxiv.org/abs/2008.13535) |
| **DLRM** | Single-Task | Bottom MLP + Pairwise dot interactions | [Naumov et al., 2019](https://arxiv.org/abs/1906.00091) |
| **MMoE** | Multi-Task | Shared experts with task-specific gates | [Ma et al., KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3220007) |

## Dataset

Uses the **Yandex Yambda** dataset (same as Chapter 6):
- ~50M user-item interactions
- Listen completion as primary task
- Engagement, like, and dislike prediction as additional tasks (for MMoE)

## Directory Structure

```
chapter7_advanced_ranking/
├── config.py                 # Configuration and hyperparameters
├── models/
│   ├── __init__.py
│   ├── dcn.py               # DCN-V2 architecture
│   ├── dlrm.py              # DLRM architecture
│   └── mmoe.py              # MMoE + FocalLoss + MultiTaskLoss
├── utils/
│   ├── __init__.py
│   └── polars_pipeline.py   # Data loading and feature engineering
├── train_dcn.py             # DCN-V2 training script
├── train_dlrm.py            # DLRM training script
├── train_mmoe.py            # MMoE multi-task training script
├── outputs/                 # Experiment outputs
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Same dependencies as Chapter 6
pip install torch polars pyarrow scikit-learn
```

### 2. Set Data Path

```bash
# Windows PowerShell
$env:YAMBDA_DATA_DIR = "C:\path\to\Dataset\Yandex\flat"

# Or use --data_dir argument
```

### 3. Train Models

```bash
# DCN-V2 (30-day sample for quick iteration)
python train_dcn.py --train_days 30

# DLRM
python train_dlrm.py --train_days 30

# MMoE (multi-task: 5 tasks - engagement, completion, play_ratio[regression], like, dislike)
python train_mmoe.py --train_days 30 --use_focal_loss
```

## Model Details

### DCN-V2: Deep & Cross Network V2

**Architecture:**
```
Input → [Embeddings | Dense] → Concat → ┬→ Cross Network → ┐
                                         │                   ├→ Concat → Output
                                         └→ Deep Network  → ┘
```

**Key Innovation:** The cross network uses matrix W (not vector w) with low-rank factorization:
- W = U · V^T (reduces parameters while maintaining expressiveness)
- x_{l+1} = x_0 ⊙ (W · x_l + b) + x_l

**When to use:** CTR prediction where explicit feature interactions matter.

```bash
python train_dcn.py --train_days 30 --cross_layers 3 --cross_rank 32 --structure parallel
```

### DLRM: Deep Learning Recommendation Model

**Architecture:**
```
Sparse Features → Embeddings ────────────────┐
                                              ├→ Dot Product Interactions → Top MLP → Output
Dense Features  → Bottom MLP → [embed_dim] ──┘
```

**Key Innovation:** 
- Bottom MLP transforms dense features to embedding space
- Pairwise dot products capture all feature interactions
- Designed for distributed training at scale

**When to use:** Large-scale systems with significant dense features.

```bash
python train_dlrm.py --train_days 30 --bottom_mlp_dims 64,32,16 --top_mlp_dims 256,128,64
```

### MMoE: Multi-gate Mixture-of-Experts

**Architecture (5 Tasks: 4 Binary + 1 Regression):**
```
                         ┌→ Gate 1 → Tower 1 → P(Engagement) [Binary]
Input → [Expert 1] → ┐   │
        [Expert 2] → ├───┼→ Gate 2 → Tower 2 → P(Completion) [Binary]
        [Expert 3] → │   │
                     │   ├→ Gate 3 → Tower 3 → E[PlayRatio]  [Regression]
        [Expert 4] → ┤   ├→ Gate 3 → Tower 3 → P(Like)
        [Expert 5] → │   │
        [Expert 6] → ┘   └→ Gate 4 → Tower 4 → P(Dislike)
```

**Key Innovation:**
- Shared experts learn general representations
- Task-specific gates weight expert contributions
- Handles task conflicts gracefully

**Tasks (4 total):**
- **Engagement:** played_ratio_pct > 0 (user started listening)
- **Completion:** played_ratio_pct >= 50 (~63% positive)
- **Like:** User liked the item (~4-8% positive, use focal loss)
- **Dislike:** User disliked the item (~1-2% positive, use focal loss)

**Value Function:**
```
value = w1*P(engagement) + w2*P(completion) + w3*P(like) - w4*P(dislike)
```

```bash
python train_mmoe.py --train_days 30 --num_experts 6 --use_focal_loss
```

## Hyperparameter Guide

### DCN-V2
| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 16 | Embedding dimension for sparse features |
| `cross_layers` | 3 | Number of cross network layers |
| `cross_rank` | 32 | Low-rank dimension (smaller = fewer params) |
| `mlp_dims` | 256,128,64 | Deep network hidden layers |
| `structure` | parallel | 'parallel' or 'stacked' |

### DLRM
| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 16 | Embedding dimension |
| `bottom_mlp_dims` | 64,32,16 | Dense feature projection (last must = embed_dim) |
| `top_mlp_dims` | 256,128,64 | Final prediction network |
| `interaction` | dot | 'dot' (pairwise) or 'cat' (concatenation) |

### MMoE
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 6 | Number of shared expert networks (6 recommended for 5 tasks) |
| `expert_dims` | 256,128 | Expert network hidden layers |
| `tower_dims` | 64,32 | Task tower hidden layers |
| `engagement_weight` | 1.0 | Loss weight for engagement task (binary) |
| `completion_weight` | 1.0 | Loss weight for completion task (binary) |
| `play_ratio_weight` | 1.0 | Loss weight for play_ratio task (regression) |
| `like_weight` | 2.0 | Loss weight for like task (binary, higher due to imbalance) |
| `dislike_weight` | 2.0 | Loss weight for dislike task (binary, higher due to imbalance) |
| `use_focal_loss` | False | Use focal loss for like/dislike tasks |

## Artifacts for Chapter 8 (Value Functions)

The MMoE training script exports two types of artifacts for Chapter 8:

### Option A: Full Inference Pipeline (Recommended)

For realistic production-like code, load the model and run inference:

```
outputs/mmoe_30d_4exp/
├── best_model.pt                    # Model checkpoint
└── inference/
    ├── feature_processor.pkl        # Encoders + scaler
    ├── polars_pipeline/             # Feature engineering pipeline
    └── inference_example.py         # Ready-to-run example script
```

**Example: Run inference on new data**

```python
# In Chapter 8, run the provided inference script:
python inference/inference_example.py \
    --model_dir outputs/mmoe_30d_6exp \
    --data_dir /path/to/yambda/flat

# Or use the functions directly:
from inference.inference_example import (
    load_inference_artifacts,
    transform_for_inference,
    predict_with_value_function,
)

# Load trained model
model, config, processor_state, pipeline = load_inference_artifacts(model_dir)

# Transform new data
sparse_data, dense_data = transform_for_inference(new_df, processor_state)

# Predict with custom value function weights (5 tasks)
results = predict_with_value_function(
    model, sparse_data, dense_data,
    weights={
        'engagement': 1.0,
        'completion': 2.0,
        'play_ratio': 1.0,  # Regression: predicted listen percentage
        'like': 5.0,
        'dislike': -10.0  # Negative! Penalize dislikes
    }
)
```

### Option B: Pre-computed Predictions (Quick Analysis)

For rapid iteration on value function weights without re-running inference:

| File | Description | Usage |
|------|-------------|-------|
| `predictions.parquet` | Test set predictions | Value function optimization |
| `calibration.json` | Calibration curves | Reliability diagrams |
| `value_function_config.json` | Task stats & suggested weights | Starting point |
| `gate_weights_sample.parquet` | Expert routing | Interpretability |

**predictions.parquet Schema (5 tasks: 4 binary + 1 regression):**

```python
columns = [
    'uid', 'item_id', 'timestamp',
    # True labels (note: label_play_ratio is continuous [0,1])
    'label_engagement', 'label_completion', 'label_like', 'label_dislike',
    # Predicted probabilities
    'prob_engagement', 'prob_completion', 'prob_like', 'prob_dislike',
    # Raw logits (for temperature scaling / calibration)
    'logit_engagement', 'logit_completion', 'logit_like', 'logit_dislike',
]
```

**Example: Quick value function analysis**

```python
import pandas as pd

# Load pre-computed predictions
df = pd.read_parquet('outputs/mmoe_30d_6exp/predictions.parquet')

# Experiment with different value function weights
experiments = [
    {
        'name': 'equal', 
        'engagement': 1.0, 'completion': 1.0, 'like': 1.0, 'dislike': -1.0
    },
    {
        'name': 'engagement_focused', 
        'engagement': 1.0, 'completion': 2.0, 'like': 5.0, 'dislike': -10.0
    },
    {
        'name': 'satisfaction_focused', 
        'engagement': 0.5, 'completion': 1.0, 'like': 10.0, 'dislike': -20.0
    },
]

for exp in experiments:
    df[f"value_{exp['name']}"] = (
        exp['engagement'] * df['prob_engagement'] +
        exp['completion'] * df['prob_completion'] + 
        exp['like'] * df['prob_like'] +
        exp['dislike'] * df['prob_dislike']  # Negative weight!
    )
    
# Compare rankings - note how dislike penalty affects ordering
print(df[['uid', 'item_id', 'prob_dislike', 'value_equal', 'value_satisfaction_focused']].head(10))
```

### Calibration Analysis

```python
import json
import matplotlib.pyplot as plt

with open('outputs/mmoe_30d_6exp/calibration.json') as f:
    cal_data = json.load(f)

# Plot reliability diagrams for 4 binary tasks (play_ratio is regression)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, task in zip(axes.flat, ['engagement', 'completion', 'like', 'dislike']):
# Note: play_ratio is a regression task - use MSE/MAE for evaluation instead
    ax.plot(cal_data[task]['mean_predicted'], 
            cal_data[task]['fraction_positives'], 
            'o-', label='Model')
    ax.plot([0, 1], [0, 1], '--', label='Perfect')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Positive Rate')
    ax.set_title(f'{task.title()} Task Calibration')
    ax.legend()
plt.tight_layout()
```

## Expected Results

| Model | Train Days | Completion AUC | Like AUC | Dislike AUC | Notes |
|-------|------------|----------------|----------|-------------|-------|
| DCN-V2 | 30 | ~0.56-0.58 | - | - | Single-task |
| DLRM | 30 | ~0.55-0.57 | - | - | Single-task |
| MMoE | 30 | ~0.55-0.57 | ~0.65-0.70 | ~0.60-0.65 | 4-task |
| DeepFM (Ch6) | 30 | ~0.55-0.58 | - | - | Baseline |
| XGBoost (Ch6) | 30 | ~0.58-0.60 | - | - | Feature engineering |

Note: Results depend on hardware, random seeds, and hyperparameters. Engagement AUC is typically high (>0.95) since almost all listens have played_ratio_pct > 0 in the dataset.

## Comparison with Chapter 6

| Aspect | Chapter 6 | Chapter 7 |
|--------|-----------|-----------|
| **Models** | XGBoost, DeepFM | DCN-V2, DLRM, MMoE |
| **Tasks** | Single (completion) | Single + 4-task Multi-task |
| **MTL Tasks** | - | Engagement, Completion, Like, Dislike |
| **Feature Interactions** | FM (2nd order) | Cross Network, Dot Products |
| **Industry Use** | Baseline, Production | State-of-the-art |
| **Complexity** | Moderate | Higher |

## References

1. **DCN V2**: Wang et al. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (WWW 2021)
2. **DLRM**: Naumov et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (Facebook, 2019)
3. **MMoE**: Ma et al. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (KDD 2018)
4. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
5. **Yambda Dataset**: [HuggingFace - yandex/yambda](https://huggingface.co/datasets/yandex/yambda)

