# Chapter 8: Value Functions and Diversity Optimization

This chapter covers the **Ordering Stage** of a recommender system - transforming raw model scores into a final, polished ranked list.

## Overview

After the Scoring Stage (Chapters 6-7) produces pointwise predictions, the Ordering Stage applies **listwise logic** to:

1. **Blend multiple objectives** into a single value score
2. **Enforce diversity** to avoid monotonous recommendations
3. **Apply business rules** (pacing, slotting, policies)

## Module Structure

```
chapter8_value_functions/
├── __init__.py              # Package exports
├── value_function.py        # Multi-objective blending
├── mmr_diversity.py         # Maximal Marginal Relevance
├── business_rules.py        # Artist pacing, slot allocation
├── pipeline.py              # End-to-end orchestration
├── model_interface.py       # Chapter 7 model contract
├── evaluate.py              # Diversity and ranking metrics
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Prerequisites

### From Chapter 7 (Multi-Task Model)

This chapter requires a multi-task model trained in Chapter 7 with the following outputs:

| Output | Type | Description |
|--------|------|-------------|
| `p_listen` | Tensor (batch,) | P(listen) probability |
| `p_like` | Tensor (batch,) | P(like\|listen) probability |
| `e_engagement` | Tensor (batch,) | E[play_ratio] in [0, 1] |
| `p_dislike` | Tensor (batch,) | P(dislike\|listen) probability (optional) |

**Required artifacts** in `chapter7_advanced_ranking/outputs/`:
- `mtl_model.pt` - Model weights
- `mtl_config.json` - Model architecture config
- `feature_processor.pkl` - Fitted feature encoders

### From Yambda Dataset

- `embeddings.parquet` - Audio embeddings for MMR diversity
- `artist_item_mapping.parquet` - Item to artist mapping
- `album_item_mapping.parquet` - Item to album mapping

## Quick Start

### 1. Value Function (Section 8.1)

```python
from chapter8_value_functions import ValueFunction, ValueFunctionConfig

# Configure weights based on business goals
config = ValueFunctionConfig(
    w_listen=1.0,      # Base engagement
    w_like=2.0,        # Explicit positive feedback (2x weight)
    w_engagement=1.5,  # Engagement depth
    w_dislike_penalty=1.0,
)

vf = ValueFunction(config)

# Blend multi-task model outputs
value_scores = vf.compute_value(
    p_listen=model_outputs['p_listen'],
    p_like=model_outputs['p_like'],
    e_engagement=model_outputs['e_engagement'],
)
```

### 2. MMR Diversity (Section 8.3)

```python
from chapter8_value_functions import MMRReranker, DiversityConfig
from chapter8_value_functions.mmr_diversity import load_yambda_embeddings

# Load audio embeddings
embeddings = load_yambda_embeddings("data/yambda/embeddings.parquet")

# Configure diversity
config = DiversityConfig(
    lambda_param=0.7,  # 0.7 relevance, 0.3 diversity
    top_k=20,
)

reranker = MMRReranker(embeddings, config)

# Rerank with diversity
candidates = list(zip(item_ids, value_scores))
diverse_ranking = reranker.rerank(candidates)
```

### 3. Business Rules (Section 8.4)

```python
from chapter8_value_functions import ArtistPacer, PacingConfig
from chapter8_value_functions.business_rules import load_artist_mapping

# Load mappings
artist_mapping = load_artist_mapping("data/yambda/artist_item_mapping.parquet")

# Configure pacing
config = PacingConfig(max_per_artist=2, max_per_album=1)
pacer = ArtistPacer(artist_mapping, config=config)

# Apply pacing
final_ranking = pacer.apply_pacing(diverse_ranking)
```

### 4. End-to-End Pipeline

```python
from chapter8_value_functions import OrderingPipeline
from chapter8_value_functions.pipeline import PipelineConfig

config = PipelineConfig(
    model_dir="chapter7_advanced_ranking/outputs",
    embeddings_path="data/yambda/embeddings.parquet",
    artist_mapping_path="data/yambda/artist_item_mapping.parquet",
)

pipeline = OrderingPipeline(config)

# Rerank candidates
final_ranking = pipeline.rerank(
    candidate_item_ids=candidates,
    user_features=user_features,
    item_features=item_features,
    top_k=20,
)
```

## Key Concepts

### Value Function Formula

```
Value = w_listen × P(listen) 
      + w_like × P(listen) × P(like|listen)
      + w_engagement × P(listen) × E[play_ratio]
      - w_dislike × P(listen) × P(dislike|listen)
```

The conditional composition (multiplying by P(listen)) ensures downstream signals only contribute when there's a reasonable chance of initial engagement.

### MMR Formula

```
MMR(i) = λ × Relevance(i) - (1-λ) × max_{j ∈ S} Similarity(i, j)
```

- λ = 1.0: Pure relevance (no diversity)
- λ = 0.0: Pure diversity (ignores relevance)
- λ = 0.7: Recommended starting point

### Evaluation Metrics

| Metric | Measures | Higher is Better? |
|--------|----------|-------------------|
| NDCG@k | Ranking quality | Yes |
| ILD | Intra-List Diversity | Yes |
| Artist Entropy | Artist diversity | Yes |
| Coverage | Catalog exploration | Yes |

## Running Examples

```bash
# Test value function
python -m chapter8_value_functions.value_function

# Test MMR diversity
python -m chapter8_value_functions.mmr_diversity

# Test business rules
python -m chapter8_value_functions.business_rules

# Test evaluation metrics
python -m chapter8_value_functions.evaluate
```

## References

- [Shaped.ai - The Anatomy of Modern Ranking Architectures: Part 4](https://shaped.ai/blog/the-anatomy-of-a-modern-ranking-architectures-part-4)
- Carbonell & Goldstein (1998) - "The Use of MMR, Diversity-Based Reranking"
- [Yandex Yambda Dataset](https://huggingface.co/datasets/yandex/yambda)

