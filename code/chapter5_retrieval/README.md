# Chapter 5: Retrieval

This chapter demonstrates **two-tower retrieval models** for e-commerce recommendations, progressing from simple to sophisticated approaches.

## Scripts Overview

| Script | Description | Key Learning |
|--------|-------------|--------------|
| `similar_items_retrieval.py` | Non-personalized "items like this" | Embedding similarity, FAISS indexing |
| `personalized_retrieval.py` | Unsupervised user-item retrieval | Mean pooling vs last-item aggregation |
| `supervised_two_tower.py` | Learned two-tower model | Attention, Transformers, contrastive learning |
| `test_supervised_two_tower.py` | Unit tests for supervised model | Testing neural network components |

---

## 1. Similar Items Retrieval (Non-Personalized)

**Use Case**: "Items similar to this" carousel on product pages.

```
Hero Item → [Item Embedding] → FAISS Search → Top-K Similar Items
```

**Key findings from ablation study**:
- Description embeddings alone: **Best performance**
- Image embeddings alone: Lower performance
- Combined (concatenated): No improvement over description-only

---

## 2. Personalized Retrieval (Unsupervised)

**Use Case**: "Recommended for you" based on browsing history.

```
User History [item1, item2, ..., itemN] → Aggregate → User Embedding → FAISS Search → Top-K Items
```

**Aggregation methods compared**:
| Method | Hit Rate@10 | Description |
|--------|-------------|-------------|
| Mean Pooling | ~11% | Average of all history embeddings |
| Last Item | ~15% | Only use most recent item |

**Key finding**: The "last item" is surprisingly hard to beat—recent intent matters most.

---

## 3. Supervised Two-Tower Model

**Use Case**: Learn optimal sequence encoding for next-item prediction.

```
User History → [User Tower (Attention/Transformer)] → User Embedding
                                                            ↓
Item Catalog → [Item Tower (Projection)] → Item Embeddings → Dot Product → Ranking
```

**Architecture options**:
- **Attention Pooling**: Learns which history items matter most
- **Transformer Encoder**: Full self-attention with [CLS] token aggregation

**Training**: In-batch contrastive loss with softmax cross-entropy

**Key finding**: Despite architectural complexity, supervised models struggled to beat the simple "last item" baseline on this dataset.

---

## Dataset: SIGIR E-commerce (Coveo)

- **Product catalog**: ~48K products with pre-computed description & image embeddings
- **User sessions**: ~36M browsing events
- **Actions**: detail, add, purchase, remove

## Evaluation Setup

**Ground truth**: Next-item prediction
- Input: First N-1 items in session (history)
- Target: Nth item (ground truth)

**Metrics**:
- **Hit Rate@K**: Was ground truth in top-K?
- **MRR@K**: Reciprocal rank of ground truth

**Data split**: Time-based (train on older sessions, test on newer)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run similar items (non-personalized)
python similar_items_retrieval.py

# Run personalized retrieval (unsupervised baselines)
python personalized_retrieval.py

# Run supervised two-tower model
python supervised_two_tower.py

# Run unit tests
python test_supervised_two_tower.py
```

## Configuration

Key parameters in each script:

```python
# Data paths
DATA_PATH = Path("path/to/SIGIR-ecom-data-challenge/train")

# Model settings
EMBEDDING_DIM = 50          # Pre-computed embedding dimension
MAX_SEQ_LEN = 20            # Maximum history length
USER_TOWER_TYPE = 'attention'  # or 'transformer'

# Evaluation
TOP_K_RETRIEVAL = [5, 10, 20, 50]
GROUND_TRUTH_ACTIONS = ['detail', 'add', 'purchase']
```

---

## Key Takeaways for the Book

1. **Simple baselines are powerful**: The "last item" heuristic achieved 15.6% Hit Rate@10, beating all learned approaches.

2. **More parameters ≠ better**: Transformer user encoder (more params) performed similarly to simple attention pooling.

3. **Pre-trained embeddings matter**: When embeddings already capture semantic relationships, learned aggregation may add noise rather than signal.

4. **Evaluation rigor**: Time-based splits are best practice, though for independent sessions with frozen embeddings, the impact is minimal.

5. **Always include baselines**: Random (~0%), Popularity (~0.5%), and unsupervised methods provide essential context for interpreting model performance.

---

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [SIGIR E-commerce Data Challenge](https://github.com/coveooss/SIGIR-ecom-data-challenge)
- [Two-Tower Models (Google)](https://research.google/pubs/pub48840/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
