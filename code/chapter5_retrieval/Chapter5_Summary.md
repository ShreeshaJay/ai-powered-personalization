# Chapter 5: Retrieval - Summary of Experiments and Findings

This document summarizes the experiments, design decisions, and key learnings from building retrieval models for e-commerce recommendations.

---

## 1. Problem Setup

### Use Case: "Items Similar to This" → "Recommended for You"

We progressed through three increasingly sophisticated approaches:

1. **Non-personalized**: Given a "hero" item, find similar items (content-based)
2. **Unsupervised personalized**: Given user history, aggregate item embeddings to form a user representation
3. **Supervised personalized**: Learn how to encode user history for next-item prediction

### Dataset: SIGIR E-commerce (Coveo)

- **Product catalog**: ~48K products with pre-computed description and image embeddings (50-dimensional)
- **User sessions**: ~36M browsing events
- **Actions**: `detail`, `add`, `purchase`, `remove`

---

## 2. Ground Truth Definition

### For Similar Items (Non-personalized)

**Session co-occurrence**: If a user viewed item A, then engaged with items B, C, D in the same session, those items are considered "similar" from the user's perspective.

- Query: First item in session
- Ground truth: Subsequent items in session
- Filtered actions: `detail`, `add`, `purchase` (excluding `remove`)

### For Personalized Retrieval

**Next-item prediction**: Given a user's browsing history (items 1 to N-1), predict the next item (item N).

```
Session: [item1 → item2 → item3 → item4 → item5]
                                    ↓
         history = [1,2,3,4]    ground_truth = [5]
```

**Key insight**: The within-session chronology is built into the task definition itself—we always use past items to predict future items.

---

## 3. Evaluation Methodology

### Metrics

| Metric | Description |
|--------|-------------|
| **Hit Rate@K** | Was the ground truth item in the top-K retrieved? |
| **MRR@K** | Mean Reciprocal Rank—what was the rank of the first relevant item? |
| **Recall@K** | What fraction of relevant items were retrieved? (for multi-item ground truth) |

### Data Splitting: Time-Based vs Random

We explored both approaches:

**Time-based split (recommended)**:
- Train: Oldest 70% of sessions
- Validation: Next 15%
- Test: Most recent 15%

**Random split**:
- Sessions shuffled randomly across splits

**Finding**: For our setup (independent sessions, frozen pre-trained embeddings), the choice of split had minimal impact on relative model rankings. However, time-based splitting is the principled approach because:
1. It simulates realistic deployment (train on past, predict future)
2. It can detect temporal drift in model performance
3. It's good practice for when it *does* matter (e.g., learned embeddings, user-level models)

---

## 4. Baseline Models

### Why Baselines Matter

Baselines provide essential context for interpreting model performance. Without them, it's impossible to know if 10% Hit Rate is good or bad.

### Random Baseline

```python
class RandomBaseline:
    def retrieve_for_user(self, history_items, top_k=10):
        candidates = catalog - set(history_items)
        return random.sample(candidates, k=top_k)
```

**Result**: ~0.02% Hit Rate@10

This is the absolute floor—if any model performs at or below this, something is fundamentally wrong.

### Popularity Baseline

```python
class PopularityBaseline:
    def __init__(self, engagement_counts):
        # engagement = count of (detail + add + purchase) events
        self.sorted_items = sorted(items, key=engagement_counts.get, reverse=True)
    
    def retrieve_for_user(self, history_items, top_k=10):
        return [item for item in self.sorted_items if item not in history_items][:top_k]
```

**Result**: ~0.5-1% Hit Rate@10

Popularity is a surprisingly strong baseline in many recommendation settings. It represents "what's popular globally" without any personalization.

---

## 5. Unsupervised Personalized Retrieval

### Architecture

```
User History [item1, item2, ..., itemN]
            ↓
    [Aggregation Method]
            ↓
      User Embedding
            ↓
   FAISS Nearest Neighbor Search
            ↓
       Top-K Items
```

### Aggregation Methods

**Mean Pooling**:
```python
user_embedding = np.mean([item_embeddings[i] for i in history], axis=0)
```
Treats all history items equally.

**Last Item**:
```python
user_embedding = item_embeddings[history[-1]]
```
Uses only the most recent item—assumes current intent is what matters most.

### Results

| Method | Hit Rate@10 | MRR@10 |
|--------|-------------|--------|
| Mean Pooling | 11.12% | 0.0523 |
| **Last Item** | **13.60%** | **0.0756** |

**Key Finding**: The "last item" approach outperformed mean pooling by 22%. This suggests that for session-based recommendations, **recent behavior is more indicative of current intent** than historical average.

---

## 6. Supervised Two-Tower Model

### Motivation

Can we learn a better way to aggregate user history than simple heuristics?

### Architecture

```
User Tower                          Item Tower
    ↓                                   ↓
[User History Sequence]           [Item Embedding]
    ↓                                   ↓
[Learned Encoder]                 [Projection Layer]
    ↓                                   ↓
User Embedding (50-dim)  ←→  Item Embedding (50-dim)
                    ↓
            Dot Product Similarity
```

### User Tower Options

#### Option 1: Attention Pooling (Simple)

```python
class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, mask):
        # x: (batch, seq_len, embedding_dim)
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
```

This learns a scalar weight for each item based on its embedding, then computes a weighted sum. Each item's weight is computed independently—no item-item interaction.

**Note**: This is NOT the same as Transformer attention. It's simpler "additive attention" or "Bahdanau-style attention" used for sequence pooling.

#### Option 2: Transformer Encoder (Complex)

```python
class TransformerUserTower(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len+1, embedding_dim))
        self.transformer = nn.TransformerEncoder(...)
    
    def forward(self, x, mask):
        # Prepend [CLS] token
        x = torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x, src_key_padding_mask=mask)
        return x[:, 0, :]  # Return [CLS] token output
```

Full self-attention where each item attends to all other items. More expressive but more parameters.

### Training: In-Batch Contrastive Loss

**Negative Sampling Strategy**: In-batch negatives

For a batch of N users:
- Each user has 1 positive item (their ground truth)
- The other N-1 positive items in the batch serve as negatives for each user

```python
# Logits: (batch, batch) matrix
logits = torch.matmul(user_emb, item_emb.T) / temperature

# Labels: diagonal is positive
labels = torch.arange(batch_size)  # [0, 1, 2, ..., N-1]

# Loss: push diagonal similarities high, off-diagonal low
loss = F.cross_entropy(logits, labels)
```

**Why In-Batch Negatives?**
- Free (no extra sampling needed)
- Diverse (other users' positives are realistic items)
- Moderately hard (better signal than random negatives)

### Sequence Padding

Sequences shorter than `max_seq_len` are **padded at the START**, not the end:

```
Original:     [item1, item2, item3]
Padded:       [PAD, PAD, ..., item1, item2, item3]
Mask:         [False, False, ..., True, True, True]
```

**Why pad at start?** The most recent item should always be at a consistent position (the last position), which helps the model learn recency patterns.

### Results

| User Tower | Hit Rate@10 | MRR@10 |
|------------|-------------|--------|
| Attention Pooling | 10.42% | 0.0453 |
| Transformer | 9.88% | 0.0400 |

**Surprising Finding**: Both supervised approaches performed WORSE than the simple "last item" baseline (15.60%).

---

## 7. Full Model Comparison

| Model | Hit Rate@10 | MRR@10 | Lift vs Random |
|-------|-------------|--------|----------------|
| Random | 0.02% | 0.0000 | baseline |
| Popularity | 0.66% | 0.0021 | +3,200% |
| Unsupervised (mean) | 11.58% | 0.0550 | +57,800% |
| **Unsupervised (last)** | **15.60%** | **0.0864** | **+77,900%** |
| Supervised (attention) | 10.42% | 0.0453 | +52,000% |
| Supervised (transformer) | 9.88% | 0.0400 | +49,300% |

---

## 8. Key Takeaways

### 1. Simple Baselines Are Powerful

The "last item" heuristic achieved 15.6% Hit Rate@10, beating all learned approaches. In session-based recommendations, recency is an extremely strong signal.

### 2. More Parameters ≠ Better Performance

The Transformer user encoder (more parameters, full self-attention) performed similarly to simple attention pooling. With limited training data and short sequences, the added complexity doesn't help.

### 3. Pre-trained Embeddings Already Encode Relationships

When item embeddings already capture semantic relationships (from pre-training), learning additional transformations may add noise rather than signal. The supervised model is trying to improve upon embeddings that are already good.

### 4. Evaluation Rigor Matters

- Always include baselines (random, popularity, simple heuristics)
- Time-based splits are best practice for temporal data
- Within-session chronology is what makes the task valid—history → future

### 5. When Would Supervised Models Win?

The supervised approach would likely outperform if:
- Item embeddings are learned end-to-end (not frozen pre-trained)
- User histories are longer (more signal to aggregate)
- There's more training data (neural networks are data-hungry)
- The task involves complex user intent patterns that simple heuristics can't capture

---

## 9. Ablation Study: Embedding Modalities

For the similar items task, we compared:

| Embedding | Hit Rate@10 |
|-----------|-------------|
| Description only | **Best** |
| Image only | Lower |
| Combined (concatenated) | No improvement |

**Finding**: For this e-commerce dataset, text descriptions were more informative than images for finding similar items. Adding image embeddings didn't help—possibly because the image embeddings captured visual similarity that didn't align with user-perceived similarity.

---

## 10. Code Architecture

### Scripts

| Script | Purpose |
|--------|---------|
| `similar_items_retrieval.py` | Non-personalized item-to-item similarity |
| `personalized_retrieval.py` | Unsupervised user-item retrieval with baselines |
| `supervised_two_tower.py` | Learned two-tower model with attention/transformer |
| `test_supervised_two_tower.py` | Unit tests for model components |

### Key Classes

```python
# Item retrieval
class SimilarItemsRetriever:
    def build_index(self, embeddings)  # FAISS IndexFlatIP
    def retrieve(self, query_embedding, top_k)

# Baselines
class RandomBaseline
class PopularityBaseline

# Unsupervised personalized
class UnsupervisedRetriever:  # mean or last aggregation

# Supervised model
class AttentionPooling(nn.Module)
class TransformerUserTower(nn.Module)
class UserTower(nn.Module)
class ItemTower(nn.Module)
class TwoTowerModel(nn.Module)
```

---

## 11. Practical Recommendations

1. **Start with baselines**: Random and popularity establish the floor and ceiling for non-personalized approaches.

2. **Try "last item" first**: For session-based recommendations, this simple heuristic is often hard to beat.

3. **Don't over-engineer early**: Mean pooling of history embeddings is a reasonable starting point that's easy to deploy.

4. **Consider supervised models when**:
   - You have abundant training data
   - User histories are long and complex
   - You can learn embeddings end-to-end
   - Simple heuristics demonstrably fail

5. **Always validate with proper evaluation**:
   - Time-based splits for temporal data
   - Multiple metrics (Hit Rate, MRR, Recall)
   - Statistical significance if comparing models

---

*This summary was generated from experiments conducted on the SIGIR E-commerce dataset for "AI-Powered Personalization" Chapter 5: Retrieval.*


