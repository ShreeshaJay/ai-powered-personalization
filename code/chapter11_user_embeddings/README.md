# Chapter 11: User/Customer Embeddings

This chapter builds **user embedding representations** from the item embeddings produced in Chapter 10. A user embedding captures individual preferences and can be reused for retrieval, ranking, and personalization.

The fundamental question:

```
user_embedding = f(item_embedding_1, item_embedding_2, ..., item_embedding_N)
```

What should `f` be? Each section gives a progressively more powerful answer.

## Chapter Structure

| Section | Approach | Training? | Key Idea |
|---------|----------|-----------|----------|
| 11.1 | Aggregation baselines | No | A user is the (weighted) sum of their interactions |
| 11.2 | Sequence models (SASRec / GRU4Rec) | Yes | Learn *which* items in the history matter most |
| 11.3 | Graph Neural Networks (LightGCN) | Yes | Learn embeddings from the user-item interaction graph |

- **Section 11.1** compares training-free aggregations over Chapter 10's item embeddings: simple mean, last-K mean, exponential recency decay, TF-IDF weighting, and TF-IDF + recency.
- **Section 11.2** trains SASRec (causal Transformer) and GRU4Rec (stacked GRU) on frozen SBERT item embeddings with MultipleNegativesRankingLoss (in-batch negatives). A separate item-ID variant learns embeddings from scratch for comparison.
- **Section 11.3** trains LightGCN on the MIND user-item click graph (BPR and MNR losses), includes a cold-start evaluation slice, and an SBERT-initialized variant that combines content and collaborative signals.

## Code Structure

```
chapter11_user_embeddings/
├── config.py                     # Paths, model settings, hyperparameters
├── data/
│   ├── amazon_session_loader.py  # Amazon KDD 2023 session loader
│   ├── mind_user_loader.py       # MIND user history loader
│   └── mind_graph_builder.py     # Bipartite graph construction for LightGCN
├── models/
│   ├── aggregators.py            # 11.1 aggregation methods
│   ├── sequence_models.py        # 11.2 SASRec / GRU4Rec (SBERT-feature input)
│   ├── sequence_models_itemid.py # 11.2 item-ID embedding variant
│   └── lightgcn.py               # 11.3 LightGCN
├── utils/
│   ├── metrics.py                # HR@K, NDCG@K, MRR
│   └── faiss_index.py            # FAISS index helpers
├── data_analysis.py              # Dataset statistics
├── evaluate_aggregation.py       # ★ 11.1 evaluation
├── train_sequence.py             # ★ 11.2 training (SBERT features)
├── evaluate_sequence.py          # 11.2 evaluation
├── train_sequence_itemid.py      # 11.2 training (item-ID embeddings)
├── evaluate_sequence_itemid.py   # 11.2 evaluation (item-ID variant)
├── train_lightgcn.py             # ★ 11.3 training
├── evaluate_lightgcn.py          # 11.3 evaluation
├── requirements.txt              # Dependencies
└── docs/
    └── Chapter11_Design_Notes.md # Detailed design decisions and results
```

## Prerequisites

### 1. Datasets

| Dataset | Used For | Download |
|---------|----------|----------|
| Amazon KDD Cup 2023 | Session embeddings (anonymous users) | https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge |
| MIND (small) | User embeddings (persistent user IDs) | https://msnews.github.io/ |

`config.py` expects datasets under a `Dataset/` folder three levels above the chapter directory (see `PROJECT_ROOT` in `config.py`). Update `DATASET_ROOT` and the dataset-specific paths for your environment.

### 2. Chapter 10 Item Embeddings (Sections 11.1 and 11.2)

Sections 11.1 and 11.2 consume pre-computed item embeddings from Chapter 10 (expected in `chapter10_item_embeddings/outputs/embeddings/`). Run Chapter 10's `encode_items.py` first to generate them. Section 11.3 (LightGCN) and the item-ID sequence variant learn their own embeddings and do not require Chapter 10 outputs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Dataset statistics
python data_analysis.py

# Section 11.1: Aggregation baselines (no training, CPU-friendly)
python evaluate_aggregation.py

# Section 11.2: Sequence models
python train_sequence.py
python evaluate_sequence.py

# Section 11.2 variant: item-ID embeddings learned from scratch
python train_sequence_itemid.py
python evaluate_sequence_itemid.py

# Section 11.3: LightGCN on the MIND click graph
python train_lightgcn.py
python evaluate_lightgcn.py
```

All scripts accept command-line options; run any script with `--help` to see them.

## Hardware Expectations

- **Section 11.1** (aggregation) runs on a laptop CPU.
- **Sections 11.2 and 11.3** (training) run on CPU for small samples, but a CUDA GPU (local or Colab) is recommended for full training runs. Full-data item-ID sequence models and LightGCN training are the most compute-intensive parts of the chapter.

## Evaluation

Next-item prediction with leave-last-out splits, reported as HR@K, NDCG@K, and MRR. LightGCN additionally reports a cold-start user slice. See `docs/Chapter11_Design_Notes.md` for the full protocol, hyperparameter grounding, results, and known evaluation biases.

## References

- Kang & McAuley, "Self-Attentive Sequential Recommendation" (SASRec), ICDM 2018
- Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks" (GRU4Rec), ICLR 2016
- He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020
- Henderson et al., "Efficient Natural Language Response Suggestion" (MNR loss), 2017
- MIND: https://msnews.github.io/
- Amazon KDD Cup 2023: https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge
