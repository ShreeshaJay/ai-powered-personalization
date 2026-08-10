## Chapter 10: Item Embeddings - Design Notes

### Section 10.1: Pre-trained Text Encoder Baselines

**Date:** February 14, 2026  
**Status:** Complete ✅

---

## Overview

Section 10.1 demonstrates how to generate item embeddings using pre-trained text encoders (zero-shot, no training required). This establishes a strong baseline that can be deployed immediately and serves as a comparison point for behavioral embeddings (Section 10.2) and fine-tuned models (Section 10.3).

**Key Learning Objectives:**
1. Encode items with rich text metadata (MIND news articles)
2. Evaluate embedding quality via category retrieval, co-click retrieval, and qualitative NN inspection
3. Compare general-purpose SBERT vs. domain-specialized RexBERT (with mean-pooling wrapper)
4. Understand why zero-shot encoding fails on anonymized/hashed features (IJCAI homework)

---

## Code Execution Flowcharts

### Section 10.1: Zero-Shot Text Encoders

**MIND Evaluation** (`evaluate_mind.py`):

```
main()
 |
 +--[--compare_models?]----> compare_models()
 |   no                       |  for each model in [MiniLM, mpnet]:
 |                            |    run_full_evaluation() ----+
 v                            |  print summary DataFrame     |
 run_full_evaluation()  <-----+------------------------------+
 |
 +-- MINDDataset(data_dir)
 |     .load_news()               <-- data/mind_dataset.py
 |     .load_behaviors()
 |     .get_item_texts(template)
 |
 +-- create_encoder(model_name)   <-- models/text_encoders.py
 |     returns TextEncoder or RexBERTEncoder
 |
 +-- encoder.encode_dict(texts)
 |     .encode() -> SentenceTransformer.encode()
 |     returns Dict[news_id -> embedding]
 |
 +-- np.savez_compressed(...)     <-- cache embeddings .npz
 |
 +-- Eval 1: evaluate_category_retrieval()
 |     sim_matrix = embeddings @ embeddings.T
 |     np.argpartition -> top-K neighbors
 |     evaluate_ranking_batch()   <-- utils/metrics.py
 |     print_ranking_metrics()
 |
 +-- Eval 2: build_coclick_ground_truth()
 |            evaluate_coclick_retrieval()
 |              same pattern: sim_matrix -> top-K -> metrics
 |
 +-- Eval 3: print_qualitative_neighbors()
 |     one article per category, top-5 NNs
 |
 +-- Save JSON results
```

**Amazon KDD Evaluation** (`evaluate_amazon.py`):

```
main()
 |
 +-- AmazonKDDDataset(data_dir, locale="UK")
 |     .load_products(chunk_size)     <-- data/amazon_kdd_dataset.py
 |     .load_sessions(product_ids)
 |     .get_next_item_pairs()         <-- temporal ground truth
 |
 +--[--compare_models?]----> compare_models()
 |   no                       |  for each model in [MiniLM, mpnet, RexBERT]:
 |                            |    run_full_evaluation() ----+
 v                            |  print comparison table      |
 run_full_evaluation()  <-----+------------------------------+
 |
 +-- dataset.get_item_texts(template)
 |
 +-- load_or_encode(model, texts)
 |     check cache .npz
 |     if miss: create_encoder() -> encode_dict() -> save .npz
 |     returns (embeddings, item_ids)
 |
 +-- Eval 1: evaluate_next_item_retrieval()
 |     build_faiss_index(embeddings)    <-- utils/faiss_index.py
 |       FaissIndexWrapper(IndexFlatIP)
 |     faiss_index.search(queries, K)
 |     evaluate_ranking_batch()         <-- utils/metrics.py
 |     print_ranking_metrics()
 |
 +-- Eval 2: print_qualitative_neighbors()
 |     FAISS batch search, print examples
 |
 +-- Save JSON results
```

### Section 10.2: Item2Vec

**Training** (`item2vec.py`):

```
main()
 |
 +--[--dataset=amazon?]
 |   |
 |   +-- load_amazon_sessions()
 |   |     AmazonKDDDataset.load_products()
 |   |     AmazonKDDDataset.load_sessions()
 |   |     AmazonKDDDataset.get_sessions_as_sequences()
 |   |     returns List[List[str]]
 |   |
 |   +-- train_item2vec(sessions, config)
 |   |     resolve_min_count(config, sessions)
 |   |       compute_min_count() -- percentile of freq dist
 |   |     gensim.Word2Vec(sentences, sg=1, negative=10, ...)
 |   |     save model .model + embeddings .npz + meta .json
 |   |
 |   +-- print_model_summary()
 |         model.wv.most_similar() -- sample NNs
 |
 +--[--dataset=yandex?]
     |
     +-- load_yandex_sessions()
     |     YandexDataset(data_dir)       <-- data/yandex_dataset.py
     |     dataset.build_sequences(mode) -- "full_history" or "session"
     |
     +-- train_item2vec(sessions, config)
     +-- print_model_summary()
```

**Evaluation** (`evaluate_item2vec.py`):

```
main()
 |
 +--[--dataset=amazon]
 |   |
 |   +-- AmazonKDDDataset -> load_products, load_sessions
 |   |   dataset.get_next_item_pairs()   <-- ground truth
 |   |
 |   +--[--item2vec_only?]
 |   |   yes: load_item2vec_embeddings() -> next_item_retrieval()
 |   |   no:  run_comparison()
 |   |         |
 |   |         +-- load_item2vec_embeddings("amazon_item2vec")
 |   |         +-- load_text_embeddings(zero-shot tag)
 |   |         +-- load_text_embeddings(fine-tuned tag)  [if exists]
 |   |         |
 |   |         +-- Eval 1: Full catalog
 |   |         |   for each model:
 |   |         |     next_item_retrieval(embeds, ids, ground_truth)
 |   |         |       build_faiss_index()    <-- utils/faiss_index.py
 |   |         |       faiss search -> evaluate_ranking_batch()
 |   |         |
 |   |         +-- Eval 2: Common subset
 |   |         |   intersect item sets -> restrict_to=common_items
 |   |         |   next_item_retrieval(... restrict_to)
 |   |         |
 |   |         +-- print_comparison_table()
 |   |         +-- Save CSV + JSON
 |   |
 +--[--dataset=yandex]
     |
     +-- evaluate_yandex_item2vec()
           load embeddings, build ground truth
           next_item_retrieval() -> metrics
           optionally compare vs audio CNN embeddings
```

### Section 10.3: Contrastive Fine-Tuning

**Training** (`finetune_contrastive.py`):

```
main()
 |
 +--[--dataset=amazon?]
 |   build_amazon_training_pairs(config)
 |     AmazonKDDDataset -> get_item_texts, get_next_item_pairs
 |     pairs = [(query_text, next_item_text), ...]
 |     split train/val by train_fraction
 |
 +--[--dataset=mind?]
 |   build_mind_training_pairs(config)
 |     MINDDataset -> get_item_texts, get_coclick_pairs
 |     pairs = [(article_a_text, article_b_text), ...]
 |     split train/val
 |
 +-- train(dataset, config, output_dir)
       |
       +-- SentenceTransformer(base_model)
       +-- AdamW optimizer + LinearLR warmup + GradScaler (FP16)
       |
       +-- for epoch in 1..N:
       |     shuffle train_pairs
       |     for batch in train_pairs:
       |       tokenize_batch(anchors)
       |       tokenize_batch(positives)
       |       model(anchor_enc) -> anchor_emb (L2-norm)
       |       model(positive_enc) -> positive_emb (L2-norm)
       |       mnr_loss(anchor_emb, positive_emb)
       |         cosine_sim * 20.0 -> cross_entropy(diagonal)
       |       backward, clip_grad, optimizer.step
       |     evaluate_val(model, val_pairs)
       |       encode pairs -> mean cosine sim
       |     save best model if improved
       |
       +-- save final model + training_meta.json
```

**Evaluation** (`evaluate_finetuned.py`):

```
main()
 |
 +-- find_finetuned_model(dataset, base_model)
 |     looks in MODELS_DIR/contrastive_finetuned/
 |
 +--[--dataset=amazon]
 |   compare_amazon()
 |     AmazonKDDDataset -> get_item_texts, get_next_item_pairs
 |     |
 |     +-- Zero-shot:
 |     |   encode_items(base_model, texts, cache_tag)
 |     |     check cache -> create_encoder -> encode_dict -> save
 |     |   evaluate_amazon(embeds, ids, ground_truth)
 |     |     build_faiss_index -> search -> evaluate_ranking_batch
 |     |
 |     +-- Fine-tuned:
 |     |   encode_items(finetuned_path, texts, cache_tag)
 |     |     SentenceTransformer(path) -> encode_dict -> save
 |     |   evaluate_amazon(embeds, ids, ground_truth)
 |     |
 |     +-- print delta table, save CSV + JSON
 |
 +--[--dataset=mind]
     compare_mind()
       MINDDataset -> get_item_texts, build coclick map
       |
       +-- for model in [zero-shot, fine-tuned]:
       |     encode_items(model, texts, cache_tag)
       |     evaluate_mind_category()
       |       FAISS search -> same-category relevance
       |     evaluate_mind_coclick()
       |       FAISS search -> co-click relevance
       |
       +-- print delta table, save CSV + JSON
```

### Section 10.4: Multi-Modal Content Fusion

**Training** (`multimodal_fusion.py`):

```
main()
 |
 +-- Music4allDataset(data_dir, config)    <-- data/music4all_dataset.py
 |     .load_audio()      -- id_ivec256.tsv.bz2       (100-dim)
 |     .load_lyrics()     -- id_lyrics_word2vec.tsv.bz2 (300-dim)
 |     .load_genre()      -- id_genres_tf-idf.tsv.bz2  (685-dim)
 |     .align_modalities() -- intersect track IDs, align arrays
 |
 +--[--mode=pca or both]
 |   pca_fusion(dataset, n_components=128)
 |     concat [audio|lyrics|genre] -> (N, 1085)
 |     PCA(128).fit_transform -> (N, 128)
 |     L2-normalize
 |     save pca_fused.npz
 |
 +--[--mode=clip or both]
 |   train_clip(dataset, config)
 |     90/10 train/val split
 |     CLIPFusionModel:
 |       audio_proj: 100 -> 256 -> 128  (ProjectionHead)
 |       lyrics_proj: 300 -> 256 -> 128 (ProjectionHead)
 |       learnable temperature
 |     for epoch in 1..20:
 |       model(audio, lyrics) -> InfoNCE loss
 |         sim = (a @ l.T) / temperature
 |         loss = (CE(sim, labels) + CE(sim.T, labels)) / 2
 |       optimizer.step, scheduler.step
 |       validate, track best
 |     save clip_fusion_model.pt + metadata .json
 |   encode_clip(model, dataset)
 |     batch project all tracks through audio_proj, lyrics_proj
 |     fused = avg(audio_proj, lyrics_proj), re-normalize
 |     save clip_fused.npz
 |
 +-- Save single-modality .npz: audio_raw, lyrics_raw, genre_raw
```

**Evaluation** (`evaluate_multimodal.py`):

```
main()
 |
 +-- Music4allDataset(data_dir, config)
 |     load_audio, load_lyrics, load_genre, align_modalities
 |     .load_interactions()           -- userid_trackid_count.tsv.bz2
 |     .get_colisten_pairs(max_users) -- co-listen ground truth
 |       group by user -> user_tracks
 |       for each user: all pairs of their tracks = positives
 |       returns [(query_track, {positive_tracks}), ...]
 |
 +-- for method in [audio, lyrics, genre, pca, clip]:
 |     load_embeddings(method, emb_dir)
 |       maps to {method}_raw.npz / pca_fused.npz / clip_fused.npz
 |       returns (ids, embeddings)
 |     colisten_retrieval(ids, embeds, pairs, k_values)
 |       build FAISS IndexFlatIP (or brute-force fallback)
 |       search(query_embeds, max_k)
 |       for each query: precision, recall, ndcg, mrr
 |       average across queries
 |
 +-- print_results_table(all_results)
 |     Method | Dim | P@1 | P@5 | P@10 | MRR | Queries
 |
 +-- Save JSON results
```

### Appendix: Superlinked Multi-Modal Integration

**Full Demo** (`appendix_superlinked/superlinked_demo.py`):

```
main()
 |
 +-- prepare_mind_data(mind_cfg)
 |     MINDDataset(data_dir)            <-- data/mind_dataset.py
 |       .load_news()
 |       .load_behaviors()
 |     extract unique categories (16), subcategories (~200)
 |     build article dicts: {id, body, category, subcategory}
 |     .get_coclick_pairs(min_support=2) -> co-click ground truth
 |     build coclick_map: article_id -> {co-clicked IDs}
 |     returns (articles, lookup, valid_queries, cats, subcats)
 |
 +-- build_superlinked_app(articles, cats, subcats, model)
 |     [Superlinked DSL]
 |     Schema:  NewsArticle(id, body, category, subcategory)
 |     Spaces:  TextSimilaritySpace(body, "all-MiniLM-L6-v2")
 |              CategoricalSimilaritySpace(category, 16 labels)
 |              CategoricalSimilaritySpace(subcategory, ~200 labels)
 |     Index:   sl.Index([text, cat, subcat])
 |     Query:   .find(article)
 |              .similar(text, Param("query_text"))
 |              .similar(cat, Param("query_category"))
 |              .similar(subcat, Param("query_subcategory"))
 |              weights={text: Param("w_text"), ...}
 |     Executor: InMemoryExecutor -> app = executor.run()
 |     Ingest:  source.put(batch) x N batches of 500
 |     returns (app, query_template, schema)
 |
 +-- run_all_evaluations(app, query, ...)
 |     sample query_ids (up to max_queries)
 |     for preset in [text_only, balanced, cat_heavy, subcat_heavy]:
 |       evaluate_preset(app, query, query_ids, weights)
 |         for each query article:
 |           app.query(query_template,
 |             query_text=body, query_category=cat,
 |             query_subcategory=subcat,
 |             w_text=w, w_cat=w, w_subcat=w, limit=K+1)
 |           sl.PandasConverter.to_pandas(result)
 |           exclude self-match
 |           compute precision, recall, ndcg, mrr
 |         average across queries
 |       store results[preset] = metrics
 |
 +-- print_comparison_table()
 |     Preset | w_text | w_cat | w_sub | P@1 | MRR | ...
 |
 +-- run_qualitative_examples()    [unless --skip-qualitative]
 |     pick 1 article per category (up to 3)
 |     for each: query with text_only vs balanced weights
 |     print top-5 neighbors with category labels
 |
 +-- Save JSON results
 +-- Print summary: best preset vs text_only delta
```

---

## Dataset Selection Rationale

### MIND (Primary for Rich Text)

**Why MIND?**
- **Rich free-text metadata:** Title, abstract, category, subcategory, entity annotations
- **Multiple modalities:** Text + knowledge graph entities (WikiData embeddings provided)
- **Real-world structure:** Mirrors actual news recommendation systems
- **Co-click signal:** User click histories provide positive pairs for Section 10.3 (contrastive learning)

**Limitations:**
- Limited to news domain
- No explicit search queries (can't demonstrate query-item bi-encoder fully)

### IJCAI CVR (Homework Exercise)

**Why demoted to homework?**
The IJCAI dataset uses hashed/anonymized category and property IDs — 19-digit
numbers like `7908382889764677758` instead of human-readable text like "clothing".
Sub-word tokenizers split these into digit fragments (`"790"`, `"877"`, `"688"`...)
that are common across many different IDs. As a result, zero-shot text encoders
produce near-identical embeddings for all items (cosine similarity 0.94-0.96 for
*everything*), yielding all-zero retrieval metrics.

This is a valuable pedagogical exercise: students run the evaluation, observe the
failure, and examine diagnostic output to understand WHY it happens.

**Key takeaway:** Zero-shot text encoders require semantically meaningful text.
Hashed IDs, even from the same category, look like random number sequences to the
model. This motivates behavioral embeddings (Section 10.2) and fine-tuning
(Section 10.3) for datasets without human-readable text.

---

## Architecture Decisions

### 1. Text Encoder Choice: Sentence-BERT

**Primary Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Pros:** Fast (384-dim), good quality, widely used baseline
- **Cons:** Lower quality than larger models

**Comparison Model:** `sentence-transformers/all-mpnet-base-v2`
- **Pros:** Higher quality (768-dim), better for semantic similarity
- **Cons:** 2x slower, 2x memory

**Domain-Specialized Model:** `thebajajra/RexBERT-base` (100M params)
- **Pros:** Trained on 2.3T+ e-commerce tokens (Ecom-niverse corpus), domain-adapted 
  to retail/shopping text. Outperforms larger general-purpose encoders on e-commerce 
  benchmarks despite 2-3x fewer parameters. [arXiv 2602.04605]
- **Cons:** Fill-Mask model, NOT a sentence-transformer out of the box. Requires 
  wrapping with a mean-pooling layer to produce sentence embeddings.
- **Why include it:** IJCAI is Taobao e-commerce data — this is exactly RexBERT's 
  domain. Comparing a domain-specialized encoder against general-purpose SBERT models
  is a key teaching point: does domain pre-training help for structured metadata?

Available sizes: micro (17M), mini (68M), base (100M), large (400M).
We use `RexBERT-base` for comparable size to `all-mpnet-base-v2`.

### 2. Text Serialization Strategy (IJCAI)

**Template Design:**
```python
"Category: {categories} | Properties: {properties} | Brand: {brand} | Price: {price_level} | Sales: {sales_level} | City: {city}"
```

**Design Choices:**
1. **Hierarchical categories:** "cat0 > cat1 > cat2" format preserves structure
2. **Descriptive level names:** Map numeric levels (0-6) to text ("low", "medium", "high") for better token semantics
3. **Delimiter choice:** Use "|" to separate major fields, "," for lists
4. **Shop features (optional):** Can append shop reputation metrics if desired

**Alternatives Considered:**
- JSON format: Too verbose, wastes token budget
- Natural language: "This is a medium-priced dress with cotton material..." - more fluent but harder to generate programmatically at scale

### 3. MIND Evaluation Design (Primary, evaluate_mind.py)

The MIND evaluation uses three complementary lenses:

**Evaluation 1 — Category Retrieval:**
- For each article, retrieve K nearest neighbors in embedding space
- Measure what fraction share the same category (or subcategory)
- Intrinsic evaluation: good embeddings should cluster articles by topic
- No behavioral data needed — purely tests embedding geometry

**Evaluation 2 — Co-Click Retrieval:**
- From `behaviors.tsv`, extract articles co-clicked by the same user
- For each article, check if co-clicked partners appear in top-K neighbors
- Tests whether embedding similarity aligns with user interest signals
- Ground truth comes from real user behavior

**Evaluation 3 — Qualitative Nearest Neighbors:**
- Pick one article per major category, print its top-5 neighbors
- Human-inspectable check: do neighbors look topically related?
- Complements quantitative metrics with interpretability

**Metrics:** Precision@K, Recall@K, NDCG@K for K ∈ {1, 5, 10, 20, 50}, MRR

### 3b. IJCAI Bi-Encoder Evaluation (Homework, evaluate_biencoder.py)

Kept as a teaching exercise. See the "IJCAI CVR (Homework Exercise)" section
above for why this produces near-zero results.

---

## Technical Implementation Details

### 1. Embedding Normalization

**Decision:** Always L2-normalize embeddings

**Rationale:**
- For normalized vectors: cosine similarity = dot product (more efficient)
- FAISS IndexFlatIP (inner product) works directly
- Prevents magnitude dominance in similarity computation

```python
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
```

### 2. FAISS vs. Brute-Force Search

**When to use FAISS:**
- Item catalog > 10K items
- Query set > 1K queries
- Production deployment

**When brute-force is fine:**
- Small datasets (< 10K items)
- One-time evaluation
- Debugging

**Implementation:**
```python
if use_faiss and faiss_available:
    index = build_faiss_index(item_embeddings, index_type="IndexFlatIP")
    distances, indices = index.search(query_embeddings, k=50)
else:
    # Fallback: brute-force
    similarities = query_embeddings @ item_embeddings.T
    indices = np.argsort(-similarities, axis=1)[:, :50]
```

### 3. Sparse Relevance Handling

**Challenge:** Most query-item pairs have no interaction in the data.

**Solution:** Only evaluate on queries that have at least one relevant item in the dataset.

```python
query_relevances = defaultdict(dict)
for query_idx, item_idx, relevance in relevance_data:
    query_relevances[query_idx][item_idx] = relevance

# Filter queries with at least one relevant item
valid_queries = [q for q in query_relevances if any(query_relevances[q].values())]
```

This prevents artificially low metrics from queries with zero relevant items in the catalog.

### 4. Batch Encoding for Memory Efficiency

**Challenge:** Encoding large datasets (100K+ items) can exceed GPU memory.

**Solution:** Batch encoding with progress tracking.

```python
encoder = TextEncoder(model_name="...", batch_size=64)
embeddings = encoder.encode(texts, batch_size=64, show_progress=True)
```

sentence-transformers handles batching internally.

---

## Actual Results — Consolidated

### Section 10.1: Zero-Shot Text Encoders (Amazon KDD, 20K queries, ~500K products)

| Model | Params | Dim | P@1 | P@5 | MRR | R@20 | R@50 |
|-------|--------|-----|-----|-----|-----|------|------|
| all-MiniLM-L6-v2 | 22M | 384 | 0.222 | 0.108 | 0.309 | 0.499 | 0.619 |
| all-mpnet-base-v2 | 109M | 768 | ~0.22 | ~0.11 | ~0.31 | ~0.50 | ~0.62 |
| RexBERT-base (mean-pooled) | 100M | 768 | 0.010 | 0.003 | 0.013 | 0.011 | 0.017 |

MiniLM matches mpnet despite 5x fewer parameters. RexBERT (a fill-mask model,
not a sentence encoder) produces near-random results — confirms that domain
MLM pre-training alone does not yield useful sentence embeddings.

### Section 10.2: Item2Vec (Amazon KDD, 20K queries)

| Model | Dim | Coverage | P@1 | MRR | R@20 | R@50 |
|-------|-----|----------|-----|-----|------|------|
| Item2Vec (behavioral) | 128 | 494K | 0.188 | 0.247 | 0.364 | 0.468 |
| Zero-shot (MiniLM) | 384 | 500K | 0.222 | 0.309 | 0.499 | 0.619 |

Item2Vec trails zero-shot text by ~6 points MRR. See Section 10.2 design
notes below for detailed analysis.

### Section 10.3: Contrastive Fine-Tuning (Amazon KDD, 20K queries)

| Model | Dim | P@1 | MRR | R@50 | vs Zero-Shot |
|-------|-----|-----|-----|------|-------------|
| Zero-shot (MiniLM) | 384 | 0.222 | 0.309 | 0.619 | baseline |
| Fine-tuned (MNR) | 384 | 0.224 | 0.319 | 0.668 | +1 MRR, +4.9 R@50 |

### 3-Way Comparison (All sections combined)

| Model | Signal | P@1 | MRR | R@50 |
|-------|--------|-----|-----|------|
| Item2Vec (10.2) | Behavior only | 0.188 | 0.247 | 0.468 |
| Zero-shot SBERT (10.1) | Content only | 0.222 | 0.309 | 0.619 |
| Fine-tuned SBERT (10.3) | Content + Behavior | **0.224** | **0.319** | **0.668** |

The progression validates the chapter's pedagogical arc: each section builds
on the last.  Content alone is strong; behavior alone is weaker on this
metadata-rich dataset; combining both (fine-tuning) yields the best results.

### Section 10.4: Multi-Modal Fusion (Music4all, 5K queries, 109K tracks)

| Method | Dim | P@1 | P@10 | MRR |
|--------|-----|-----|------|-----|
| **Genre (TF-IDF)** | 685 | **0.219** | **0.197** | **0.358** |
| PCA fusion (audio+lyrics+genre) | 128 | 0.191 | 0.140 | 0.318 |
| Audio (i-vectors) | 100 | 0.136 | 0.095 | 0.243 |
| CLIP fusion (audio ↔ lyrics) | 128 | 0.109 | 0.096 | 0.228 |
| Lyrics (Word2Vec) | 300 | 0.066 | 0.060 | 0.153 |

Genre alone outperforms all fusion approaches. See Section 10.4 design notes
for detailed analysis of why naive multi-modal fusion can dilute strong signals.

### IJCAI Bi-Encoder (Homework Exercise)

Produces near-zero metrics across all models due to hashed/anonymized IDs.
See the "IJCAI CVR (Homework Exercise)" section above for explanation.

---

## Gotchas and Troubleshooting

### 1. Missing Abstracts in MIND

**Problem:** Some news articles have no abstract (empty string).

**Solution:** Fill with empty string and let the model handle it. The template still works:
```
"Title here [SEP]  [SEP] category subcategory"
```

### 2. IJCAI Category List Parsing

**Problem:** Category lists are semicolon-separated, but some have inconsistent formatting.

**Solution:** Robust parsing with error handling:
```python
if pd.notna(categories):
    categories = " > ".join(str(categories).split(";"))
else:
    categories = "unknown"
```

### 3. FAISS Installation Issues

**CPU vs. GPU:**
- `faiss-cpu`: Easy install via pip, no CUDA required
- `faiss-gpu`: Requires CUDA, conda install recommended

**Fallback:** If FAISS not available, use brute-force search (provided in `faiss_index.py`).

### 4. Memory Issues with Large Datasets

**Symptom:** OOM when encoding 100K+ items on GPU.

**Solution 1:** Reduce batch size
```python
encoder.batch_size = 32  # Default is 64
```

**Solution 2:** Use CPU (slower but more memory)
```python
encoder = TextEncoder(device="cpu")
```

**Solution 3:** Encode in chunks and save incrementally
```python
for chunk in chunks(item_texts, chunk_size=10000):
    chunk_embeddings = encoder.encode(chunk)
    np.save(f"chunk_{i}.npy", chunk_embeddings)
```

### 5. Incorrect Similarity Metric

**Problem:** Using L2 distance on normalized embeddings instead of inner product.

**Why it matters:** For normalized vectors:
- Inner product = cosine similarity (✅ correct)
- L2 distance ≠ cosine similarity (❌ wrong ranking)

**Solution:** Always use `IndexFlatIP` (inner product) for normalized embeddings.

---

## Code Quality and Testing

### Unit Tests

Each module has standalone testing:
```bash
python data/mind_dataset.py        # Test data loading
python models/text_encoders.py     # Test encoding
python utils/metrics.py            # Test metric computation
python utils/faiss_index.py        # Test FAISS indexing
```

### Integration Test

End-to-end pipeline:
```bash
# 1. Encode items
python encode_items.py --dataset mind --sample_size 1000

# 2. Evaluate bi-encoder
python evaluate_biencoder.py --dataset ijcai --sample_size 5000
```

### Configuration Validation

`config.py` includes path validation:
```python
if __name__ == "__main__":
    print(f"MIND data exists: {MIND_DATA_PATH.exists()}")
    print(f"IJCAI data exists: {IJCAI_DATA_PATH.exists()}")
```

---

## Lessons Learned

### 1. Structured Metadata Serialization is Common
In production, most item catalogs have structured fields, not free-text descriptions. Text serialization is a practical, widely-used technique. Don't over-engineer the template — simple works.

### 2. Zero-Shot Embeddings are Surprisingly Good
P@1 of ~22% across a 500K product catalog with no task-specific training is a strong baseline that can be deployed immediately. This sets a high bar for any learned approach.

### 3. Domain-Specific Pre-training ≠ Better Sentence Embeddings
RexBERT was pre-trained on 2.3T+ e-commerce tokens but underperforms general-purpose SBERT models on next-item retrieval. The reason: RexBERT is a fill-mask model that produces token-level representations, not sentence embeddings. Mean-pooling over token outputs is a weak proxy for semantic similarity. Purpose-built sentence encoders (trained with contrastive objectives) consistently outperform.

### 4. Smaller Models Can Match Larger Ones
MiniLM (384-dim, 22M params) matches mpnet (768-dim, 109M params) on both MIND and Amazon KDD. For production systems where latency and storage matter, this is a significant practical finding.

### 5. Next-Item Retrieval is a Clean Evaluation
Using temporal ordering within sessions (last viewed item → next engaged item) provides a realistic, directional evaluation that avoids the symmetry assumptions of co-engagement pair approaches.

### 6. Behavioral Embeddings Are Dataset-Dependent
Item2Vec underperforms zero-shot text on Amazon KDD (rich metadata), but this result is specific to metadata-rich datasets. On datasets with poor/hashed metadata (like IJCAI), Item2Vec would be the only viable approach. The relative value of content vs. behavioral embeddings depends heavily on the quality of available text.

### 7. Session Sparsity Limits Shallow Models
The median Amazon KDD item appears in only 5 sessions. Each item generates ~20-50 Skip-gram training examples. Compare this to SBERT's pre-training on billions of text examples. When behavioral signal is sparse, a model that starts from scratch (Item2Vec) cannot match one that leverages massive pre-trained knowledge (SBERT).

### 8. Content and Behavior Capture Different Signals
Text embeddings capture semantic similarity ("blue fleece blanket" ≈ "grey sherpa throw"). Behavioral embeddings capture complementary associations ("phone case" ≈ "screen protector"). Neither subsumes the other. The 3-way comparison shows that fine-tuning (content + behavior) beats both pure approaches, motivating the fusion strategies in Chapters 10.4, 10.5, and 11.

### 9. More Modalities ≠ Better Embeddings
On Music4all, genre TF-IDF alone (MRR 0.358) beats every fusion method — PCA (0.318) and CLIP (0.228). Adding weaker modalities (audio, lyrics) dilutes the strongest one. Always evaluate single modalities first before investing in fusion.

### 10. Naive Fusion Methods Can Hurt Performance
PCA over concatenated features loses 11% MRR compared to genre alone. The dimensionality reduction mixes strong genre signals with irrelevant variation from audio and lyrics. Task-aware fusion (learned weights or attention) is needed to avoid this pitfall — motivating Chapter 11.

### 11. CLIP-Style Alignment Serves Cross-Modal Retrieval, Not Same-Modal
CLIP training aligns audio and lyrics into a shared space, which is useful for cross-modal queries ("find songs that sound like these lyrics"). But for track-to-track co-listen retrieval, the original single-modality features (especially genre) are more directly predictive. Match the fusion method to the retrieval task.

### 12. Multi-Modal Fusion = Weighted Concatenation (Including Hybrid Search)
Combining any heterogeneous embeddings — dense text (384d), sparse BM25 (50K+d), categorical one-hots, numeric features — reduces to the same operation: L2-normalize each sub-vector, weight them, concatenate, and dot-product. This is exactly what production "hybrid search" systems (Vespa, Weaviate, Pinecone) do under the hood. PCA fusion (10.4) destroys per-modality weight control; keeping sub-vectors separate preserves it. No special library needed — `np.concatenate([w1 * emb1, w2 * emb2])` is the complete implementation.

---

## Section 10.2: Item2Vec — Design Decisions

### What Is Item2Vec?

Item2Vec (Barkan & Koenigstein, 2016) applies the Word2Vec algorithm to
recommendation data. The core analogy: **sessions are sentences, items are
words**. Just as Word2Vec learns word embeddings from textual co-occurrence,
Item2Vec learns item embeddings from behavioral co-occurrence in user sessions.

### How the Learning Process Works

Item2Vec is a **shallow neural network** — a single hidden-layer model with
two weight matrices and no activation functions:

1. **Input embeddings (W):** shape `[vocab_size × embedding_dim]` — lookup
   table mapping each item ID to a dense vector
2. **Output embeddings (W'):** same shape — used only during training for
   the negative sampling objective

**Skip-gram training procedure:**

For each item in a session, the model creates (center_item, context_item)
pairs from a sliding window. For a session `[A, B, C, D, E]` with window=5,
item C generates pairs: (C,A), (C,B), (C,D), (C,E).

For each pair:
1. Look up the center item's vector from W
2. Look up the context item's vector from W'
3. Compute the dot product → should be high (positive pair)
4. Sample `k` random items as negatives (we use k=10), compute their dot
   products with the center → should be low
5. Update both matrices via SGD to push the positive dot product up and
   negative dot products down

The loss is binary cross-entropy on each sigmoid(dot product) — pushing
positive pairs toward 1 and negative pairs toward 0. This **negative
sampling** avoids the expensive full softmax over all ~500K items.

After training, W' is discarded. Each row of W becomes an item's embedding.

**Why this works despite being shallow:**

The expressive power comes from the **distributional hypothesis** — items
are defined by the company they keep. If items X and Y appear in similar
sessions (surrounded by similar other items), their embeddings converge
even if X and Y never co-occurred directly. The embedding for an item is
effectively a compressed summary of "what items typically surround me in
sessions."

### Algorithm Choice: Word2Vec Skip-gram

We use `gensim.Word2Vec` with Skip-gram and negative sampling, exactly as
described in the original Item2Vec paper.

**Why Skip-gram over CBOW?**
- Skip-gram predicts context from center; CBOW predicts center from context
- Skip-gram typically produces better representations for rare items
- Skip-gram is the standard choice in the Item2Vec literature

### Hyperparameter Defaults

- `embedding_dim=128`: smaller than SBERT (384) — collaborative embeddings
  need less capacity since they only encode co-occurrence patterns, not
  rich semantics
- `window_size=5`: captures items within 5 positions in a sequence
- `min_count`: derived from the **5th percentile of the item frequency
  distribution** rather than an arbitrary constant. This adapts to the data.
  Can be overridden with `--min-count` for manual control.
- `negative=10`: number of negative samples per positive pair
- `epochs=10`: standard for medium-sized corpora

### Comparison to Contrastive Fine-Tuning (Section 10.3)

| Aspect | Item2Vec (10.2) | MNR Fine-tuning (10.3) |
|--------|----------------|----------------------|
| Architecture | 2 lookup matrices (~126M params) | 22M-param BERT transformer |
| Input | Item IDs only (no metadata) | Item text strings |
| What's learned | Embedding lookup table from scratch | Adjustments to pre-trained text encoder |
| Training signal | Session co-occurrence | Session co-occurrence (same data!) |
| Cold-start items | No embedding (item must be in vocab) | Always has embedding (from text) |
| Training time | ~5 min on CPU | ~30 min on GPU |

Both methods learn from the **same behavioral signal** (session co-occurrence),
but Item2Vec learns a lookup table from scratch while MNR fine-tuning adjusts
a pre-trained text encoder. The 3-way evaluation reveals what each approach
captures that the other misses.

### Training Data Design

**Amazon KDD sessions:** Each session `[prev_items..., next_item]` is treated
as a "sentence". Items co-occurring within the same session are treated as
co-occurring "words". The skip-gram window determines how far apart two items
can be while still being considered co-occurring. Sessions are pre-defined by
the dataset (each row is a session), so no arbitrary splitting is needed.

**Yandex Yambda:** The default is `sequence_mode="full_history"` — each user's
complete chronological listening history forms one "sentence". The `window_size`
parameter naturally limits how far apart two tracks can be while co-occurring.
This follows the original Item2Vec paper and avoids introducing an arbitrary
session-gap threshold. An alternative `sequence_mode="session"` splits by
30-minute gaps, which is appropriate when sessions represent distinct user
intents but adds a tunable hyperparameter. Only listens where ≥ 50% of the
track was played count (filters accidental plays / skips).

### Evaluation Strategy: Two Comparison Modes

A key challenge with Item2Vec is that it only produces embeddings for items
appearing ≥ `min_count` times. Text-based models cover 100% of the catalog.

We report two evaluation modes:

1. **Full catalog:** Each model evaluated on all items it covers.
   - Text models have a natural advantage: they cover long-tail items
   - Item2Vec only covers items with sufficient behavioral signal
   - This reflects the real production trade-off: coverage vs. quality

2. **Common subset:** All models restricted to the Item2Vec vocabulary.
   - Apples-to-apples comparison on the same items

### Actual Results (Amazon KDD)

**Training:** 1,182,181 sessions → 6.05M item occurrences (avg 5.1 items/session).
494,409 unique items in vocabulary. 128-dim, Skip-gram, 10 epochs, 339 seconds on CPU.

**Item frequency distribution (from training data):**
- p5=1, p25=3, p50=5, p75=11, p95=45
- p5 = 1 → `min_count=1` → 100% of items retained (494,409 / 494,409)

**3-Way Comparison — Full Catalog (20K queries, ~500K products):**

| Model | Dim | Coverage | P@1 | P@5 | MRR | R@20 | R@50 |
|-------|-----|----------|-----|-----|-----|------|------|
| Item2Vec (behavioral) | 128 | 494K | 0.188 | 0.086 | 0.247 | 0.364 | 0.468 |
| Zero-shot (all-MiniLM-L6-v2) | 384 | 500K | 0.222 | 0.108 | 0.309 | 0.499 | 0.619 |
| Fine-tuned (MNR) | 384 | 500K | **0.224** | **0.112** | **0.319** | **0.527** | **0.668** |

**Common Subset (494K items shared across all models):**

| Model | P@1 | P@5 | MRR | R@20 | R@50 |
|-------|-----|-----|-----|------|------|
| Item2Vec (behavioral) | 0.188 | 0.086 | 0.247 | 0.364 | 0.468 |
| Zero-shot (all-MiniLM-L6-v2) | 0.222 | 0.108 | 0.309 | 0.499 | 0.619 |
| Fine-tuned (MNR) | **0.224** | **0.112** | **0.319** | **0.527** | **0.668** |

### Analysis of Results

**1. Text embeddings outperform Item2Vec across all metrics.**

This was *not* the expected outcome. We hypothesized Item2Vec would excel
on the common subset since it learns directly from behavioral co-occurrence,
which is exactly what next-item retrieval measures. Instead, text models
beat Item2Vec by 6-7 points on MRR and ~20 points on R@50.

**2. Why text embeddings win — the "rich metadata" effect:**

Amazon KDD products have exceptionally detailed metadata: titles, brands,
descriptions, colors, materials. When a user views a "blue fleece blanket"
and then a "grey sherpa throw", the text encoder recognizes these as similar
products from their descriptions alone. Item2Vec can only learn this
association if these specific items co-occurred in enough sessions.

**3. Full catalog ≈ Common subset — the dense dataset effect:**

Because the Amazon KDD dataset is dense (even the p5 item appears once),
Item2Vec's vocabulary covers 494K of 500K products. The ~6K missing items
are products that never appeared in any session. This means the "coverage
advantage" of text models is minimal here — the full catalog and common
subset results are essentially identical.

**4. Item2Vec's relative weakness comes from session sparsity, not model capacity:**

The median item appears in only 5 sessions. With window=5 and avg session
length 5.1, each item generates ~20-50 training examples. Compare this to
SBERT which was pre-trained on billions of text examples. The behavioral
signal is simply thinner than the pre-trained text knowledge.

**5. The fine-tuned model (10.3) combines the best of both worlds:**

Fine-tuning takes a strong text baseline and injects behavioral signal on
top. The result (MRR 0.319) beats both pure text (0.309) and pure behavior
(0.247). This validates the chapter's progression: content → behavior →
content+behavior.

### Pedagogical Takeaways

1. **Pre-trained text encoders are a surprisingly strong baseline** — even
   without any domain-specific training, SBERT's general text understanding
   captures product similarity well enough to beat a model trained explicitly
   on behavioral data.

2. **Item2Vec shines when text metadata is absent or poor.** The IJCAI
   dataset (hashed IDs) is a case where Item2Vec would dominate, since text
   encoders produce meaningless embeddings. The Amazon KDD result is
   *specific to datasets with rich text metadata*.

3. **Session sparsity limits Item2Vec.** With more interaction data (e.g.,
   millions of sessions per item as in production systems), Item2Vec's
   performance would be substantially stronger.

4. **Content and behavior are complementary.** Even though text dominates
   here, Item2Vec captures patterns that text cannot — e.g., "people who
   bought phone cases also bought screen protectors" is a behavioral
   association that no amount of text similarity would surface. Chapter 11
   will explore how to fuse both signals.

---

## Section 10.3: Contrastive Fine-Tuning — Design Decisions

### Why MultipleNegativesRankingLoss (MNR)?

MNR is the standard loss for fine-tuning sentence encoders because:
1. **Sample efficient:** A batch of B pairs yields B*(B-1) negative comparisons for free
2. **No explicit negative mining needed:** In-batch negatives are diverse by construction
3. **Same loss used by Sentence-BERT:** So the model architecture is already optimized for it
4. **Scales to large datasets:** 500K+ pairs train in minutes on a single GPU

**MNR vs Triplet Loss — detailed comparison:**

| Aspect | MNR | Triplet Loss |
|--------|-----|-------------|
| Input | (anchor, positive) pairs | (anchor, positive, negative) triplets |
| Negatives per anchor | B-1 (in-batch) | 1 (or small fixed count) |
| Effective signal / batch | B*(B-1) comparisons | B comparisons |
| Negative quality | Diverse (random batch) | Depends on mining strategy |
| Engineering overhead | Minimal | Need negative mining pipeline |
| Hyperparameters | Batch size, LR | Margin, mining strategy, LR |
| Failure modes | Rare | Margin too large → no gradient; too small → weak signal |

With batch_size=128, MNR produces 128*127 = 16,256 contrastive comparisons
per batch.  Triplet loss with the same batch produces 128.  This is why MNR
converges faster and typically reaches better quality with less tuning.

MNR is mathematically equivalent to InfoNCE/NT-Xent (the loss behind SimCLR,
CLIP, DPR) — the dominant paradigm in modern contrastive learning.  Triplet
loss can outperform when hard negatives are very carefully curated, but that
requires a mature embedding pipeline to begin with (chicken-and-egg problem).

Other alternatives considered:
- **InfoNCE:** Equivalent to MNR with explicit temperature scaling
- **Cosine Similarity Loss:** Less discriminative, no in-batch negative sharing

### Training Data Design

**Amazon KDD (primary):**
- Positive pairs: (last viewed item → next engaged item) from session data
- Temporal ordering is preserved within sessions — no data leakage
- Up to 500K pairs (configurable cap to control training time)
- 90/10 train/val split on the flattened pair list

**MIND (secondary):**
- Positive pairs: articles co-clicked by the same user (min_support=2)
- Up to 200K pairs
- Demonstrates the technique generalizes beyond e-commerce

### Evaluation Strategy

The key comparison is **zero-shot vs fine-tuned on the exact same items**:
- Both use the same item text representations
- Both are evaluated with the same FAISS-based retrieval pipeline
- Both use the same query set (fixed random seed)

This eliminates confounders — any improvement is directly attributable to
the contrastive training objective.

### Actual Results (Amazon KDD)

Training: 450K pairs, 3 epochs, batch_size=128, MNR loss, fp16 on GPU.
Best checkpoint at step 3500 (end of epoch 1), mild overfitting in epochs 2-3.

| Metric | Zero-Shot | Fine-Tuned | Delta |
|--------|-----------|-----------|-------|
| P@1 | 0.222 | 0.224 | +0.002 |
| MRR | 0.309 | 0.319 | +0.010 |
| R@50 | 0.619 | 0.668 | +0.049 |

Improvements are consistent but modest.  Why:
1. Zero-shot SBERT is already a strong baseline on well-formed product text
2. Each item appears in ~1-2 training pairs — sparse behavioral signal
3. Next-item pairs capture browsing intent, not semantic equivalence
4. Loss decreased steadily (1.61 → 1.47 → 1.41) but val sim peaked at epoch 1

**Lesson for practitioners:** Contrastive fine-tuning always helps, but the
magnitude depends on (a) how strong the base model already is and (b) how
much task-specific data you have.  Production systems with billions of
interactions see much larger gains.

---

## Section 10.4: Multi-Modal Fusion — Design Decisions

### Motivation

Sections 10.1–10.3 work exclusively with text. Real-world items are richer:
music has audio, video, lyrics, and genre metadata. Section 10.4 explores
whether combining multiple content modalities into a single embedding
improves item similarity — specifically for music co-listen prediction.

**Key constraint:** This section uses only *content* features (no behavioral
data for training). Combining content + behavior is deferred to Chapter 11.

### Dataset: Music4all-Onion

109,269 tracks with pre-extracted features from multiple modalities:

| Modality | File | Dimensions | Signal Type |
|----------|------|------------|-------------|
| Audio (i-vectors) | `id_ivec256.tsv.bz2` | 100 | Timbral / acoustic patterns |
| Lyrics (Word2Vec) | `id_lyrics_word2vec.tsv.bz2` | 300 | Semantic content of lyrics |
| Genre (TF-IDF) | `id_genres_tf-idf.tsv.bz2` | 685 | Categorical metadata |

User interactions: 50M listening records from 119K users (`userid_trackid_count`).
Evaluation uses co-listen retrieval: do nearest neighbors in embedding space
overlap with tracks listened to by the same users?

### Approach A: PCA Fusion (Baseline)

Concatenate audio (100) + lyrics (300) + genre (685) = 1,085 dims, then
apply PCA to 128 dims. No learning — purely geometric dimensionality reduction.
Retained **90.8% of variance** in 128 components.

**Why PCA?** It's the simplest fusion method that a practitioner would try
first. It establishes whether the raw concatenation already captures useful
signal, before investing in more complex learned approaches.

### Approach B: CLIP-Style Contrastive Alignment

Inspired by CLIP (Radford et al., 2021), we train projection heads to align
audio and lyrics representations in a shared 128-dim space.

**Architecture:**
- Audio projection: Linear(100 → 256) → GELU → Dropout → Linear(256 → 128) → L2-norm
- Lyrics projection: Linear(300 → 256) → GELU → Dropout → Linear(256 → 128) → L2-norm
- Total parameters: **168,705**

**Training signal (InfoNCE):**

For a batch of N tracks, the positive pair is (audio_i, lyrics_i) — the
audio and lyrics of the *same song*. The (N-1) cross-modal pairs are negatives.
The loss is symmetric:

```
loss = (CrossEntropy(audio @ lyrics.T / τ, labels) +
        CrossEntropy(lyrics @ audio.T / τ, labels)) / 2
```

where τ (temperature) is learnable (initialized at 0.07, learned to 0.088).

**What this learns:** The model discovers which audio patterns correspond to
which lyrical content. For example, acoustic guitar recordings might cluster
near lyrics about nature/folk themes; heavy bass/synths near electronic/dance
lyrics. The fused embedding is the average of both projections, re-normalized.

**Training details:**
- 109K tracks: 90% train / 10% val
- 20 epochs, batch_size=512, AdamW, cosine LR schedule
- Best val_loss at epoch 7 (5.727), mild overfitting after
- Total training time: **55 seconds on GPU**

**Loss analysis:** Random baseline for InfoNCE with B=512 is -log(1/512) ≈ 6.24.
Best val loss = 5.73, meaning the model correctly identifies the matching
pair among 512 candidates significantly above chance, but the cross-modal
gap between audio and lyrics is inherently difficult.

### Actual Results (Music4all, Co-Listen Retrieval)

5,000 queries, 10,000 sampled users, 109,269 tracks, avg 3,852 positives/query.

| Method | Dim | P@1 | P@5 | P@10 | MRR |
|--------|-----|-----|-----|------|-----|
| **Genre (TF-IDF)** | 685 | **0.219** | **0.210** | **0.197** | **0.358** |
| PCA fusion | 128 | 0.191 | 0.154 | 0.140 | 0.318 |
| Audio (i-vectors) | 100 | 0.136 | 0.108 | 0.095 | 0.243 |
| CLIP fusion | 128 | 0.109 | 0.103 | 0.096 | 0.228 |
| Lyrics (Word2Vec) | 300 | 0.066 | 0.060 | 0.060 | 0.153 |

### Analysis of Results

**1. Genre alone outperforms all fusion methods.**

MRR 0.358 — the clear winner. This is because co-listening is primarily
a genre-level phenomenon: if you listen to jazz, your co-listened tracks
are overwhelmingly other jazz tracks. Genre TF-IDF captures this directly
with a 685-dimensional one-hot-like representation over fine-grained
genre labels (e.g., "jazz trumpet", "funeral doom", "celtic metal").

**2. PCA fusion actually *hurts* compared to genre alone.**

MRR drops from 0.358 → 0.318. When PCA compresses all three modalities,
the 685-dim genre space dominates the variance, but the weaker audio (100)
and lyrics (300) dimensions add noise. The "explained variance" of 90.8%
sounds good, but the retained components mix genre signal with acoustically
irrelevant variation. This is a textbook case of **signal dilution**.

**3. Audio captures within-genre similarity.**

MRR 0.243 — second-best single modality. I-vectors encode timbral properties
(instruments, production style, energy). Two jazz tracks may sound very
different, but i-vectors capture acoustic similarity that partially
correlates with co-listening. However, it's weaker than genre because users
don't choose music by acoustic features alone.

**4. Lyrics is a poor co-listen predictor.**

MRR 0.153 — weakest by far. Lyrical similarity (word2vec averaged over song
words) has almost no predictive power for co-listening. A jazz ballad about
love and a metal ballad about love share similar lyrics but very different
audiences. Users choose music by sound and genre, not lyrical themes.

**5. CLIP fusion falls between audio and lyrics.**

MRR 0.228. The CLIP model produces a shared audio-lyrics space. By averaging
both projections, it inherits some of audio's strength but gets pulled down
by the lyrics signal. The CLIP loss *does* learn meaningful correspondences
(loss well below random), but those correspondences don't align with
co-listen behavior.

### Pedagogical Takeaways

1. **More modalities ≠ better embeddings.** Naive fusion (PCA, averaging)
   can dilute the strongest modality with noise from weaker ones. Before
   fusing, assess which modalities actually carry signal for your task.

2. **The "right" modality depends on the task.** Genre dominates co-listen
   prediction, but for a *different* task — say, "find songs with similar
   mood" — lyrics sentiment or audio valence might dominate instead.

3. **CLIP-style alignment is most valuable for cross-modal retrieval.**
   Its real power is queries like "find songs that sound like these lyrics"
   — projecting *different* modalities into a shared space for cross-modal
   search. For same-modality retrieval (track → track), the original
   single-modality features are often sufficient.

4. **Task-aware fusion is critical.** A learned fusion that uses behavioral
   signal (Chapter 11) can weight modalities by their relevance to the
   prediction task, unlike PCA which is task-agnostic. This is where
   attention-based and graph-based approaches shine.

5. **Baseline first, always.** Testing single modalities independently
   *before* fusion reveals which modalities carry signal. If we had jumped
   straight to PCA fusion, we'd have reported MRR 0.318 and missed that
   genre alone achieves 0.358.

### The General Principle: Weighted Concatenation and Hybrid Retrieval

Section 10.4's PCA fusion and the CLIP-style alignment are specific
instances of a broader pattern that practitioners encounter constantly:
**combining heterogeneous vector representations into a single retrieval
score.** Understanding this pattern is more valuable than any single library.

**The core operation:**

```
score(query, item) = w1 * sim(query_v1, item_v1) + w2 * sim(query_v2, item_v2) + ...
```

This is mathematically equivalent to concatenating weighted, normalized
sub-vectors and computing a single dot product:

```
query_combined  = [w1 * normalize(v1_query)  | w2 * normalize(v2_query)  | ...]
item_combined   = [normalize(v1_item)        | normalize(v2_item)        | ...]
score           = dot(query_combined, item_combined)
```

**Where this pattern appears in production:**

| System | Modality 1 | Modality 2 | Combination |
|--------|-----------|-----------|-------------|
| **Hybrid search** | Dense (SBERT 384d) | Sparse (BM25/TF-IDF 50K+d) | Weighted sum |
| **E-commerce** | Text embedding | Price + Brand one-hot | Weighted concat |
| **Music rec** | Audio features | Genre one-hot | Weighted concat |
| **News rec** | Article text | Category one-hot | Weighted concat |
| **Ads ranking** | Query-ad embedding | CTR prediction | Weighted sum |

**Hybrid search is the most common production instance.** Dense embeddings
(sentence-transformers) capture semantic meaning ("cheap flights" matches
"budget airfare"), while sparse vectors (BM25) capture exact keyword
matching ("iPhone 15 Pro Max" must match literally). Neither alone is
sufficient — dense misses exact matches, sparse misses synonyms. Systems
like Vespa, Weaviate, Pinecone, and Elasticsearch all offer "hybrid search"
as a first-class feature. Under the hood, it's the same weighted
concatenation.

**Why PCA (Section 10.4) fails as a fusion method:**

PCA compresses all sub-vectors into a single fixed-dimensional space. This:
1. Destroys the ability to weight modalities differently per query
2. Mixes strong signals with noise from weaker modalities
3. Produces a single static embedding — no query-time flexibility

**The better approach: keep sub-vectors separate, weight at query time.**

```python
# Instead of PCA fusion:
pca_embedding = PCA(128).fit_transform(np.concatenate([text, genre, audio], axis=1))

# Do this — preserves per-modality control:
def hybrid_score(query_text, query_genre, item_text, item_genre, w_text, w_genre):
    return (w_text * cosine_sim(query_text, item_text) +
            w_genre * cosine_sim(query_genre, item_genre))
```

This costs more storage (you store the full concatenated vector, not a
128d PCA projection), but gains query-time flexibility. In production
systems with multiple use cases (search, browsing, recommendation), this
trade-off almost always favors keeping sub-vectors separate.

**Connection to Chapter 11:** The weighted concatenation approach uses
*fixed* weights per query. Chapter 11 will explore *learned* weights —
attention mechanisms and graph neural networks that dynamically weight
modalities based on the user, context, and item, producing personalized
fusion rather than static presets.

### Homework Suggestions for Students

- Swap audio features: try MFCC stats, essentia, or ResNet video embeddings
- Add genre as a third modality in CLIP (3-way contrastive alignment)
- Implement a weighted fusion: learn per-modality weights via a small MLP
- Compare with a "late fusion" approach: retrieve top-K from each modality
  separately, then merge the ranked lists (e.g., reciprocal rank fusion)
- **Hybrid search exercise:** For the Amazon KDD dataset, combine SBERT
  dense embeddings (384d) with TF-IDF sparse vectors over product titles.
  Implement the weighted concatenation approach and sweep `w_dense` vs
  `w_sparse` to find the best blend for next-item retrieval. Compare
  against dense-only (10.1) and sparse-only baselines.

### Appendix: Superlinked Multi-Modal Integration

Moved to `appendix_superlinked/` for longevity reasons (startup library,
API may change). The core concept — weighted concatenation of per-modality
vectors with query-time weight tuning — is durable and can be implemented
with ~50 lines of numpy. See `appendix_superlinked/Appendix_Superlinked_Design_Notes.md`.

---

## References

1. **Sentence-BERT:** Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
2. **Dense Passage Retrieval:** Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", EMNLP 2020
3. **MIND Dataset:** Wu et al., "MIND: A Large-scale Dataset for News Recommendation", ACL 2020
4. **Item2Vec:** Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative Filtering", IEEE MLSP 2016
5. **Word2Vec:** Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality", NeurIPS 2013
6. **Negative Sampling:** Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", ICLR Workshop 2013
7. **RexBERT:** Bajaj et al., "RexBERT: Pre-trained Language Models for E-commerce", arXiv:2602.04605, 2026
8. **CLIP:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
9. **InfoNCE:** van den Oord et al., "Representation Learning with Contrastive Predictive Coding", arXiv:1807.03748, 2018
10. **Music4all:** Santana et al., "Music4all-Onion — A Large-Scale Multi-Modal Music Dataset", ACM ICMR 2020
11. **Superlinked:** Open-source framework for multi-modal vector search, https://github.com/superlinked/superlinked (see Appendix)

---

## Change Log

- **2026-02-28:** Superlinked demo moved to `appendix_superlinked/`
  - Startup library — not suitable for main chapter longevity
  - Core concept (weighted concat with query-time weights) documented in Lesson #12
  - Config and script preserved in appendix for interested readers
- **2026-02-28:** Section 10.4 complete — Multi-Modal Fusion on Music4all
  - PCA baseline (audio+lyrics+genre → 128d): MRR 0.318
  - CLIP-style audio ↔ lyrics alignment (128d, 20 epochs): MRR 0.228
  - Genre TF-IDF alone dominates co-listen retrieval: MRR 0.358
  - Key insight: naive multi-modal fusion can dilute the strongest modality
  - `multimodal_fusion.py`: PCA + CLIP training; `evaluate_multimodal.py`: co-listen eval
  - `data/music4all_dataset.py`: Music4all-Onion data loader (3 modalities + interactions)
- **2026-02-25:** Section 10.2 results — Item2Vec trained & evaluated on Amazon KDD
  - 494K items embedded (128-dim) from 1.18M sessions in 339s on CPU
  - 3-way comparison (Item2Vec vs zero-shot vs fine-tuned) on 20K queries
  - Item2Vec MRR=0.247 vs zero-shot 0.309 vs fine-tuned 0.319
  - Key insight: rich product metadata gives text encoders a strong advantage
  - Full catalog ≈ common subset due to high item coverage (p5=1 → min_count=1)
- **2026-02-12:** Section numbering swapped: Item2Vec → 10.2, Contrastive → 10.3
  (ascending complexity: no training → Word2Vec → BERT fine-tuning)
- **2026-02-12:** Section 10.2 implemented — Item2Vec
  - `item2vec.py`: Word2Vec training on Amazon KDD + Yandex sessions
  - `evaluate_item2vec.py`: behavioral vs content comparison + 3-way after 10.3
  - `data/yandex_dataset.py`: Yandex Yambda data loader
  - Two evaluation modes: full catalog and common subset
  - Percentile-based min_count; full_history sequence mode for Yandex
- **2026-02-23:** Section 10.3 complete — Amazon KDD results
  - Fine-tuned MiniLM on 450K next-item pairs (MNR loss, 3 epochs)
  - MRR +1 point, R@50 +4.9 points vs zero-shot baseline
  - Best checkpoint at epoch 1; mild overfitting in epochs 2-3
  - Manual PyTorch training loop (avoids HF Trainer version issues)
- **2026-02-22:** Section 10.3 implemented
  - `finetune_contrastive.py`: MNR loss training on Amazon KDD + MIND
  - `evaluate_finetuned.py`: head-to-head zero-shot vs fine-tuned comparison
  - `ContrastiveFineTuneConfig` in config.py
- **2026-02-22:** Section 10.1 complete — final results
  - Amazon KDD: 500K UK products encoded, 1.18M sessions, next-item retrieval eval
  - MIND: 51K articles, category + co-click retrieval eval
  - Embedding cache (.npz) for instant re-evaluation
  - FAISS-based search (memory-efficient for 500K items on 16GB RAM)
  - RexBERT comparison confirms domain MLM < purpose-built sentence encoders
- **2026-02-14:** Initial implementation of Section 10.1
  - MIND data loader with text generation
  - IJCAI data loader with structured metadata serialization
  - Text encoder wrapper (sentence-transformers + RexBERT)
  - Bi-encoder evaluation with Precision@K, NDCG@K, MRR
  - FAISS integration for fast search

---

## Next Milestone

**Chapter 10 complete.** Four main sections + one appendix:
- 10.1: Zero-shot text encoders (MIND + Amazon KDD)
- 10.2: Item2Vec collaborative embeddings (Amazon KDD + Yandex)
- 10.3: Contrastive fine-tuning (Amazon KDD + MIND)
- 10.4: Multi-modal content fusion (Music4all)
- Appendix: Superlinked integration (MIND) — in `appendix_superlinked/`

**Next:** Chapter 11 — Customer Embeddings
- Aggregate item embeddings into user representations
- Transformer-based sequence models (SASRec, GRU4Rec)
- Graph neural networks for joint user/item embeddings
