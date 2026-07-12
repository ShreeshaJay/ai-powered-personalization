# Design Note: GR2-Style Recommendation Reranking

## Purpose

This exercise demonstrates how a GR2-style generative reranker can be framed as the final stage of a recommender-system funnel:

```text
candidate retrieval -> fixed slate -> listwise generative reranker -> final order
```

It is a tutorial prototype, not a benchmark claim. The goal is to make the design, failure modes, and evaluation tradeoffs visible to readers.

## Dataset

Primary dataset:

- [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Category: `Beauty_and_Personal_Care`
- Review file: `raw/review_categories/Beauty_and_Personal_Care.jsonl`
- Metadata file: `raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl`

The earlier `All_Beauty` category was useful for smoke tests, but too small after 5-core filtering. `Beauty_and_Personal_Care` keeps the same broad domain while giving a more useful tutorial scale.

## Task

For each user:

1. Sort interactions chronologically by timestamp.
2. Use the prefix as user history.
3. Hold out the last item as the target.
4. Retrieve a top-K candidate slate.
5. Rerank the slate.

Default candidate slate size:

```text
K = 10
```

The split follows a TIGER/Diffusion-GR2-style leave-last-out protocol.

## Milestone 1: Retrieval and Baselines

Milestone 1 does not use LLMs. It validates the data pipeline, retrieval stack, and baseline metrics.

### Retriever 1: SASRec

SASRec is used as a behavioral sequential recommender. It encodes a user's item history and scores candidate items by dot product against learned item embeddings.

In the experimental workbench, SASRec initially scores the full catalog per eval user. That is simple and transparent but slow for larger catalogs, so BPC runs should use eval caps or a batched scoring implementation.

### Retriever 2: SBERT + FAISS

SBERT encodes item metadata text such as title, category, features, and description. A user's query vector is derived from their history item embeddings, then FAISS retrieves nearest item texts.

This retriever tests whether item text similarity alone provides useful candidate slates.

### Retriever 3: Popularity

Popularity retrieves train-visible items by interaction frequency, excluding items in the user's history.

It is a simple sanity baseline.

## Evaluation Modes

### Natural Slate

The retriever returns K candidates as-is.

Natural-slate metrics measure candidate-generation quality:

- target-in-slate rate
- Recall@K
- NDCG@K
- MRR

If the target is absent from the natural slate, no reranker can recover it.

### Oracle Slate

If the target is missing, replace the last candidate with the held-out target.

Oracle slates isolate the reranking question:

> Given a fixed slate that contains the relevant item, can the reranker move it upward?

This is common for reranking research because retrieval recall and ordering quality are different questions.

## Milestone 2: Zero-Shot Generative Reranking

Milestone 2 will prompt an instruction-tuned LLM, such as Qwen, with:

- user history
- candidate item IDs
- compact item metadata
- optional retriever scores

The model returns a permutation of candidate IDs. The evaluation should report both ranking metrics and output-validity metrics.

Required validity metrics:

- valid JSON rate
- valid permutation rate
- duplicate candidate rate
- missing candidate rate
- out-of-candidate ID rate
- parse failure rate
- latency and token usage

Initial Milestone 2 should use oracle slates from SASRec and SBERT/FAISS so the task is truly reranking.

## Roadmap After Milestone 1

The current work starts with a pragmatic retrieval and reranking scaffold. It does not yet reproduce the full GR2 or Diffusion-GR2 architecture. The planned progression is:

### M2: Zero-Shot Listwise Reranking

Use Qwen on corrected oracle slates from SASRec and SBERT/FAISS. The goal is to test the mechanics of listwise generative reranking before training:

- prompt construction
- candidate permutation output
- output parser and validator
- valid-output metrics
- NDCG/MRR on oracle slates

This milestone demonstrates GR2-style reranking behavior, not full end-to-end recommendation quality.

### M3: Teacher Reasoning Traces

Generate a small, budgeted set of teacher traces with a stronger API model. Each trace should contain:

- user history summary
- candidate-level evidence
- concise ranking rationale
- final candidate permutation
- model/provider/prompt provenance

This stage creates supervised data for a student model while keeping cost bounded.

### M4: Student SFT

Fine-tune a Qwen or Gemma student on the teacher traces. Compare:

- zero-shot Qwen
- SFT Qwen/Gemma
- non-LLM baselines on the same slates

Use the same parser and validity metrics as M2 so format reliability remains visible.

### M5: Semantic IDs

Add item Semantic IDs using an RQ-VAE/TIGER-style tokenizer. This is the first step toward a closer GR2/TIGER-style architecture.

Questions to test:

- Do Semantic IDs compact item representation?
- Do they improve ranking behavior versus plain metadata prompts?
- Do they help with training stability or generalization?

### M6: RL Reward Design

Document the reward-design layer from GR2-style systems:

- ranking reward from NDCG/MRR or related listwise metrics
- format reward for valid permutations
- anti-degeneracy checks, such as penalizing blindly preserving input order
- held-out validation to prevent prompt/reward overfitting

Implementation is optional. The minimum reader-facing deliverable is a clear design note explaining how verifiable rewards would be constructed.

### Stretch: Diffusion-GR2 Context

Diffusion-GR2-style block diffusion or parallel decoding is out of scope for the initial tutorial. It can be discussed as architecture context after M2-M6 are stable, but should not block the practical exercise.

These milestones should only proceed once retrieval, slate construction, parser validation, and baseline evaluation are stable.

## Leakage Controls

Milestone 1 should enforce:

- user histories exclude held-out targets
- popularity counts use train prefixes only
- SASRec trains on train prefixes only
- item mappings and catalog filters are logged
- test labels are not included in LLM prompts
- generated outputs are stored outside the public repo

Known limitation:

The leave-last-out split is per-user, not a global temporal split. Cross-user temporal leakage can remain because future interactions by one user may affect train-time item popularity for another user. This is documented rather than hidden.

## Artifact Policy

Do not commit large generated files to GitHub. Store them in an artifact repository such as Hugging Face or Google Drive.

Candidate large artifacts:

- `split_bundle.json`
- candidate slate parquet/JSONL files
- trained SASRec checkpoints
- Qwen reranking outputs
- metrics bundles
- model cards and dataset cards for generated artifacts

The public GitHub folder should remain reader-facing and reproducible: docs, scripts, configs, and clean notebooks only.

## Current Caveats

The first capped BPC run showed natural Recall@10 below 1% for SASRec, SBERT/FAISS, and popularity. This suggests the retrieval baseline is weak at K=10 and should not be over-interpreted.

For the GR2 reranking tutorial, the immediate next requirement is corrected oracle-slate metrics. Natural retrieval can be improved separately through larger K, stronger sequential models, batched SASRec scoring, or a two-stage candidate approach.
