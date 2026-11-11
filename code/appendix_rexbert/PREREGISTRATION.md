# Preregistration: Appendix A, RexBERT as a sentence encoder

This file locks the questions, the decision rule, the run matrix, and the protocol constants before any result file exists. It contains no measured metrics. It is committed to the repository before run R3. `run_manifest.json` records the SHA-256 of this file and leaves `preregistration_committed_before_r3` unset until that commit is confirmed.

Reported numbers come from the `gpu_full` profile only. The `cpu_small` profile exercises the same code on a smaller sample. Its metrics are teaching numbers and are not the appendix's results.

## Questions

1. **Q1, replication.** On the Chapter 10 zero-shot protocol, how do all-MiniLM-L6-v2, all-mpnet-base-v2, RexBERT-base, and RexBERT-micro compare? Chapter 10, paragraph 86, states that general-purpose SBERT outperforms domain-specialized RexBERT on retrieval on the Amazon KDD data. The direction check is MiniLM against RexBERT-micro and, on `gpu_full`, against RexBERT-base. Ettin-17m is encoded in the same run as a capacity-matched zero-shot reference. It is not part of the published Chapter 10 comparison.
2. **Q2, conversion.** How much does contrastive fine-tuning change RexBERT-micro relative to its own zero-shot encoder, under the appendix protocol?
3. **Q3, controlled comparison.** Under one fine-tuning recipe, how do fine-tuned RexBERT-micro, fine-tuned Ettin-17m, and fine-tuned all-MiniLM-L6-v2 compare on held-out sessions?

## Decision rule

The primary decision metric is MRR at full rank. Recall@50 is a secondary check and does not select the branch.

For a paired comparison, the code averages the per-query metric across training seeds, then draws 1,000 bootstrap resamples of queries with seed 42. The interval is the 2.5 and 97.5 percentiles of those resample means.

A difference is **supported** only when both of these hold:

- the paired 95% interval of the seed-averaged per-query difference excludes zero, and
- all three training seeds agree in sign, each strictly positive or each strictly negative.

Otherwise the difference is **not supported**. One seed, as in `cpu_small`, cannot support a difference.

Q3 applies the rule to fine-tuned RexBERT-micro minus fine-tuned Ettin-17m.

- Branch (a): the difference is supported and positive.
- Branch (c): the difference is supported and negative.
- Branch (b): every other outcome.

MiniLM's position is descriptive. It does not select the branch. Its prior contrastive training and its width (384 against 256) are part of that description.

Q2 applies the same rule to fine-tuned RexBERT-micro minus its zero-shot encoder. The zero-shot vector is repeated once per training seed so the paired interval is defined.

## Run matrix

| Run | Content | Profile |
|---|---|---|
| R0 | Load the UK catalog and sessions, split, build pairs, print the data card | both |
| R1 | Zero-shot encoding and both retrieval protocols | both, with the model lists below |
| R2 | Learning-rate sweep, one epoch, seed 0 | `gpu_full` only. `cpu_small` records the single pre-registered rate and does not train |
| R3 | Main fine-tuning from the original checkpoint, not from the sweep checkpoint | both |
| R4 | Encode fine-tuned models, retrieval, bootstrap, slices, figures, decision JSON | both |

`gpu_full` models:

- R1 zero-shot: MiniLM, mpnet, RexBERT-base, RexBERT-micro, Ettin-17m.
- R3 fine-tune: RexBERT-micro, Ettin-17m, MiniLM. Seeds 0, 1, 2. Up to 3 epochs. 500,000 pairs.
- R2: those three models at 2e-5, 5e-5, and 1e-4, one epoch, 100,000 pairs, seed 0. The subset is drawn with `RandomState(0)` and the chosen indices are sorted. The selected rate is the highest validation MRR@100. A tie takes the smaller rate. R3 refuses to start on `gpu_full` if this sweep file is missing.

`cpu_small` models: zero-shot MiniLM, RexBERT-micro, and Ettin-17m. Fine-tune RexBERT-micro and Ettin-17m only, seed 0, one epoch, 50,000 pairs, learning rate 5e-5. RexBERT-base and mpnet are not loaded. This profile does not answer Q1 or Q3 for the appendix.

R5, a base-size scale check, is out of scope for this build.

## Protocol constants

| Item | Value |
|---|---|
| Locale | UK |
| Text template | `{title} [SEP] {brand} [SEP] {description}`, then whitespace collapsed |
| Session split | SHA-256 of `{seed}:{filtered_row_index}` with seed 42. The first 8 bytes, as a big-endian integer, modulo 10. Buckets 0-7 train, 8 validation, 9 test |
| Filtered row index | Position, in file order, among UK sessions whose query item and next item are both in the UK catalog |
| Training pairs | (last previous item, next item) from non-degenerate training sessions. Exact duplicate pairs keep the first session. Cap 500,000 on `gpu_full` and 50,000 on `cpu_small`, drawn with `RandomState(42)` and stored in sorted index order |
| Seen pair | The exact (query, next) pair occurs in the deduplicated training set before the cap |
| Degenerate session | Next item equals the query item. Dropped from training pairs and from validation and test queries. Counts are reported by split |
| Warm target | The item appears in any training session, as a previous item or as the next item, including degenerate training sessions. Cold means otherwise |
| Validation | 2,000 validation sessions. Corpus of 50,000 items that includes every validation target. Selection metric MRR@100. The query item is masked so it cannot be a neighbor of itself |
| Test queries, `gpu_full` | Seeded sample of 20,000 test sessions, seed 42, sorted sample indices. The full test split is larger. The cap is fixed here so retrieval against the full UK catalog stays inside the planned R4 window |
| Test queries, `cpu_small` | 2,000. Teaching only |
| Retrieval corpus, `gpu_full` | All UK products. `IndexFlatIP`. The query item is excluded |
| Retrieval corpus, `cpu_small` | 50,000 products that include the sampled test targets, the Chapter 10 queries and their targets, and the validation queries and targets. If that required set exceeds 50,000, the corpus grows and the run records the expansion. Metrics on this corpus are labelled `corpus_is_full_catalog: false` and `reported_protocol: false` |
| Metrics | Recall@10, Recall@50, MRR (full rank), MRR@100, NDCG@10. For one relevant item, Recall@K equals hit rate at K. NDCG uses discount `1/log2(rank+1)` with rank starting at 1. With one relevant item, IDCG is 1, so NDCG@10 is `1/log2(rank+1)` when rank is at most 10, and 0 otherwise |
| Tie policy | If the target is inside the FAISS top 100, its rank is its position in that self-excluded list. Otherwise the rank is 1 plus the count of strictly higher inner products. A startup check compares FAISS top-5 scores with exact inner products at absolute tolerance 1e-3 |
| Max sequence length | 128 |
| Batch size | 128 on `gpu_full`. Each anchor uses the other 127 batch rows as negatives. 64 on `cpu_small`, because the teaching machine has 16 GB of RAM |
| Epochs | Up to 3 on `gpu_full`, checkpoint with the best validation MRR@100. One epoch on `cpu_small` |
| Learning rates | Sweep `{2e-5, 5e-5, 1e-4}` on `gpu_full` as in R2. `cpu_small` uses 5e-5 and does not sweep |
| Other training settings | Warmup 10% of steps, passed as `warmup_steps=0.1` because Transformers 5 retired `warmup_ratio`. Weight decay 0.01. MultipleNegativesRankingLoss scale 20 (temperature 0.05). `BatchSamplers.NO_DUPLICATES`. bf16 on GPU only when CUDA bf16 is supported. `cpu_small` does not request bf16 |
| Seeds | Training seeds 0, 1, and 2 on `gpu_full`. Seed 0 only on `cpu_small`. Pair, query, catalog, and sweep draws each use their own `numpy.random.RandomState`. The duplicate-free batch sampler uses a PyTorch generator seeded with the training seed. Python, NumPy, and PyTorch are seeded per run. The session split does not use that RNG |
| Bootstrap | 1,000 resamples, seed 42, percentiles 2.5 and 97.5, paired for differences |
| Test-session use | Appendix test sessions are used in R1 and R4. They are not used for training or for model selection |

## Chapter 10 protocol (Q1 only)

This path is labelled separately from the appendix protocol. It reproduces the published settings, including their leakage.

- Query is `prev_items[-1]`.
- Targets are aggregated across all sessions, including test sessions.
- A session enters this map when it has at least two previous items and is not degenerate. Chapter 10's minimum session length of 3 is this same cutoff.
- 5,000 queries, drawn with `numpy.random.RandomState(42).choice` without sorting.
- Catalog order is file order. Search is `IndexFlatIP`. The query item is excluded.
- On `cpu_small` the query set is drawn from the full catalog and then forced into the 50,000-item corpus, so the distractor set is smaller than Chapter 10's. Those metrics are not a numeric replication. Full numeric replication is an R1 job on `gpu_full`.

## Model revisions

Loading passes `revision=` and `trust_remote_code=False`.

| Key | Hub id | Revision |
|---|---|---|
| rexbert_micro | thebajajra/RexBERT-micro | 25587117376236b769710249fdcb66d6041edcee |
| rexbert_base | thebajajra/RexBERT-base | 55bd1c52eb0714beefcb61cae5e304352197f44c |
| ettin_17m | jhu-clsp/ettin-encoder-17m | 987607455c61e7a5bbc85f7758e0512ea6d0ae4c |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| mpnet | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf5434bc2275fc9bab85f82640a19130 |

## Baselines

Both baselines rank inside the evaluation catalog and exclude the query item.

- **Popularity.** Count of times the item is the next item in a non-degenerate training session. Higher count ranks first. Ties break toward the earlier catalog row.
- **Co-visitation.** Consecutive ordered transitions in training sequences (`prev_items`, then `next_item`), self-loops skipped, non-degenerate training sessions only. Rank is lexicographic: higher transition count, then higher popularity, then earlier catalog index. Items with a zero transition count fall through to that popularity order. This is a backoff, not a scaled popularity term.

## What this file does not claim

The `gpu_full` runtimes in `CLOUD_RUN.md` are planning estimates. They are not results. Fill-mask prompts in the code are a demonstration and are not a result. Derived fine-tuned weights are not part of the published tree.
