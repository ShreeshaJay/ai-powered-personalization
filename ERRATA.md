# Errata

Errata, 2026-11-09. Applies to chapters published before this date; corrected in the republished Chapters 4, 5, 6, 7, 8, 10 and 16.

## Chapter 4: Scalable Data Pipelines and Feature Management

- **References.** The entry for Shukla et al. (2023), "D3: An Automated System to Detect Data Drifts," now lists all five authors (Shukla, A., Nath, K., Vats, N., Tatipathri, V., and Jagannathan, D.) and links to https://www.uber.com/us/en/blog/d3-an-automated-system-to-detect-data-drifts/.

## Chapter 5: Retrieval: Finding Candidates at Scale

- **Section 5.3.1.** Typo corrected ("Searcch").

## Chapter 6: First-Stage Ranking: From Candidates to a Shortlist

- **Section 6.2.2, XGBRanker example.** The `fit` call now monitors the validation set (`eval_set=[(X_val, y_val)], eval_group=[val_groups]`) rather than the test set, matching the pointwise example in Section 6.2.1 and the companion code. Evaluating on the test set during training lets test data influence model selection.
- **Section 6.3.1, historical aggregate features.** The text and code snippet were corrected. Aggregates such as a user's average completion rate are computed from the same column as the label. Fitting them on the training rows and then applying them to those same rows puts each row's own label into its features. Training rows now take statistics from an earlier history window, and validation and test rows take statistics from the training window. The companion code applies the same rule: no row's aggregates come from its own window or a later one (see the Chapter 6 design notes).

## Chapter 7: Second-Stage Ranking and Multi-Task Learning

- **Section 7.10.2, artifact handoff.** The pre-computed predictions are now described as two files: `val_predictions.parquet` (validation set), for iterating on value function weights and calibration, and `predictions.parquet` (test set), for reporting the chosen configuration once. The companion code exports both.

## Chapter 8: Value Functions: From Scores to Decisions

- **Section 8.2.3.** Weight configurations should be explored on the validation predictions file (`val_predictions.parquet`), with the test predictions file (`predictions.parquet`) kept for reporting the chosen weights.
- **Section 8.9.4.** The text now tells the reader to run the end-to-end evaluation on the test set predictions and report the results, with weights fixed in advance and any weight tuning done on the validation predictions. No script in the companion code reports this evaluation automatically.

## Chapter 10: Reusable Item Embeddings

- **Section 10.4, model comparison.** RexBERT was pre-trained on about 2.3 trillion tokens in total, including about 350 billion tokens of e-commerce text, not 2.3 trillion e-commerce tokens. Source: Bajaj, R. and Garg, A. (2026). RexBERT: Context Specialized Bidirectional Encoders for E-commerce. arXiv:2602.04605, https://arxiv.org/abs/2602.04605; RexBERT-base model card, https://huggingface.co/thebajajra/RexBERT-base.
- **Sections 10.1, 10.2 and 10.6, cross-references.** Several section numbers were off by one and now match the headings: zero-shot encoders (including text serialization) are Section 10.4, Item2Vec is Section 10.5, contrastive fine-tuning is Section 10.6, and multi-modal fusion (including the Music4all-Onion dataset) is Section 10.7.
- **Sections 10.4 and 10.6, MIND co-click pairs.** Co-click pairs are built from each user's click history, not from single browsing sessions. The co-click evaluation and the contrastive training pairs both use the whole history.
- **Section 10.6.** Co-click pairs teach the model to place articles read by the same person near each other, not that such articles share a topic.
- **Section 10.3.** The comparison of compact dense embeddings with sparse, inverted-index representations is in the Appendix on Hybrid Search with Vector Databases (forthcoming).
- **Section 10.7.** The pointer to a discussion of task-aware fusion now refers to the Appendix on Hybrid Search with Vector Databases (forthcoming). Chapter 11 does not contain that discussion.
- Minor wording and punctuation corrections in Sections 10.1, 10.2, 10.3, 10.4 and 10.6.

## Chapter 16: Experimentation and Bandits

- **Section 16.10.3.** The appendix on agentic research loops is Appendix C. The appendices were re-lettered: A, RexBERT; B, Hybrid Search with Vector Databases; C, agentic research loops; D, structured-output benchmark. The previously listed Superlinked appendix was dropped.
