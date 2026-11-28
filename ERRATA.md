# Errata

Errata, 2026-11-10, updated 2026-11-28. Applies to chapters published before the date of each correction; corrected in the republished Chapters 1 (2026-11-18), 4, 5, 6, 7, 8, 10 (again 2026-11-18 and 2026-11-28), 12 (2026-11-23), 14 and 16, and Appendix A (2026-11-18, again 2026-11-23).

## Chapter 1: Introduction to Recommender Systems

- **Figure 1.2.** The caption cites Grainger et al. (2024), AI-Powered Search. That book is now in the References: Grainger, T., Turnbull, D., and Irwin, M. (2024). AI-Powered Search. Manning.
- **Section 1.7.** The roadmap sentence for Chapter 12 now says the chapter explores generative AI for recommender systems, including large language models and models trained on behavior sequences.
- **Section 1.7.** The roadmap sentence for Chapter 15 now covers monitoring, reliability, and retraining, matching the chapter's title.

## Chapter 4: Scalable Data Pipelines and Feature Management

- **References.** The entry for Shukla et al. (2023), "D3: An Automated System to Detect Data Drifts," now lists all five authors (Shukla, A., Nath, K., Vats, N., Tatipathri, V., and Jagannathan, D.) and links to https://www.uber.com/us/en/blog/d3-an-automated-system-to-detect-data-drifts/.

## Chapter 5: Retrieval: Finding Candidates at Scale

- **Section 5.3.1.** Typo corrected ("Searcch").

## Chapter 6: First-Stage Ranking: From Candidates to a Shortlist

- **Section 6.2.2, XGBRanker example.** The `fit` call now monitors the validation set (`eval_set=[(X_val, y_val)], eval_group=[val_groups]`) rather than the test set, matching the pointwise example in Section 6.2.1 and the companion code. Evaluating on the test set during training lets test data influence model selection.
- **Section 6.3.1, historical aggregate features.** The text and code snippet were corrected. Aggregates such as a user's average completion rate are computed from the same column as the label. Fitting them on the training rows and then applying them to those same rows puts each row's own label into its features. Training rows now take statistics from an earlier history window, and validation and test rows take statistics from the training window. The companion code applies the same rule: no row's aggregates come from its own window or a later one (see the Chapter 6 design notes).
- **Section 6.4.3.** The note on the Yambda tutorial's split is now ordinary text rather than an author's note: the tutorial uses the Global Temporal Split (GTS) protocol from the Yambda paper, with 300 days of training data, a 30-minute gap to prevent leakage, and 1 day of test data.

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
- **Section 10.3.** Dense embeddings are now contrasted with bag-of-words and learned sparse models, whose vocabulary-sized sparse vectors are served from an inverted index. That comparison is in the Appendix on Hybrid Search with Vector Databases (forthcoming).
- **Section 10.7.** The pointer to a discussion of task-aware fusion now refers to the Appendix on Hybrid Search with Vector Databases (forthcoming). Chapter 11 does not contain that discussion.
- Minor wording and punctuation corrections in Sections 10.1, 10.2, 10.3, 10.4 and 10.6.
- **Section 10.4 (2026-11-18).** RexBERT is no longer described as having deep knowledge of product terminology and catalog structure. The sentence now says it has seen far more product terminology than general-purpose encoders, and it cites Bajaj and Garg (2026), now in the References.
- **Section 10.4 (2026-11-18).** The zero-shot SBERT-versus-RexBERT comparison is now attributed to the Appendix on Domain-Specialized Encoders as Sentence-Embedding Models: on Amazon KDD sessions, a general-purpose contrastively trained SBERT model used zero-shot retrieves better than an MLM-only domain model. The appendix also fine-tunes the domain model with a contrastive objective and finds any advantage over a general sibling to be small.
- **Section 10.6 (2026-11-18).** The closing paragraph no longer says this section shows what happens when a contrastive objective is added to the domain model. That comparison is in the same appendix; Section 10.6 fine-tunes SBERT on behavioral pairs.
- **Sections 10.6 and 10.12 (2026-11-28).** Measured outcomes of the contrastive fine-tune are no longer stated in the prose. Section 10.6 now says the companion code trains the encoder on the pairs, scores it on held-out sessions and users, and records the metrics, including the comparison of the saved checkpoint with the zero-shot encoder. The summary says that comparison is in the companion code. The discussion still notes that the zero-shot baseline is strong, that the pairs encode browsing patterns, and that production systems with more interaction data typically see larger gains.

## Chapter 12: Generative AI for Recommender Systems

- **Exercise 6 (2026-11-23).** The position-bias probe now notes that Appendix D, Section D.6.1, measures a related position effect for typed-decision scorers.

## Chapter 14: Latency Optimization

- **Section 14.9.3.** The opening paragraph now states that Chapter 13's orchestrator already skips a stage when too little budget remains for it. This section treats that skip as the ladder's first level, adds triggers based on load, and measures what each level costs in quality.

## Chapter 16: Experimentation and Bandits

- **Section 16.10.3.** The appendix on agentic research loops is Appendix C. The appendices were re-lettered: A, RexBERT; B, Hybrid Search with Vector Databases; C, agentic research loops; D, structured-output benchmark. The previously listed Superlinked appendix was dropped.

## Appendix A: Domain-Specialized Encoders as Sentence-Embedding Models

- **Title (2026-11-18).** The appendix was retitled from "RexBERT: Turning a Domain-Specialized Encoder into a Sentence-Embedding Model". Its content is unchanged.
- **Section A.3.4 (2026-11-23).** Combining dense and sparse retrieval now points to Appendix B (Hybrid Search with Vector Databases).
- **Section A.6 (2026-11-23).** Combining the encoder with a lexical index, and exact identifier match, now point to Appendix B.
- **Exercise A.3 (2026-11-23).** Hybrid retrieval now stays in Appendix B.
