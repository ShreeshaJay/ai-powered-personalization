# Chapter 12 Design Notes

Decisions behind the Chapter 12 companion code, and the limits a reader should keep in mind when reading its numbers.

## 1. Data and splits

**Random session split.** The Amazon KDD Cup 2023 session file has no user IDs and no timestamps. A per-user or time-based split is impossible, so sessions are split 80/10/10 at random with a fixed seed. Two consequences follow. Sessions from the same (unknown) shopper can land in different splits, so some test behaviour may resemble training behaviour more than a production holdout would. And no result here says anything about drift over time.

**Stable session IDs.** `session_id` is the row number in `sessions_train.csv` before any filtering. Every artifact that is keyed by session (slates, LLM intents, ranker rows) joins on that ID, never on row order, and the tests check alignment after shuffling.

**Train-only statistics.** Popularity, cold and warm labels, and every count used for negatives come from the train split. Validation and test sessions only ever contribute their own history at inference time.

**Small catalogs for small runs.** `--catalog referenced` keeps the items the kept sessions mention plus random distractors, so a CPU run does not encode 500K items. Retrieval over a smaller catalog is easier, which is one more reason the `tiny` numbers are only a code check.

## 2. Text encoder and leakage

Chapter 10 fine-tunes MiniLM on next-item pairs drawn from every session. Those pairs include this chapter's validation and test sessions, so reusing that model would put test behaviour into the item vectors, and every downstream comparison (Semantic IDs, retrieval slates, ranker features) would inherit the leak. The default is the base `all-MiniLM-L6-v2`. A reader who wants a fine-tuned encoder should fine-tune it on train-split sessions only.

All vectors that are compared with each other must come from one encoder. Each `.npz` stores the encoder name, and loaders refuse to mix them. LLM texts are embedded with the encoder named in `item_embeddings.npz`.

`tfidf-svd` exists so that the synthetic pipeline runs without downloads and slow machines can still run the real-data steps. It is a weaker encoder; its results are not comparable with MiniLM's.

## 3. Semantic IDs (RQ-VAE)

- **Three levels plus a collision suffix.** Three residual levels give a coarse-to-fine code; a fourth level numbers items that share all three codes, as in TIGER. This keeps IDs short for beam search and still makes every ID unique.
- **EMA codebooks with k-means initialisation.** Codebooks are buffers updated by exponential moving averages of assigned latents; they start from k-means on a sample of encoder outputs, so early training does not pile every item onto a few codes. Laplace smoothing only stabilises the EMA division.
- **Relative dead-code reset.** A codeword whose EMA usage falls below a fraction of the mean usage is reset to a random latent from the current batch. A relative threshold works at any codebook size and batch size, where an absolute count does not.
- **One straight-through estimator.** The decoder receives `z_e + (z_q - z_e).detach()` for the summed quantization. Applying the estimator per level multiplies the encoder gradient by the number of levels. The commitment loss reaches the encoder through the residuals.
- **Usage is reported.** Codes used and perplexity per level are written to `metrics/rqvae.json`, computed from actual assignments, so collapse is visible.

## 4. TIGER

- **Fixed-length IDs, no separators.** Every item is exactly L tokens, so item boundaries are implicit. Token ids are offset per level, giving each level its own vocabulary slice and its own softmax.
- **Positions count back from the most recent item,** so the most recent item always has the same position embedding however long the history is.
- **Constrained beam search.** A prefix tree of real Semantic IDs masks invalid continuations at each step, so every beam maps to an item. Search is batched over sessions; the tests check that beam scores equal teacher-forced log-probabilities.
- **History removal for every method.** Items already in the history are removed from TIGER's beams and from both baselines. Extra beams (`--history-margin`) keep the reported list depth after removal. `target_in_history_rate` reports how often a removed item was the target.
- **Model selection on validation loss; test used once.**

## 5. Candidate slates

The slate decides what a ranking number means. Earlier versions of this experiment retrieved the candidates most similar to the session history and, when retrieval missed the true item, overwrote the last candidate with it. An inserted target is then, by construction, the least history-like candidate, which pushes any similarity-based ranker below random and changes what an LLM ranker is being compared against. Earlier versions also trained the Yes/No model on random negatives and tested it on retrieved candidates, which look nothing alike. The fix is a shared slate file per split and mode:

- `retrieved` (default) never inserts the target. Metrics are reported over all sessions and over sessions whose slate holds the target, with the retrieval hit rate alongside.
- `injected` is kept as a diagnostic and labelled as such.
- `random` and `popular` slates are easy and are reported for contrast.
- Candidates are stored in shuffled display order with each candidate's similarity rank, so position effects and the similarity baseline are measured on identical slates.

## 6. LLMs as rankers

- **Licences.** Defaults are Apache-2.0 Qwen2.5 Instruct models (0.5B on CPU, 1.5B on GPU, 7B as an option). Qwen2.5-3B-Instruct is distributed under the Qwen Research License and is not a default. No script sets `trust_remote_code`.
- **Zero-shot listwise.** Greedy decoding. The parser reads only the `Ranking:` line (or the first line), keeps the first occurrence of each valid letter, and appends omitted candidates in a seeded random order so silence is never rewarded. Validity counts and a position-bias check (first choice equals first displayed candidate, against 1/N) are reported with every run.
- **Pointwise scoring.** P(Yes) / (P(Yes) + P(No)) at the first answer position, with left padding and explicit position ids so a candidate's score does not depend on its batch (tested).
- **TALLRec-style tuning.** Negatives come from the same mode of train-split slates as the test slates, so training and test candidates look alike. Labels cover only the answer tokens and the end-of-turn token. Prompts that are too long lose their oldest history items. The base model is evaluated with the adapter disabled on the same slates, so the tuning effect is measured on one model. LoRA weights are kept in fp32; on GPUs without bf16 the loss is scaled.

Product titles from public Amazon data may appear in an LLM's pretraining data. This chapter cannot rule that out, and a reader should treat zero-shot results with that in mind.

## 7. LLMs as feature generators (KAR-style)

- **Complete coverage or an explicit override.** A feature present for a small share of rows mostly teaches the ranker that presence correlates with the label. The ranker stops below 99% coverage unless `--allow-partial-coverage` is given, and presence flags are always inputs.
- **Session intent from history only.** The prompt never sees the target.
- **One vector space.** LLM texts are embedded with the item encoder, which lets the ranker take cosines between LLM text and item content.
- **A control.** `--backend copy-input` embeds the prompt text itself. If the LLM features do no better than this control, the LLM has added nothing beyond re-encoding the product text.

## 8. Augmented ranker

- **Feature sets renamed to say what they compute:** `content`, `content_history`, `content_history_llm_item`, `full`. The history representation is a mean of item vectors; earlier versions called it "SASRec", which it is not.
- **Frozen tables, on-the-fly features.** Embedding tables are buffers and features are computed per batch, so no large feature matrix is materialised. Checkpoints store only the MLP weights and the configuration.
- **Negatives.** Hard negatives are nearest neighbours of the history mean that are neither the target nor in the history; random negatives are added. Early stopping uses validation-split sessions.

## 9. HSTU

See `hstu_yambda/README.md` for the split, training windows, loss masking and the three evaluation views.

## 10. Reading the small-run numbers

- The `synthetic` profile has planted structure and exists to show the pipeline runs end to end.
- The `tiny` profile keeps 3,000 sessions over a catalog of thousands of items. Almost every item is seen in at most a handful of training sessions, so cold and warm splits are thin (the `tiny` profile sets the cold threshold to 1, meaning never seen in training), TIGER is undertrained, and LLM steps run on a handful of sessions. None of these numbers support a conclusion about the methods.
- The `ranker` and `full` profiles are the runs to read, and they need a GPU (see `CLOUD_RUN.md`).
