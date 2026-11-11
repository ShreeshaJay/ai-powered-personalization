# Design notes: RexBERT as a sentence encoder

These notes record the model facts checked on 2026-11-11 and the training choices locked in `PREREGISTRATION.md`. They do not report retrieval results.

## Facts checked on 2026-11-11

Citation used by the appendix: Bajaj, R. and Garg, A. (2026). RexBERT: Context Specialized Bidirectional Encoders for E-commerce. arXiv:2602.04605.

The arXiv abstract page (https://arxiv.org/abs/2602.04605) gives that title and those authors. The abstract says Ecom-niverse is a 350 billion token corpus, that the recipe has three phases (general pre-training, context extension, and annealed domain specialization), and that the family ranges from 17M to 400M parameters. The abstract does not state a total of 2.3 trillion tokens and does not single out 150M.

The paper body on the same HTML page does. The contributions list four scales: 17M, 68M, 150M, and 400M. The curriculum figure caption states 1.7 trillion tokens of mixed data, then 250 billion tokens of context extension to 8,192, then 350 billion tokens of Ecom-niverse annealing. Those three phases sum to 2.3 trillion tokens, so the e-commerce slice sits inside the total.

The Hugging Face cards for `thebajajra/RexBERT-base` and `thebajajra/RexBERT-micro`, read the same day, say the base model is a 150M-parameter encoder and the micro model is a 17M-parameter encoder, both Apache-2.0. Both cards also say the model "has been trained on 2.3T+ tokens along with 350B+ e-commerce-specific tokens." The phrase "along with" can be read as adding 350B on top of 2.3T. The phase sum in the paper is the reading this appendix uses: about 2.3T tokens in total, of which about 350B are e-commerce text. The appendix therefore says "about 150M parameters" for RexBERT-base, "about 2.3T pre-training tokens in total," and "about 350B of e-commerce text."

The sentence-encoder wrapper keeps the Transformer stack, mean pooling, and L2 normalization. It does not keep the masked-language-model head, so a parameter count taken from the wrapper can sit below the card's 150M. A CPU load check on 2026-11-11, with `trust_remote_code=False`, recorded 16,797,440 parameters for RexBERT-micro, 149,014,272 for RexBERT-base, and 16,797,440 for Ettin-17m. The hidden sizes were 256, 768, and 256. The transformers load report listed the masked-language-model head weights as unused by that wrapper. The dimension check the code enforces is 768 for RexBERT-base and 256 for RexBERT-micro and Ettin-17m.

`jhu-clsp/ettin-encoder-17m` is the capacity-matched control. Its card, also read on 2026-11-11, lists a 17M encoder under the MIT license, with 7 layers, hidden size 256, intermediate size 384, and 4 heads. Its decay phase is 100B tokens of general-domain data, after a 1.7T pre-training phase and a 250B context-extension phase. RexBERT-micro's third phase is the 350B e-commerce anneal. The two models match in architecture and in the size of the first two phases. They differ in what the third phase contains and in how many tokens that phase uses (about 350B against about 100B). That token-count gap is a confound, and the appendix states it whenever the comparison is interpreted.

`sentence-transformers/all-MiniLM-L6-v2` is the practical reference. Its card says it was fine-tuned on a 1B-sentence-pair collection (the card's table sums to about 1.17 billion pairs) and that inputs longer than 256 word pieces are truncated. Its vectors are 384-dimensional. RexBERT-micro and Ettin-17m are 256-dimensional. That width gap is a second confound, so MiniLM's position is described and does not select the Q3 branch.

None of the three masked-language-model configs used here advertises an `auto_map`. Loading passes `trust_remote_code=False`.

## Why the comparison is RexBERT-micro against Ettin-17m

RexBERT-base is the model Chapter 10 discussed. It is part of the zero-shot replication (Q1) on `gpu_full`. The controlled question (Q3) uses the micro model because Ettin publishes a sibling with the same depth and width. A comparison of RexBERT-base with MiniLM would mix domain pre-training, contrastive pre-training, and a change in width. Holding the architecture fixed leaves the third-phase data as the intended contrast, with the 350B versus 100B confound stated above.

## Conversion

`encoders.build_sentence_encoder` builds a `SentenceTransformer` from the masked language model: a `Transformer` module at the pinned revision, mean pooling, then L2 normalization. Existing sentence encoders (MiniLM, mpnet) load through `SentenceTransformer` at their pinned revisions. The maximum sequence length for this appendix is 128.

Fine-tuning uses `SentenceTransformerTrainer` and `MultipleNegativesRankingLoss` with scale 20, which is a temperature of 0.05. The batch sampler is `BatchSamplers.NO_DUPLICATES`, so one product does not appear twice in a batch and the in-batch negatives stay distinct. On `gpu_full` the batch size is 128, so each anchor has 127 in-batch negatives. Training runs for up to three epochs. The checkpoint with the best validation MRR@100 is kept. The learning rate is chosen by a one-epoch sweep over `{2e-5, 5e-5, 1e-4}` on 100,000 pairs at seed 0. The main runs start again from the original checkpoint. Warmup is 10% of steps. Transformers 5 retired `warmup_ratio`, so the code passes `warmup_steps=0.1`. Weight decay is 0.01. bf16 is requested on GPU when the device supports it.

Mean pooling is the pooling used by the MiniLM card's reference implementation and by the RexBERT card's feature-extraction example. The loss is multiple-negatives ranking because each training row is one positive next item and the rest of the batch supplies negatives. The code does not train a triplet loss, CoSENT, or TSDAE. Those are separate exercises, not this protocol.

## Splits and baselines

Sessions are split before any pair is built. The split is a SHA-256 of the filtered row index, seed 42, with buckets 0-7 train, 8 validation, and 9 test. Pairs are (last previous item, next item) from non-degenerate training sessions only. Validation selection uses an information-retrieval evaluator on validation sessions, with MRR@100, and the query item is masked. Test sessions are read in the zero-shot run and the final retrieval run.

The popularity baseline ranks by how often an item is the next item in non-degenerate training sessions. The co-visitation baseline counts consecutive transitions in those sessions and breaks remaining ties by popularity, then by catalog order. Neither baseline reads product text.

The fill-mask call in `diagnostics.py` is a demonstration on three generic prompts. It is not a retrieval result. If it fails, the run records the error type and continues.

## What the teaching profile changes

`cpu_small` uses one seed, one epoch, 50,000 pairs, a batch of 64, 2,000 test queries, and a 50,000-item catalog. It does not sweep the learning rate. It uses 5e-5, the middle sweep candidate, so a noisy short sweep cannot choose the teaching rate. Those choices are for exercising the code. They are not the reported protocol.
