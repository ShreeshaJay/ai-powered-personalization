# Hybrid Search with Vector Databases

This folder is the companion code for Appendix B of *AI-Powered Personalization*. It retrieves products for shopping queries with BM25, three public SPLADE checkpoints, and one dense encoder, then fuses those lists and scores them on the human ESCI labels.

The catalog is the US English `large_version` of the Amazon shopping-queries dataset. Task 1 of that dataset used `small_version`. This appendix uses `large_version` for every retriever.

## What the code measures

Protocol P1 ranks each query's judged products and reports nDCG, with gains Exact 1.0, Substitute 0.1, Complement 0.01, and Irrelevant 0. Protocol P2 treats unjudged products as unjudged. It reports recall of Exact and of Exact or Substitute at 100 and 1,000, condensed nDCG at 10, and judged-at-10. An empty list scores zero and stays in the count. A query whose labels are all Irrelevant is left out of nDCG only.

`analysis_plan.yaml` fixes the splits, model revisions, fusion grids, and the ten primary comparisons. `analysis_plan.sha256` is the SHA-256 of that file. Scoring the test split a second time, under the same digest, stops.

## Data

Download the shopping-queries release from the amazon-science/esci-data repository. The licence is Apache-2.0. Place these two files in one directory:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`

Pass that directory with `--data-dir`, or set the environment variable `ESCI_DATA_DIR`. There is no default path.

The Qdrant ESCI SPLADE checkpoint was trained from `tasksource/esci` at revision `8113b17a5d4099e20243282c926f1bc1a08a4d13`. Its document text is that dataset's `product_text` field: title, brand, color, description, and bullet point, joined by newlines, with a missing value written as `None`. Protocol P1 uses that text for this checkpoint only. Every other model, and protocol P2 for every model, uses `{title}. {bullet_point}. Brand: {brand}`. The split code replays the training prefix and keeps those query texts out of development. A test query whose text matches that prefix, under the same comparison, is left out of the test split, and every system is scored on the queries that remain. That dataset card also uses the Apache-2.0 licence.

Reciprocal rank fusion starts at rank 1 with k = 60, so the first document contributes 1/61. The Qdrant client's local-mode formula is 1/(k + position) and its positions start at 0, so the client constant that matches this script is 61. The client's default constant is 2.

## Models and licences

Every Hugging Face load uses the revision in `analysis_plan.yaml` and `trust_remote_code=False`.

| Role | Repository | Licence |
| --- | --- | --- |
| SPLADE | `naver/splade-cocondenser-ensembledistil` | CC BY-NC-SA 4.0 |
| SPLADE | `prithivida/Splade_PP_en_v2` | Apache-2.0 |
| SPLADE | `thierrydamiba/splade-ecommerce-esci` | Apache-2.0 |
| Dense | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| Cross-encoder, optional | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |

Naver vectors and indexes are an evaluation cache. Do not redistribute them. The other checkpoints are Apache-2.0.

`prithivida/Splade_PP_en_v2` is a document encoder. Its model card says a separate query model will be released. This code uses that one checkpoint for queries and documents, with query length 24 and document length 128.

## CPU teaching slice

These times were measured on a laptop CPU, Python 3.13.15, on a schema fixture of 8 products, 4 test queries, and 1 development query. They include the first load of each checkpoint. They are teaching measurements. They are not catalog timings, and the scores from this slice are not results for the book.

| Step | Seconds | RSS at end (bytes) | Process peak working set after the step (bytes) |
| --- | ---: | ---: | ---: |
| Fixture and splits | 0.381 | 111,173,632 | 111,173,632 |
| BM25 | 7.438 | 280,694,784 | 281,759,744 |
| Qdrant parity | 4.924 | 327,766,016 | 327,766,016 |
| Dense encode and search | 52.013 | 655,335,424 | 656,408,576 |
| ESCI SPLADE encode and search | 1.969 | 1,008,496,640 | 1,014,833,152 |
| Fusion | 0.138 | 1,008,513,024 | 1,014,833,152 |
| Evaluation | 0.114 | 1,008,689,152 | 1,014,833,152 |
| Slices | 0.000 | 1,008,689,152 | 1,014,833,152 |
| Cross-encoder rerank | 1.570 | 1,111,093,248 | 1,111,093,248 |
| One fine-tune step | 10.045 | 1,633,792,000 | 2,192,551,936 |
| Label-file conversion | 0.118 | 1,634,234,368 | 2,192,551,936 |
| Total | 78.710 |  | 2,192,551,936 |

The peak working set is the Windows process peak since the process started, so it does not fall after a later step uses less memory. The highest value was 2,192,551,936 bytes, about 2.04 GiB, during the one-step fine-tune.

Create a virtual environment named `venv`, install a CPU build of PyTorch from the PyTorch CPU index when the machine has no CUDA build, then install this folder's requirements with `--upgrade-strategy only-if-needed`. That flag keeps an existing CUDA build in place. `requirements.txt` does not pin torch.

```bash
python -m unittest discover -s tests -v
python run_cpu_slice.py --output-dir outputs/cpu_slice --device cpu
```

Exercise 1 builds a larger slice: the judged products of 1,000 test queries plus 30,000 distractors. Slice metrics on that exercise are inflated relative to the full catalog. Exercise 2 runs the same hybrid in Qdrant local mode. Exercise 3 is the optional fine-tune. Exercise 4 needs a label file that is not included here.

## Full catalog

The GPU times below are planning estimates. They were not measured in this folder, and they were not recomputed from a 20,000-product smoke.

| Stage | Where | Planning estimate |
| --- | --- | --- |
| S0 smoke, 20,000 products and 200 queries | Laptop CPU | 20 to 40 minutes |
| S1 encode and retrieve | Colab L4 | 2.5 to 4.5 hours |
| S2 BM25, fusion, evaluation | Colab CPU or a CPU virtual machine | 1.5 to 3 hours |
| S3 cross-encoder, optional | Colab L4 | 15 to 30 minutes |
| S4 fine-tune, optional | Colab L4 | 2 to 4 hours |
| S5 score the saved runs again and compare summary hashes | CPU virtual machine | 1.5 to 3 hours |

The required path is S0, S1, S2, and S5: about 3 to 5.5 L4 hours. Adding S3 and S4 brings the L4 time to about 6 to 10 hours.

On a Colab runtime that already has torch `2.14.1+cu130`, install with:

```bash
pip install -r requirements.txt --upgrade-strategy only-if-needed
python run_pipeline.py --stages s1,s2 --data-dir /content/esci --output-dir outputs --device auto --prefix-source tasksource/esci
python run_pipeline.py --stages s5 --data-dir /content/esci --output-dir outputs --device auto
```

Optional stages:

```bash
python run_pipeline.py --stages s3 --data-dir /content/esci --output-dir outputs --device auto
python run_pipeline.py --stages s4 --data-dir /content/esci --output-dir outputs --device cuda
```

A step whose output files already exist is skipped. Encoding writes a checkpoint every 50,000 products. After each stage the runner can execute a command you pass with `--sync-cmd`. The placeholders are `{stage}`, `{step}`, and `{output}`. The code does not contain a destination.

```bash
python run_pipeline.py --stages s1,s2,s5 --data-dir /content/esci --output-dir outputs --device auto --sync-cmd "your-command {stage} {step} {output}"
```

The notebooks `colab/encode_and_retrieve.ipynb` and `colab/finetune_splade.ipynb` call the same runner. L4 batch sizes are in `analysis_plan.yaml`: 32 for the BERT SPLADE checkpoints, 64 for the DistilBERT SPLADE checkpoint and the cross-encoder, and 256 for the dense encoder.

## Labels

`provenance/llm_labeler.py` refuses to run. An earlier draft called a paid language-model API. That client is not included, and this folder does not call one. `llm_labels_convert.py` only reads a label file you already have. Protocol P3 and Exercise 4 are not reported. The saved label file does not store the sampling parameters, so the query sample is not reconstructed.
