# Appendix: Structured-Output Scoring Benchmark

This companion package scores **typed decision models** on four ecommerce search
tasks. Each model answers the same frozen questions over a compact item state
and returns calibrated labels (Choice / Score / Noul), not free-form text.

The tasks are:

1. Product–accessory **compatibility** (four-level label).
2. Commerce-query **segmentation** (goal, object, specificity, commerce scope).
3. Query-to-**brand** and query-to-**category** classification.
4. Amazon **ESCI** E/S/C/I classification on a balanced human US test slice.

It sits next to [`code/appendix_hybrid_search/`](../appendix_hybrid_search/README.md).
That appendix compares retrieval methods. This one compares structured scorers
that can sit on top of retrieved candidates.

## Finished experiments in this drop

The committed comparison covers six **completed** zero-shot runs:

| Adapter | Checkpoint / endpoint | Where it ran |
|---------|------------------------|--------------|
| `majority` | class prior | CPU |
| `laya` | `convaiinnovations/laya` (`typed-decisions`) | Laptop GPU |
| `jev` | TypeSafe `jev-1.13.0` | Hosted `/v1/systemone` API |
| `kev` | `jaredpalmer/kev-0.8b` | Laptop GPU |
| `kev` | `jaredpalmer/kev-4b` | Colab A100 |
| `jevlite` | `vagmi/jev-lite` | Colab A100 |

Headline numbers are in
[`outputs/benchmark/ZERO_SHOT_RESULTS.md`](outputs/benchmark/ZERO_SHOT_RESULTS.md).
Machine-readable `summary.json` files are under `outputs/benchmark/<model>/`.
Raw prediction JSONL files are not shipped.

Compatibility, query, and brand/category use dual-judge field consensus
(Claude Opus 5 and GPT-5.6 Sol). Agreement is the label; disagreement is
withheld. ESCI uses existing Amazon human labels only — no new Opus/Sol ESCI
labels.

## What is not shipped

The public clone is the runnable harness, not the private workspace dump.

- **Reference manifests and consensus labels** (`references/`, ~20 MB) are not
  published here. Rebuild them with the scripts below, or copy the JSONL files
  into `references/` if you have the private workspace.
- **Raw predictions, caches, smoke runs, and Colab zips** stay out of git.
- **Isolated virtualenvs** (`.venvs/`) are local-only.
- **Source catalogs** used to *build* the pilots and reference sets (Amazon
  ESCI parquet, ORCAS-I, enrichment outputs) live outside this folder.

Without `references/`, `run_benchmark.py --dataset reference` and two unit
tests that load the 2,000-item manifests will skip or fail. The 100-item
pilots under `pilots/` **are** included.

## Install

```bash
cd code/appendix_system_one_benchmark
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `TYPESAFE_API_KEY` only if you will call
hosted Jev. Do not commit `.env`.

Optional stacks:

- **Laya:** `torch`, `transformers`, and the `laya` package, plus a GPU if you
  want the published latency profile.
- **Kev / JevLite:** an isolated Python 3.12 environment. The Laya/transformers
  stack and Kev's `transformers>=5.17` requirement should not share one venv.

```bash
uv python install 3.12
uv venv .venvs/kev --python 3.12
uv pip install --python .venvs/kev "kev[serve] @ git+https://github.com/jaredpalmer/kev.git"
uv pip install --python .venvs/kev --reinstall-package torch \
  --index-url https://download.pytorch.org/whl/cu124 torch
```

On Windows, skip fused Qwen3.5 / Triton kernels (no official `win_amd64`
wheel). The Kev adapter starts `kev.serve` on `127.0.0.1:8008` when that
endpoint is down.

## Tests

From this folder:

```bash
python -m pytest tests -q
```

Most tests are offline (schemas, metrics, payload conversion, cost
arithmetic). Tests that need `references/` skip when those files are absent.

## Rebuild labels (optional)

Pilots (100 items/task) are already in `pilots/`. To rebuild them you need the
private ESCI / enrichment inputs that `build_pilots.py` reads:

```bash
python build_pilots.py
python validate_pilots.py --json-output pilots/validation_summary.json
```

Scaled 2,000-item reference sets:

```bash
python build_reference_sets.py
python validate_reference_sets.py --json-output references/validation_summary.json
python estimate_reference_costs.py --mode batch
python adjudicate_pilots.py --dataset reference --max-items-per-task 2000 --execute
python export_consensus.py --dataset reference --output-dir references/consensus
```

Adjudication calls Anthropic and OpenAI. Dry-run first (no `--execute`) and
set a `--budget-usd` cap. ESCI evaluation uses
`build_esci_eval_slice.py` against the existing human US test labels.

## Run a finished model

Place reference manifests and consensus JSONL under `references/` (and
`references/consensus/`), then:

```bash
python run_benchmark.py --model majority --device cpu
python run_benchmark.py --model laya --batch-size 8 --device cuda
python estimate_jev_costs.py --json-output outputs/benchmark/jev_cost_estimate.json
python run_benchmark.py --model jev --batch-size 32 --concurrency 8 --budget-usd 2
python run_benchmark.py --model kev --model-id jaredpalmer/kev-0.8b \
  --batch-size 2 --concurrency 2 --device cuda
python write_comparison_report.py
```

`run_benchmark.py` writes `outputs/benchmark/<slug>/summary.json` and
`BENCHMARK_RESULTS.md`. Predictions cache in `outputs/benchmark_cache.sqlite`
and can be resumed.

A 100-item smoke uses `--dataset pilot` once pilot consensus rows exist under
`outputs/consensus/`.

## Colab notebook

[`colab/open_models_colab.ipynb`](colab/open_models_colab.ipynb) is the
reproduction path for JevLite and Kev-4B (Pro+ L4 or A100). Pack a bundle
**only if** you have the private reference JSONL files:

```bash
python pack_colab_bundle.py
```

Upload the notebook, set a Hugging Face token if the checkpoints are gated,
and upload `outputs/benchmark/colab_bundle.zip` when asked. Download
`open_model_results.zip` at the end. The committed `summary.json` files and
`ZERO_SHOT_RESULTS.md` already include those full-slice numbers.

## Layout

```
code/appendix_system_one_benchmark/
├── adapters/                 # majority, Laya, Jev, Kev, JevLite
├── eval/                     # loaders, metrics, runner
├── rubrics/                  # dual-judge instructions
├── tests/
├── colab/                    # Pro+ notebook + server launcher
├── pilots/                   # 100-item manifests
├── config/reference_costs.json
├── run_benchmark.py
├── write_comparison_report.py
└── outputs/benchmark/        # committed summaries + ZERO_SHOT_RESULTS.md
```

## Secrets

`.env` with a live `TYPESAFE_API_KEY` must never be committed. Use
`.env.example`. The appendix `.gitignore` also blocks `.venvs/`, `references/`,
prediction dumps, caches, and zip artifacts. The repo-root `.gitignore` only
covers `code/chapter*/outputs/` and `code/chapter*/cache/`.
