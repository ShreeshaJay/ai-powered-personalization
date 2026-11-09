# Appendix: Structured-Output Scoring Benchmark

This companion package scores **typed decision models** on four ecommerce search
tasks. Each model answers the same frozen questions over a compact item state
and returns typed labels (Choice / Score / Noul) with a probability for every
allowed option, not free-form text.

The tasks are:

1. Product–accessory **compatibility** (four-level label).
2. Commerce-query **segmentation** (goal, object, specificity, commerce scope).
3. Query-to-**brand** and query-to-**category** classification.
4. Amazon **ESCI** E/S/C/I classification on a balanced human US test slice.

It sits next to [`code/appendix_hybrid_search/`](../appendix_hybrid_search/README.md).
That appendix compares retrieval methods. This one compares structured scorers
that can sit on top of retrieved candidates.

## Correction note (2026-11-09)

The calibration column previously labelled "ECE" in
`outputs/benchmark/ZERO_SHOT_RESULTS.md` was computed with 10 bins on each
adapter's reported `confidence` field. For most adapters that field is not a
probability (it is a rescaled margin, or one minus normalised entropy), so the
column did not measure calibration. The table now reports top-label ECE
computed from the probability vectors (15 equal-width and 15 equal-mass bins)
and keeps ECE on `confidence` only as a labelled comparison column. The
query-segmentation multi-target field, where the reference is one value for
99.95% of items, is reported separately as degenerate. The runs table no longer
names GPU types, which the runs did not record. Per-item predictions and
reference labels are now committed as a text-free replay pack, so every metric
in the committed summaries can be recomputed on a CPU (the per-category
ORCAS-I accuracies in aggregate). The pilot summary now names its private
inputs by file name only and its pilot files by relative path; its hashes and
counts are unchanged.

## Finished experiments in this drop

The committed comparison covers six **completed** zero-shot runs:

| Adapter | Checkpoint / endpoint | Where it ran (as recorded) |
|---------|------------------------|----------------------------|
| `majority` | class prior | CPU |
| `laya` | `convaiinnovations/laya` (`typed-decisions`) | Local GPU (`cuda`, 8,188 MB device memory; type not recorded) |
| `jev` | TypeSafe `jev-1.13.0` | Hosted `/v1/systemone` API |
| `kev` | `jaredpalmer/kev-0.8b` | Local GPU (`cuda`, 8,188 MB device memory; type not recorded) |
| `kev` | `jaredpalmer/kev-4b` | Colab GPU (`cuda`, 40,960 MB device memory; type not recorded) |
| `jevlite` | `vagmi/jev-lite` | Colab GPU (`cuda`, 40,960 MB device memory; type not recorded) |

`laya_typed_decisions` ran with the adapter's `max_len=512` (`adapters/laya.py`), although the checkpoint accepts 1,024 tokens. A tokenizer check on 2026-11-09 found inputs cut at 512 tokens for 1,996 of the 2,000 brand/category items and 3 of the 2,000 compatibility items; no input exceeded 1,024 tokens. The checkpoint's 256-token question limit kept 127 to 215 of the 452 instruction tokens in each query-segmentation question. ESCI inputs were not cut. Laya's results measure this configuration, not the checkpoint's best case.

The appendix refers to each scorer by its role label. The model key is the name
used in `outputs/benchmark/<model key>/`, the replay pack and the results files.

| Role label (appendix) | Model key | Adapter | What it is |
|---|---|---|---|
| Majority prior | `majority` | `majority` | One fixed class per field, with probability one |
| Encoder decision model | `laya_typed_decisions` | `laya` | Laya `typed-decisions` checkpoint: an encoder fine-tuned on non-commerce workflows |
| Hosted typed-decision API | `jev_1_13_0` | `jev` | TypeSafe `jev-1.13.0`, a closed hosted model |
| Open adapter, small | `kev_0_8b` | `kev` | `jaredpalmer/kev-0.8b`, an adapter on Qwen3.5-0.8B-Base |
| Open adapter, mid | `kev_4b` | `kev` | `jaredpalmer/kev-4b`, an adapter on Qwen3.5-4B-Base |
| Option-letter adapter | `jevlite` | `jevlite` | `vagmi/jev-lite`, an adapter on Gemma 4 E4B that reads the logits of option letters |

Headline numbers are in
[`outputs/benchmark/ZERO_SHOT_RESULTS.md`](outputs/benchmark/ZERO_SHOT_RESULTS.md),
which `write_comparison_report.py` regenerates from the summaries and the
replay pack. Machine-readable `summary.json` files are under
`outputs/benchmark/<model>/`. Intervals, paired differences, label-source
sensitivity and per-stratum results are in
[`outputs/replay/REPLAY_RESULTS.md`](outputs/replay/REPLAY_RESULTS.md).

Compatibility, query, and brand/category use dual-judge field consensus
(Claude Opus 5 and GPT-5.6 Sol). Agreement is the label; disagreement is
withheld. ESCI uses existing Amazon human labels only — no new Opus/Sol ESCI
labels.

## Replay on a CPU (no keys, no GPU)

[`replay/`](replay/) holds every item's reference labels (consensus and both
judges' labels) and every model's prediction (label, probability vector,
reported `confidence`, latency) for the 2,000-item reference sets and the
8,000-item ESCI slice. It contains item IDs, option IDs and numbers only: no
query, product or candidate text. `replay/MANIFEST.json` records file hashes,
row counts, the hash of each committed `summary.json` and rubric, the label
sources, the judges and the licences below. `replay/label_provenance.json`
records judge agreement by stratum and aggregate label-generation counts.

```bash
cd code/appendix_system_one_benchmark
python -m pip install -r requirements.txt
python -m pytest tests -q
python replay_scores.py verify    # exit 0 pass; 1 metric or results-file mismatch; 2 manifest mismatch
python replay_scores.py report    # outputs/replay/REPLAY_RESULTS.md and replay_metrics.json
python replay_scores.py figures   # outputs/replay/figures/*.png (needs matplotlib)
```

`verify` recomputes every metric stored in every committed `summary.json` with
the committed `eval/runner.py` and `eval/metrics.py`, compares at stored
precision, and regenerates `ZERO_SHOT_RESULTS.md` in memory to compare it with
the committed file. It writes `outputs/replay/verify_report.json`. `report`
uses a seeded paired bootstrap (2,000 resamples) for intervals and paired
differences, so repeated runs write identical files. Measured runtimes are in
[`CLOUD_RUN.md`](CLOUD_RUN.md).

### What the replay can and cannot check

- **Recomputed exactly:** per field, accuracy, macro- and micro-F1, per-label
  precision, recall and F1, confusion counts, ECE, Brier score, log loss, mean
  confidence, coverage-accuracy and per-stratum accuracy; per task, item
  counts, complete exact match, and latency percentiles and throughput (from
  the per-item latencies in the pack); Jev's input-token counts and cost where
  the summary stores them.
- **Recomputed in aggregate:** the pack merges all ORCAS-I items into the
  single stratum `orcas_i`, while the published summaries keep
  query-segmentation accuracy aggregated by ORCAS-I intent category. Those
  per-category entries are checked by merging them into `orcas_i`.
- **Recorded, not recomputable:** wall time, device memory and cache counts,
  and the conditions behind the latencies (GPU type was not recorded).
- **Not in the pack:** item text, the inputs used to build the item sets, and
  the judges' raw responses. Rebuilding the labels needs the private inputs and
  paid API keys (see below).

The `ece` key in each `summary.json` is the original 10-bin ECE on reported
`confidence`. It is kept unchanged so the summaries stay verifiable; the
calibration numbers to quote are the top-label ECE columns in
`ZERO_SHOT_RESULTS.md` and `REPLAY_RESULTS.md`.

### Label sources

- `dual_judge_consensus`: field consensus of `judge_a` (Anthropic
  `claude-opus-5`) and `judge_b` (OpenAI `gpt-5.6-sol`) under the rubric in
  `rubrics/`. A field is labelled only where both judges agree; otherwise its
  reference is null and it is not scored for that item. Both judges' labels are
  in the pack, so results can be rescored against either judge.
- `human`: the ESCI human relevance labels (E, S, C or I) from the Amazon
  Shopping Queries Dataset.

## Data and licences

Code in this repository is under the repository [LICENSE](../../LICENSE).

| Source | Licence | Use here |
|--------|---------|----------|
| [Amazon Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data) | Apache-2.0 | Queries and products behind the compatibility, brand/category and ESCI items, 1,000 of the 2,000 query-segmentation queries, and the ESCI human labels for the ESCI slice. No query or product text is in the replay pack. |
| [ESCI-S product metadata](https://github.com/shuttie/esci-s) | Apache-2.0 (LICENSE file and README checked 2026-11-09; the README states the metadata was scraped from product pages) | US category paths, shown to models and judges as the brand/category task's category options. The pack stores only option IDs. |
| ORCAS-I gold (Alexander, Kusa and de Vries, SIGIR 2022) | CC BY-NC 4.0 | 1,000 of the 2,000 query-segmentation queries were sampled from ORCAS-I. The queries and their ORCAS-I intent labels are not redistributed; those items carry the stratum `orcas_i`. The per-category accuracy aggregates in the published `summary.json` files derive from ORCAS-I and are credited under the same licence. |
| GPT-5.1 enrichment of ESCI queries | OpenAI model output; not redistributed except as stratum names | Query taxonomy, complement candidates and category matches used to form sampling strata and candidate pools. The only enrichment-derived values in the pack are stratum names: each `esci_*` stratum is the query's GPT-5.1 product-type intent, and several compatibility and brand/category strata name pools built from the enrichment. No enrichment text is in the pack, and none was shown to models or judges except the ESCI-S category paths it matched. |
| Judge labels: `judge_a` (Anthropic `claude-opus-5`), `judge_b` (OpenAI `gpt-5.6-sol`), and their field consensus | AI-generated outputs of the two named models, released with this repository as evaluation labels (Anthropic Commercial Terms and OpenAI Services Agreement and Sharing and Publication Policy checked 2026-11-09) | Reference labels for compatibility, query segmentation and brand/category. |

The judge labels are AI-generated by `claude-opus-5` and `gpt-5.6-sol`, not
human judgments, and they may be wrong. They are released as evaluation labels
so that readers can check this benchmark's numbers; the ESCI human labels are
the only human labels in the pack.

## What is not shipped

The public clone is the runnable harness and the replay pack, not the private
workspace dump.

- **Reference manifests and consensus labels with item text** (`references/`,
  ~20 MB) are not published here. Rebuild them with the scripts below, or copy
  the JSONL files into `references/` if you have the private workspace. The
  replay pack carries the same labels without text.
- **Raw prediction JSONL with item text, caches, smoke runs, and Colab zips**
  stay out of git.
- **Isolated virtualenvs** (`.venvs/`) are local-only.
- **Source catalogs** used to *build* the pilots and reference sets (Amazon
  ESCI parquet, ORCAS-I, enrichment outputs) live outside this folder.
  `build_pilots.py` reads them from `--esci-dir` and `--enrichment-dir` (or
  `APPE_ESCI_DIR` and `APPE_ENRICHMENT_DIR`; default `data/esci` and
  `data/enrichment`). `build_reference_sets.py` and `build_esci_eval_slice.py`
  have no path flags and read the same two locations;
  `build_reference_sets.py` also reads `data/orcas_i/ORCAS-I-gold.tsv`.
  `build_replay_pack.py` reads from `--private-root` (or `APPE_PRIVATE_ROOT`).
  `pilots/pilot_summary.json` names those inputs by file name only.

Without `references/`, `run_benchmark.py --dataset reference` cannot run, and
three tests that load the 2,000-item manifests or consensus labels skip. The
100-item pilots under `pilots/` **are** included.

## Install

```bash
cd code/appendix_system_one_benchmark
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `TYPESAFE_API_KEY` only if you will call
hosted Jev. Do not commit `.env`.

Optional stacks:

- **Laya:** `torch`, `transformers`, and the `laya` package, plus a GPU if you
  want comparable latency.
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

The tests are offline (schemas, metrics, calibration, bootstrap, payload
conversion, cost arithmetic, replay verification on a synthetic pack, and the
committed pack's schema and hashes). `tests/test_hygiene.py` checks that no
file git would track here (including the decompressed pack files) contains an
absolute Windows or home-directory path or a private workspace folder name,
that no `.env` file is present, and that no file exceeds 50 MB. Tests that need
`references/` skip when those files are absent.

## Rebuild labels (optional)

Pilots (100 items/task) are already in `pilots/`. To rebuild them you need the
private ESCI / enrichment inputs that `build_pilots.py` reads:

```bash
python build_pilots.py --esci-dir /path/to/esci --enrichment-dir /path/to/enrichment
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

`build_replay_pack.py --private-root /path/to/private/benchmark` rebuilds
`replay/` from the private references, consensus labels, predictions and
adjudication cache. It refuses to write any string outside a closed vocabulary
of IDs and labels, then runs `verify` on the new pack. To reproduce the
committed `MANIFEST.json` byte for byte, pass the `--build-date` and
`--builder-commit` values it records.

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
reproduction path for JevLite and Kev-4B on a Colab GPU runtime. The committed
runs did not record the GPU type or the checkpoint revisions. Pack a bundle
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
├── eval/                     # loaders, metrics, runner, calibration, intervals, replay pack
├── rubrics/                  # dual-judge instructions
├── tests/
├── colab/                    # Colab notebook + server launcher
├── pilots/                   # 100-item manifests
├── config/reference_costs.json
├── replay/                   # text-free labels and predictions + MANIFEST.json
├── run_benchmark.py
├── write_comparison_report.py
├── build_replay_pack.py      # rebuilds replay/ from the private workspace
├── replay_scores.py          # verify / report / figures on a CPU
├── make_figures.py
├── CLOUD_RUN.md
└── outputs/
    ├── benchmark/            # committed summaries + ZERO_SHOT_RESULTS.md
    └── replay/               # verify_report.json, REPLAY_RESULTS.md, replay_metrics.json, figures/
```

## Secrets

`.env` with a live `TYPESAFE_API_KEY` must never be committed. Use
`.env.example`. The appendix `.gitignore` blocks `.env`, `.venvs/`,
`references/`, prediction dumps, caches, and zip artifacts. The repo-root
`.gitignore` adds the standard Python ignores (including `.env`) and
`code/chapter*/outputs/` and `code/chapter*/cache/`.
