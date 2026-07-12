# GR2 Recommendations Methodology

This folder contains the reader-facing methodology draft for a GR2-inspired recommendation reranking exercise for Chapter 12, **LLMs for Recommenders**.

The experiment studies a multi-stage recommendation funnel on Amazon Reviews 2023 `Beauty_and_Personal_Care`:

```text
history-based retrieval -> candidate slate -> listwise generative reranking
```

The current public folder intentionally contains methodology and run instructions only. Experimental code, checkpoints, cached datasets, candidate slates, and generated outputs remain outside this repository until reviewed.

## Goals

- Show how a GR2-style listwise reranker fits after retrieval.
- Compare behavioral retrieval (SASRec), text retrieval (SBERT/FAISS), and simple popularity baselines.
- Separate **retrieval quality** from **reranking quality** using natural and oracle candidate slates.
- Prepare for zero-shot Qwen/Gemma reranking before any teacher-trace, SFT, or RL work.

## Current Experimental Status

The local workbench lives outside this repository:

```text
RecSys book code/chapter12_llms_for_recs/gr2_experiment/
```

Current state:

- Milestone 1 is implemented in the local workbench.
- Dataset switched from `All_Beauty` to `Beauty_and_Personal_Care` for a more useful tutorial scale.
- Capped Colab run used:

```bash
python run_milestone1.py --max-users-for-split 10000 --max-eval-users 2000 --sasrec-epochs 3
```

Preliminary BPC scale from that run:

| Statistic | Value |
| --- | ---: |
| Users | 10,000 |
| Items | 130,479 |
| Interactions | 828,846 |
| Avg sequence length | 82.88 |

The first capped run exposed an oracle-slate target-injection bug in the experimental code. Corrected metrics should be generated before reporting final Milestone 1 results.

## What Is Not Committed Here

Large or generated artifacts should not be committed to GitHub:

- raw Amazon Reviews data
- cached parquet files
- Hugging Face caches
- trained model checkpoints
- candidate slate dumps
- generated labels
- notebook scratch outputs
- `outputs/` directories

Use Hugging Face datasets/models, Google Drive, or another artifact store for large outputs. This GitHub repo should contain clean code, configs, notebooks with outputs cleared, and documentation.

## Folder Contents

```text
gr2_recommendations/
├── README.md
├── DESIGN_NOTE.md
└── RUNBOOK_COLAB.md
```

## Milestone Plan

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | Retrieval and non-LLM baselines | Experimental workbench implemented; corrected BPC metrics pending |
| M2 | Zero-shot Qwen listwise reranking on fixed/oracle slates | Next |
| M3 | Teacher traces and SFT | Later |
| M4 | RL / verifiable rewards / distillation | Design-only unless needed |

## Related Track

Track B is a separate search-focused effort on Amazon ESCI. Keep Track A recommendation code independent unless a small, reviewed shared utility becomes useful.
