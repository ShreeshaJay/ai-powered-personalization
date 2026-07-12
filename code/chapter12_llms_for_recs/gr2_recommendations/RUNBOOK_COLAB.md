# Colab Runbook: GR2 Recommendation Experiments

This runbook documents the intended cloud workflow for the Track A recommendation experiments. It is written for a clean reader-facing repository, so generated outputs and large data files are assumed to live outside GitHub.

## Runtime

Use Google Colab when:

- running `Beauty_and_Personal_Care` preprocessing at category scale
- running Qwen/Gemma listwise reranking
- local memory or GPU availability is limited

Recommended runtime:

- High-RAM Colab runtime for preprocessing
- GPU runtime for SASRec/SBERT/Qwen
- A100 preferred for Qwen 7B-class models

Check runtime resources:

```python
import os
import torch

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
os.system("free -h")
```

## Suggested Drive Layout

```text
MyDrive/
└── RecSys/
    └── gr2_experiment/
        ├── source/
        ├── outputs/
        └── artifacts/
```

Use:

- `source/` for code uploaded from the reviewed workbench
- `outputs/` for copied metrics
- `artifacts/` for copied checkpoints or candidate slates, if needed

Do not commit these generated outputs to GitHub.

## Notebook Setup

Create the notebook beside `source/`, for example:

```text
MyDrive/RecSys/gr2_experiment/milestone1_bpc_colab.ipynb
```

Mount Drive and enter the source folder:

```python
from google.colab import drive
drive.mount("/content/drive")

import os
import sys

ROOT = "/content/drive/MyDrive/RecSys/gr2_experiment/source"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

print(os.getcwd())
```

Install dependencies:

```python
!pip install -q huggingface_hub polars pyarrow numpy sentence-transformers faiss-cpu
```

For Milestone 2 Qwen inference:

```python
!pip install -q transformers accelerate bitsandbytes
```

## Milestone 1: Capped BPC Run

The `Beauty_and_Personal_Care` category is much larger than `All_Beauty`. Use caps until SASRec scoring is batched or otherwise optimized.

Example capped run:

```python
!python run_milestone1.py \
  --max-users-for-split 10000 \
  --max-eval-users 2000 \
  --sasrec-epochs 3
```

Use `--force-download` only when intentionally rebuilding local cache files:

```python
!python run_milestone1.py \
  --max-users-for-split 10000 \
  --max-eval-users 2000 \
  --sasrec-epochs 3 \
  --force-download
```

Expected outputs:

```text
outputs/metrics/milestone1_latest.json
outputs/artifacts/split_bundle.json
outputs/artifacts/sasrec_milestone1.pt
```

Copy useful outputs to Drive after each successful run:

```python
!mkdir -p "/content/drive/MyDrive/RecSys/gr2_experiment/outputs/milestone1_bpc"
!cp -r outputs/metrics "/content/drive/MyDrive/RecSys/gr2_experiment/outputs/milestone1_bpc/"
!cp -r outputs/artifacts "/content/drive/MyDrive/RecSys/gr2_experiment/outputs/milestone1_bpc/"
```

## Metric Inspection

Print key retriever metrics:

```python
import json
from pathlib import Path

summary = json.loads(Path("outputs/metrics/milestone1_latest.json").read_text())

print("Run:", summary["run_id"])
print("Smoke:", summary["smoke"])
print("Split:", summary["split"])
print("Eval instances:", summary["num_eval_instances"])
print("Data stats:", summary["data_stats"])
print()

for result in summary["retriever_results"]:
    retriever = result["retriever"]
    natural = result["metrics"]["natural_slate"]["incoming_order"]
    oracle = result["metrics"]["oracle_slate"]["incoming_order"]

    print(f"=== {retriever} ===")
    print("natural target_in_slate_rate:", round(natural.get("target_in_slate_rate", 0), 4))
    print("natural recall@10:", round(natural.get("recall@10", 0), 4))
    print("natural ndcg@10:", round(natural.get("ndcg@10", 0), 4))
    print("oracle recall@10:", round(oracle.get("recall@10", 0), 4))
    print("oracle ndcg@10:", round(oracle.get("ndcg@10", 0), 4))
    print("oracle mrr:", round(oracle.get("mrr", 0), 4))
    print()
```

## Milestone 2: Qwen Listwise Reranking

Milestone 2 should load fixed/oracle candidate slates and ask an instruction-tuned model to return a candidate permutation.

Recommended initial models:

| Model | Use |
| --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | smoke tests |
| `Qwen/Qwen2.5-7B-Instruct` | stronger Colab/A100 run |

Example 4-bit loading pattern:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "Qwen/Qwen2.5-7B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
)
```

Milestone 2 reports should include both ranking metrics and output-validity metrics:

- valid JSON rate
- valid permutation rate
- duplicate candidate rate
- missing candidate rate
- out-of-candidate ID rate
- parse failure rate
- latency and token usage

## Artifact Storage

Store large generated outputs outside GitHub. A Hugging Face dataset/model repository is a natural fit for:

- candidate slate files
- trained checkpoints
- model outputs
- metrics bundles
- dataset cards
- model cards

The GitHub repo should describe how to reproduce or download those artifacts, not store them directly.
