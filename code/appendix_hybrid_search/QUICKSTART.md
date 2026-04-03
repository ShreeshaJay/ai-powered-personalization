# Appendix Hybrid Search - Quick Start Commands

## Setup (run once per new PowerShell window)

```powershell
cd "C:\Users\shree\OneDrive\Desktop\RecSys ML Book Writing\RecSys book code\Appendix Hybrid Search"
& "C:\Users\shree\OneDrive\Desktop\RecSys ML Book Writing\recsys_venv\Scripts\Activate.ps1"
```

## Evaluation

### Curated Pool (all 3 SPLADE variants)
```powershell
python hybrid_search.py --mode evaluate --eval_mode curated --max_queries 2000 --splade_models "prithivi,qdrant_esci,naver"
```

### Curated Pool (Prithivi only — faster)
```powershell
python hybrid_search.py --mode evaluate --eval_mode curated --max_queries 2000
```

## LLM Labeling (Gemini 2.5 Flash Lite)

### Run labeler (resumes automatically from last checkpoint)
```powershell
python llm_labeler.py --top_k 100 --max_queries 10000 --concurrency 100
```

## Encoding (only needed once, or after code changes)

### Dense (SBERT) encoding
```powershell
python dense_search.py --mode encode
```

### SPLADE encoding (one variant at a time)
```powershell
python splade_search.py --mode encode --splade_model prithivi
python splade_search.py --mode encode --splade_model qdrant_esci
python splade_search.py --mode encode --splade_model naver
```

## Useful Checks

### Check LLM labeling progress
```powershell
python -c "import json; d=json.load(open('outputs/metrics/llm_labels.json')); print(f'Labels: {len(d):,}')"
```

### Check evaluation results
```powershell
python -c "import json; d=json.load(open('outputs/metrics/curated_pool_results.json')); [print(f'{k}: NDCG@10={v[\"ndcg@10\"]:.4f}, MRR@10={v[\"mrr@10\"]:.4f}') for k,v in d.items()]"
```
