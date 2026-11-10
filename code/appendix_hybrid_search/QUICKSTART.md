# Appendix B: Hybrid Search - Quick Start Commands

## Setup (once)

From the repository root:

```powershell
cd code\appendix_hybrid_search
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Evaluation

### Curated Pool (all 3 SPLADE variants)
```powershell
python hybrid_search.py --mode evaluate --eval_mode curated --max_queries 2000 --splade_models "prithivi,qdrant_esci,naver"
```

### Curated Pool (Prithivi only, faster)
```powershell
python hybrid_search.py --mode evaluate --eval_mode curated --max_queries 2000
```

## LLM Labeling (Gemini 2.5 Flash Lite; paid API, needs an API key)

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
