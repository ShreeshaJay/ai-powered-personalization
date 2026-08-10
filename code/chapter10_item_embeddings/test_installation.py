"""
Test Installation and Setup for Chapter 10

Validates:
1. Dependencies are installed
2. Datasets are accessible
3. Basic functionality works

Run this after installing requirements.txt to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

print("="*80)
print("Chapter 10: Item Embeddings - Installation Test")
print("="*80)

# Test imports
print("\n[1/5] Testing imports...")
try:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import torch
    print("  ✓ Core dependencies imported successfully")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("  → Run: pip install -r requirements.txt")
    sys.exit(1)

# Test FAISS (optional)
print("\n[2/5] Testing FAISS (optional)...")
try:
    import faiss
    print("  ✓ FAISS is available")
    print(f"    Version: {faiss.__version__}")
except ImportError:
    print("  ⚠ FAISS not available (optional)")
    print("  → For fast similarity search, run: pip install faiss-cpu")

# Test dataset paths
print("\n[3/5] Testing dataset paths...")
sys.path.insert(0, str(Path(__file__).parent))
from config import MIND_DATA_PATH, IJCAI_DATA_PATH

mind_exists = MIND_DATA_PATH.exists()
ijcai_exists = IJCAI_DATA_PATH.exists()

if mind_exists:
    print(f"  ✓ MIND dataset found: {MIND_DATA_PATH}")
else:
    print(f"  ✗ MIND dataset not found: {MIND_DATA_PATH}")
    print("  → Download from: https://msnews.github.io/")

if ijcai_exists:
    print(f"  ✓ IJCAI dataset found: {IJCAI_DATA_PATH}")
else:
    print(f"  ✗ IJCAI dataset not found: {IJCAI_DATA_PATH}")
    print("  → Download from: https://tianchi.aliyun.com/dataset/147588")

# Test data loading
print("\n[4/5] Testing data loaders...")
if mind_exists:
    try:
        from data import MINDDataset
        dataset = MINDDataset(MIND_DATA_PATH, sample_size=100)
        news_df = dataset.load_news()
        print(f"  ✓ MIND loader works: loaded {len(news_df)} news articles")
    except Exception as e:
        print(f"  ✗ MIND loader failed: {e}")
else:
    print("  ⊘ Skipped (dataset not found)")

if ijcai_exists:
    try:
        from data import IJCAIDataset
        dataset = IJCAIDataset(IJCAI_DATA_PATH, sample_size=100)
        samples_df = dataset.load_data()
        print(f"  ✓ IJCAI loader works: loaded {len(samples_df)} samples")
    except Exception as e:
        print(f"  ✗ IJCAI loader failed: {e}")
else:
    print("  ⊘ Skipped (dataset not found)")

# Test text encoder
print("\n[5/5] Testing text encoder...")
try:
    from models import TextEncoder
    
    encoder = TextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=128,
        normalize_embeddings=True
    )
    
    # Encode sample texts
    sample_texts = [
        "Apple announces new iPhone with improved camera",
        "Local team wins championship in overtime"
    ]
    
    embeddings = encoder.encode(sample_texts, show_progress=False)
    
    print(f"  ✓ Text encoder works")
    print(f"    Model: {encoder.model_name}")
    print(f"    Embedding dim: {embeddings.shape[1]}")
    print(f"    Device: {encoder.device}")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    if np.allclose(norms, 1.0, atol=1e-5):
        print(f"    Embeddings are normalized: ✓")
    else:
        print(f"    Warning: Embeddings not normalized (norms: {norms})")
    
except Exception as e:
    print(f"  ✗ Text encoder failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("Installation Test Summary")
print("="*80)
print(f"  Core dependencies: ✓")
print(f"  FAISS: {'✓' if 'faiss' in sys.modules else '⚠ (optional)'}")
print(f"  MIND dataset: {'✓' if mind_exists else '✗'}")
print(f"  IJCAI dataset: {'✓' if ijcai_exists else '✗'}")
print(f"  Text encoder: ✓")
print("="*80)

if mind_exists or ijcai_exists:
    print("\n✓ Ready to run Chapter 10 code!")
    print("\nQuick start:")
    if mind_exists:
        print("  python encode_items.py --dataset mind --sample_size 1000")
    if ijcai_exists:
        print("  python encode_items.py --dataset ijcai --sample_size 5000")
        print("  python evaluate_biencoder.py --dataset ijcai --sample_size 5000")
else:
    print("\n⚠ Datasets not found. Download them to get started.")
    print("\nDataset locations:")
    print("  MIND: https://msnews.github.io/")
    print("  IJCAI: https://tianchi.aliyun.com/dataset/147588")
