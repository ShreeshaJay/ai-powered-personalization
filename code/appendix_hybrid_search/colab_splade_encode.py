"""
SPLADE Encoding for Google Colab (A100 GPU).

Run this script on Colab to encode 1.2M products with SPLADE models.
Download the resulting cache files and place them in your local cache/ directory.

Setup on Colab:
  1. Upload the ESCI dataset parquet files to Colab (or mount Google Drive)
  2. pip install sentence-transformers>=5.0 transformers torch scipy numpy tqdm
  3. Run this script with the desired model variant
  4. Download the output .npz and .npy files

Usage:
  python colab_splade_encode.py --model qdrant_esci --data_dir /content/esci_data
  python colab_splade_encode.py --model naver --data_dir /content/esci_data
  python colab_splade_encode.py --model prithivi --data_dir /content/esci_data

Resume after disconnect (automatically detects checkpoint):
  python colab_splade_encode.py --model qdrant_esci --data_dir /content/esci_data
"""

import os
# Fix CUDA memory fragmentation — must be set before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import scipy.sparse as sp
import torch
import time
import argparse
import logging
import json
import gc
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# Model configs (mirrors config.py but standalone for Colab)
# =============================================================================

SPLADE_MODELS = {
    "prithivi": {
        "model_name": "prithivida/Splade_PP_en_v2",
        "use_sparse_encoder": False,
        "label": "SPLADE (Prithivi)",
    },
    "qdrant_esci": {
        "model_name": "thierrydamiba/splade-ecommerce-esci",
        "use_sparse_encoder": True,
        "label": "SPLADE (Qdrant ESCI)",
    },
    "naver": {
        "model_name": "naver/splade-cocondenser-ensembledistil",
        "use_sparse_encoder": True,
        "label": "SPLADE (Naver)",
    },
}

VOCAB_SIZE = 30522  # BERT vocabulary size
TEXT_TEMPLATE = "{title}. {bullet_point}. Brand: {brand}"

# Checkpoint every N products (saves partial .npz to disk)
CHECKPOINT_INTERVAL = 50000


def load_products(data_dir: str, max_products: int = 0):
    """Load ESCI product catalog and serialize to text."""
    import pandas as pd

    products_path = Path(data_dir) / "shopping_queries_dataset_products.parquet"
    logger.info(f"Loading products from {products_path}...")

    products_df = pd.read_parquet(products_path)
    products_df = products_df[products_df["product_locale"] == "us"].copy()

    text_cols = ["product_title", "product_bullet_point",
                 "product_description", "product_brand", "product_color"]
    for col in text_cols:
        products_df[col] = products_df[col].fillna("")

    if max_products > 0 and len(products_df) > max_products:
        products_df = products_df.head(max_products)

    logger.info(f"  Products loaded: {len(products_df):,}")

    # Serialize to text
    product_ids = []
    product_texts = []
    for _, row in tqdm(products_df.iterrows(), total=len(products_df), desc="Serializing"):
        text = TEXT_TEMPLATE.format(
            title=row["product_title"],
            bullet_point=row["product_bullet_point"],
            brand=row["product_brand"],
        )
        text = " ".join(text.split())
        product_ids.append(row["product_id"])
        product_texts.append(text)

    logger.info(f"  Product texts serialized: {len(product_texts):,}")
    return product_ids, product_texts


def get_checkpoint_path(output_dir):
    """Return paths for checkpoint files."""
    output_dir = Path(output_dir)
    return {
        "vectors": output_dir / "checkpoint_vectors.npz",
        "meta": output_dir / "checkpoint_meta.json",
    }


def save_checkpoint(output_dir, chunks, n_encoded, total):
    """Save encoding checkpoint to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cp = get_checkpoint_path(output_dir)

    # Stack all chunks into one sparse matrix and save
    partial = sp.vstack(chunks, format="csr")
    sp.save_npz(cp["vectors"], partial)

    meta = {"n_encoded": n_encoded, "total": total, "timestamp": time.time()}
    with open(cp["meta"], "w") as f:
        json.dump(meta, f)

    logger.info(f"  💾 Checkpoint saved: {n_encoded:,}/{total:,} products ({n_encoded/total*100:.1f}%)")


def load_checkpoint(output_dir):
    """Load checkpoint if it exists. Returns (sparse_matrix, n_encoded) or (None, 0)."""
    cp = get_checkpoint_path(output_dir)
    if cp["meta"].exists() and cp["vectors"].exists():
        with open(cp["meta"]) as f:
            meta = json.load(f)
        vectors = sp.load_npz(cp["vectors"])
        n_encoded = meta["n_encoded"]
        logger.info(f"  ✅ Checkpoint found: {n_encoded:,}/{meta['total']:,} products already encoded")
        logger.info(f"     Resuming from product {n_encoded:,}...")
        return vectors, n_encoded
    return None, 0


def clear_checkpoint(output_dir):
    """Remove checkpoint files after successful completion."""
    cp = get_checkpoint_path(output_dir)
    for f in cp.values():
        p = Path(f)
        if p.exists():
            p.unlink()
    logger.info("  Checkpoint files cleaned up.")


def encode_with_sparse_encoder(model_name, texts, batch_size=64, device="cuda",
                                output_dir=None):
    """Encode texts using SparseEncoder (sentence-transformers v5) with checkpointing."""
    from sentence_transformers import SparseEncoder

    # Check for existing checkpoint
    checkpoint_vectors, start_idx = load_checkpoint(output_dir)

    if start_idx >= len(texts):
        logger.info("  All products already encoded from checkpoint!")
        return checkpoint_vectors

    remaining_texts = texts[start_idx:]

    logger.info(f"Loading SparseEncoder: {model_name}")
    encoder = SparseEncoder(model_name, device=device)

    logger.info(f"Encoding {len(remaining_texts):,} texts (batch_size={batch_size})...")
    logger.info("  (convert_to_tensor=False to avoid GPU OOM)")

    # Encode in chunks with checkpointing
    all_chunks = []
    if checkpoint_vectors is not None:
        all_chunks.append(checkpoint_vectors)

    n_encoded = start_idx
    chunk_outputs = []
    t0 = time.time()

    # Process in mini-batches, accumulate, checkpoint periodically
    # Use smaller encode chunks to prevent GPU memory buildup
    encode_chunk_size = min(batch_size, 32)  # Cap at 32 to limit GPU memory

    for batch_start in tqdm(range(0, len(remaining_texts), encode_chunk_size),
                            total=(len(remaining_texts) + encode_chunk_size - 1) // encode_chunk_size,
                            desc=f"SPLADE (from {start_idx:,})"):
        batch_texts = remaining_texts[batch_start:batch_start + encode_chunk_size]

        batch_out = encoder.encode(
            batch_texts, batch_size=encode_chunk_size, show_progress_bar=False,
            convert_to_tensor=False, convert_to_sparse_tensor=False,
        )

        # Convert batch output to CSR (on CPU — keeps GPU free)
        if isinstance(batch_out, (sp.spmatrix, sp.sparray)):
            chunk_outputs.append(sp.csr_matrix(batch_out))
        elif isinstance(batch_out, list) and len(batch_out) > 0 and torch.is_tensor(batch_out[0]):
            batch_np = torch.stack(batch_out).cpu().numpy()
            chunk_outputs.append(sp.csr_matrix(batch_np))
        else:
            batch_np = np.array(batch_out)
            chunk_outputs.append(sp.csr_matrix(batch_np))

        # Free GPU memory after every batch
        del batch_out
        torch.cuda.empty_cache()

        n_encoded += len(batch_texts)

        # Checkpoint every CHECKPOINT_INTERVAL products
        if n_encoded % CHECKPOINT_INTERVAL < encode_chunk_size and n_encoded > start_idx + encode_chunk_size:
            merged = sp.vstack(all_chunks + chunk_outputs, format="csr")
            save_checkpoint(output_dir, [merged], n_encoded, len(texts))
            # Keep merged as single chunk going forward
            all_chunks = [merged]
            chunk_outputs = []
            gc.collect()
            torch.cuda.empty_cache()
            elapsed = time.time() - t0
            rate = (n_encoded - start_idx) / elapsed
            eta = (len(texts) - n_encoded) / rate if rate > 0 else 0
            logger.info(f"    Rate: {rate:.1f} products/sec, ETA: {eta/60:.1f}min")

    elapsed = time.time() - t0
    logger.info(f"  Encoding completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Final merge
    all_chunks.extend(chunk_outputs)
    result = sp.vstack(all_chunks, format="csr")

    if result.shape[1] != VOCAB_SIZE:
        result = sp.csr_matrix(
            (result.data, result.indices, result.indptr),
            shape=(result.shape[0], VOCAB_SIZE),
        )

    logger.info(f"  Result: {result.shape}, nnz={result.nnz:,}")
    clear_checkpoint(output_dir)
    return result


def encode_with_automodel(model_name, texts, batch_size=32, device="cuda",
                           output_dir=None):
    """Encode texts using AutoModelForMaskedLM (Prithivi model) with checkpointing."""
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    # Check for existing checkpoint
    checkpoint_vectors, start_idx = load_checkpoint(output_dir)

    if start_idx >= len(texts):
        logger.info("  All products already encoded from checkpoint!")
        return checkpoint_vectors

    remaining_texts = texts[start_idx:]

    logger.info(f"Loading AutoModelForMaskedLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    logger.info(f"Encoding {len(remaining_texts):,} texts (batch_size={batch_size})...")

    all_chunks = []
    if checkpoint_vectors is not None:
        all_chunks.append(checkpoint_vectors)

    chunk_rows = []
    n_encoded = start_idx
    n_batches = (len(remaining_texts) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in tqdm(range(0, len(remaining_texts), batch_size),
                          total=n_batches,
                          desc=f"SPLADE (from {start_idx:,})"):
        batch_texts = remaining_texts[batch_idx:batch_idx + batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="pt", max_length=256,
            truncation=True, padding=True,
        ).to(device)

        with torch.no_grad():
            output = model(**tokens)
            logits = output.logits
            splade_vectors = torch.log1p(torch.relu(logits))
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            splade_vectors = splade_vectors * attention_mask
            splade_vectors = splade_vectors.max(dim=1).values

        batch_np = splade_vectors.cpu().numpy()
        chunk_rows.append(sp.csr_matrix(batch_np))
        n_encoded += len(batch_texts)

        # Checkpoint every CHECKPOINT_INTERVAL products
        if n_encoded % CHECKPOINT_INTERVAL < batch_size and n_encoded > start_idx + batch_size:
            merged = sp.vstack(all_chunks + chunk_rows, format="csr")
            save_checkpoint(output_dir, [merged], n_encoded, len(texts))
            all_chunks = [merged]
            chunk_rows = []
            elapsed = time.time() - t0
            rate = (n_encoded - start_idx) / elapsed
            eta = (len(texts) - n_encoded) / rate if rate > 0 else 0
            logger.info(f"    Rate: {rate:.1f} products/sec, ETA: {eta/60:.1f}min")

    elapsed = time.time() - t0
    logger.info(f"  Encoding completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    all_chunks.extend(chunk_rows)
    result = sp.vstack(all_chunks, format="csr")
    logger.info(f"  Result: {result.shape}, nnz={result.nnz:,}")
    clear_checkpoint(output_dir)
    return result


def save_index(product_ids, product_vectors, output_dir):
    """Save SPLADE index to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_file = output_dir / "splade_product_vectors.npz"
    ids_file = output_dir / "splade_product_ids.npy"

    sp.save_npz(vectors_file, product_vectors)
    np.save(ids_file, np.array(product_ids))

    logger.info(f"  Saved vectors: {vectors_file} ({vectors_file.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"  Saved IDs: {ids_file}")


def main():
    parser = argparse.ArgumentParser(description="SPLADE Encoding for Colab")
    parser.add_argument("--model", type=str, required=True,
                        choices=["prithivi", "qdrant_esci", "naver"],
                        help="SPLADE model variant")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ESCI dataset directory (with parquet files)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./cache/splade/{model})")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding (A100 can handle 128+)")
    parser.add_argument("--max_products", type=int, default=0,
                        help="Max products to encode (0 = all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    model_config = SPLADE_MODELS[args.model]
    output_dir = args.output_dir or f"./cache/splade/{args.model}"

    logger.info(f"\n{'='*60}")
    logger.info(f"  SPLADE Encoding: {model_config['label']}")
    logger.info(f"  Model: {model_config['model_name']}")
    logger.info(f"  Backend: {'SparseEncoder' if model_config['use_sparse_encoder'] else 'AutoModelForMaskedLM'}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'='*60}")

    # Check GPU
    if args.device == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("  CUDA not available, falling back to CPU")
            args.device = "cpu"

    # Load data
    product_ids, product_texts = load_products(args.data_dir, args.max_products)

    # Encode (with automatic checkpoint resume)
    t0 = time.time()
    if model_config["use_sparse_encoder"]:
        product_vectors = encode_with_sparse_encoder(
            model_config["model_name"], product_texts,
            batch_size=args.batch_size, device=args.device,
            output_dir=output_dir,
        )
    else:
        product_vectors = encode_with_automodel(
            model_config["model_name"], product_texts,
            batch_size=args.batch_size, device=args.device,
            output_dir=output_dir,
        )
    total_time = time.time() - t0

    logger.info(f"\n  Total encoding time: {total_time:.1f}s ({total_time/60:.1f}m)")

    # Save final index
    save_index(product_ids, product_vectors, output_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE! Copy these files to your local machine:")
    logger.info(f"  {output_dir}/splade_product_vectors.npz")
    logger.info(f"  {output_dir}/splade_product_ids.npy")
    logger.info(f"")
    logger.info(f"  Place them in: cache/splade/{args.model}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
