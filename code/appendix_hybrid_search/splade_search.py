"""
SPLADE learned sparse search using prithivida/Splade_PP_en_v2.

SPLADE uses a BERT masked language model to produce sparse vocabulary-sized
vectors where each dimension corresponds to a BERT wordpiece token. The model
learns to expand queries and documents with semantically related terms,
enabling both exact matching AND semantic matching in sparse space.

Key insight: SPLADE activates tokens beyond those literally present in the text.
For 'wireless headphones', it might activate 'bluetooth', 'earbuds', 'audio'.

Uses Apache 2.0 licensed model (NOT the Naver CC-BY-NC-SA models).

Usage:
  python splade_search.py --mode encode --device cuda
  python splade_search.py --mode evaluate --max_queries 5000
  python splade_search.py --mode visualize --demo_query "notebook for travel"
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

from config import (
    SPLADEConfig, DEFAULT_SPLADE_CONFIG, DEFAULT_DATA_CONFIG,
    DEFAULT_EVAL_CONFIG, METRICS_DIR, PLOTS_DIR, SPLADE_MODELS,
)
from data_loader import ESCIDataset
from evaluate import (
    evaluate_candidate_scoring, evaluate_retrieval,
    print_evaluation_results, save_results_json,
    plot_score_boxplots,
)

logger = logging.getLogger(__name__)


class SPLADEEncoder:
    """SPLADE sparse encoder supporting two backends.

    Backend 1 (AutoModelForMaskedLM) — for prithivida/Splade_PP_en_v2:
      1. Tokenize text with BERT tokenizer
      2. Forward pass through BERT MLM to get per-token logits (vocab_size)
      3. Apply ReLU + log(1 + x) activation (SPLADE transform)
      4. Max-pool across token positions -> single vocab-sized vector

    Backend 2 (SparseEncoder from sentence-transformers v5) — for Qdrant/Naver models:
      Uses sentence_transformers.SparseEncoder which handles the full pipeline
      internally. Required for models that don't support raw AutoModelForMaskedLM.

    Result: a sparse vector where non-zero dimensions indicate which
    vocabulary terms are relevant and how strongly.
    """

    def __init__(self, config: SPLADEConfig = DEFAULT_SPLADE_CONFIG):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.sparse_encoder = None  # SparseEncoder backend
        self.device = None

    def load_model(self, device: str = "cpu") -> None:
        """Load SPLADE model using the appropriate backend."""
        self.device = torch.device(device)
        logger.info(f"Loading SPLADE model: {self.config.model_name}")
        logger.info(f"  Backend: {'SparseEncoder' if self.config.use_sparse_encoder else 'AutoModelForMaskedLM'}")
        logger.info(f"  Device: {self.device}")

        if self.config.use_sparse_encoder:
            from sentence_transformers import SparseEncoder
            self.sparse_encoder = SparseEncoder(
                self.config.model_name, device=str(self.device)
            )
            # Get tokenizer for visualization (term expansion)
            self.tokenizer = self.sparse_encoder.tokenizer
            logger.info(f"  SparseEncoder loaded. Vocab size: {self.tokenizer.vocab_size:,}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"  AutoModelForMaskedLM loaded. Vocab size: {self.tokenizer.vocab_size:,}")

    def _sparse_output_to_dense(self, sparse_output) -> np.ndarray:
        """Convert SparseEncoder output to a dense vocab-sized vector."""
        vector = np.zeros(self.config.vocab_size, dtype=np.float32)
        if hasattr(sparse_output, 'indices') and hasattr(sparse_output, 'values'):
            # sentence-transformers SparseEncoder returns objects with .indices/.values
            indices = sparse_output.indices.cpu().numpy() if torch.is_tensor(sparse_output.indices) else np.array(sparse_output.indices)
            values = sparse_output.values.cpu().numpy() if torch.is_tensor(sparse_output.values) else np.array(sparse_output.values)
            vector[indices] = values
        elif isinstance(sparse_output, dict):
            for idx, val in sparse_output.items():
                vector[int(idx)] = float(val)
        elif isinstance(sparse_output, (sp.spmatrix, sp.sparray)):
            dense = sparse_output.toarray().flatten()
            vector[:len(dense)] = dense
        return vector

    def _sparse_outputs_to_csr(self, sparse_outputs) -> sp.csr_matrix:
        """Convert batch of SparseEncoder outputs to scipy CSR matrix."""
        rows, cols, data = [], [], []
        for row_idx, output in enumerate(sparse_outputs):
            if hasattr(output, 'indices') and hasattr(output, 'values'):
                indices = output.indices.cpu().numpy() if torch.is_tensor(output.indices) else np.array(output.indices)
                values = output.values.cpu().numpy() if torch.is_tensor(output.values) else np.array(output.values)
            elif isinstance(output, dict):
                indices = np.array([int(k) for k in output.keys()])
                values = np.array([float(v) for v in output.values()])
            else:
                continue
            rows.extend([row_idx] * len(indices))
            cols.extend(indices.tolist())
            data.extend(values.tolist())

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(sparse_outputs), self.config.vocab_size),
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to a dense SPLADE vector (for visualization).

        Returns a vocab_size-dimensional vector (not sparse).
        """
        if self.config.use_sparse_encoder:
            outputs = self.sparse_encoder.encode(
                [text], convert_to_tensor=False, convert_to_sparse_tensor=False,
            )
            if isinstance(outputs, (sp.spmatrix, sp.sparray)):
                return outputs.toarray().flatten()
            elif isinstance(outputs, list) and len(outputs) > 0:
                v = outputs[0]
                if torch.is_tensor(v):
                    v = v.cpu().numpy()
                elif not isinstance(v, np.ndarray):
                    v = np.array(v)
                # Pad to vocab_size if needed
                if v.shape[0] < self.config.vocab_size:
                    padded = np.zeros(self.config.vocab_size, dtype=v.dtype)
                    padded[:v.shape[0]] = v
                    return padded
                return v
            elif torch.is_tensor(outputs):
                return outputs[0].cpu().numpy()
            return self._sparse_output_to_dense(outputs[0])

        # AutoModelForMaskedLM path
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)
            logits = output.logits  # (1, seq_len, vocab_size)

            # SPLADE transform: log(1 + ReLU(logits))
            splade_vector = torch.log1p(torch.relu(logits))

            # Max-pool across token positions, mask padding
            attention_mask = tokens["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            splade_vector = splade_vector * attention_mask
            splade_vector = splade_vector.max(dim=1).values  # (1, vocab_size)

        return splade_vector.squeeze(0).cpu().numpy()

    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> sp.csr_matrix:
        """Encode texts to sparse SPLADE vectors.

        Returns a scipy CSR matrix of shape (N, vocab_size).
        Each row is a sparse SPLADE representation.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if self.config.use_sparse_encoder:
            logger.info(f"  Encoding {len(texts):,} texts with SparseEncoder...")
            logger.info(f"  (convert_to_tensor=False to avoid GPU OOM on large catalogs)")
            outputs = self.sparse_encoder.encode(
                texts, batch_size=batch_size, show_progress_bar=show_progress,
                convert_to_tensor=False, convert_to_sparse_tensor=False,
            )
            # Convert outputs to scipy CSR matrix
            if isinstance(outputs, (sp.spmatrix, sp.sparray)):
                result = sp.csr_matrix(outputs)
            elif isinstance(outputs, list) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                # List of dense torch tensors (30522-dim each) — convert to CSR
                # Process in chunks to avoid memory spike from stacking all at once
                chunk_size = 10000
                sparse_chunks = []
                for i in range(0, len(outputs), chunk_size):
                    chunk = outputs[i:i + chunk_size]
                    chunk_np = torch.stack(chunk).numpy()
                    sparse_chunks.append(sp.csr_matrix(chunk_np))
                    if i % 100000 == 0 and i > 0:
                        logger.info(f"    Converting to sparse: {i:,}/{len(outputs):,}")
                result = sp.vstack(sparse_chunks, format="csr")
            else:
                result = self._sparse_outputs_to_csr(outputs)
            # Ensure correct vocab dimension
            if result.shape[1] != self.config.vocab_size:
                result = sp.csr_matrix(
                    (result.data, result.indices, result.indptr),
                    shape=(result.shape[0], self.config.vocab_size),
                )
            logger.info(f"  SPLADE encoding complete: {result.shape}, "
                        f"nnz={result.nnz:,}, "
                        f"density={result.nnz / (result.shape[0] * result.shape[1]):.6f}")
            return result

        # AutoModelForMaskedLM path
        all_rows = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="SPLADE encoding", total=n_batches)

        for start in iterator:
            batch_texts = texts[start:start + batch_size]

            tokens = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**tokens)
                logits = output.logits  # (batch, seq_len, vocab_size)

                # SPLADE transform
                splade_vectors = torch.log1p(torch.relu(logits))

                # Max-pool with attention mask
                attention_mask = tokens["attention_mask"].unsqueeze(-1)
                splade_vectors = splade_vectors * attention_mask
                splade_vectors = splade_vectors.max(dim=1).values  # (batch, vocab_size)

            # Convert to scipy sparse (per batch to save memory)
            batch_np = splade_vectors.cpu().numpy()
            batch_sparse = sp.csr_matrix(batch_np)
            all_rows.append(batch_sparse)

        # Vertical stack all batches
        result = sp.vstack(all_rows, format="csr")
        logger.info(f"  SPLADE encoding complete: {result.shape}, "
                    f"nnz={result.nnz:,}, "
                    f"density={result.nnz / (result.shape[0] * result.shape[1]):.6f}")
        return result


class SPLADEIndex:
    """Sparse retrieval index for SPLADE vectors.

    Retrieval uses: product_vectors @ query_vector.T  (N, vocab) x (vocab, 1) -> (N, 1)
    This avoids transposing the large product matrix (which causes OOM).
    Efficient because both query and document vectors are very sparse.
    """

    def __init__(
        self,
        product_ids: List[str],
        product_vectors: sp.csr_matrix,
    ):
        self.product_ids = product_ids
        self.product_vectors = product_vectors
        self.id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[List[str], np.ndarray]:
        """Search full catalog. Returns (product_ids, scores)."""
        # Convert query to sparse column vector (vocab, 1)
        if isinstance(query_vector, np.ndarray):
            query_col = sp.csc_matrix(query_vector.reshape(-1, 1))
        else:
            query_col = sp.csc_matrix(query_vector.toarray().reshape(-1, 1))

        # Sparse dot product: (N, vocab) x (vocab, 1) -> (N, 1)
        # This avoids transposing the (N, vocab) product matrix
        scores = (self.product_vectors @ query_col).toarray().flatten()

        # Top-K
        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        result_ids = [self.product_ids[idx] for idx in top_indices]
        result_scores = scores[top_indices]
        return result_ids, result_scores

    def score_candidates(
        self,
        query_vector: np.ndarray,
        candidate_ids: List[str],
    ) -> np.ndarray:
        """Score specific candidates using sparse dot product."""
        if isinstance(query_vector, np.ndarray):
            query_col = sp.csc_matrix(query_vector.reshape(-1, 1))
        else:
            query_col = sp.csc_matrix(query_vector.toarray().reshape(-1, 1))

        candidate_indices = []
        for pid in candidate_ids:
            idx = self.id_to_idx.get(pid)
            if idx is not None:
                candidate_indices.append(idx)
            else:
                candidate_indices.append(-1)

        scores = np.zeros(len(candidate_ids), dtype=np.float64)
        valid_indices = [i for i, idx in enumerate(candidate_indices) if idx >= 0]
        if valid_indices:
            valid_doc_indices = [candidate_indices[i] for i in valid_indices]
            candidate_vectors = self.product_vectors[valid_doc_indices]
            dot_scores = (candidate_vectors @ query_col).toarray().flatten()
            for i, score in zip(valid_indices, dot_scores):
                scores[i] = score

        return scores

    def save(self, path: Optional[Path] = None) -> None:
        """Save SPLADE index to disk."""
        save_dir = path or DEFAULT_SPLADE_CONFIG.cache_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        sp.save_npz(save_dir / DEFAULT_SPLADE_CONFIG.product_vectors_file,
                     self.product_vectors)
        np.save(save_dir / DEFAULT_SPLADE_CONFIG.product_ids_file,
                np.array(self.product_ids))
        logger.info(f"  SPLADE index saved to {save_dir}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["SPLADEIndex"]:
        """Load SPLADE index from disk. Returns None if files don't exist."""
        load_dir = path or DEFAULT_SPLADE_CONFIG.cache_dir
        vectors_file = load_dir / DEFAULT_SPLADE_CONFIG.product_vectors_file
        ids_file = load_dir / DEFAULT_SPLADE_CONFIG.product_ids_file

        if not vectors_file.exists() or not ids_file.exists():
            return None

        try:
            product_vectors = sp.load_npz(vectors_file)
            product_ids = np.load(ids_file, allow_pickle=True).tolist()
            logger.info(f"  SPLADE index loaded: {product_vectors.shape}, "
                        f"nnz={product_vectors.nnz:,}")
            return cls(product_ids, product_vectors)
        except Exception as e:
            logger.warning(f"  Failed to load SPLADE index: {e}")
            return None


def visualize_splade_expansion(
    encoder: SPLADEEncoder,
    query: str,
    top_k_tokens: int = 20,
) -> None:
    """Visualize SPLADE term expansion -- the 'aha moment'.

    Shows how SPLADE activates vocabulary tokens beyond those literally
    present in the query. For example, 'wireless headphones' might activate
    'bluetooth', 'earbuds', 'audio', 'earphone'.

    This demonstrates SPLADE's key advantage: semantic expansion in sparse space.
    """
    print(f"\n{'='*60}")
    print(f'  SPLADE Term Expansion: "{query}"')
    print(f"{'='*60}")

    # Encode query
    vector = encoder.encode_single(query)
    nonzero_count = np.count_nonzero(vector)
    print(f"  Non-zero dimensions: {nonzero_count} / {len(vector)}")

    # Get top-K activated tokens
    top_indices = np.argsort(-vector)[:top_k_tokens]

    # Find which tokens are in the original query
    query_tokens_raw = encoder.tokenizer.tokenize(query.lower())
    query_token_set = set(t.replace("##", "") for t in query_tokens_raw)

    print(f"\n  Top-{top_k_tokens} activated tokens:")
    print(f"  {'Token':<20s} {'Weight':>8s}  {'Source'}")
    print(f"  {'-'*50}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens_for_plot = []
    weights_for_plot = []
    colors_for_plot = []

    for idx in top_indices:
        weight = vector[idx]
        if weight <= 0:
            break
        token = encoder.tokenizer.decode([idx]).strip()
        is_original = token.lower().replace("##", "") in query_token_set
        source = "ORIGINAL" if is_original else "EXPANDED"
        marker = "*" if is_original else "+"
        print(f"  {marker} {token:<18s} {weight:>8.3f}  {source}")

        tokens_for_plot.append(token)
        weights_for_plot.append(weight)
        colors_for_plot.append("#2ecc71" if is_original else "#3498db")

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(tokens_for_plot))
    ax.barh(y_pos, weights_for_plot, color=colors_for_plot, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens_for_plot, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("SPLADE Weight", fontsize=11)
    ax.set_title(f'SPLADE Term Expansion: "{query}"', fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Original term"),
        Patch(facecolor="#3498db", label="Expanded term"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    output_path = PLOTS_DIR / "splade_expansion.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Expansion plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SPLADE Learned Sparse Search")
    parser.add_argument("--mode", type=str,
                        choices=["encode", "evaluate", "visualize"],
                        default="evaluate", help="Mode to run")
    parser.add_argument("--max_queries", type=int, default=5000,
                        help="Max queries for evaluation (0 = all)")
    parser.add_argument("--max_products", type=int, default=0,
                        help="Max products to encode (0 = all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--splade_model", type=str,
                        choices=list(SPLADE_MODELS.keys()) + ["all"],
                        default="prithivi",
                        help="SPLADE variant to use (default: prithivi)")
    parser.add_argument("--demo_query", type=str,
                        default="notebook for travel",
                        help="Query for visualization mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Determine which SPLADE model(s) to use
    if args.splade_model == "all":
        model_keys = list(SPLADE_MODELS.keys())
    else:
        model_keys = [args.splade_model]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    for model_key in model_keys:
        splade_config = SPLADE_MODELS[model_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"  Processing: {splade_config.label}")
        logger.info(f"{'='*60}")

        # Load SPLADE model
        encoder = SPLADEEncoder(config=splade_config)
        encoder.load_model(device=device)

        if args.mode == "visualize":
            visualize_splade_expansion(encoder, args.demo_query)
            continue

        # Load dataset
        data_config = DEFAULT_DATA_CONFIG
        data_config.max_products = args.max_products
        dataset = ESCIDataset(data_config)
        dataset.load_examples()
        dataset.load_products()

        # Encode or load product vectors
        index = SPLADEIndex.load(path=splade_config.cache_dir)

        if args.mode == "encode" or index is None:
            product_ids, product_texts = dataset.get_product_texts_list()
            logger.info(f"Encoding {len(product_ids):,} products with {splade_config.label}...")
            t0 = time.time()
            product_vectors = encoder.encode_batch(
                product_texts, batch_size=args.batch_size
            )
            elapsed = time.time() - t0
            logger.info(f"  Encoding completed in {elapsed:.1f}s")

            index = SPLADEIndex(product_ids, product_vectors)
            index.save(path=splade_config.cache_dir)

        if args.mode == "evaluate":
            method_name = splade_config.label
            logger.info(f"Running {method_name} evaluation...")

            # Score-based evaluation
            query_candidates = dataset.get_query_candidates(max_queries=args.max_queries)

            def splade_scoring_fn(query_text, candidate_ids):
                query_vector = encoder.encode_single(query_text)
                return index.score_candidates(query_vector, candidate_ids)

            scoring_results = evaluate_candidate_scoring(
                query_candidates=query_candidates,
                scoring_fn=splade_scoring_fn,
                method_name=method_name,
                eval_config=DEFAULT_EVAL_CONFIG,
            )
            print_evaluation_results(scoring_results)

            # Full-catalog retrieval evaluation
            eval_queries = dataset.get_evaluation_queries(max_queries=args.max_queries)
            retrieved = {}
            for qid, qdata in tqdm(eval_queries.items(), desc=f"{method_name} Retrieval"):
                query_vector = encoder.encode_single(qdata["query_text"])
                result_ids, _ = index.search(query_vector, top_k=100)
                retrieved[qid] = result_ids

            retrieval_results = evaluate_retrieval(
                retrieved, eval_queries, DEFAULT_EVAL_CONFIG.k_values
            )
            scoring_results["retrieval_metrics"] = retrieval_results

            results_filename = f"splade_{model_key}_results.json"
            save_results_json([scoring_results], METRICS_DIR / results_filename)

            # Plot
            plot_score_boxplots([scoring_results],
                                output_path=PLOTS_DIR / f"splade_{model_key}_boxplots.png")


if __name__ == "__main__":
    main()
