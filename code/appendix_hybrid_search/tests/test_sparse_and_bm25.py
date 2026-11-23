"""CPU sparse agreement and BM25 direct score lookup."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bm25_search import BM25Index
from fusion import ranking_from_scores
from splade_search import encode_documents, relative_score_gap, score_csc, score_rows, score_torch, to_csr, top_ids


class SparseAndBM25Tests(unittest.TestCase):
    def test_posting_walk_matches_dense_dot(self) -> None:
        rng = np.random.default_rng(0)
        n_docs, vocab = 40, 25
        rows, cols, data = [], [], []
        for doc in range(n_docs):
            for term in rng.choice(vocab, size=4, replace=False):
                rows.append(doc)
                cols.append(int(term))
                data.append(float(rng.random()))
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_docs, vocab)).tocsc()
        query = np.zeros(vocab, dtype=np.float64)
        query[rng.choice(vocab, size=3, replace=False)] = rng.random(3)
        terms = np.flatnonzero(query)
        walked = score_csc(matrix, terms, query[terms])
        dotted = np.asarray(matrix.T.dot(query)).ravel() if False else np.asarray(matrix @ query).ravel()
        self.assertLess(relative_score_gap(walked, dotted), 1e-4)

    def test_torch_cpu_matches_posting_walk(self) -> None:
        rng = np.random.default_rng(1)
        n_docs, vocab = 30, 20
        rows, cols, data = [], [], []
        for doc in range(n_docs):
            for term in rng.choice(vocab, size=3, replace=False):
                rows.append(doc)
                cols.append(int(term))
                data.append(float(rng.random()))
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_docs, vocab)).tocsc()
        query = np.zeros((2, vocab), dtype=np.float64)
        query[0, [1, 4, 7]] = [0.2, 0.5, 0.9]
        query[1, [0, 3]] = [0.4, 0.8]
        torch_scores = score_torch(matrix, query, "cpu")
        product_ids = [f"p{i:02d}" for i in range(n_docs)]
        for row in range(2):
            terms = np.flatnonzero(query[row])
            walked = score_csc(matrix, terms, query[row, terms])
            self.assertLess(relative_score_gap(walked, torch_scores[row]), 1e-4)
            self.assertEqual(top_ids(product_ids, walked, 10), top_ids(product_ids, torch_scores[row], 10))

    def test_score_rows_is_identical_for_csr_and_csc(self) -> None:
        rng = np.random.default_rng(2)
        n_docs, vocab = 200, 60
        rows, cols, data = [], [], []
        for doc in range(n_docs):
            for term in rng.choice(vocab, size=8, replace=False):
                rows.append(doc)
                cols.append(int(term))
                data.append(float(rng.random()))
        coo = sp.coo_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(n_docs, vocab))
        csc, csr = coo.tocsc(), coo.tocsr()
        for trial in range(20):
            query = sp.csr_matrix(
                (rng.random(6).astype(np.float32), (np.zeros(6, dtype=int), rng.choice(vocab, 6, replace=False))),
                shape=(1, vocab),
            )
            chosen = rng.choice(n_docs, size=int(rng.integers(1, 30)), replace=False).tolist()
            self.assertEqual(score_rows(csr, query, chosen).tobytes(), score_rows(csc, query, chosen).tobytes())
        self.assertEqual(len(score_rows(csr, query, [])), 0)

    def test_document_chunks_are_reused_only_for_the_same_products(self) -> None:
        import tempfile

        class Encoder:
            def __init__(self) -> None:
                self.calls = 0

            def encode_document(self, texts, **_kwargs):
                self.calls += 1
                return sp.csr_matrix(np.array([[float(len(text)), float(text.count("a"))] for text in texts]))

        first_ids = [f"d{index}" for index in range(5)]
        first_texts = ["a" * (index + 1) for index in range(5)]
        second_ids = [f"t{index}" for index in range(7)]
        second_texts = ["b" * (index + 10) for index in range(7)]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "cache"
            encoder = Encoder()
            first = encode_documents(encoder, first_ids, first_texts, directory, 2, 16, checkpoint_every=2)
            self.assertEqual(first.shape[0], 5)
            calls = encoder.calls
            (directory / "documents.npz").unlink()
            again = encode_documents(encoder, first_ids, first_texts, directory, 2, 16, checkpoint_every=2)
            self.assertEqual(encoder.calls, calls)
            self.assertEqual((again != first).nnz, 0)
            second = encode_documents(encoder, second_ids, second_texts, directory, 2, 16, checkpoint_every=2)
            expected = np.array([[float(len(text)), 0.0] for text in second_texts])
            self.assertTrue(np.array_equal(second.toarray(), expected))
            (directory / "chunk_ids.sha256").unlink()
            (directory / "documents.npz").unlink()
            calls = encoder.calls
            encode_documents(encoder, second_ids, second_texts, directory, 2, 16, checkpoint_every=2)
            self.assertEqual(encoder.calls, calls + 4)

    def test_bm25_candidate_lookup_matches_full_scores(self) -> None:
        ids = ["b", "a", "c", "d"]
        texts = [
            "red leather shoes for running",
            "blue cotton hat",
            "red shoes",
            "garden hose",
        ]
        index = BM25Index()
        index.build(ids, texts)
        full = index.scores("red shoes")
        looked = index.score_products("red shoes", ["c", "d", "missing"])
        self.assertAlmostEqual(looked[0], full[index.id_to_row["c"]])
        self.assertAlmostEqual(looked[1], full[index.id_to_row["d"]])
        self.assertEqual(looked[2], 0.0)
        hits = index.search("red shoes", top_k=4)
        self.assertEqual(
            [pid for pid, _ in hits],
            [pid for pid, _ in ranking_from_scores(ids, full, 4)],
        )
        again = index.search("red shoes", top_k=4)
        self.assertEqual(hits, again)

    def test_tie_order_is_stable_across_calls(self) -> None:
        ids = ["b", "a"]
        index = BM25Index()
        index.build(ids, ["same tokens here", "same tokens here"])
        first = index.search("same tokens", top_k=2)
        second = index.search("same tokens", top_k=2)
        self.assertEqual(first, second)
        self.assertEqual([pid for pid, _ in first], ["a", "b"])

    def test_bfloat16_tensors_convert_without_numpy_view(self) -> None:
        import torch

        dense = torch.tensor([1.5, 0.25], dtype=torch.bfloat16)
        with self.assertRaises(TypeError):
            np.asarray(dense)
        hits = ranking_from_scores(["b", "a"], dense)
        self.assertEqual([item[0] for item in hits], ["b", "a"])
        self.assertAlmostEqual(hits[0][1], 1.5, places=2)
        coo = torch.sparse_coo_tensor(
            torch.tensor([[0, 0], [0, 1]]),
            torch.tensor([0.5, 1.25], dtype=torch.bfloat16),
            size=(1, 4),
        ).coalesce()
        matrix = to_csr(coo)
        self.assertAlmostEqual(float(matrix[0, 0]), 0.5, places=2)
        self.assertAlmostEqual(float(matrix[0, 1]), 1.25, places=2)
        csr = torch.sparse_csr_tensor(
            torch.tensor([0, 2]),
            torch.tensor([0, 1]),
            torch.tensor([0.5, 1.25], dtype=torch.bfloat16),
            size=(1, 4),
        )
        converted = to_csr(csr)
        self.assertAlmostEqual(float(converted[0, 1]), 1.25, places=2)


if __name__ == "__main__":
    unittest.main()
