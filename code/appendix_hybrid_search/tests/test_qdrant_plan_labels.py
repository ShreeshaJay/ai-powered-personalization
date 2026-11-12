"""Qdrant parity, the protocol digest, and label-sample reconstruction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import require_plan_hash, sha256_file, PLAN_HASH_PATH, PLAN_PATH
from llm_labels_convert import select_query_ids
from qdrant_demo import run_parity


class QdrantPlanLabelTests(unittest.TestCase):
    def test_plan_sidecar_matches(self) -> None:
        self.assertTrue(PLAN_HASH_PATH.exists(), "analysis_plan.sha256 is missing")
        self.assertEqual(require_plan_hash(), sha256_file(PLAN_PATH))
        self.assertEqual(len(require_plan_hash()), 64)

    def test_selection_matches_random_state(self) -> None:
        scores = {f"q{i}": float(i) / 10 for i in range(10)}
        chosen = select_query_ids(list(scores), scores, max_queries=5, high_disagree_fraction=0.6, seed=42)
        sorted_ids = sorted(scores, key=lambda key: scores[key], reverse=True)
        high = sorted_ids[:3]
        remaining = sorted_ids[3:]
        uniform = np.random.RandomState(42).choice(remaining, size=2, replace=False).tolist()
        self.assertEqual(chosen, [str(item) for item in high + uniform])

    def test_qdrant_parity_on_synthetic_vectors(self) -> None:
        rng = np.random.default_rng(7)
        product_ids = [f"p{i:02d}" for i in range(15)]
        dense = rng.normal(size=(15, 8)).astype(np.float32)
        sparse = sp.random(15, 12, density=0.25, format="csr", random_state=7)
        query_dense = rng.normal(size=(5, 8)).astype(np.float32)
        query_sparse = sp.random(5, 12, density=0.25, format="csr", random_state=8)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_parity(
                product_ids,
                dense,
                sparse,
                query_dense,
                query_sparse,
                Path(tmp) / "qdrant",
                prefetch_limit=15,
                top_k=5,
            )
        self.assertTrue(report["variants"]["aligned_k_61"]["pass"], report["variants"]["aligned_k_61"])
        self.assertTrue(report["variants"]["server_default_rrf"]["pass"], report["variants"]["server_default_rrf"])
        self.assertTrue(report["variants"]["explicit_k_60"]["pass"], report["variants"]["explicit_k_60"])
        self.assertTrue(report["variants"]["dbsf"]["pass"], report["variants"]["dbsf"])
        self.assertTrue(report["pass"])


if __name__ == "__main__":
    unittest.main()
