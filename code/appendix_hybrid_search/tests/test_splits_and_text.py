"""Template, split disjointness, and prefix exclusion."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import sha256_text
from data_loader import load_esci, render_product_text, render_qdrant_text
from splits import build_splits, exercise_slice, write_bundle
from tests.support import write_tiny_esci


class TextAndSplitTests(unittest.TestCase):
    def test_empty_parts_do_not_leave_separators(self) -> None:
        self.assertEqual(render_product_text("Hello", "", ""), "Hello")
        self.assertEqual(render_product_text("Hello", "", "Acme"), "Hello. Brand: Acme")
        self.assertEqual(render_product_text("Hello", "Soft", "Acme"), "Hello. Soft. Brand: Acme")
        self.assertEqual(render_product_text("  Hello   world ", "", ""), "Hello world")
        self.assertNotIn(". .", render_product_text("Hello", "", "Acme"))

    def test_query_in_both_splits_keeps_the_test_rows(self) -> None:
        root = Path(tempfile.mkdtemp())
        write_tiny_esci(root)
        examples_path = root / "shopping_queries_dataset_examples.parquet"
        examples = pd.read_parquet(examples_path)
        extra = examples.iloc[0].copy()
        extra["example_id"] = 99
        extra["query_id"] = "q5"
        extra["query"] = "phone case"
        extra["product_id"] = "p8"
        extra["esci_label"] = "S"
        extra["split"] = "train"
        pd.concat([examples, pd.DataFrame([extra])], ignore_index=True).to_parquet(examples_path)
        data = load_esci(root)
        self.assertEqual(data.queries["q5"].split, "test")
        self.assertEqual(set(data.queries["q5"].labels), {"p1", "p7"})
        kept = data.examples[(data.examples["query_id"] == "q5") & (data.examples["split"] == "train")]
        self.assertTrue(kept.empty)

    def test_qdrant_sensitivity_format(self) -> None:
        text = render_qdrant_text("Mug", "Acme", "red", None, "one\ntwo")
        self.assertEqual(text, "Mug\nAcme\nred\nNone\none\ntwo")
        self.assertEqual(render_qdrant_text("Mug", "Acme", None, None, None), "Mug\nAcme\nNone\nNone\nNone")

    def test_line_hash_matches_joined_bytes(self) -> None:
        from config import sha256_text
        from data_loader import _hash_lines

        lines = ["a", "b c", ""]
        self.assertEqual(_hash_lines(lines), sha256_text("\n".join(lines) + "\n"))
        self.assertEqual(_hash_lines([]), sha256_text(""))
        self.assertEqual(_hash_lines(iter(["only"])), sha256_text("only\n"))

    def test_import_does_not_create_directories(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "config.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Call):
                func = node.value.func
                name = getattr(func, "attr", "") or getattr(func, "id", "")
                self.assertNotIn("mkdir", name)

    def test_splits_are_disjoint_and_prefix_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = load_esci(write_tiny_esci(Path(tmp)))
            bundle = build_splits(
                data,
                prefix_texts=["red shoes"],
                prefix_source="fixture",
                development_size=1,
                development_seed=20261110,
                exercise3_pairs=1,
            )
            self.assertEqual(set(bundle.test_ids) & set(bundle.development_ids), set())
            self.assertEqual(set(bundle.development_ids) & set(bundle.exercise3_query_ids), set())
            self.assertNotIn("q1", bundle.development_ids)
            self.assertNotIn("q1", bundle.exercise3_query_ids)
            self.assertTrue(set(bundle.test_ids).isdisjoint(bundle.exercise3_query_ids))
            self.assertEqual(bundle.exercise3_pairs[0][0], "q2")
            self.assertEqual(bundle.excluded_test_ids, [])

    def test_test_query_matching_prefix_text_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tiny_esci(Path(tmp))
            examples_path = root / "shopping_queries_dataset_examples.parquet"
            examples = pd.read_parquet(examples_path)
            extras = []
            for number, query_id, query in (
                (100, "q9", "red shoes"),
                (101, "q10", "  red   shoes "),
            ):
                row = examples.iloc[0].copy()
                row["example_id"] = number
                row["query_id"] = query_id
                row["query"] = query
                row["product_id"] = "p1"
                row["esci_label"] = "E"
                row["split"] = "test"
                extras.append(row)
            pd.concat([examples, pd.DataFrame(extras)], ignore_index=True).to_parquet(examples_path)
            data = load_esci(root)
            bundle = build_splits(
                data,
                prefix_texts=["red shoes"],
                prefix_source="fixture",
                development_size=1,
                development_seed=20261110,
                exercise3_pairs=1,
            )
            self.assertEqual(bundle.excluded_test_ids, ["q10", "q9"])
            self.assertEqual(bundle.as_dict()["n_excluded_test_ids"], 2)
            self.assertEqual(
                bundle.digests["excluded_test_query_ids"],
                sha256_text("\n".join(bundle.excluded_test_ids) + "\n"),
            )
            self.assertNotIn("q9", bundle.test_ids)
            self.assertNotIn("q10", bundle.test_ids)
            for query_id in ("q5", "q6", "q7", "q8"):
                self.assertIn(query_id, bundle.test_ids)
            self.assertEqual(len(bundle.test_ids), 4)
            path = Path(tmp) / "splits.json"
            write_bundle(path, bundle)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["excluded_test_ids"], ["q10", "q9"])
            self.assertEqual(payload["n_excluded_test_ids"], 2)
            sliced = exercise_slice(
                data,
                n_test_queries=10,
                n_distractors=0,
                seed=1,
                prefix_texts=["red shoes"],
            )
            self.assertNotIn("q9", sliced["query_ids"])
            self.assertNotIn("q10", sliced["query_ids"])
            self.assertEqual(set(sliced["query_ids"]), {"q5", "q6", "q7", "q8"})


if __name__ == "__main__":
    unittest.main()
