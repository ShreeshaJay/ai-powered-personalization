"""Hold-out seed and Amazon cache-tag tests. Neither loads a model."""

import sys
import unittest
from pathlib import Path

import pandas as pd

CHAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER))

from cache_tags import amazon_finetuned_cache_tag
from data.holdout import split_amazon_sessions, split_mind_users


class AmazonCacheTagTests(unittest.TestCase):
    def test_checkpoint_path_changes_the_cache_name(self):
        base = "sentence-transformers/all-MiniLM-L6-v2"
        first = amazon_finetuned_cache_tag(base, "checkpoints/epoch_1")
        second = amazon_finetuned_cache_tag(base, "checkpoints/epoch_2")
        again = amazon_finetuned_cache_tag(base, "checkpoints/epoch_1")

        self.assertNotEqual(first, second)
        self.assertEqual(first, again)
        self.assertNotEqual(f"{first}.npz", f"{second}.npz")
        self.assertEqual(f"{first}.npz", f"{again}.npz")

    def test_writer_and_reader_resolve_the_same_cache_path(self):
        base = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = Path("embeddings")
        models = Path("models")
        checkpoint = models / "contrastive_finetuned" / f"{base.replace('/', '_')}_amazon"
        writer = embeddings / f"{amazon_finetuned_cache_tag(base, checkpoint)}.npz"
        reader_dir = models / "contrastive_finetuned" / f"{base.replace('/', '_')}_amazon"
        reader = embeddings / f"{amazon_finetuned_cache_tag(base, reader_dir)}.npz"
        old = embeddings / f"amazon_kdd_embeddings_finetuned_{base.replace('/', '_')}.npz"
        self.assertEqual(writer, reader)
        self.assertNotEqual(writer, old)

class HoldoutSeedTests(unittest.TestCase):
    def test_amazon_seeds_differ_and_repeat(self):
        sessions = pd.DataFrame({
            "prev_items": [f"prev-{i}" for i in range(40)],
            "next_item": [f"next-{i}" for i in range(40)],
        })
        train_a, eval_a = split_amazon_sessions(sessions, 0.2, seed=0)
        train_b, eval_b = split_amazon_sessions(sessions, 0.2, seed=0)
        train_c, _eval_c = split_amazon_sessions(sessions, 0.2, seed=1)

        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(eval_a, eval_b)
        self.assertFalse(train_a["next_item"].tolist() == train_c["next_item"].tolist())

    def test_mind_seeds_differ_and_repeat(self):
        behaviors = pd.DataFrame({
            "user_id": [f"user-{i}" for i in range(40)],
            "history_list": [[] for _ in range(40)],
        })
        train_a, eval_a = split_mind_users(behaviors, 0.2, seed=0)
        train_b, eval_b = split_mind_users(behaviors, 0.2, seed=0)
        train_c, _eval_c = split_mind_users(behaviors, 0.2, seed=1)

        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(eval_a, eval_b)
        self.assertFalse(train_a["user_id"].tolist() == train_c["user_id"].tolist())


if __name__ == "__main__":
    unittest.main()
