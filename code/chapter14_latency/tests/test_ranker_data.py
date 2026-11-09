import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from ranker.data import load_yambda

D = 86_400


def listens(rows):
    uid, item, ts, played = zip(*rows)
    return pl.DataFrame({
        "uid": pl.Series(uid, dtype=pl.UInt32), "timestamp": pl.Series(ts, dtype=pl.UInt32),
        "item_id": pl.Series(item, dtype=pl.UInt32), "is_organic": pl.Series([1] * len(rows), dtype=pl.UInt8),
        "played_ratio_pct": pl.Series(played, dtype=pl.UInt16),
        "track_length_seconds": pl.Series([200] * len(rows), dtype=pl.UInt32)})


class YambdaWindowTests(unittest.TestCase):
    """Two training days: test is the last day, history the window before each row's window."""

    @classmethod
    def setUpClass(cls):
        rows = [(9, 99, 1 * D, 100),                  # before every window
                (1, 10, 6 * D, 100),                  # history for training rows
                (1, 20, 8 * D, 100),                  # training
                (2, 20, 8 * D + 3 * 3_600 + 10, 10),  # training, hour 3 of day 8
                (3, 30, 9 * D - 3_600, 100),          # training, an hour before the test day
                (4, 40, 9 * D - 900, 100),            # inside the 30-minute gap
                (2, 10, 9 * D + D // 2, 20),          # test
                (1, 20, 10 * D, 80)]                  # test, the last timestamp
        with tempfile.TemporaryDirectory() as tmp:
            listens(rows).write_parquet(Path(tmp) / "listens.parquet")
            cls.data = load_yambda(Path(tmp), train_days=2, max_train_rows=100, max_test_rows=100, seed=0)

    def by_user(self, split):
        uid = split.group // 1000
        return {int(u): (int(split.user_history[i]), int(split.item_plays[i]), float(split.label[i]), i)
                for i, u in enumerate(uid)}

    def test_windows_are_measured_in_seconds(self):
        self.assertEqual(sorted(self.by_user(self.data.train)), [1, 2, 3])
        self.assertEqual(sorted(self.by_user(self.data.test)), [1, 2])
        self.assertEqual(self.data.vocab["uid"], 4)

    def test_history_excludes_the_row_window(self):
        tr = self.by_user(self.data.train)
        self.assertEqual(tr[1][:3], (1, 0, 1.0))  # earlier listen counts, its own does not
        self.assertEqual(tr[2][:3], (0, 0, 0.0))
        self.assertEqual(tr[3][:3], (0, 0, 1.0))
        te = self.by_user(self.data.test)
        self.assertEqual(te[1][:3], (1, 2, 1.0))  # test history is the training window
        self.assertEqual(te[2][:3], (1, 0, 0.0))

    def test_clock_features_use_seconds(self):
        i = self.by_user(self.data.train)[2][3]
        self.assertEqual(self.data.train.sparse[i, 2], 3 + 1)
        self.assertEqual(self.data.train.sparse[i, 3], 8 % 7 + 1)

    def test_one_group_per_user_in_the_test_day(self):
        self.assertEqual(len(np.unique(self.data.test.group)), 2)


if __name__ == "__main__":
    unittest.main()
