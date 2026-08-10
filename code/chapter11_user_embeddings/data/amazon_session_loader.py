"""
Amazon KDD Cup 2023 Session Loader for Chapter 11.

Loads anonymous browsing sessions and maps items to cached embeddings from
Chapter 10. Each session becomes a sequence of item embeddings ready for
aggregation into a "session embedding" (our proxy for a user embedding when
no user IDs are available).

Key design decisions:
- Sessions are anonymous: no user IDs, so "user embedding" = session embedding.
- Temporal ordering within a session is given by position (left-to-right).
- We reuse cached .npz item embeddings from Chapter 10 (no re-encoding).
- Leave-last-out evaluation: aggregate prev_items[:-1], predict next_item.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazonSessionDataset:
    """
    Loads Amazon KDD sessions and maps items to pre-computed embeddings.

    Usage:
        dataset = AmazonSessionDataset(config)
        dataset.load()
        sessions = dataset.get_evaluation_sessions()
    """

    def __init__(
        self,
        data_dir: Path,
        locale: str = "UK",
        min_session_length: int = 3,
        item_embedding_file: Optional[Path] = None,
        item_embedding_dim: int = 384,
    ):
        self.data_dir = Path(data_dir)
        self.locale = locale
        self.min_session_length = min_session_length
        self.item_embedding_file = item_embedding_file
        self.item_embedding_dim = item_embedding_dim

        # Loaded data (populated by load())
        self.item_embeddings: Optional[np.ndarray] = None  # (N_items, D)
        self.item_id_to_idx: Dict[str, int] = {}           # item_id -> row index
        self.item_ids: Optional[List[str]] = None           # ordered list
        self.sessions_df: Optional[pl.DataFrame] = None
        self._product_ids: Optional[set] = None

    def load(self) -> "AmazonSessionDataset":
        """Load item embeddings and session data. Returns self for chaining."""
        self._load_item_embeddings()
        self._load_sessions()
        return self

    def _load_item_embeddings(self):
        """Load cached item embeddings from Chapter 10 .npz file."""
        if self.item_embedding_file is None:
            raise ValueError("item_embedding_file must be specified")

        emb_path = Path(self.item_embedding_file)
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Item embeddings not found at {emb_path}. "
                f"Run Chapter 10 first to generate them."
            )

        logger.info(f"Loading item embeddings from {emb_path.name}...")
        data = np.load(emb_path, allow_pickle=True)

        self.item_embeddings = data["embeddings"]  # (N, D) float32
        self.item_ids = list(data["item_ids"])
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self._product_ids = set(self.item_ids)

        logger.info(
            f"  Loaded {len(self.item_ids):,} item embeddings, "
            f"dim={self.item_embeddings.shape[1]}"
        )

    def _parse_prev_items(self, raw: str) -> List[str]:
        """Parse prev_items string like "['B09W9FND7K' 'B09JSPLN1M']" into list."""
        # Strip brackets and split on whitespace + quotes
        cleaned = raw.strip("[]")
        items = re.findall(r"'([^']+)'", cleaned)
        return items

    def _load_sessions(self):
        """Load sessions from CSV, filter by locale and product catalog coverage."""
        sessions_path = self.data_dir / "sessions_train.csv"
        logger.info(f"Loading sessions from {sessions_path.name}...")

        # Use Polars for memory efficiency
        df = pl.read_csv(sessions_path)

        # Filter by locale
        df = df.filter(pl.col("locale") == self.locale)
        logger.info(f"  Sessions for locale '{self.locale}': {len(df):,}")

        # Parse prev_items into lists
        df = df.with_columns(
            pl.col("prev_items")
            .map_elements(self._parse_prev_items, return_dtype=pl.List(pl.Utf8))
            .alias("prev_items_list")
        )

        # Compute session length
        df = df.with_columns(
            pl.col("prev_items_list").list.len().alias("session_length")
        )

        # Filter by minimum session length
        df = df.filter(pl.col("session_length") >= self.min_session_length)
        logger.info(
            f"  Sessions with >= {self.min_session_length} prev_items: {len(df):,}"
        )

        # Filter: all items in session must have embeddings
        def all_items_have_embeddings(prev_items: List[str], next_item: str) -> bool:
            if next_item not in self._product_ids:
                return False
            return all(item in self._product_ids for item in prev_items)

        df = df.filter(
            pl.struct(["prev_items_list", "next_item"]).map_elements(
                lambda row: all_items_have_embeddings(
                    row["prev_items_list"], row["next_item"]
                ),
                return_dtype=pl.Boolean,
            )
        )
        logger.info(f"  Sessions with full embedding coverage: {len(df):,}")

        self.sessions_df = df

    def get_session_length_stats(self) -> Dict[str, float]:
        """Compute session length distribution statistics for hyperparameter grounding."""
        if self.sessions_df is None:
            raise RuntimeError("Call load() first")

        lengths = self.sessions_df["session_length"].to_numpy()
        stats = {
            "count": len(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "p25": float(np.percentile(lengths, 25)),
            "p75": float(np.percentile(lengths, 75)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
        }
        return stats

    def get_item_frequency(self) -> Dict[str, int]:
        """Compute item frequency across sessions (for IDF calculation).

        Returns dict mapping item_id -> number of sessions containing that item.
        IDF is computed on training data only (call this before splitting).
        """
        if self.sessions_df is None:
            raise RuntimeError("Call load() first")

        freq: Dict[str, int] = {}
        for row in self.sessions_df.iter_rows(named=True):
            # Count each item once per session (binary TF)
            seen_in_session = set(row["prev_items_list"])
            seen_in_session.add(row["next_item"])
            for item_id in seen_in_session:
                freq[item_id] = freq.get(item_id, 0) + 1
        return freq

    def get_evaluation_sessions(
        self,
        max_sessions: Optional[int] = None,
        seed: int = 42,
    ) -> List[Dict]:
        """Prepare sessions for leave-last-out evaluation.

        For each session with prev_items = [i1, i2, ..., iN] and next_item = iN+1:
        - history_items = [i1, i2, ..., iN]   (all prev_items for aggregation)
        - target_item = iN+1                   (next_item to predict)

        Note on leave-last-out bias: This always evaluates with the longest
        possible history per session.  See Chapter11_Design_Notes.md for the
        random-truncation alternative (homework exercise).

        Returns:
            List of dicts with keys:
                - history_item_ids: List[str] of item IDs for aggregation
                - history_embeddings: np.ndarray (L, D) of item embeddings
                - target_item_id: str
                - target_item_idx: int (index into the full item embedding matrix)
                - session_length: int
        """
        if self.sessions_df is None:
            raise RuntimeError("Call load() first")

        df = self.sessions_df

        # Optionally subsample for evaluation speed
        if max_sessions is not None and len(df) > max_sessions:
            df = df.sample(n=max_sessions, seed=seed)
            logger.info(f"  Sampled {max_sessions:,} sessions for evaluation")

        sessions = []
        for row in df.iter_rows(named=True):
            prev_items = row["prev_items_list"]
            next_item = row["next_item"]

            # History = all prev_items (aggregation input)
            history_ids = prev_items
            history_embs = np.stack([
                self.item_embeddings[self.item_id_to_idx[iid]]
                for iid in history_ids
            ])

            sessions.append({
                "history_item_ids": history_ids,
                "history_embeddings": history_embs,
                "target_item_id": next_item,
                "target_item_idx": self.item_id_to_idx[next_item],
                "session_length": len(history_ids),
            })

        logger.info(f"  Prepared {len(sessions):,} evaluation sessions")
        return sessions

    def get_all_item_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Return the full item embedding matrix and ID list (for FAISS index)."""
        return self.item_embeddings, self.item_ids

    def print_stats(self):
        """Print summary statistics."""
        if self.sessions_df is None:
            print("Dataset not loaded. Call load() first.")
            return

        print(f"\n{'=' * 60}")
        print(f"Amazon KDD Session Dataset (locale={self.locale})")
        print(f"{'=' * 60}")
        print(f"  Total sessions:         {len(self.sessions_df):>10,}")
        print(f"  Items with embeddings:  {len(self.item_ids):>10,}")
        print(f"  Embedding dimension:    {self.item_embeddings.shape[1]:>10}")

        stats = self.get_session_length_stats()
        print(f"\n  Session Length Distribution:")
        print(f"    Mean:                 {stats['mean']:>10.1f}")
        print(f"    Median:               {stats['median']:>10.1f}")
        print(f"    Std:                  {stats['std']:>10.1f}")
        print(f"    Min:                  {stats['min']:>10}")
        print(f"    Max:                  {stats['max']:>10}")
        print(f"    P25:                  {stats['p25']:>10.1f}")
        print(f"    P75:                  {stats['p75']:>10.1f}")
        print(f"    P90:                  {stats['p90']:>10.1f}")
        print(f"    P95:                  {stats['p95']:>10.1f}")
        print(f"{'=' * 60}\n")
