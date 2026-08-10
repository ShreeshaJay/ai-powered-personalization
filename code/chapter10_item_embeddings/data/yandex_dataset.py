"""
Yandex Yambda Music Dataset Loader for Item Embeddings

Loads listening history from the Yambda dataset (50M interactions)
and converts it into item sequences suitable for Item2Vec training.

The flat format has one row per listen event:
    uid, item_id, timestamp, is_organic, played_ratio_pct, track_length_seconds

We group by user, sort by timestamp, optionally filter by played_ratio,
and split into sessions based on time gaps.

Dataset reference:
    https://huggingface.co/datasets/yandex/yambda
    Paper: https://arxiv.org/pdf/2505.22238
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YandexDataset:
    """
    Yandex Yambda dataset loader for music track embeddings.

    Provides:
    - Listening sessions split by time gaps
    - Audio CNN embeddings for comparison with learned embeddings
    """

    def __init__(
        self,
        data_dir: Path,
        min_played_ratio: int = 50,
        min_listens_per_user: int = 10,
        max_listens_per_user: int = 5000,
        session_gap_seconds: int = 1800,
    ):
        self.data_dir = Path(data_dir)
        self.min_played_ratio = min_played_ratio
        self.min_listens_per_user = min_listens_per_user
        self.max_listens_per_user = max_listens_per_user
        self.session_gap_seconds = session_gap_seconds

        self.listens_df: Optional[pd.DataFrame] = None
        self.sessions: Optional[List[List[str]]] = None
        self.audio_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.stats = {}

    def load_listens(self) -> pd.DataFrame:
        """
        Load and filter listening events from listens.parquet.

        Filters:
        - played_ratio_pct >= min_played_ratio (meaningful listens only)
        - Users with at least min_listens_per_user events
        - Cap users at max_listens_per_user (memory control)

        Returns:
            DataFrame sorted by (uid, timestamp)
        """
        flat_dir = self.data_dir / "flat"
        listens_path = flat_dir / "listens.parquet"
        if not listens_path.exists():
            raise FileNotFoundError(f"Listens file not found: {listens_path}")

        logger.info(f"Loading listens from {listens_path}")

        cols = ["uid", "item_id", "timestamp", "played_ratio_pct"]
        df = pd.read_parquet(listens_path, columns=cols)
        logger.info(f"  Raw listens: {len(df):,}")

        if self.min_played_ratio > 0:
            df = df[df["played_ratio_pct"] >= self.min_played_ratio]
            logger.info(f"  After played_ratio >= {self.min_played_ratio}%: {len(df):,}")

        user_counts = df["uid"].value_counts()
        valid_users = user_counts[user_counts >= self.min_listens_per_user].index
        df = df[df["uid"].isin(valid_users)]
        logger.info(f"  After min_listens >= {self.min_listens_per_user}: {len(df):,} "
                     f"({len(valid_users):,} users)")

        if self.max_listens_per_user:
            df = df.groupby("uid").head(self.max_listens_per_user)
            logger.info(f"  After capping at {self.max_listens_per_user}/user: {len(df):,}")

        df = df.sort_values(["uid", "timestamp"]).reset_index(drop=True)

        # Convert item_id to string for consistency with other datasets
        df["item_id"] = df["item_id"].astype(str)

        self.listens_df = df
        self.stats["total_listens"] = len(df)
        self.stats["unique_users"] = df["uid"].nunique()
        self.stats["unique_tracks"] = df["item_id"].nunique()

        logger.info(f"Loaded {len(df):,} listens "
                     f"({self.stats['unique_users']:,} users, "
                     f"{self.stats['unique_tracks']:,} tracks)")

        return df

    def build_sequences(self, mode: str = "full_history") -> List[List[str]]:
        """
        Build item sequences for Item2Vec training.

        Args:
            mode: Strategy for constructing sequences.
                "full_history" — one sequence per user (entire chronological
                    history). The Word2Vec window_size naturally limits
                    co-occurrence distance. Simpler; follows the original
                    Item2Vec paper (Barkan & Koenigstein, 2016).
                "session" — split each user's history into sessions by time
                    gaps (session_gap_seconds). Prevents cross-session
                    co-occurrence, which can be useful if sessions represent
                    distinct user intents (e.g., workout vs. dinner music).

        Returns:
            List of sequences, where each is a list of item_id strings.
        """
        if self.listens_df is None:
            self.load_listens()

        if mode == "full_history":
            return self._build_full_history_sequences()
        elif mode == "session":
            return self._build_session_sequences()
        else:
            raise ValueError(f"Unknown sequence_mode: {mode!r}. "
                             f"Use 'full_history' or 'session'.")

    def _build_full_history_sequences(self) -> List[List[str]]:
        """One sequence per user: their complete chronological listening history."""
        logger.info("Building sequences (mode=full_history, one per user)...")

        sequences = []
        for uid, group in self.listens_df.groupby("uid", sort=False):
            items = group["item_id"].tolist()
            if len(items) >= 2:
                sequences.append(items)

        self.sessions = sequences
        self._log_sequence_stats(sequences, "full_history")
        return sequences

    def _build_session_sequences(self) -> List[List[str]]:
        """Split each user's history into sessions by time gaps."""
        logger.info(f"Building sequences (mode=session, "
                     f"gap threshold: {self.session_gap_seconds}s)...")

        gap_threshold_units = self.session_gap_seconds // 5

        sessions = []
        current_session = []
        prev_uid = None
        prev_ts = None

        for _, row in self.listens_df.iterrows():
            uid = row["uid"]
            ts = row["timestamp"]
            item = row["item_id"]

            if uid != prev_uid:
                if len(current_session) >= 2:
                    sessions.append(current_session)
                current_session = [item]
                prev_uid = uid
                prev_ts = ts
                continue

            if prev_ts is not None and (ts - prev_ts) > gap_threshold_units:
                if len(current_session) >= 2:
                    sessions.append(current_session)
                current_session = [item]
            else:
                current_session.append(item)

            prev_ts = ts

        if len(current_session) >= 2:
            sessions.append(current_session)

        self.sessions = sessions
        self._log_sequence_stats(sessions, "session")
        return sessions

    def _log_sequence_stats(self, sequences: List[List[str]], mode: str):
        """Compute and log statistics for the built sequences."""
        lengths = [len(s) for s in sequences]
        self.stats["sequence_mode"] = mode
        self.stats["total_sequences"] = len(sequences)
        self.stats["avg_sequence_length"] = np.mean(lengths)
        self.stats["median_sequence_length"] = np.median(lengths)

        logger.info(f"Built {len(sequences):,} sequences "
                     f"(avg length: {np.mean(lengths):.1f}, "
                     f"median: {np.median(lengths):.1f})")

    # Keep build_sessions as an alias for backward compatibility
    def build_sessions(self) -> List[List[str]]:
        """Alias for build_sequences(mode='session')."""
        return self.build_sequences(mode="session")

    def load_audio_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load pre-computed CNN audio embeddings for tracks.

        The embeddings.parquet file sits in the dataset root directory
        (not in the flat/ subdirectory).

        Returns:
            Dict mapping item_id (str) -> embedding (numpy array)
        """
        emb_path = self.data_dir / "embeddings.parquet"
        if not emb_path.exists():
            logger.warning(f"Audio embeddings not found: {emb_path}")
            return {}

        logger.info(f"Loading audio embeddings from {emb_path}")
        df = pd.read_parquet(emb_path)

        embeddings = {}
        emb_col = "normalized_embed" if "normalized_embed" in df.columns else "embed"

        for _, row in df.iterrows():
            item_id = str(row["item_id"])
            emb = np.array(row[emb_col], dtype=np.float32)
            embeddings[item_id] = emb

        self.audio_embeddings = embeddings

        if embeddings:
            sample_dim = next(iter(embeddings.values())).shape[0]
            logger.info(f"Loaded {len(embeddings):,} audio embeddings (dim={sample_dim})")
        else:
            logger.warning("No audio embeddings loaded")

        return embeddings

    def get_next_item_pairs(self) -> Dict[str, set]:
        """
        Build next-item ground truth from sessions for evaluation.

        Same interface as AmazonKDDDataset.get_next_item_pairs() —
        last item in session -> next item.

        For Yandex, we use consecutive items within sessions:
        for each session [A, B, C, D], we get pairs (A,B), (B,C), (C,D).
        """
        if self.sessions is None:
            self.build_sessions()

        from collections import defaultdict
        next_item_map: Dict[str, set] = defaultdict(set)
        total = 0

        for session in self.sessions:
            for i in range(len(session) - 1):
                query = session[i]
                target = session[i + 1]
                if query != target:
                    next_item_map[query].add(target)
                    total += 1

        logger.info(f"Next-item ground truth: {len(next_item_map):,} query items, "
                     f"{total:,} pairs")
        return next_item_map

    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "=" * 80)
        print("Yandex Yambda Dataset Statistics")
        print("=" * 80)
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.2f}")
            else:
                print(f"  {key:.<50} {value:,}" if isinstance(value, int) else
                      f"  {key:.<50} {value}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import YANDEX_DATA_PATH

    print("Testing Yandex Dataset Loader")
    print("=" * 80)

    dataset = YandexDataset(
        data_dir=YANDEX_DATA_PATH,
        min_played_ratio=50,
        min_listens_per_user=10,
        session_gap_seconds=1800,
    )

    df = dataset.load_listens()
    print(f"\nListens shape: {df.shape}")
    print(f"Sample:\n{df.head()}")

    # Default: full_history mode
    sequences = dataset.build_sequences(mode="full_history")
    print(f"\nSample sequences (full_history):")
    for s in sequences[:3]:
        print(f"  {s[:8]}{'...' if len(s) > 8 else ''} (len={len(s)})")

    dataset.print_stats()
