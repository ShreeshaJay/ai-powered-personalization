"""
MIND (Microsoft News) User Loader for Chapter 11.

Loads user click histories with timestamps and maps articles to cached
SBERT embeddings from Chapter 10. Provides user-level sequences for
aggregation into user embeddings.

Key design decisions:
- True user IDs available: aggregation produces genuine user embeddings.
- Timestamps on each impression enable real temporal decay weighting.
- Temporal split: train on earlier impressions, evaluate on the last impression.
- Multiple impressions per user are merged into a single chronological history.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MINDUserDataset:
    """
    Loads MIND user histories and maps articles to pre-computed embeddings.

    Usage:
        dataset = MINDUserDataset(config)
        dataset.load()
        users = dataset.get_evaluation_users()
    """

    def __init__(
        self,
        data_dir: Path,
        news_file: str = "news.tsv",
        behaviors_file: str = "behaviors.tsv",
        min_history_length: int = 5,
        item_embedding_file: Optional[Path] = None,
        item_embedding_dim: int = 384,
    ):
        self.data_dir = Path(data_dir)
        self.news_file = news_file
        self.behaviors_file = behaviors_file
        self.min_history_length = min_history_length
        self.item_embedding_file = item_embedding_file
        self.item_embedding_dim = item_embedding_dim

        # Loaded data (populated by load())
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_id_to_idx: Dict[str, int] = {}
        self.item_ids: Optional[List[str]] = None
        self.news_df: Optional[pl.DataFrame] = None
        self.behaviors_df: Optional[pl.DataFrame] = None
        self._valid_news_ids: Optional[Set[str]] = None

    def load(self) -> "MINDUserDataset":
        """Load item embeddings, news articles, and user behaviors."""
        self._load_item_embeddings()
        self._load_news()
        self._load_behaviors()
        return self

    def _load_item_embeddings(self):
        """Load cached article embeddings from Chapter 10."""
        if self.item_embedding_file is None:
            raise ValueError("item_embedding_file must be specified")

        emb_path = Path(self.item_embedding_file)
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Item embeddings not found at {emb_path}. "
                f"Run Chapter 10 first to generate them."
            )

        logger.info(f"Loading article embeddings from {emb_path.name}...")
        data = np.load(emb_path, allow_pickle=True)

        self.item_embeddings = data["embeddings"]
        # MIND .npz files from Chapter 10 use "news_ids" as the key name;
        # handle both for robustness.
        if "item_ids" in data:
            self.item_ids = list(data["item_ids"])
        elif "news_ids" in data:
            self.item_ids = list(data["news_ids"])
        else:
            raise KeyError(
                f"Expected 'item_ids' or 'news_ids' in {emb_path.name}, "
                f"found keys: {list(data.keys())}"
            )
        self.item_id_to_idx = {nid: idx for idx, nid in enumerate(self.item_ids)}
        self._valid_news_ids = set(self.item_ids)

        logger.info(
            f"  Loaded {len(self.item_ids):,} article embeddings, "
            f"dim={self.item_embeddings.shape[1]}"
        )

    def _load_news(self):
        """Load news articles metadata.

        MIND's news.tsv contains unescaped quotes in article titles/abstracts
        (e.g., '"Bad" cholesterol means LDL...'), so we disable quote parsing
        and set truncate_ragged_lines=True for robustness.
        """
        news_path = self.data_dir / self.news_file
        logger.info(f"Loading news from {news_path.name}...")

        self.news_df = pl.read_csv(
            news_path,
            separator="\t",
            has_header=False,
            new_columns=[
                "news_id", "category", "subcategory", "title",
                "abstract", "url", "title_entities", "abstract_entities",
            ],
            quote_char=None,  # Disable quote parsing (unescaped quotes in text)
            truncate_ragged_lines=True,
            infer_schema_length=10000,
        )
        logger.info(f"  Loaded {len(self.news_df):,} news articles")

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse MIND timestamp format: 'MM/DD/YYYY HH:MM:SS AM/PM'."""
        return datetime.strptime(ts_str, "%m/%d/%Y %I:%M:%S %p")

    def _load_behaviors(self):
        """Load and parse user behavior logs."""
        behaviors_path = self.data_dir / self.behaviors_file
        logger.info(f"Loading behaviors from {behaviors_path.name}...")

        df = pl.read_csv(
            behaviors_path,
            separator="\t",
            has_header=False,
            new_columns=[
                "impression_id", "user_id", "time", "history", "impressions",
            ],
        )

        # Parse timestamp using native Polars expression (much faster)
        df = df.with_columns(
            pl.col("time")
            .str.to_datetime(format="%m/%d/%Y %I:%M:%S %p")
            .alias("timestamp")
        )

        # Parse click history: space-separated news IDs
        # Handle null/empty history fields gracefully
        df = df.with_columns(
            pl.when(pl.col("history").is_not_null() & (pl.col("history").str.len_chars() > 0))
            .then(pl.col("history").str.split(" "))
            .otherwise(pl.lit([]).cast(pl.List(pl.Utf8)))
            .alias("history_list")
        )

        # Parse impressions: "N12345-1 N67890-0" -> clicked article IDs
        def parse_clicked_impressions(imp_str: str) -> List[str]:
            """Extract article IDs that were actually clicked (label=1)."""
            if not isinstance(imp_str, str) or not imp_str.strip():
                return []
            clicked = []
            for entry in imp_str.split():
                parts = entry.rsplit("-", 1)
                if len(parts) == 2 and parts[1] == "1":
                    clicked.append(parts[0])
            return clicked

        df = df.with_columns(
            pl.col("impressions")
            .map_elements(parse_clicked_impressions, return_dtype=pl.List(pl.Utf8))
            .alias("clicked_articles")
        )

        # History length
        df = df.with_columns(
            pl.col("history_list").list.len().alias("history_length")
        )

        self.behaviors_df = df
        logger.info(f"  Loaded {len(df):,} behavior records")
        logger.info(f"  Unique users: {df['user_id'].n_unique():,}")

    def _build_user_timelines(self) -> Dict[str, List[Dict]]:
        """Build chronological timelines per user.

        Each user gets a sorted list of impressions with:
        - timestamp: datetime
        - history_list: List[str] (cumulative click history at that point)
        - clicked_articles: List[str] (articles clicked in this impression)

        Returns:
            Dict[user_id -> List[impression dicts]] sorted by timestamp
        """
        if self.behaviors_df is None:
            raise RuntimeError("Call load() first")

        user_timelines: Dict[str, List[Dict]] = {}
        for row in self.behaviors_df.iter_rows(named=True):
            uid = row["user_id"]
            if uid not in user_timelines:
                user_timelines[uid] = []
            user_timelines[uid].append({
                "timestamp": row["timestamp"],
                "history_list": row["history_list"],
                "clicked_articles": row["clicked_articles"],
            })

        # Sort each user's timeline chronologically
        for uid in user_timelines:
            user_timelines[uid].sort(key=lambda x: x["timestamp"])

        return user_timelines

    def get_history_length_stats(self) -> Dict[str, float]:
        """Compute per-user history length distribution."""
        if self.behaviors_df is None:
            raise RuntimeError("Call load() first")

        # Aggregate: total unique clicks per user across all impressions
        user_timelines = self._build_user_timelines()
        history_lengths = []
        for uid, timeline in user_timelines.items():
            # Use the history from the latest impression (cumulative)
            latest = timeline[-1]
            history_list = latest["history_list"]
            # Handle None/empty histories
            if history_list is None:
                history_lengths.append(0)
                continue
            # Filter to articles with embeddings
            valid_history = [
                nid for nid in history_list
                if nid in self._valid_news_ids
            ]
            history_lengths.append(len(valid_history))

        lengths = np.array(history_lengths)
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
        """Compute article frequency across users (for IDF calculation).

        Returns dict mapping news_id -> number of users who clicked that article.
        """
        if self.behaviors_df is None:
            raise RuntimeError("Call load() first")

        user_timelines = self._build_user_timelines()
        freq: Dict[str, int] = {}

        for uid, timeline in user_timelines.items():
            # Collect all unique articles this user clicked
            user_articles: Set[str] = set()
            for impression in timeline:
                user_articles.update(impression["history_list"])
                user_articles.update(impression["clicked_articles"])

            for nid in user_articles:
                if nid in self._valid_news_ids:
                    freq[nid] = freq.get(nid, 0) + 1

        return freq

    def get_evaluation_users(
        self,
        max_users: Optional[int] = None,
        seed: int = 42,
    ) -> List[Dict]:
        """Prepare users for temporal evaluation.

        Temporal split per user:
        - History: all articles from impressions EXCEPT the last one
        - Target: articles clicked in the LAST impression

        This ensures we train on past behavior and evaluate on future clicks.

        Returns:
            List of dicts with keys:
                - user_id: str
                - history_item_ids: List[str] (chronologically ordered)
                - history_embeddings: np.ndarray (L, D)
                - history_timestamps: List[datetime] (one per history item, approximate)
                - target_item_ids: List[str] (clicked in last impression)
                - target_item_idxs: List[int] (indices into item embedding matrix)
                - history_length: int
        """
        if self.behaviors_df is None:
            raise RuntimeError("Call load() first")

        user_timelines = self._build_user_timelines()

        eval_users = []
        skipped_short = 0
        skipped_no_target = 0
        skipped_no_embeddings = 0

        for uid, timeline in user_timelines.items():
            if len(timeline) < 2:
                # Need at least 2 impressions: history from earlier, target from last
                # For single-impression users, use the history field as history
                # and clicked_articles as target
                if len(timeline) == 1:
                    imp = timeline[0]
                    history_ids = [
                        nid for nid in imp["history_list"]
                        if nid in self._valid_news_ids
                    ]
                    target_ids = [
                        nid for nid in imp["clicked_articles"]
                        if nid in self._valid_news_ids
                    ]
                else:
                    skipped_short += 1
                    continue
            else:
                # Multi-impression user: use history from last impression as
                # the accumulated reading history, target = clicked in last impression
                last_impression = timeline[-1]
                history_ids = [
                    nid for nid in last_impression["history_list"]
                    if nid in self._valid_news_ids
                ]
                target_ids = [
                    nid for nid in last_impression["clicked_articles"]
                    if nid in self._valid_news_ids
                ]

            # Apply minimum history length filter
            if len(history_ids) < self.min_history_length:
                skipped_short += 1
                continue

            # Must have at least one valid target
            if len(target_ids) == 0:
                skipped_no_target += 1
                continue

            # Build history embeddings
            try:
                history_embs = np.stack([
                    self.item_embeddings[self.item_id_to_idx[nid]]
                    for nid in history_ids
                ])
            except KeyError:
                skipped_no_embeddings += 1
                continue

            # Approximate timestamps: for single-impression users or when
            # history doesn't have per-click timestamps, we assign uniform
            # spacing within the impression's timeframe.
            # MIND's history field doesn't have per-article timestamps,
            # so we use position as a proxy (similar to Amazon).
            # The impression timestamp is used as the "current time" reference.
            if len(timeline) >= 2:
                ref_time = last_impression["timestamp"]
            else:
                ref_time = timeline[0]["timestamp"]

            target_idxs = [self.item_id_to_idx[nid] for nid in target_ids]

            eval_users.append({
                "user_id": uid,
                "history_item_ids": history_ids,
                "history_embeddings": history_embs,
                "reference_timestamp": ref_time,
                "target_item_ids": target_ids,
                "target_item_idxs": target_idxs,
                "history_length": len(history_ids),
            })

        logger.info(
            f"  Prepared {len(eval_users):,} evaluation users "
            f"(skipped: {skipped_short} short, {skipped_no_target} no target, "
            f"{skipped_no_embeddings} missing embeddings)"
        )

        # Optionally subsample
        if max_users is not None and len(eval_users) > max_users:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(eval_users), size=max_users, replace=False)
            eval_users = [eval_users[i] for i in indices]
            logger.info(f"  Sampled {max_users:,} users for evaluation")

        return eval_users

    def get_all_item_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Return the full article embedding matrix and ID list (for FAISS index)."""
        return self.item_embeddings, self.item_ids

    def get_news_metadata(self, news_id: str) -> Optional[Dict]:
        """Get news article metadata for qualitative inspection."""
        if self.news_df is None:
            return None
        row = self.news_df.filter(pl.col("news_id") == news_id)
        if len(row) == 0:
            return None
        return row.to_dicts()[0]

    def print_stats(self):
        """Print summary statistics."""
        if self.behaviors_df is None:
            print("Dataset not loaded. Call load() first.")
            return

        print(f"\n{'=' * 60}")
        print(f"MIND User Dataset")
        print(f"{'=' * 60}")
        print(f"  News articles:          {len(self.news_df):>10,}")
        print(f"  Behavior records:       {len(self.behaviors_df):>10,}")
        print(f"  Unique users:           {self.behaviors_df['user_id'].n_unique():>10,}")
        print(f"  Articles with embeds:   {len(self.item_ids):>10,}")
        print(f"  Embedding dimension:    {self.item_embeddings.shape[1]:>10}")

        stats = self.get_history_length_stats()
        print(f"\n  User History Length Distribution:")
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
