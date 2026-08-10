"""
Music4all-Onion dataset loader for Section 10.4: Multi-Modal Fusion.

Loads pre-extracted features from three modalities (audio, lyrics, genre)
and user interaction data for co-listen evaluation.
"""

import bz2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Music4allDataset:
    """
    Loader for the Music4all-Onion multi-modal music dataset.

    Each track has pre-extracted feature vectors for multiple modalities.
    Files are tab-separated, bz2-compressed, with a header row. The first
    column is always the track ID.
    """

    def __init__(self, data_dir: Path, config=None):
        self.data_dir = Path(data_dir)
        self.config = config

        # Aligned data (populated after align_modalities())
        self.track_ids: Optional[np.ndarray] = None
        self.audio_features: Optional[np.ndarray] = None
        self.lyrics_features: Optional[np.ndarray] = None
        self.genre_features: Optional[np.ndarray] = None
        self.interactions_df: Optional[pd.DataFrame] = None

        # Raw per-modality data (before alignment)
        self._raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._id_to_idx: Optional[Dict[str, int]] = None

    def _load_feature_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a bz2-compressed TSV feature file.

        Returns:
            (track_ids, features) — ids shape (N,), features shape (N, D) float32.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Feature file not found: {filepath}")

        logger.info(f"Loading {filename}...")
        df = pd.read_csv(filepath, sep="\t", compression="bz2", dtype={"id": str})
        ids = df["id"].values.astype(str)
        features = df.drop(columns=["id"]).values.astype(np.float32)

        # Replace any NaN with 0 (sparse genre features may have gaps)
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            logger.info(f"  Replacing {nan_count:,} NaN values with 0")
            features = np.nan_to_num(features, nan=0.0)

        logger.info(f"  Loaded {len(ids):,} tracks × {features.shape[1]} dims")
        return ids, features

    def load_audio(self, filename: Optional[str] = None) -> int:
        """Load audio features. Returns number of tracks loaded."""
        fname = filename or (self.config.audio_file if self.config else "id_ivec256.tsv.bz2")
        ids, feats = self._load_feature_file(fname)
        self._raw["audio"] = (ids, feats)
        return len(ids)

    def load_lyrics(self, filename: Optional[str] = None) -> int:
        """Load lyrics features. Returns number of tracks loaded."""
        fname = filename or (self.config.lyrics_file if self.config else "id_lyrics_word2vec.tsv.bz2")
        ids, feats = self._load_feature_file(fname)
        self._raw["lyrics"] = (ids, feats)
        return len(ids)

    def load_genre(self, filename: Optional[str] = None) -> int:
        """Load genre features. Returns number of tracks loaded."""
        fname = filename or (self.config.genre_file if self.config else "id_genres_tf-idf.tsv.bz2")
        ids, feats = self._load_feature_file(fname)
        self._raw["genre"] = (ids, feats)
        return len(ids)

    def align_modalities(self):
        """
        Align all loaded modalities to a common set of track IDs.
        After calling this, all feature arrays share the same row ordering
        and self.track_ids is the canonical ID list.
        """
        if not self._raw:
            raise ValueError("No modalities loaded. Call load_audio/lyrics/genre first.")

        id_sets = [set(ids) for ids, _ in self._raw.values()]
        common_ids = sorted(id_sets[0].intersection(*id_sets[1:]))

        logger.info(f"Aligned modalities: {len(common_ids):,} tracks in common")
        for name, (ids, _) in self._raw.items():
            logger.info(f"  {name}: {len(ids):,} → {len(common_ids):,}")

        self.track_ids = np.array(common_ids)
        self._id_to_idx = {tid: i for i, tid in enumerate(common_ids)}

        for name, (ids, feats) in self._raw.items():
            id_map = {tid: i for i, tid in enumerate(ids)}
            indices = [id_map[tid] for tid in common_ids]
            aligned = feats[indices]

            if name == "audio":
                self.audio_features = aligned
            elif name == "lyrics":
                self.lyrics_features = aligned
            elif name == "genre":
                self.genre_features = aligned

        dims = []
        for name, arr in [("audio", self.audio_features),
                          ("lyrics", self.lyrics_features),
                          ("genre", self.genre_features)]:
            if arr is not None:
                dims.append(f"{name}={arr.shape[1]}")
        logger.info(f"  Dimensions: {', '.join(dims)}")

    def load_interactions(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load user-track interaction counts.

        Returns DataFrame with columns: user_id, track_id, count
        """
        fname = filename or (
            self.config.interactions_file if self.config else "userid_trackid_count.tsv.bz2"
        )
        filepath = self.data_dir / fname
        if not filepath.exists():
            raise FileNotFoundError(f"Interactions file not found: {filepath}")

        logger.info(f"Loading interactions from {fname}...")
        df = pd.read_csv(
            filepath, sep="\t", compression="bz2",
            dtype={"user_id": str, "track_id": str, "count": int},
        )
        logger.info(f"  Raw: {len(df):,} user-track pairs")

        min_count = self.config.min_listen_count if self.config else 2
        df = df[df["count"] >= min_count]
        logger.info(f"  After min_count>={min_count}: {len(df):,} pairs")

        if self.track_ids is not None:
            valid_tracks = set(self.track_ids)
            df = df[df["track_id"].isin(valid_tracks)]
            logger.info(f"  Restricted to aligned tracks: {len(df):,} pairs")

        min_t = self.config.min_tracks_per_user if self.config else 10
        max_t = self.config.max_tracks_per_user if self.config else 500
        user_counts = df.groupby("user_id")["track_id"].nunique()
        valid_users = user_counts[
            (user_counts >= min_t) & (user_counts <= max_t)
        ].index
        df = df[df["user_id"].isin(valid_users)]
        logger.info(f"  Users with {min_t}-{max_t} tracks: "
                    f"{len(valid_users):,} users, {len(df):,} pairs")

        self.interactions_df = df
        return df

    def get_colisten_pairs(
        self,
        max_users: int = 10_000,
        seed: int = 42,
    ) -> List[Tuple[str, Set[str]]]:
        """
        Build co-listen retrieval ground truth: for each track, the set of
        other tracks listened to by the same users.

        Returns list of (query_track_id, {positive_track_ids}).
        """
        if self.interactions_df is None:
            self.load_interactions()

        df = self.interactions_df
        rng = np.random.RandomState(seed)

        all_users = df["user_id"].unique()
        if len(all_users) > max_users:
            sampled = rng.choice(all_users, max_users, replace=False)
            df = df[df["user_id"].isin(set(sampled))]
            logger.info(f"Sampled {max_users:,} users for co-listen evaluation")

        user_tracks = df.groupby("user_id")["track_id"].apply(set).to_dict()

        track_positives: Dict[str, Set[str]] = {}
        for tracks in user_tracks.values():
            for tid in tracks:
                if tid not in track_positives:
                    track_positives[tid] = set()
                track_positives[tid].update(tracks - {tid})

        pairs = [(tid, pos) for tid, pos in track_positives.items() if len(pos) >= 1]

        avg_pos = np.mean([len(p) for _, p in pairs]) if pairs else 0
        logger.info(f"Co-listen pairs: {len(pairs):,} query tracks, "
                    f"avg {avg_pos:.1f} positives/track")
        return pairs

    def get_track_index(self, track_id: str) -> Optional[int]:
        """Get the row index for a track ID, or None if not in aligned set."""
        if self._id_to_idx is None:
            return None
        return self._id_to_idx.get(track_id)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DEFAULT_MUSIC4ALL_CONFIG as cfg

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ds = Music4allDataset(cfg.data_dir, config=cfg)
    ds.load_audio()
    ds.load_lyrics()
    ds.load_genre()
    ds.align_modalities()

    print(f"\nAligned tracks: {len(ds.track_ids):,}")
    print(f"Audio shape:  {ds.audio_features.shape}")
    print(f"Lyrics shape: {ds.lyrics_features.shape}")
    print(f"Genre shape:  {ds.genre_features.shape}")

    ds.load_interactions()
    pairs = ds.get_colisten_pairs(max_users=1000)
    print(f"Co-listen pairs (1K users): {len(pairs):,}")
