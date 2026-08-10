"""
Chapter 6: Historical Feature Engineering
==========================================
Compute aggregated historical features for users, items, and user-item pairs.

IMPORTANT: To prevent data leakage, these features are computed from the
TRAINING SET ONLY and then applied to train/val/test sets.

Point-in-Time (PIT) Correctness Note:
-------------------------------------
This pandas-based implementation uses LOOKUP TABLES (features computed from
the full training window). For true PIT correctness where `previous_listen_count`
is computed as a cumulative count up to each row's timestamp, use the Polars
pipeline in `utils/polars_pipeline.py`.

In production, a Feature Store would provide PIT-correct values for all
historical aggregates.

Features:
---------
User Features:
    - user_total_listens: Count of past listens
    - user_avg_completion: Mean played_ratio_pct historically
    - user_artist_concentration: HHI of artist distribution (how focused)
    - user_organic_ratio: % of organic vs recommended listens

Item Features:
    - item_total_plays: Global play count
    - item_avg_completion: Mean completion rate
    - item_unique_listeners: Number of unique users
    - item_like_ratio: likes / (likes + dislikes) if available

User-Item Features:
    - has_listened_before: 0/1 flag
    - previous_listen_count: How many times user played this item
    
REMOVED Features (due to data leakage):
    - user_item_avg_completion: REMOVED - computed from ALL interactions
      including the current row's label, causing severe look-ahead bias
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class HistoricalFeatureBuilder:
    """
    Build historical aggregate features from training data.
    
    Usage:
    ------
    >>> builder = HistoricalFeatureBuilder()
    >>> builder.fit(train_df, likes_df, dislikes_df, artist_mapping_df)
    >>> train_enriched = builder.transform(train_df)
    >>> test_enriched = builder.transform(test_df)
    """
    
    def __init__(self):
        self.user_features: Optional[pd.DataFrame] = None
        self.item_features: Optional[pd.DataFrame] = None
        self.user_item_features: Optional[pd.DataFrame] = None
        self.user_artist_features: Optional[pd.DataFrame] = None
        self.fitted = False
        
        # Default values for cold-start (unseen users/items)
        self.user_defaults = {}
        self.item_defaults = {}
        
    def fit(
        self,
        train_df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame] = None,
        dislikes_df: Optional[pd.DataFrame] = None,
        artist_mapping_df: Optional[pd.DataFrame] = None
    ) -> 'HistoricalFeatureBuilder':
        """
        Compute aggregate features from training data.
        
        Args:
            train_df: Training listens DataFrame (uid, item_id, timestamp, played_ratio_pct, is_organic)
            likes_df: Optional likes DataFrame (uid, item_id, timestamp)
            dislikes_df: Optional dislikes DataFrame (uid, item_id, timestamp)
            artist_mapping_df: Optional artist-item mapping (artist_id, item_id)
            
        Returns:
            Self for chaining
        """
        logger.info("Computing historical features from training data...")
        
        # Compute user-level features
        self._compute_user_features(train_df)
        
        # Compute item-level features
        self._compute_item_features(train_df, likes_df, dislikes_df)
        
        # Compute user-item pair features
        self._compute_user_item_features(train_df)
        
        # Compute user-artist features if mapping available
        if artist_mapping_df is not None:
            self._compute_user_artist_features(train_df, artist_mapping_df)
        
        # Compute global defaults for cold-start
        self._compute_defaults(train_df)
        
        self.fitted = True
        logger.info("Historical feature computation complete")
        
        return self
    
    def _compute_user_features(self, train_df: pd.DataFrame) -> None:
        """Compute user-level aggregate features."""
        logger.info("  Computing user features...")
        
        user_agg = train_df.groupby('uid').agg(
            user_total_listens=('item_id', 'count'),
            user_avg_completion=('played_ratio_pct', 'mean'),
            user_std_completion=('played_ratio_pct', 'std'),
            user_median_completion=('played_ratio_pct', 'median'),
            user_unique_items=('item_id', 'nunique'),
            user_organic_ratio=('is_organic', 'mean'),
            user_min_timestamp=('timestamp', 'min'),
            user_max_timestamp=('timestamp', 'max'),
        ).reset_index()
        
        # Fill NaN std with 0 (users with single listen)
        user_agg['user_std_completion'] = user_agg['user_std_completion'].fillna(0)
        
        # Compute listening span (diversity of time)
        user_agg['user_active_span'] = user_agg['user_max_timestamp'] - user_agg['user_min_timestamp']
        
        # Compute listen rate (listens per time unit active)
        user_agg['user_listen_rate'] = user_agg['user_total_listens'] / (user_agg['user_active_span'] + 1)
        
        # Drop intermediate columns
        user_agg = user_agg.drop(columns=['user_min_timestamp', 'user_max_timestamp'])
        
        self.user_features = user_agg
        logger.info(f"    Computed features for {len(user_agg):,} users")
    
    def _compute_item_features(
        self,
        train_df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame],
        dislikes_df: Optional[pd.DataFrame]
    ) -> None:
        """Compute item-level aggregate features."""
        logger.info("  Computing item features...")
        
        item_agg = train_df.groupby('item_id').agg(
            item_total_plays=('uid', 'count'),
            item_avg_completion=('played_ratio_pct', 'mean'),
            item_std_completion=('played_ratio_pct', 'std'),
            item_unique_listeners=('uid', 'nunique'),
            item_organic_ratio=('is_organic', 'mean'),
        ).reset_index()
        
        # Fill NaN std with 0
        item_agg['item_std_completion'] = item_agg['item_std_completion'].fillna(0)
        
        # Compute repeat listen ratio (total plays / unique listeners)
        item_agg['item_repeat_ratio'] = item_agg['item_total_plays'] / item_agg['item_unique_listeners']
        
        # Add like ratio if data available
        if likes_df is not None and dislikes_df is not None:
            likes_count = likes_df.groupby('item_id').size().reset_index(name='item_likes')
            dislikes_count = dislikes_df.groupby('item_id').size().reset_index(name='item_dislikes')
            
            item_agg = item_agg.merge(likes_count, on='item_id', how='left')
            item_agg = item_agg.merge(dislikes_count, on='item_id', how='left')
            
            item_agg['item_likes'] = item_agg['item_likes'].fillna(0)
            item_agg['item_dislikes'] = item_agg['item_dislikes'].fillna(0)
            
            # Like ratio: likes / (likes + dislikes), with smoothing
            total_feedback = item_agg['item_likes'] + item_agg['item_dislikes']
            item_agg['item_like_ratio'] = (item_agg['item_likes'] + 1) / (total_feedback + 2)  # Laplace smoothing
        else:
            item_agg['item_like_ratio'] = 0.5  # Default neutral
        
        self.item_features = item_agg
        logger.info(f"    Computed features for {len(item_agg):,} items")
    
    def _compute_user_item_features(self, train_df: pd.DataFrame) -> None:
        """
        Compute user-item pair features.
        
        NOTE ON DATA LEAKAGE:
        ---------------------
        `user_item_avg_completion` has been REMOVED because it causes severe 
        data leakage: it computes the average completion from ALL interactions,
        including the current row's label, directly predicting the target.
        
        `previous_listen_count` in this pandas implementation uses a lookup-table
        approach (total count per user-item pair). For true Point-in-Time (PIT)
        correctness, use the Polars pipeline in `utils/polars_pipeline.py` which
        computes a cumulative count up to (but not including) each row's timestamp.
        
        In production, a Feature Store would provide PIT-correct values for all
        historical aggregates.
        """
        logger.info("  Computing user-item features...")
        
        # NOTE: Only counting interactions, not averaging completion (removed due to leakage)
        user_item_agg = train_df.groupby(['uid', 'item_id']).agg(
            previous_listen_count=('timestamp', 'count'),
            # user_item_avg_completion REMOVED - causes severe data leakage
            # because it includes the current row's label in the average
        ).reset_index()
        
        # has_listened_before is implicit (if in this table, yes)
        user_item_agg['has_listened_before'] = 1
        
        self.user_item_features = user_item_agg
        logger.info(f"    Computed features for {len(user_item_agg):,} user-item pairs")
        logger.info("    NOTE: For PIT-correct previous_listen_count, use Polars pipeline")
    
    def _compute_user_artist_features(
        self,
        train_df: pd.DataFrame,
        artist_mapping_df: pd.DataFrame
    ) -> None:
        """Compute user-artist affinity features."""
        logger.info("  Computing user-artist features...")
        
        # Join items with artists
        train_with_artist = train_df.merge(
            artist_mapping_df[['item_id', 'artist_id']],
            on='item_id',
            how='left'
        )
        
        # User-artist play counts
        user_artist = train_with_artist.groupby(['uid', 'artist_id']).agg(
            user_artist_plays=('item_id', 'count'),
            user_artist_avg_completion=('played_ratio_pct', 'mean'),
        ).reset_index()
        
        # User's total plays (for computing concentration)
        user_totals = train_df.groupby('uid')['item_id'].count().reset_index(name='user_total')
        user_artist = user_artist.merge(user_totals, on='uid', how='left')
        
        # Artist share for each user
        user_artist['artist_share'] = user_artist['user_artist_plays'] / user_artist['user_total']
        
        self.user_artist_features = user_artist
        logger.info(f"    Computed features for {len(user_artist):,} user-artist pairs")
    
    def _compute_defaults(self, train_df: pd.DataFrame) -> None:
        """Compute global defaults for cold-start handling."""
        self.user_defaults = {
            'user_total_listens': 0,
            'user_avg_completion': train_df['played_ratio_pct'].mean(),
            'user_std_completion': train_df['played_ratio_pct'].std(),
            'user_median_completion': train_df['played_ratio_pct'].median(),
            'user_unique_items': 0,
            'user_organic_ratio': train_df['is_organic'].mean(),
            'user_active_span': 0,
            'user_listen_rate': 0,
        }
        
        self.item_defaults = {
            'item_total_plays': 0,
            'item_avg_completion': train_df['played_ratio_pct'].mean(),
            'item_std_completion': train_df['played_ratio_pct'].std(),
            'item_unique_listeners': 0,
            'item_organic_ratio': train_df['is_organic'].mean(),
            'item_repeat_ratio': 1.0,
            'item_like_ratio': 0.5,
        }
        
        self.user_item_defaults = {
            'previous_listen_count': 0,
            'has_listened_before': 0,
            # user_item_avg_completion REMOVED due to severe data leakage
        }
    
    def transform(
        self,
        df: pd.DataFrame,
        artist_mapping_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add historical features to a DataFrame.
        
        Args:
            df: DataFrame with uid, item_id columns
            artist_mapping_df: Optional artist mapping for artist features
            
        Returns:
            DataFrame with additional historical feature columns
        """
        if not self.fitted:
            raise ValueError("HistoricalFeatureBuilder not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Add user features
        df = df.merge(self.user_features, on='uid', how='left')
        for col, default in self.user_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        # Add item features
        item_cols = [c for c in self.item_features.columns if c != 'item_id']
        df = df.merge(self.item_features, on='item_id', how='left')
        for col, default in self.item_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        # Add user-item features
        # NOTE: user_item_avg_completion removed due to severe data leakage
        user_item_cols = ['uid', 'item_id', 'previous_listen_count', 'has_listened_before']
        available_cols = [c for c in user_item_cols if c in self.user_item_features.columns]
        df = df.merge(
            self.user_item_features[available_cols],
            on=['uid', 'item_id'],
            how='left'
        )
        
        # Fill user-item defaults
        df['has_listened_before'] = df['has_listened_before'].fillna(0).astype(int)
        df['previous_listen_count'] = df['previous_listen_count'].fillna(0)
        
        # Add user-artist features if available
        if self.user_artist_features is not None and artist_mapping_df is not None:
            # Get artist for each item
            df = df.merge(
                artist_mapping_df[['item_id', 'artist_id']],
                on='item_id',
                how='left'
            )
            
            # Get user-artist affinity
            df = df.merge(
                self.user_artist_features[['uid', 'artist_id', 'artist_share', 'user_artist_avg_completion']],
                on=['uid', 'artist_id'],
                how='left'
            )
            
            df['artist_share'] = df['artist_share'].fillna(0)
            df['user_artist_avg_completion'] = df['user_artist_avg_completion'].fillna(
                self.user_defaults['user_avg_completion']
            )
            
            # Drop artist_id (not needed as feature)
            df = df.drop(columns=['artist_id'], errors='ignore')
        
        return df
    
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame] = None,
        dislikes_df: Optional[pd.DataFrame] = None,
        artist_mapping_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(train_df, likes_df, dislikes_df, artist_mapping_df)
        return self.transform(train_df, artist_mapping_df)
    
    def get_feature_names(self) -> list:
        """Get list of feature names added by this builder."""
        features = []
        
        # User features
        if self.user_features is not None:
            features.extend([c for c in self.user_features.columns if c != 'uid'])
        
        # Item features
        if self.item_features is not None:
            features.extend([c for c in self.item_features.columns if c != 'item_id'])
        
        # User-item features (user_item_avg_completion removed due to leakage)
        features.extend(['has_listened_before', 'previous_listen_count'])
        
        # User-artist features
        if self.user_artist_features is not None:
            features.extend(['artist_share', 'user_artist_avg_completion'])
        
        return features
    
    def save(self, path: str) -> None:
        """Save feature builder to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'user_features': self.user_features,
            'item_features': self.item_features,
            'user_item_features': self.user_item_features,
            'user_artist_features': self.user_artist_features,
            'user_defaults': self.user_defaults,
            'item_defaults': self.item_defaults,
            'user_item_defaults': self.user_item_defaults,
            'fitted': self.fitted,
        }
        
        with open(path / 'historical_features.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Historical feature builder saved to {path}")
    
    def load(self, path: str) -> 'HistoricalFeatureBuilder':
        """Load feature builder from disk."""
        path = Path(path)
        
        with open(path / 'historical_features.pkl', 'rb') as f:
            state = pickle.load(f)
        
        self.user_features = state['user_features']
        self.item_features = state['item_features']
        self.user_item_features = state['user_item_features']
        self.user_artist_features = state['user_artist_features']
        self.user_defaults = state['user_defaults']
        self.item_defaults = state['item_defaults']
        self.user_item_defaults = state['user_item_defaults']
        self.fitted = state['fitted']
        
        logger.info(f"Historical feature builder loaded from {path}")
        return self


def get_historical_feature_columns() -> Dict[str, list]:
    """
    Get dictionary of historical feature columns by category.
    
    Returns:
        Dictionary mapping category to list of column names
    """
    return {
        'user_features': [
            'user_total_listens',
            'user_avg_completion',
            'user_std_completion',
            'user_median_completion',
            'user_unique_items',
            'user_organic_ratio',
            'user_active_span',
            'user_listen_rate',
        ],
        'item_features': [
            'item_total_plays',
            'item_avg_completion',
            'item_std_completion',
            'item_unique_listeners',
            'item_organic_ratio',
            'item_repeat_ratio',
            'item_like_ratio',
        ],
        'user_item_features': [
            'has_listened_before',
            'previous_listen_count',
            # user_item_avg_completion REMOVED due to data leakage
        ],
        'user_artist_features': [
            'artist_share',
            'user_artist_avg_completion',
        ],
    }

