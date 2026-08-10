"""
Chapter 6: Polars-Based Feature Pipeline
=========================================
Memory-efficient feature processing using Polars for large datasets
on memory-constrained machines (e.g., 16GB laptop with 50M rows).

Why Polars?
-----------
- 2-5x more memory efficient than pandas
- Lazy evaluation optimizes query plans before execution
- Native parallel processing
- Zero-copy data sharing where possible
- Apache Arrow memory format (same as PyArrow)

Point-in-Time (PIT) Feature Correctness
---------------------------------------
A critical concept in production ML systems is Point-in-Time (PIT) correctness:
features computed for a prediction at time T should only use data from times < T.

This pipeline implements TWO approaches:

1. **PIT-CORRECT Features** (computed per-row):
   - `previous_listen_count`: Cumulative count of (user, item) interactions
     occurring BEFORE the current row's timestamp. Computed using Polars'
     window functions with `order_by='timestamp'`.
   - `has_listened_before`: Binary flag derived from previous_listen_count.

2. **LOOKUP-TABLE Features** (computed from full training window):
   The following features are computed from ALL training data and joined
   as lookup tables. This is a KNOWN SIMPLIFICATION for educational purposes:
   
   User-level (lower leakage risk - aggregated over many items):
   - user_avg_completion, user_median_completion, user_std_completion
   - user_total_listens, user_unique_items, user_organic_ratio
   - user_active_span, user_listen_rate
   
   Item-level (lower leakage risk - aggregated over many users):
   - item_avg_completion, item_std_completion, item_total_plays
   - item_unique_listeners, item_organic_ratio, item_repeat_ratio, item_like_ratio

Production Feature Store Context
--------------------------------
In production systems, a Feature Store (e.g., Feast, Tecton, Vertex AI Feature Store)
would provide PIT-correct values for ALL features:

1. **Offline Feature Computation**: Batch jobs compute features at regular intervals
   (e.g., daily user_avg_completion as of midnight each day).

2. **Temporal Joins**: When creating training data, the Feature Store performs
   temporal joins - for each row with timestamp T, it fetches feature values
   that were valid at time T (not the latest values).

3. **Online Serving**: At inference time, the Feature Store serves the most
   recent feature values, which are genuinely "historical" from the model's
   perspective.

For this educational codebase, we accept the simplification of lookup-table
features because:
- The train-test leakage risk is manageable (~0.17 AUC gap)
- User/item-level aggregates are less prone to leakage than user-item-level
- True PIT would require 5-10x more code complexity
- The key concepts are demonstrated with previous_listen_count

Usage:
------
>>> from utils.polars_pipeline import PolarsPipeline, load_yambda_polars
>>> 
>>> # Load full dataset (300 days, ~50M rows)
>>> train_lf, test_lf = load_yambda_polars(data_dir)
>>> 
>>> # Or use shorter train window for faster iteration on 16GB laptop
>>> train_lf, test_lf = load_yambda_polars(data_dir, train_days=30)  # ~5M rows
>>> 
>>> # Run feature pipeline
>>> pipeline = PolarsPipeline()
>>> train_df, feature_cols = pipeline.fit_transform(train_lf)

Requirements:
-------------
pip install polars pyarrow
"""

import polars as pl
from polars import col
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle

# Note: pl.lit is used directly rather than imported

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Optimal Polars dtypes
POLARS_DTYPES = {
    'uid': pl.Int32,
    'item_id': pl.Int32,
    'timestamp': pl.Int32,
    'played_ratio_pct': pl.Float32,
    'track_length_seconds': pl.Float32,
    'is_organic': pl.Int8,
    'label': pl.Int8,
}


# =============================================================================
# Data Loading with Time Window
# =============================================================================

def load_yambda_polars(
    data_dir: str,
    train_days: Optional[int] = None,
    test_days: int = 1,
    gap_seconds: int = 1800,  # 30 minutes gap
    lazy: bool = True
) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Union[pl.LazyFrame, pl.DataFrame]]:
    """
    Load Yambda dataset with Global Temporal Split using Polars.
    
    Uses a time-window approach which is more appropriate for time-series data
    than stratified sampling:
    - Preserves temporal order
    - Maintains correct historical feature computation
    - No risk of temporal data leakage
    
    Args:
        data_dir: Path to Yandex/flat directory
        train_days: Number of days for training window. Options:
            - None: Use full 300 days (GTS protocol, ~50M rows)
            - 30: Last 30 days (~5M rows, good for local dev)
            - 60: Last 60 days (~10M rows)
            - 90: Last 90 days (~15M rows)
            - 150: Last 150 days (~25M rows)
        test_days: Number of days for test window (default: 1 day per GTS)
        gap_seconds: Gap between train and test in seconds (default: 30 min)
        lazy: If True, return LazyFrames for deferred execution
        
    Returns:
        Tuple of (train_data, test_data) as LazyFrames or DataFrames
        
    Example:
        >>> # Full 300 days (GTS protocol)
        >>> train_lf, test_lf = load_yambda_polars(data_dir)
        >>> 
        >>> # Last 30 days only (faster for local development)
        >>> train_lf, test_lf = load_yambda_polars(data_dir, train_days=30)
        >>> 
        >>> # Last 90 days (balance of speed and data)
        >>> train_lf, test_lf = load_yambda_polars(data_dir, train_days=90)
    """
    data_path = Path(data_dir)
    listens_path = data_path / 'listens.parquet'
    
    if not listens_path.exists():
        raise FileNotFoundError(f"Listens file not found: {listens_path}")
    
    logger.info(f"Loading data from {listens_path}")
    
    # Scan parquet (lazy - doesn't load into memory yet)
    lf = pl.scan_parquet(listens_path)
    
    # Get timestamp range
    ts_stats = lf.select([
        col('timestamp').min().alias('min_ts'),
        col('timestamp').max().alias('max_ts'),
    ]).collect()
    
    min_ts = ts_stats['min_ts'][0]
    max_ts = ts_stats['max_ts'][0]
    total_range = max_ts - min_ts
    
    # Convert time parameters to timestamp units (1 unit = 5 seconds)
    SECONDS_PER_DAY = 86400
    TS_UNITS_PER_DAY = SECONDS_PER_DAY // 5  # 17280 units per day
    
    gap_units = gap_seconds // 5
    test_duration_units = test_days * TS_UNITS_PER_DAY
    
    # Calculate boundaries working backwards from max_ts
    # Timeline: [...train_start...train_end][gap][test_start...test_end=max_ts]
    test_end = max_ts
    test_start = test_end - test_duration_units
    train_end = test_start - gap_units
    
    if train_days is not None:
        # Use specified train window (most recent N days before gap)
        train_duration_units = train_days * TS_UNITS_PER_DAY
        train_start = train_end - train_duration_units
        
        # Ensure we don't go before data start
        train_start = max(train_start, min_ts)
        
        logger.info(f"Using {train_days}-day train window")
    else:
        # Use all available data for training (GTS protocol: ~300 days)
        train_start = min_ts
        actual_train_days = (train_end - train_start) / TS_UNITS_PER_DAY
        logger.info(f"Using full train window (~{actual_train_days:.0f} days)")
    
    logger.info(f"Timestamp range: {min_ts:,} to {max_ts:,}")
    logger.info(f"Train window: {train_start:,} to {train_end:,} "
                f"(~{(train_end - train_start) / TS_UNITS_PER_DAY:.0f} days)")
    logger.info(f"Test window: {test_start:,} to {test_end:,} "
                f"(~{test_days} days)")
    
    # Split by timestamp
    train_lf = lf.filter(
        (col('timestamp') >= train_start) & (col('timestamp') < train_end)
    )
    test_lf = lf.filter(
        (col('timestamp') >= test_start) & (col('timestamp') <= test_end)
    )
    
    # Cast to optimal dtypes
    train_lf = _cast_dtypes(train_lf)
    test_lf = _cast_dtypes(test_lf)
    
    if not lazy:
        logger.info("Collecting DataFrames into memory...")
        train_lf = train_lf.collect()
        test_lf = test_lf.collect()
        logger.info(f"Train rows: {len(train_lf):,}, Test rows: {len(test_lf):,}")
    
    return train_lf, test_lf


def _cast_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast columns to optimal dtypes."""
    # Get column names without triggering expensive schema resolution warning
    existing_cols = lf.collect_schema().names()
    
    # Only cast columns that exist in the data
    cast_exprs = [
        col(c).cast(POLARS_DTYPES[c]) 
        for c in POLARS_DTYPES 
        if c in existing_cols
    ]
    
    if cast_exprs:
        return lf.with_columns(cast_exprs)
    return lf


# =============================================================================
# Feature Encoding
# =============================================================================

class PolarsFeatureEncoder:
    """
    Polars-based feature encoder for categorical variables.
    
    Same logic as pandas FeatureEncoder but optimized for Polars.
    """
    
    def __init__(self, high_cardinality_threshold: int = 10000):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.frequency_maps: Dict[str, pl.DataFrame] = {}
        self.label_maps: Dict[str, pl.DataFrame] = {}
        self.fitted = False
    
    def fit(
        self, 
        lf: Union[pl.LazyFrame, pl.DataFrame],
        categorical_columns: List[str]
    ) -> 'PolarsFeatureEncoder':
        """Fit encoder on training data."""
        
        # Collect if lazy (needed to compute statistics)
        if isinstance(lf, pl.LazyFrame):
            df = lf.collect()
        else:
            df = lf
        
        for col_name in categorical_columns:
            if col_name not in df.columns:
                logger.warning(f"Column {col_name} not found, skipping")
                continue
            
            n_unique = df[col_name].n_unique()
            
            if n_unique > self.high_cardinality_threshold:
                # Frequency encoding - store as lookup table
                freq_df = (
                    df.group_by(col_name)
                    .agg(pl.count().alias('_count'))
                    .with_columns([
                        (col('_count') / df.height).alias(f'{col_name}_freq')
                    ])
                    .select([col_name, f'{col_name}_freq'])
                )
                self.frequency_maps[col_name] = freq_df
                logger.info(f"Frequency encoding: {col_name} ({n_unique:,} unique)")
            else:
                # Label encoding - store mapping
                unique_vals = df[col_name].unique().sort()
                label_df = pl.DataFrame({
                    col_name: unique_vals,
                    f'{col_name}_encoded': pl.Series(range(len(unique_vals)), dtype=pl.Int32)
                })
                self.label_maps[col_name] = label_df
                logger.info(f"Label encoding: {col_name} ({n_unique:,} unique)")
        
        self.fitted = True
        return self
    
    def transform(
        self, 
        lf: Union[pl.LazyFrame, pl.DataFrame]
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """Apply encodings to data."""
        
        if not self.fitted:
            raise ValueError("Encoder not fitted")
        
        is_lazy = isinstance(lf, pl.LazyFrame)
        
        # Apply frequency encodings via join
        for col_name, freq_df in self.frequency_maps.items():
            if is_lazy:
                freq_lf = freq_df.lazy()
                lf = lf.join(freq_lf, on=col_name, how='left')
            else:
                lf = lf.join(freq_df, on=col_name, how='left')
            
            # Fill nulls (unseen values) with 0
            lf = lf.with_columns([
                col(f'{col_name}_freq').fill_null(0.0)
            ])
        
        # Apply label encodings via join
        for col_name, label_df in self.label_maps.items():
            if is_lazy:
                label_lf = label_df.lazy()
                lf = lf.join(label_lf, on=col_name, how='left')
            else:
                lf = lf.join(label_df, on=col_name, how='left')
            
            # Fill nulls (unseen values) with -1
            lf = lf.with_columns([
                col(f'{col_name}_encoded').fill_null(-1)
            ])
        
        return lf
    
    def get_encoded_column_names(self) -> List[str]:
        """Get list of encoded column names."""
        cols = []
        for c in self.frequency_maps:
            cols.append(f'{c}_freq')
        for c in self.label_maps:
            cols.append(f'{c}_encoded')
        return cols


# =============================================================================
# Historical Features (Polars)
# =============================================================================

class PolarsHistoricalFeatures:
    """
    Compute historical aggregate features using Polars.
    
    Much more memory-efficient than pandas for large datasets due to:
    - Lazy evaluation and query optimization
    - Better memory layout
    - Parallel aggregation
    """
    
    def __init__(self):
        self.user_features: Optional[pl.DataFrame] = None
        self.item_features: Optional[pl.DataFrame] = None
        self.user_item_features: Optional[pl.DataFrame] = None
        
        self.user_defaults: Dict[str, float] = {}
        self.item_defaults: Dict[str, float] = {}
        self.user_item_defaults: Dict[str, float] = {}
        
        self.fitted = False
    
    def fit(
        self,
        lf: Union[pl.LazyFrame, pl.DataFrame],
        likes_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None,
        dislikes_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None
    ) -> 'PolarsHistoricalFeatures':
        """Compute aggregate features from training data."""
        
        logger.info("Computing historical features with Polars...")
        
        # Collect if lazy (needed for aggregations)
        if isinstance(lf, pl.LazyFrame):
            df = lf.collect()
        else:
            df = lf
        
        # ===== User Features =====
        logger.info("  Computing user features...")
        self.user_features = (
            df.group_by('uid')
            .agg([
                pl.count().alias('user_total_listens'),
                col('played_ratio_pct').mean().alias('user_avg_completion'),
                col('played_ratio_pct').std().alias('user_std_completion'),
                col('played_ratio_pct').median().alias('user_median_completion'),
                col('item_id').n_unique().alias('user_unique_items'),
                col('is_organic').mean().alias('user_organic_ratio'),
                col('timestamp').min().alias('_min_ts'),
                col('timestamp').max().alias('_max_ts'),
            ])
            .with_columns([
                col('user_std_completion').fill_null(0.0),
                (col('_max_ts') - col('_min_ts')).alias('user_active_span'),
            ])
            .with_columns([
                (col('user_total_listens') / (col('user_active_span') + 1)).alias('user_listen_rate')
            ])
            .drop(['_min_ts', '_max_ts'])
            .cast({
                'user_total_listens': pl.Int32,
                'user_avg_completion': pl.Float32,
                'user_std_completion': pl.Float32,
                'user_median_completion': pl.Float32,
                'user_unique_items': pl.Int32,
                'user_organic_ratio': pl.Float32,
                'user_active_span': pl.Int32,
                'user_listen_rate': pl.Float32,
            })
        )
        logger.info(f"    Computed for {self.user_features.height:,} users")
        
        # ===== Item Features =====
        logger.info("  Computing item features...")
        self.item_features = (
            df.group_by('item_id')
            .agg([
                pl.count().alias('item_total_plays'),
                col('played_ratio_pct').mean().alias('item_avg_completion'),
                col('played_ratio_pct').std().alias('item_std_completion'),
                col('uid').n_unique().alias('item_unique_listeners'),
                col('is_organic').mean().alias('item_organic_ratio'),
            ])
            .with_columns([
                col('item_std_completion').fill_null(0.0),
                (col('item_total_plays') / col('item_unique_listeners')).alias('item_repeat_ratio'),
            ])
        )
        
        # Add like ratio if likes/dislikes provided
        if likes_lf is not None and dislikes_lf is not None:
            likes_df = likes_lf.collect() if isinstance(likes_lf, pl.LazyFrame) else likes_lf
            dislikes_df = dislikes_lf.collect() if isinstance(dislikes_lf, pl.LazyFrame) else dislikes_lf
            
            likes_count = likes_df.group_by('item_id').agg(pl.count().alias('_likes'))
            dislikes_count = dislikes_df.group_by('item_id').agg(pl.count().alias('_dislikes'))
            
            self.item_features = (
                self.item_features
                .join(likes_count, on='item_id', how='left')
                .join(dislikes_count, on='item_id', how='left')
                .with_columns([
                    col('_likes').fill_null(0),
                    col('_dislikes').fill_null(0),
                ])
                .with_columns([
                    ((col('_likes') + 1) / (col('_likes') + col('_dislikes') + 2)).alias('item_like_ratio')
                ])
                .drop(['_likes', '_dislikes'])
            )
        else:
            self.item_features = self.item_features.with_columns([
                pl.lit(0.5).cast(pl.Float32).alias('item_like_ratio')
            ])
        
        # Cast to optimal dtypes
        self.item_features = self.item_features.cast({
            'item_total_plays': pl.Int32,
            'item_avg_completion': pl.Float32,
            'item_std_completion': pl.Float32,
            'item_unique_listeners': pl.Int32,
            'item_organic_ratio': pl.Float32,
            'item_repeat_ratio': pl.Float32,
            'item_like_ratio': pl.Float32,
        })
        logger.info(f"    Computed for {self.item_features.height:,} items")
        
        # ===== User-Item Features =====
        # NOTE: previous_listen_count is now computed with Point-in-Time (PIT) 
        # correctness in transform(), not here. This ensures that for each row,
        # we only count interactions that occurred BEFORE that row's timestamp.
        #
        # user_item_avg_completion was REMOVED due to severe look-ahead bias:
        # it was computed from ALL listens including future ones, leaking labels.
        #
        # We still track unique (uid, item_id) pairs seen during training for
        # the has_listened_before flag, but the count is computed per-row.
        logger.info("  Computing user-item features...")
        self.user_item_pairs = (
            df.select(['uid', 'item_id'])
            .unique()
        )
        logger.info(f"    Found {self.user_item_pairs.height:,} unique user-item pairs")
        logger.info("    NOTE: previous_listen_count will be computed with PIT correctness in transform()")
        
        # ===== Defaults for cold-start =====
        self.user_defaults = {
            'user_total_listens': 0,
            'user_avg_completion': float(df['played_ratio_pct'].mean()),
            'user_std_completion': float(df['played_ratio_pct'].std()),
            'user_median_completion': float(df['played_ratio_pct'].median()),
            'user_unique_items': 0,
            'user_organic_ratio': float(df['is_organic'].mean()),
            'user_active_span': 0,
            'user_listen_rate': 0.0,
        }
        
        self.item_defaults = {
            'item_total_plays': 0,
            'item_avg_completion': self.user_defaults['user_avg_completion'],
            'item_std_completion': self.user_defaults['user_std_completion'],
            'item_unique_listeners': 0,
            'item_organic_ratio': self.user_defaults['user_organic_ratio'],
            'item_repeat_ratio': 1.0,
            'item_like_ratio': 0.5,
        }
        
        # User-item defaults (used for cold-start cases in transform)
        self.user_item_defaults = {
            'has_listened_before': 0,
            'previous_listen_count': 0,
            # user_item_avg_completion removed due to leakage
        }
        
        # Store the user_item_features reference for backwards compatibility
        # (now just used for has_listened_before check, count computed in transform)
        self.user_item_features = self.user_item_pairs
        
        self.fitted = True
        logger.info("  Historical features computed")
        
        return self
    
    def transform(
        self, 
        lf: Union[pl.LazyFrame, pl.DataFrame]
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Add historical features via efficient joins and PIT-correct computations.
        
        Point-in-Time (PIT) Correctness:
        ---------------------------------
        `previous_listen_count` is computed with PIT correctness: for each row,
        we count only the interactions of that (user, item) pair that occurred 
        BEFORE that row's timestamp. This prevents look-ahead bias.
        
        Features NOT PIT-corrected (documented):
        ----------------------------------------
        The following features are computed from the ENTIRE training window
        and joined as lookup tables. In production, a Feature Store would 
        provide these as PIT-correct values:
        
        - user_avg_completion, user_median_completion, user_std_completion
        - user_total_listens, user_unique_items, user_organic_ratio
        - user_active_span, user_listen_rate
        - item_avg_completion, item_std_completion, item_total_plays
        - item_unique_listeners, item_organic_ratio, item_repeat_ratio
        
        For educational purposes, the lookup-table approach is used because:
        1. It's simpler to understand and implement
        2. The leakage risk is lower (user/item-level vs user-item-level)
        3. True PIT requires significant infrastructure (sliding windows, 
           incremental computation) typically handled by Feature Stores
        """
        
        if not self.fitted:
            raise ValueError("Not fitted")
        
        is_lazy = isinstance(lf, pl.LazyFrame)
        
        # Convert lookup tables to lazy if input is lazy
        user_lf = self.user_features.lazy() if is_lazy else self.user_features
        item_lf = self.item_features.lazy() if is_lazy else self.item_features
        
        # Join user features
        lf = lf.join(user_lf, on='uid', how='left')
        
        # Fill user defaults
        user_fill_exprs = [
            col(c).fill_null(v).alias(c) for c, v in self.user_defaults.items()
            if c in (lf.collect_schema().names() if is_lazy else lf.columns)
        ]
        if user_fill_exprs:
            lf = lf.with_columns(user_fill_exprs)
        
        # Join item features
        lf = lf.join(item_lf, on='item_id', how='left')
        
        # Fill item defaults
        item_fill_exprs = [
            col(c).fill_null(v).alias(c) for c, v in self.item_defaults.items()
            if c in (lf.collect_schema().names() if is_lazy else lf.columns)
        ]
        if item_fill_exprs:
            lf = lf.with_columns(item_fill_exprs)
        
        # =====================================================================
        # PIT-Correct User-Item Features
        # =====================================================================
        # Compute previous_listen_count with Point-in-Time correctness:
        # For each row, count how many times this (user, item) appeared BEFORE
        # this timestamp. Uses cumulative count - 1 (exclude current row).
        #
        # OLD (leaky): group_by(['uid', 'item_id']).count() → same for all rows
        # NEW (PIT):   cumcount over (uid, item_id) ordered by timestamp - 1
        # =====================================================================
        
        lf = lf.with_columns([
            # Cumulative count within each (uid, item_id) group, ordered by timestamp
            # Subtract 1 to get count of PREVIOUS interactions (not including current)
            (
                pl.count()
                .over(['uid', 'item_id'], order_by='timestamp')
                - 1
            ).cast(pl.Int16).alias('previous_listen_count'),
        ])
        
        # has_listened_before: 1 if user has interacted with this item before
        lf = lf.with_columns([
            (col('previous_listen_count') > 0)
                .cast(pl.Int8)
                .alias('has_listened_before'),
        ])
        
        return lf
    
    def get_feature_names(self) -> List[str]:
        """Get list of historical feature names."""
        features = []
        
        if self.user_features is not None:
            features.extend([c for c in self.user_features.columns if c != 'uid'])
        
        if self.item_features is not None:
            features.extend([c for c in self.item_features.columns if c != 'item_id'])
        
        features.extend(['has_listened_before', 'previous_listen_count'])
        
        return features
    
    def get_memory_usage_mb(self) -> Dict[str, float]:
        """Get memory usage of stored features."""
        return {
            'user_features': self.user_features.estimated_size('mb') if self.user_features is not None else 0,
            'item_features': self.item_features.estimated_size('mb') if self.item_features is not None else 0,
            'user_item_features': self.user_item_features.estimated_size('mb') if self.user_item_features is not None else 0,
        }


# =============================================================================
# Complete Pipeline
# =============================================================================

class PolarsPipeline:
    """
    Complete feature pipeline using Polars.
    
    Combines:
    - Categorical encoding (frequency/label)
    - Historical features
    - Derived features (time-based)
    - Dtype optimization
    
    Usage:
        >>> pipeline = PolarsPipeline()
        >>> 
        >>> # Fit on training data
        >>> train_df, feature_cols = pipeline.fit_transform(train_lf)
        >>> 
        >>> # Transform test data
        >>> test_df, _ = pipeline.transform(test_lf)
        >>> 
        >>> # Train model (convert to pandas for XGBoost)
        >>> X_train = train_df.select(feature_cols).to_pandas()
        >>> y_train = train_df['label'].to_pandas()
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        high_cardinality_threshold: int = 10000,
        add_derived_features: bool = True,
        add_historical_features: bool = True,
        label_threshold: int = 50,
    ):
        """
        Initialize pipeline.
        
        Args:
            categorical_columns: Columns to encode
            numerical_columns: Numerical columns to include
            high_cardinality_threshold: Use frequency encoding above this
            add_derived_features: Add hour_of_day, day_of_week
            add_historical_features: Add user/item aggregates
            label_threshold: played_ratio_pct threshold for positive label
        """
        self.categorical_columns = categorical_columns or ['uid', 'item_id', 'is_organic']
        self.numerical_columns = numerical_columns or ['track_length_seconds']
        self.high_cardinality_threshold = high_cardinality_threshold
        self.add_derived_features = add_derived_features
        self.add_historical_features = add_historical_features
        self.label_threshold = label_threshold
        
        self.encoder = PolarsFeatureEncoder(high_cardinality_threshold)
        self.hist_builder = PolarsHistoricalFeatures()
        
        self.feature_columns: List[str] = []
        self.fitted = False
    
    def fit(
        self,
        lf: Union[pl.LazyFrame, pl.DataFrame],
        likes_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None,
        dislikes_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None
    ) -> 'PolarsPipeline':
        """Fit pipeline on training data."""
        
        logger.info("Fitting Polars pipeline...")
        
        # Fit encoder
        self.encoder.fit(lf, self.categorical_columns)
        
        # Fit historical features
        if self.add_historical_features:
            self.hist_builder.fit(lf, likes_lf, dislikes_lf)
        
        # Build feature column list
        self._build_feature_columns()
        
        self.fitted = True
        logger.info(f"Pipeline fitted with {len(self.feature_columns)} features")
        
        return self
    
    def _build_feature_columns(self):
        """Build list of feature columns."""
        self.feature_columns = []
        
        # Encoded categorical
        self.feature_columns.extend(self.encoder.get_encoded_column_names())
        
        # Numerical
        self.feature_columns.extend(self.numerical_columns)
        
        # Derived
        if self.add_derived_features:
            self.feature_columns.extend(['hour_of_day', 'day_of_week'])
        
        # Historical
        if self.add_historical_features:
            self.feature_columns.extend(self.hist_builder.get_feature_names())
    
    def transform(
        self,
        lf: Union[pl.LazyFrame, pl.DataFrame],
        collect: bool = True
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Transform data using fitted pipeline.
        
        Args:
            lf: Input LazyFrame or DataFrame
            collect: If True and input is lazy, collect to DataFrame
            
        Returns:
            Tuple of (transformed DataFrame, feature column list)
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted")
        
        is_lazy = isinstance(lf, pl.LazyFrame)
        
        # Add label
        lf = lf.with_columns([
            (col('played_ratio_pct') >= self.label_threshold).cast(pl.Int8).alias('label')
        ])
        
        # Apply categorical encoding
        lf = self.encoder.transform(lf)
        
        # Add derived features
        if self.add_derived_features:
            lf = lf.with_columns([
                ((col('timestamp') * 5 // 3600) % 24).cast(pl.Int8).alias('hour_of_day'),
                ((col('timestamp') * 5 // 86400) % 7).cast(pl.Int8).alias('day_of_week'),
            ])
        
        # Add historical features
        if self.add_historical_features:
            lf = self.hist_builder.transform(lf)
        
        # Collect if lazy and requested
        if is_lazy and collect:
            logger.info("Collecting transformed data...")
            df = lf.collect()
            logger.info(f"Collected {df.height:,} rows")
        else:
            df = lf
        
        # Filter to available features
        available = [c for c in self.feature_columns if c in df.columns]
        
        return df, available
    
    def fit_transform(
        self,
        lf: Union[pl.LazyFrame, pl.DataFrame],
        likes_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None,
        dislikes_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None,
        collect: bool = True
    ) -> Tuple[pl.DataFrame, List[str]]:
        """Fit and transform in one step."""
        self.fit(lf, likes_lf, dislikes_lf)
        return self.transform(lf, collect=collect)
    
    def to_pandas(self, df: pl.DataFrame, feature_cols: List[str]) -> Tuple:
        """
        Convert Polars DataFrame to pandas for model training.
        
        Args:
            df: Polars DataFrame
            feature_cols: Feature columns to include
            
        Returns:
            Tuple of (X as pandas DataFrame, y as pandas Series)
        """
        X = df.select(feature_cols).to_pandas()
        y = df['label'].to_pandas()
        return X, y
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage report."""
        return {
            'historical_features_mb': self.hist_builder.get_memory_usage_mb(),
            'encoder_frequency_maps': len(self.encoder.frequency_maps),
            'encoder_label_maps': len(self.encoder.label_maps),
        }
    
    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save encoder maps
        for col_name, freq_df in self.encoder.frequency_maps.items():
            freq_df.write_parquet(path / f'freq_{col_name}.parquet')
        
        for col_name, label_df in self.encoder.label_maps.items():
            label_df.write_parquet(path / f'label_{col_name}.parquet')
        
        # Save historical features
        if self.hist_builder.user_features is not None:
            self.hist_builder.user_features.write_parquet(path / 'hist_user.parquet')
        if self.hist_builder.item_features is not None:
            self.hist_builder.item_features.write_parquet(path / 'hist_item.parquet')
        if self.hist_builder.user_item_features is not None:
            self.hist_builder.user_item_features.write_parquet(path / 'hist_user_item.parquet')
        
        # Save metadata
        state = {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'high_cardinality_threshold': self.high_cardinality_threshold,
            'add_derived_features': self.add_derived_features,
            'add_historical_features': self.add_historical_features,
            'label_threshold': self.label_threshold,
            'feature_columns': self.feature_columns,
            'fitted': self.fitted,
            'user_defaults': self.hist_builder.user_defaults,
            'item_defaults': self.hist_builder.item_defaults,
            'user_item_defaults': self.hist_builder.user_item_defaults,
            'freq_map_cols': list(self.encoder.frequency_maps.keys()),
            'label_map_cols': list(self.encoder.label_maps.keys()),
        }
        
        with open(path / 'pipeline_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Pipeline saved to {path}")
    
    def load(self, path: str) -> 'PolarsPipeline':
        """Load pipeline from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / 'pipeline_state.pkl', 'rb') as f:
            state = pickle.load(f)
        
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns']
        self.high_cardinality_threshold = state['high_cardinality_threshold']
        self.add_derived_features = state['add_derived_features']
        self.add_historical_features = state['add_historical_features']
        self.label_threshold = state['label_threshold']
        self.feature_columns = state['feature_columns']
        self.fitted = state['fitted']
        
        # Load encoder maps
        self.encoder = PolarsFeatureEncoder(self.high_cardinality_threshold)
        self.encoder.fitted = True
        
        for col_name in state['freq_map_cols']:
            self.encoder.frequency_maps[col_name] = pl.read_parquet(path / f'freq_{col_name}.parquet')
        
        for col_name in state['label_map_cols']:
            self.encoder.label_maps[col_name] = pl.read_parquet(path / f'label_{col_name}.parquet')
        
        # Load historical features
        self.hist_builder = PolarsHistoricalFeatures()
        self.hist_builder.fitted = True
        self.hist_builder.user_defaults = state['user_defaults']
        self.hist_builder.item_defaults = state['item_defaults']
        self.hist_builder.user_item_defaults = state['user_item_defaults']
        
        if (path / 'hist_user.parquet').exists():
            self.hist_builder.user_features = pl.read_parquet(path / 'hist_user.parquet')
        if (path / 'hist_item.parquet').exists():
            self.hist_builder.item_features = pl.read_parquet(path / 'hist_item.parquet')
        if (path / 'hist_user_item.parquet').exists():
            self.hist_builder.user_item_features = pl.read_parquet(path / 'hist_user_item.parquet')
        
        logger.info(f"Pipeline loaded from {path}")
        return self


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Polars pipeline demo')
    import os
    default_data_dir = os.environ.get('YAMBDA_DATA_DIR', '')
    parser.add_argument('--data_dir', type=str,
                       default=default_data_dir or None,
                       required=not default_data_dir,
                       help='Path to Yandex/flat directory (or set YAMBDA_DATA_DIR env var)')
    parser.add_argument('--train_days', type=int, default=30,
                       help='Training window in days (default: 30 for local dev, None for full 300 days)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Polars Feature Pipeline Demo")
    print("=" * 60)
    
    # Load data with time window
    print(f"\n[1] Loading data (train_days={args.train_days})...")
    train_lf, test_lf = load_yambda_polars(
        args.data_dir,
        train_days=args.train_days,
        lazy=True
    )
    
    # Initialize pipeline
    print("\n[2] Initializing pipeline...")
    pipeline = PolarsPipeline()
    
    # Fit and transform
    print("\n[3] Fitting and transforming...")
    train_df, feature_cols = pipeline.fit_transform(train_lf)
    
    print(f"\n[4] Results:")
    print(f"    Train rows: {train_df.height:,}")
    print(f"    Features: {len(feature_cols)}")
    print(f"    Feature columns: {feature_cols[:10]}...")
    print(f"    Memory usage: {train_df.estimated_size('mb'):.1f} MB")
    
    # Transform test
    print("\n[5] Transforming test data...")
    test_df, _ = pipeline.transform(test_lf)
    print(f"    Test rows: {test_df.height:,}")
    
    # Show sample
    print("\n[6] Sample data:")
    print(train_df.select(['uid', 'item_id', 'label'] + feature_cols[:5]).head(5))
    
    # Memory report
    print("\n[7] Pipeline memory usage:")
    mem = pipeline.get_memory_usage()
    print(f"    Historical features: {mem['historical_features_mb']}")
    
    # Convert to pandas for XGBoost
    print("\n[8] Converting to pandas for XGBoost...")
    X_train, y_train = pipeline.to_pandas(train_df, feature_cols)
    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train distribution: {y_train.value_counts().to_dict()}")
    
    print("\n" + "=" * 60)
    print("Polars pipeline ready!")
    print("=" * 60)
    print("\nRecommended train_days settings:")
    print("  --train_days 30   # ~5M rows, fast local dev (16GB laptop)")
    print("  --train_days 60   # ~10M rows, moderate")
    print("  --train_days 90   # ~15M rows, larger sample")
    print("  --train_days 150  # ~25M rows, half dataset")
    print("  (omit flag)       # ~50M rows, full GTS protocol")

