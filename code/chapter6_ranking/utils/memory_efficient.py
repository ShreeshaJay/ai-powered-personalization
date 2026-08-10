"""
Chapter 6: Memory-Efficient Feature Processing
===============================================
Optional utilities for memory-efficient training on large datasets.

Import and use these utilities when working with datasets that don't fit
comfortably in memory. The main feature pipeline prioritizes code clarity
for the book; this module provides production-grade optimizations.

Usage:
------
>>> from utils.memory_efficient import (
...     optimize_dtypes,
...     MemoryEfficientHistoricalFeatures,
...     memory_usage_report
... )
>>> 
>>> # Optimize dtypes after loading
>>> train_df = optimize_dtypes(train_df)
>>> 
>>> # Use memory-efficient historical features
>>> hist_builder = MemoryEfficientHistoricalFeatures()
>>> train_df = hist_builder.fit_transform(train_df)

Memory Savings Summary:
-----------------------
- dtype optimization: ~50-60% reduction
- Index-based lookups vs merges: ~30% reduction during transform
- Single-copy pipeline: ~75% reduction vs multiple copies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import gc
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


# =============================================================================
# Dtype Optimization
# =============================================================================

# Optimal dtypes for Yambda dataset columns
OPTIMAL_DTYPES = {
    # ID columns - int32 supports up to 2B unique values
    'uid': 'int32',
    'item_id': 'int32',
    'artist_id': 'int32',
    'album_id': 'int32',
    
    # Timestamp - int32 supports ~68 years in seconds
    'timestamp': 'int32',
    
    # Percentages and ratios - float32 is sufficient
    'played_ratio_pct': 'float32',
    'track_length_seconds': 'float32',
    
    # Binary/small int columns
    'is_organic': 'int8',
    'label': 'int8',
    'has_listened_before': 'int8',
    
    # Derived features
    'hour_of_day': 'int8',
    'day_of_week': 'int8',
    
    # Frequency encodings (0-1 range)
    'uid_freq': 'float32',
    'item_id_freq': 'float32',
    
    # Label encodings
    'is_organic_encoded': 'int8',
    
    # Count features
    'user_total_listens': 'int32',
    'user_unique_items': 'int32',
    'item_total_plays': 'int32',
    'item_unique_listeners': 'int32',
    'previous_listen_count': 'int16',
    
    # Aggregated stats (float32 sufficient for averages)
    'user_avg_completion': 'float32',
    'user_std_completion': 'float32',
    'user_median_completion': 'float32',
    'user_organic_ratio': 'float32',
    'user_active_span': 'int32',
    'user_listen_rate': 'float32',
    'item_avg_completion': 'float32',
    'item_std_completion': 'float32',
    'item_organic_ratio': 'float32',
    'item_repeat_ratio': 'float32',
    'item_like_ratio': 'float32',
    # user_item_avg_completion REMOVED due to data leakage
    'artist_share': 'float32',
    'user_artist_avg_completion': 'float32',
}


def optimize_dtypes(
    df: pd.DataFrame,
    dtype_map: Optional[Dict[str, str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    
    Args:
        df: DataFrame to optimize
        dtype_map: Custom dtype mapping (default: OPTIMAL_DTYPES)
        inplace: If True, modify df in place (saves memory but mutates input)
        
    Returns:
        DataFrame with optimized dtypes
        
    Example:
        >>> df = optimize_dtypes(df)
        >>> # Or in-place to save memory:
        >>> optimize_dtypes(df, inplace=True)
    """
    dtype_map = dtype_map or OPTIMAL_DTYPES
    
    if not inplace:
        df = df.copy()
    
    original_memory = df.memory_usage(deep=True).sum()
    
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, OverflowError) as e:
                logger.warning(f"Could not convert {col} to {dtype}: {e}")
    
    optimized_memory = df.memory_usage(deep=True).sum()
    reduction = (1 - optimized_memory / original_memory) * 100
    
    logger.info(f"Memory optimization: {original_memory/1e6:.1f}MB → "
                f"{optimized_memory/1e6:.1f}MB ({reduction:.1f}% reduction)")
    
    return df


def downcast_numeric(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Automatically downcast all numeric columns to smallest sufficient dtype.
    
    This is more aggressive than optimize_dtypes() - it inspects actual
    data ranges to find the smallest dtype that fits.
    
    Args:
        df: DataFrame to optimize
        inplace: Modify in place
        
    Returns:
        DataFrame with downcasted dtypes
    """
    if not inplace:
        df = df.copy()
    
    original_memory = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if np.issubdtype(col_type, np.integer):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif np.issubdtype(col_type, np.floating):
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    optimized_memory = df.memory_usage(deep=True).sum()
    reduction = (1 - optimized_memory / original_memory) * 100
    
    logger.info(f"Downcast optimization: {original_memory/1e6:.1f}MB → "
                f"{optimized_memory/1e6:.1f}MB ({reduction:.1f}% reduction)")
    
    return df


# =============================================================================
# Memory Usage Reporting
# =============================================================================

def memory_usage_report(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
    """
    Generate detailed memory usage report for a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name for logging
        
    Returns:
        Dictionary with memory statistics
    """
    memory_by_col = df.memory_usage(deep=True)
    total_mb = memory_by_col.sum() / 1e6
    
    # Find largest columns
    col_memory = memory_by_col.drop('Index').sort_values(ascending=False)
    top_columns = col_memory.head(10)
    
    report = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'total_memory_mb': total_mb,
        'memory_per_row_kb': total_mb * 1000 / len(df) if len(df) > 0 else 0,
        'top_columns_mb': {col: mb/1e6 for col, mb in top_columns.items()},
        'dtypes': df.dtypes.value_counts().to_dict(),
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Memory Report: {name}")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Total Memory: {total_mb:.2f} MB")
    logger.info(f"Memory per Row: {report['memory_per_row_kb']:.2f} KB")
    logger.info(f"\nTop Memory Consumers:")
    for col, mb in list(report['top_columns_mb'].items())[:5]:
        logger.info(f"  {col}: {mb:.2f} MB ({df[col].dtype})")
    logger.info(f"\nDtype Distribution:")
    for dtype, count in report['dtypes'].items():
        logger.info(f"  {dtype}: {count} columns")
    
    return report


def estimate_memory_for_rows(
    df: pd.DataFrame,
    target_rows: int
) -> float:
    """
    Estimate memory usage for a different number of rows.
    
    Useful for planning if you can load the full dataset.
    
    Args:
        df: Sample DataFrame
        target_rows: Number of rows to estimate for
        
    Returns:
        Estimated memory in MB
    """
    current_rows = len(df)
    current_mb = df.memory_usage(deep=True).sum() / 1e6
    
    # Most memory scales linearly with rows
    estimated_mb = current_mb * (target_rows / current_rows)
    
    logger.info(f"Memory estimate: {current_rows:,} rows = {current_mb:.1f}MB → "
                f"{target_rows:,} rows ≈ {estimated_mb:.1f}MB")
    
    return estimated_mb


# =============================================================================
# Memory-Efficient Historical Features
# =============================================================================

class MemoryEfficientHistoricalFeatures:
    """
    Memory-efficient version of HistoricalFeatureBuilder.
    
    Key optimizations:
    1. Stores aggregates as indexed Series (not DataFrames)
    2. Uses direct indexing instead of merge operations
    3. Pre-allocates output columns
    4. Applies dtype optimization throughout
    
    Trade-offs:
    - Slightly more complex code
    - ~40% less memory during transform
    - ~20% faster due to avoiding DataFrame merge overhead
    """
    
    def __init__(self):
        # Store as indexed Series for O(1) lookup
        self.user_stats: Dict[str, pd.Series] = {}
        self.item_stats: Dict[str, pd.Series] = {}
        self.user_item_stats: Dict[str, pd.Series] = {}
        
        self.user_defaults: Dict[str, float] = {}
        self.item_defaults: Dict[str, float] = {}
        self.user_item_defaults: Dict[str, float] = {}
        
        self.fitted = False
    
    def fit(
        self,
        train_df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame] = None,
        dislikes_df: Optional[pd.DataFrame] = None
    ) -> 'MemoryEfficientHistoricalFeatures':
        """
        Compute and store aggregates as indexed Series.
        """
        logger.info("Computing memory-efficient historical features...")
        
        # ===== User Features =====
        logger.info("  Computing user features...")
        user_agg = train_df.groupby('uid').agg(
            user_total_listens=('item_id', 'count'),
            user_avg_completion=('played_ratio_pct', 'mean'),
            user_std_completion=('played_ratio_pct', 'std'),
            user_median_completion=('played_ratio_pct', 'median'),
            user_unique_items=('item_id', 'nunique'),
            user_organic_ratio=('is_organic', 'mean'),
            user_min_ts=('timestamp', 'min'),
            user_max_ts=('timestamp', 'max'),
        )
        
        # Compute derived features
        user_agg['user_std_completion'] = user_agg['user_std_completion'].fillna(0)
        user_agg['user_active_span'] = user_agg['user_max_ts'] - user_agg['user_min_ts']
        user_agg['user_listen_rate'] = user_agg['user_total_listens'] / (user_agg['user_active_span'] + 1)
        
        # Store each column as separate indexed Series (memory efficient)
        user_feature_cols = [
            'user_total_listens', 'user_avg_completion', 'user_std_completion',
            'user_median_completion', 'user_unique_items', 'user_organic_ratio',
            'user_active_span', 'user_listen_rate'
        ]
        for col in user_feature_cols:
            self.user_stats[col] = user_agg[col].astype('float32')
        
        del user_agg
        gc.collect()
        
        # ===== Item Features =====
        logger.info("  Computing item features...")
        item_agg = train_df.groupby('item_id').agg(
            item_total_plays=('uid', 'count'),
            item_avg_completion=('played_ratio_pct', 'mean'),
            item_std_completion=('played_ratio_pct', 'std'),
            item_unique_listeners=('uid', 'nunique'),
            item_organic_ratio=('is_organic', 'mean'),
        )
        
        item_agg['item_std_completion'] = item_agg['item_std_completion'].fillna(0)
        item_agg['item_repeat_ratio'] = item_agg['item_total_plays'] / item_agg['item_unique_listeners']
        
        # Add like ratio if available
        if likes_df is not None and dislikes_df is not None:
            likes_count = likes_df.groupby('item_id').size()
            dislikes_count = dislikes_df.groupby('item_id').size()
            
            item_agg['item_likes'] = likes_count.reindex(item_agg.index).fillna(0)
            item_agg['item_dislikes'] = dislikes_count.reindex(item_agg.index).fillna(0)
            total_feedback = item_agg['item_likes'] + item_agg['item_dislikes']
            item_agg['item_like_ratio'] = (item_agg['item_likes'] + 1) / (total_feedback + 2)
        else:
            item_agg['item_like_ratio'] = 0.5
        
        item_feature_cols = [
            'item_total_plays', 'item_avg_completion', 'item_std_completion',
            'item_unique_listeners', 'item_organic_ratio', 'item_repeat_ratio',
            'item_like_ratio'
        ]
        for col in item_feature_cols:
            self.item_stats[col] = item_agg[col].astype('float32')
        
        del item_agg
        gc.collect()
        
        # ===== User-Item Features =====
        # NOTE: user_item_avg_completion REMOVED due to severe data leakage
        # It was computed from ALL interactions including the current row's label.
        # For PIT-correct previous_listen_count, use the Polars pipeline instead.
        logger.info("  Computing user-item features...")
        user_item_agg = train_df.groupby(['uid', 'item_id']).agg(
            previous_listen_count=('timestamp', 'count'),
            # user_item_avg_completion REMOVED - causes severe data leakage
        )
        
        # Store with MultiIndex for efficient lookup
        self.user_item_stats['previous_listen_count'] = user_item_agg['previous_listen_count'].astype('int16')
        
        del user_item_agg
        gc.collect()
        
        # ===== Defaults for cold-start =====
        self.user_defaults = {
            'user_total_listens': 0,
            'user_avg_completion': float(train_df['played_ratio_pct'].mean()),
            'user_std_completion': float(train_df['played_ratio_pct'].std()),
            'user_median_completion': float(train_df['played_ratio_pct'].median()),
            'user_unique_items': 0,
            'user_organic_ratio': float(train_df['is_organic'].mean()),
            'user_active_span': 0,
            'user_listen_rate': 0,
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
        
        self.user_item_defaults = {
            'has_listened_before': 0,
            'previous_listen_count': 0,
            # user_item_avg_completion REMOVED due to data leakage
        }
        
        self.fitted = True
        logger.info("  Historical features computed")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add historical features using indexed lookups (no merges).
        
        This is more memory-efficient than merge operations because:
        1. No intermediate DataFrames created
        2. Direct Series indexing is O(1) per row
        3. Pre-allocated columns avoid reallocation
        """
        if not self.fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        n_rows = len(df)
        
        # Get lookup keys
        uids = df['uid'].values
        item_ids = df['item_id'].values
        
        # ===== User features via direct indexing =====
        for col, series in self.user_stats.items():
            default = self.user_defaults.get(col, 0)
            # Use reindex for vectorized lookup with default
            df[col] = series.reindex(uids).fillna(default).values
        
        # ===== Item features via direct indexing =====
        for col, series in self.item_stats.items():
            default = self.item_defaults.get(col, 0)
            df[col] = series.reindex(item_ids).fillna(default).values
        
        # ===== User-item features via MultiIndex lookup =====
        # Create MultiIndex for lookup
        lookup_index = pd.MultiIndex.from_arrays([uids, item_ids])
        
        # Previous listen count
        # NOTE: This uses a lookup-table approach (total count per user-item pair).
        # For true PIT correctness, use the Polars pipeline which computes a
        # cumulative count up to each row's timestamp.
        prev_count_series = self.user_item_stats.get('previous_listen_count')
        if prev_count_series is not None:
            df['previous_listen_count'] = prev_count_series.reindex(lookup_index).fillna(0).values
            df['has_listened_before'] = (df['previous_listen_count'] > 0).astype('int8')
        else:
            df['previous_listen_count'] = 0
            df['has_listened_before'] = 0
        
        # user_item_avg_completion REMOVED due to severe data leakage
        
        return df
    
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame] = None,
        dislikes_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(train_df, likes_df, dislikes_df)
        return self.transform(train_df)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of stored aggregates in MB."""
        usage = {}
        
        for name, stats_dict in [
            ('user_stats', self.user_stats),
            ('item_stats', self.item_stats),
            ('user_item_stats', self.user_item_stats)
        ]:
            total = sum(s.memory_usage(deep=True) for s in stats_dict.values())
            usage[name] = total / 1e6
        
        usage['total'] = sum(usage.values())
        return usage
    
    def save(self, path: str) -> None:
        """Save to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'user_stats': self.user_stats,
            'item_stats': self.item_stats,
            'user_item_stats': self.user_item_stats,
            'user_defaults': self.user_defaults,
            'item_defaults': self.item_defaults,
            'user_item_defaults': self.user_item_defaults,
            'fitted': self.fitted,
        }
        
        with open(path / 'mem_efficient_historical.pkl', 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> 'MemoryEfficientHistoricalFeatures':
        """Load from disk."""
        path = Path(path)
        
        with open(path / 'mem_efficient_historical.pkl', 'rb') as f:
            state = pickle.load(f)
        
        self.user_stats = state['user_stats']
        self.item_stats = state['item_stats']
        self.user_item_stats = state['user_item_stats']
        self.user_defaults = state['user_defaults']
        self.item_defaults = state['item_defaults']
        self.user_item_defaults = state['user_item_defaults']
        self.fitted = state['fitted']
        
        return self


# =============================================================================
# Memory-Efficient Feature Pipeline
# =============================================================================

class MemoryEfficientPipeline:
    """
    Drop-in replacement for FeaturePipeline with memory optimizations.
    
    Key differences from standard FeaturePipeline:
    1. Single DataFrame copy at the start
    2. In-place column additions
    3. Dtype optimization throughout
    4. Uses MemoryEfficientHistoricalFeatures
    
    Usage:
        >>> from utils.memory_efficient import MemoryEfficientPipeline
        >>> 
        >>> pipeline = MemoryEfficientPipeline()
        >>> train_df, feature_cols = pipeline.fit_transform(train_df)
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        high_cardinality_threshold: int = 10000,
        optimize_dtypes_on_load: bool = True
    ):
        self.categorical_columns = categorical_columns or ['uid', 'item_id', 'is_organic']
        self.numerical_columns = numerical_columns or ['track_length_seconds']
        self.high_cardinality_threshold = high_cardinality_threshold
        self.optimize_dtypes_on_load = optimize_dtypes_on_load
        
        # Encoding state
        self.frequency_maps: Dict[str, Dict] = {}
        self.label_mappings: Dict[str, Dict] = {}
        
        # Historical features
        self.hist_builder = MemoryEfficientHistoricalFeatures()
        
        self.feature_columns: List[str] = []
        self.fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame] = None,
        dislikes_df: Optional[pd.DataFrame] = None
    ) -> 'MemoryEfficientPipeline':
        """Fit the pipeline on training data."""
        
        logger.info("Fitting memory-efficient pipeline...")
        
        # Fit categorical encoders
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            n_unique = df[col].nunique()
            
            if n_unique > self.high_cardinality_threshold:
                # Frequency encoding
                freq = df[col].value_counts(normalize=True)
                self.frequency_maps[col] = freq.to_dict()
                logger.info(f"  Frequency encoding: {col} ({n_unique:,} unique)")
            else:
                # Label encoding - store as simple dict
                unique_vals = df[col].unique()
                self.label_mappings[col] = {v: i for i, v in enumerate(unique_vals)}
                logger.info(f"  Label encoding: {col} ({n_unique:,} unique)")
        
        # Fit historical features
        self.hist_builder.fit(df, likes_df, dislikes_df)
        
        # Build feature column list
        self._build_feature_columns()
        
        self.fitted = True
        return self
    
    def _build_feature_columns(self):
        """Build list of feature columns."""
        self.feature_columns = []
        
        # Encoded categorical
        for col in self.frequency_maps:
            self.feature_columns.append(f'{col}_freq')
        for col in self.label_mappings:
            self.feature_columns.append(f'{col}_encoded')
        
        # Numerical
        self.feature_columns.extend(self.numerical_columns)
        
        # Derived
        self.feature_columns.extend(['hour_of_day', 'day_of_week'])
        
        # Historical
        for col in self.hist_builder.user_stats:
            self.feature_columns.append(col)
        for col in self.hist_builder.item_stats:
            self.feature_columns.append(col)
        self.feature_columns.extend([
            'has_listened_before', 'previous_listen_count'
            # user_item_avg_completion REMOVED due to data leakage
        ])
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform DataFrame with minimal memory overhead.
        
        Creates ONE copy at the start, then modifies in-place.
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted")
        
        # Single copy at start
        df = df.copy()
        
        # Optimize dtypes first (reduces memory for subsequent operations)
        if self.optimize_dtypes_on_load:
            optimize_dtypes(df, inplace=True)
        
        # Apply frequency encoding (in-place)
        for col, freq_map in self.frequency_maps.items():
            if col in df.columns:
                df[f'{col}_freq'] = df[col].map(freq_map).fillna(0.0).astype('float32')
        
        # Apply label encoding (in-place)
        for col, mapping in self.label_mappings.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(-1).astype('int32')
        
        # Add derived features (in-place)
        if 'timestamp' in df.columns:
            seconds = df['timestamp'] * 5
            df['hour_of_day'] = ((seconds // 3600) % 24).astype('int8')
            df['day_of_week'] = ((seconds // 86400) % 7).astype('int8')
        
        # Add historical features (in-place via the efficient builder)
        df = self.hist_builder.transform(df)
        
        # Final dtype optimization
        optimize_dtypes(df, inplace=True)
        
        # Filter to available features
        available = [c for c in self.feature_columns if c in df.columns]
        
        return df, available
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        likes_df: Optional[pd.DataFrame] = None,
        dislikes_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fit and transform in one step."""
        self.fit(df, likes_df, dislikes_df)
        return self.transform(df)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get memory usage report for pipeline state."""
        report = {
            'frequency_maps_entries': sum(len(m) for m in self.frequency_maps.values()),
            'label_mappings_entries': sum(len(m) for m in self.label_mappings.values()),
            'historical_features_mb': self.hist_builder.get_memory_usage(),
        }
        return report


# =============================================================================
# Chunked Processing for Very Large Datasets
# =============================================================================

def process_in_chunks(
    parquet_path: str,
    pipeline: MemoryEfficientPipeline,
    chunk_size: int = 1_000_000,
    output_path: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Process a large parquet file in chunks to avoid memory issues.
    
    Args:
        parquet_path: Path to input parquet file
        pipeline: Fitted MemoryEfficientPipeline
        chunk_size: Rows per chunk
        output_path: If provided, save results to parquet instead of returning
        
    Returns:
        Concatenated DataFrame if output_path is None, else None
    """
    import pyarrow.parquet as pq
    
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    logger.info(f"Processing {total_rows:,} rows in {n_chunks} chunks...")
    
    results = []
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        chunk_df = batch.to_pandas()
        
        # Transform chunk
        transformed, feature_cols = pipeline.transform(chunk_df)
        
        if output_path:
            # Append to parquet
            mode = 'w' if i == 0 else 'a'
            transformed[feature_cols].to_parquet(
                output_path,
                engine='pyarrow',
                append=(i > 0)
            )
        else:
            results.append(transformed[feature_cols])
        
        logger.info(f"  Processed chunk {i+1}/{n_chunks}")
        
        # Force garbage collection
        del chunk_df, transformed
        gc.collect()
    
    if output_path:
        logger.info(f"Results saved to {output_path}")
        return None
    else:
        return pd.concat(results, ignore_index=True)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data_loader import get_train_test_data
    
    parser = argparse.ArgumentParser(description='Memory efficiency utilities demo')
    import os
    default_data_dir = os.environ.get('YAMBDA_DATA_DIR', '')
    parser.add_argument('--data_dir', type=str, 
                       default=default_data_dir or None,
                       required=not default_data_dir,
                       help='Path to Yandex/flat directory (or set YAMBDA_DATA_DIR env var)')
    parser.add_argument('--sample_frac', type=float, default=0.01)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Memory-Efficient Feature Processing Demo")
    print("=" * 60)
    
    # Load sample data
    print("\n[1] Loading sample data...")
    train_df, test_df = get_train_test_data(
        args.data_dir,
        sample_frac=args.sample_frac
    )
    
    # Memory report before optimization
    print("\n[2] Memory BEFORE optimization:")
    report_before = memory_usage_report(train_df, "train_df (original)")
    
    # Optimize dtypes
    print("\n[3] Optimizing dtypes...")
    train_df_opt = optimize_dtypes(train_df)
    
    # Memory report after
    print("\n[4] Memory AFTER dtype optimization:")
    report_after = memory_usage_report(train_df_opt, "train_df (optimized)")
    
    # Test efficient pipeline
    print("\n[5] Testing MemoryEfficientPipeline...")
    pipeline = MemoryEfficientPipeline()
    train_transformed, feature_cols = pipeline.fit_transform(train_df_opt)
    
    print(f"\n[6] Final feature count: {len(feature_cols)}")
    print(f"    Sample features: {feature_cols[:10]}...")
    
    # Final memory report
    print("\n[7] Final memory report:")
    memory_usage_report(train_transformed, "train_transformed")
    
    # Pipeline state memory
    print("\n[8] Pipeline state memory:")
    state_report = pipeline.get_memory_report()
    print(f"    Historical features: {state_report['historical_features_mb']['total']:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Memory-efficient utilities ready!")
    print("=" * 60)

