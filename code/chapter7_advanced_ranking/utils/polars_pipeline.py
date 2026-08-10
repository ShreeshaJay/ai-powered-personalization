"""
Chapter 7: Polars-Based Feature Pipeline
=========================================
Memory-efficient feature processing using Polars for large datasets.

This is a copy of Chapter 6's polars_pipeline.py to maintain independence.
Both chapters use the same feature engineering approach for fair comparison.

See Chapter 6 documentation for detailed explanation of:
- Point-in-Time (PIT) feature correctness
- Production Feature Store context
- Memory optimization strategies
"""

import polars as pl
from polars import col
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

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
    gap_seconds: int = 1800,
    lazy: bool = True
) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Union[pl.LazyFrame, pl.DataFrame]]:
    """Load Yambda dataset with Global Temporal Split using Polars."""
    data_path = Path(data_dir)
    listens_path = data_path / 'listens.parquet'
    
    if not listens_path.exists():
        raise FileNotFoundError(f"Listens file not found: {listens_path}")
    
    logger.info(f"Loading data from {listens_path}")
    
    lf = pl.scan_parquet(listens_path)
    
    ts_stats = lf.select([
        col('timestamp').min().alias('min_ts'),
        col('timestamp').max().alias('max_ts'),
    ]).collect()
    
    min_ts = ts_stats['min_ts'][0]
    max_ts = ts_stats['max_ts'][0]
    
    SECONDS_PER_DAY = 86400
    TS_UNITS_PER_DAY = SECONDS_PER_DAY // 5
    
    gap_units = gap_seconds // 5
    test_duration_units = test_days * TS_UNITS_PER_DAY
    
    test_end = max_ts
    test_start = test_end - test_duration_units
    train_end = test_start - gap_units
    
    if train_days is not None:
        train_duration_units = train_days * TS_UNITS_PER_DAY
        train_start = train_end - train_duration_units
        train_start = max(train_start, min_ts)
        logger.info(f"Using {train_days}-day train window")
    else:
        train_start = min_ts
        actual_train_days = (train_end - train_start) / TS_UNITS_PER_DAY
        logger.info(f"Using full train window (~{actual_train_days:.0f} days)")
    
    logger.info(f"Timestamp range: {min_ts:,} to {max_ts:,}")
    logger.info(f"Train window: {train_start:,} to {train_end:,} "
                f"(~{(train_end - train_start) / TS_UNITS_PER_DAY:.0f} days)")
    logger.info(f"Test window: {test_start:,} to {test_end:,} (~{test_days} days)")
    
    train_lf = lf.filter(
        (col('timestamp') >= train_start) & (col('timestamp') < train_end)
    )
    test_lf = lf.filter(
        (col('timestamp') >= test_start) & (col('timestamp') <= test_end)
    )
    
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
    existing_cols = lf.collect_schema().names()
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
    """Polars-based feature encoder for categorical variables."""
    
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
        
        for col_name, freq_df in self.frequency_maps.items():
            if is_lazy:
                freq_lf = freq_df.lazy()
                lf = lf.join(freq_lf, on=col_name, how='left')
            else:
                lf = lf.join(freq_df, on=col_name, how='left')
            lf = lf.with_columns([col(f'{col_name}_freq').fill_null(0.0)])
        
        for col_name, label_df in self.label_maps.items():
            if is_lazy:
                label_lf = label_df.lazy()
                lf = lf.join(label_lf, on=col_name, how='left')
            else:
                lf = lf.join(label_df, on=col_name, how='left')
            lf = lf.with_columns([col(f'{col_name}_encoded').fill_null(-1)])
        
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
# Historical Features
# =============================================================================

class PolarsHistoricalFeatures:
    """Compute historical aggregate features using Polars."""
    
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
        
        if isinstance(lf, pl.LazyFrame):
            df = lf.collect()
        else:
            df = lf
        
        # User Features
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
        
        # Item Features
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
        
        # User-Item pairs
        logger.info("  Computing user-item features...")
        self.user_item_pairs = df.select(['uid', 'item_id']).unique()
        logger.info(f"    Found {self.user_item_pairs.height:,} unique user-item pairs")
        logger.info("    NOTE: previous_listen_count will be computed with PIT correctness in transform()")
        
        # Defaults
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
        
        self.user_item_defaults = {
            'has_listened_before': 0,
            'previous_listen_count': 0,
        }
        
        self.user_item_features = self.user_item_pairs
        self.fitted = True
        logger.info("  Historical features computed")
        
        return self
    
    def transform(
        self, 
        lf: Union[pl.LazyFrame, pl.DataFrame]
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """Add historical features via joins and PIT-correct computations."""
        if not self.fitted:
            raise ValueError("Not fitted")
        
        is_lazy = isinstance(lf, pl.LazyFrame)
        
        user_lf = self.user_features.lazy() if is_lazy else self.user_features
        item_lf = self.item_features.lazy() if is_lazy else self.item_features
        
        lf = lf.join(user_lf, on='uid', how='left')
        
        user_fill_exprs = [
            col(c).fill_null(v).alias(c) for c, v in self.user_defaults.items()
            if c in (lf.collect_schema().names() if is_lazy else lf.columns)
        ]
        if user_fill_exprs:
            lf = lf.with_columns(user_fill_exprs)
        
        lf = lf.join(item_lf, on='item_id', how='left')
        
        item_fill_exprs = [
            col(c).fill_null(v).alias(c) for c, v in self.item_defaults.items()
            if c in (lf.collect_schema().names() if is_lazy else lf.columns)
        ]
        if item_fill_exprs:
            lf = lf.with_columns(item_fill_exprs)
        
        # PIT-correct user-item features
        lf = lf.with_columns([
            (
                pl.count()
                .over(['uid', 'item_id'], order_by='timestamp')
                - 1
            ).cast(pl.Int16).alias('previous_listen_count'),
        ])
        
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
    """Complete feature pipeline using Polars."""
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        high_cardinality_threshold: int = 10000,
        add_derived_features: bool = True,
        add_historical_features: bool = True,
        label_threshold: int = 50,
    ):
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
        
        self.encoder.fit(lf, self.categorical_columns)
        
        if self.add_historical_features:
            self.hist_builder.fit(lf, likes_lf, dislikes_lf)
        
        self._build_feature_columns()
        
        self.fitted = True
        logger.info(f"Pipeline fitted with {len(self.feature_columns)} features")
        
        return self
    
    def _build_feature_columns(self):
        """Build list of feature columns."""
        self.feature_columns = []
        self.feature_columns.extend(self.encoder.get_encoded_column_names())
        self.feature_columns.extend(self.numerical_columns)
        if self.add_derived_features:
            self.feature_columns.extend(['hour_of_day', 'day_of_week'])
        if self.add_historical_features:
            self.feature_columns.extend(self.hist_builder.get_feature_names())
    
    def transform(
        self,
        lf: Union[pl.LazyFrame, pl.DataFrame],
        collect: bool = True
    ) -> Tuple[pl.DataFrame, List[str]]:
        """Transform data using fitted pipeline."""
        if not self.fitted:
            raise ValueError("Pipeline not fitted")
        
        is_lazy = isinstance(lf, pl.LazyFrame)
        
        lf = lf.with_columns([
            (col('played_ratio_pct') >= self.label_threshold).cast(pl.Int8).alias('label')
        ])
        
        lf = self.encoder.transform(lf)
        
        if self.add_derived_features:
            lf = lf.with_columns([
                ((col('timestamp') * 5 // 3600) % 24).cast(pl.Int8).alias('hour_of_day'),
                ((col('timestamp') * 5 // 86400) % 7).cast(pl.Int8).alias('day_of_week'),
            ])
        
        if self.add_historical_features:
            lf = self.hist_builder.transform(lf)
        
        if is_lazy and collect:
            logger.info("Collecting transformed data...")
            df = lf.collect()
            logger.info(f"Collected {df.height:,} rows")
        else:
            df = lf
        
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
        """Convert Polars DataFrame to pandas for model training."""
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
        
        for col_name, freq_df in self.encoder.frequency_maps.items():
            freq_df.write_parquet(path / f'freq_{col_name}.parquet')
        
        for col_name, label_df in self.encoder.label_maps.items():
            label_df.write_parquet(path / f'label_{col_name}.parquet')
        
        if self.hist_builder.user_features is not None:
            self.hist_builder.user_features.write_parquet(path / 'hist_user.parquet')
        if self.hist_builder.item_features is not None:
            self.hist_builder.item_features.write_parquet(path / 'hist_item.parquet')
        if self.hist_builder.user_item_features is not None:
            self.hist_builder.user_item_features.write_parquet(path / 'hist_user_item.parquet')
        
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
        
        self.encoder = PolarsFeatureEncoder(self.high_cardinality_threshold)
        self.encoder.fitted = True
        
        for col_name in state['freq_map_cols']:
            self.encoder.frequency_maps[col_name] = pl.read_parquet(path / f'freq_{col_name}.parquet')
        
        for col_name in state['label_map_cols']:
            self.encoder.label_maps[col_name] = pl.read_parquet(path / f'label_{col_name}.parquet')
        
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

