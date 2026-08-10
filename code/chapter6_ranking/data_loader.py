"""
Chapter 6: Ranking - Data Loading Module
=========================================
Memory-efficient data loading for the Yandex Yambda dataset with
Global Temporal Split (GTS) implementation.

Dataset: Yandex Yambda (50M version)
Paper: https://arxiv.org/abs/2505.22238
Source: https://huggingface.co/datasets/yandex/yambda

Global Temporal Split Protocol (from paper):
- Training: First 300 days
- Gap: 30 minutes (to prevent information leakage)
- Test: Next 1 day

Note: Timestamps in Yambda are delta values binned into 5-second intervals,
NOT Unix timestamps. Data is sorted by (uid, timestamp).
"""

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging

# Import GTSConfig from centralized config
from config import GTSConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YambdaDataLoader:
    """
    Memory-efficient data loader for Yandex Yambda dataset.
    
    Uses PyArrow for efficient parquet reading without loading entire
    dataset into memory.
    
    Example usage:
    -------------
    >>> loader = YambdaDataLoader(data_dir='path/to/yandex/flat')
    >>> train_df, test_df = loader.load_with_gts()
    >>> print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    """
    
    def __init__(
        self,
        data_dir: str,
        gts_config: Optional[GTSConfig] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the Yandex/flat directory containing parquet files
            gts_config: Global Temporal Split configuration (default: paper's protocol)
        """
        self.data_dir = Path(data_dir)
        self.gts_config = gts_config or GTSConfig()
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self) -> None:
        """Validate that required data files exist."""
        required_files = ['listens.parquet', 'likes.parquet', 'dislikes.parquet']
        
        for fname in required_files:
            fpath = self.data_dir / fname
            if not fpath.exists():
                logger.warning(f"File not found: {fpath}")
                
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get metadata about the dataset without loading it.
        
        Returns:
            Dictionary with file info, row counts, schemas
        """
        info = {}
        
        for fname in ['listens.parquet', 'likes.parquet', 'dislikes.parquet', 
                      'unlikes.parquet', 'undislikes.parquet', 'multi_event.parquet']:
            fpath = self.data_dir / fname
            if fpath.exists():
                pf = pq.ParquetFile(fpath)
                info[fname] = {
                    'num_rows': pf.metadata.num_rows,
                    'num_columns': pf.metadata.num_columns,
                    'schema': [f.name for f in pf.schema_arrow],
                    'size_mb': fpath.stat().st_size / (1024**2)
                }
                
        return info
    
    def _get_timestamp_range(self, parquet_path: Path) -> Tuple[int, int]:
        """
        Get min and max timestamps from a parquet file efficiently.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Tuple of (min_timestamp, max_timestamp)
        """
        table = pq.read_table(parquet_path, columns=['timestamp'])
        min_ts = pc.min(table.column('timestamp')).as_py()
        max_ts = pc.max(table.column('timestamp')).as_py()
        return min_ts, max_ts
    
    def compute_split_boundaries(
        self,
        parquet_path: Optional[Path] = None
    ) -> Dict[str, int]:
        """
        Compute the timestamp boundaries for train/test split.
        
        Uses actual data range if available, otherwise uses config defaults.
        
        Args:
            parquet_path: Optional path to compute boundaries from actual data
            
        Returns:
            Dictionary with train_end, test_start, test_end timestamps
        """
        if parquet_path and parquet_path.exists():
            min_ts, max_ts = self._get_timestamp_range(parquet_path)
            logger.info(f"Data timestamp range: {min_ts:,} to {max_ts:,}")
            
            # For 50M dataset, scale proportionally
            # The 50M is a subset, so we use relative splits
            total_range = max_ts - min_ts
            
            # Calculate proportions based on paper's protocol
            # 300 days train + 0.02 day gap + 1 day test ≈ 301 days total
            total_days = self.gts_config.train_days + self.gts_config.test_days
            train_ratio = self.gts_config.train_days / total_days
            
            train_end = min_ts + int(total_range * train_ratio)
            test_start = train_end + self.gts_config.gap_duration_ts
            test_end = max_ts
            
        else:
            # Use absolute values from config
            train_end = self.gts_config.train_duration_ts
            test_start = train_end + self.gts_config.gap_duration_ts
            test_end = test_start + self.gts_config.test_duration_ts
            
        boundaries = {
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        }
        
        logger.info(f"Split boundaries: train_end={train_end:,}, "
                   f"test_start={test_start:,}, test_end={test_end:,}")
        
        return boundaries
    
    def load_listens_with_gts(
        self,
        columns: Optional[List[str]] = None,
        sample_frac: Optional[float] = None
    ) -> Tuple[pa.Table, pa.Table]:
        """
        Load listens data with Global Temporal Split.
        
        Args:
            columns: Specific columns to load (None = all)
            sample_frac: Optional fraction to sample (for testing)
            
        Returns:
            Tuple of (train_table, test_table) as PyArrow Tables
        """
        listens_path = self.data_dir / 'listens.parquet'
        
        if not listens_path.exists():
            raise FileNotFoundError(f"Listens file not found: {listens_path}")
            
        # Compute split boundaries from actual data
        boundaries = self.compute_split_boundaries(listens_path)
        
        logger.info("Loading listens data with GTS...")
        
        # Read full table (PyArrow is memory-efficient)
        if columns:
            # Always include timestamp for splitting
            cols_to_read = list(set(columns + ['timestamp']))
            table = pq.read_table(listens_path, columns=cols_to_read)
        else:
            table = pq.read_table(listens_path)
            
        # Apply sampling if requested (for development/testing)
        if sample_frac and sample_frac < 1.0:
            n_samples = int(table.num_rows * sample_frac)
            indices = np.random.choice(table.num_rows, n_samples, replace=False)
            indices = np.sort(indices)  # Maintain temporal order
            table = table.take(indices)
            logger.info(f"Sampled {n_samples:,} rows ({sample_frac*100:.1f}%)")
        
        # Split by timestamp
        ts_col = table.column('timestamp')
        
        # Training: timestamp < train_end
        train_mask = pc.less(ts_col, boundaries['train_end'])
        train_table = table.filter(train_mask)
        
        # Test: test_start <= timestamp < test_end
        test_mask = pc.and_(
            pc.greater_equal(ts_col, boundaries['test_start']),
            pc.less(ts_col, boundaries['test_end'])
        )
        test_table = table.filter(test_mask)
        
        logger.info(f"Train samples: {train_table.num_rows:,}")
        logger.info(f"Test samples: {test_table.num_rows:,}")
        logger.info(f"Gap samples excluded: {table.num_rows - train_table.num_rows - test_table.num_rows:,}")
        
        return train_table, test_table
    
    def load_multi_event_with_gts(
        self,
        event_types: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ) -> Tuple[pa.Table, pa.Table]:
        """
        Load multi_event data with Global Temporal Split.
        
        This is useful for multi-task learning where you need
        listen, like, and dislike events together.
        
        Args:
            event_types: Filter to specific event types 
                        (e.g., ['listen', 'like', 'dislike'])
            columns: Specific columns to load
            
        Returns:
            Tuple of (train_table, test_table) as PyArrow Tables
        """
        multi_event_path = self.data_dir / 'multi_event.parquet'
        
        if not multi_event_path.exists():
            raise FileNotFoundError(f"Multi-event file not found: {multi_event_path}")
            
        # Compute split boundaries
        boundaries = self.compute_split_boundaries(multi_event_path)
        
        logger.info("Loading multi_event data with GTS...")
        
        # Read table
        if columns:
            cols_to_read = list(set(columns + ['timestamp', 'event_type']))
            table = pq.read_table(multi_event_path, columns=cols_to_read)
        else:
            table = pq.read_table(multi_event_path)
            
        # Filter by event types if specified
        if event_types:
            event_mask = pc.is_in(table.column('event_type'), pa.array(event_types))
            table = table.filter(event_mask)
            logger.info(f"Filtered to event types: {event_types}")
            
        # Split by timestamp
        ts_col = table.column('timestamp')
        
        train_mask = pc.less(ts_col, boundaries['train_end'])
        train_table = table.filter(train_mask)
        
        test_mask = pc.and_(
            pc.greater_equal(ts_col, boundaries['test_start']),
            pc.less(ts_col, boundaries['test_end'])
        )
        test_table = table.filter(test_mask)
        
        logger.info(f"Train samples: {train_table.num_rows:,}")
        logger.info(f"Test samples: {test_table.num_rows:,}")
        
        return train_table, test_table
    
    def create_listen_completion_labels(
        self,
        table: pa.Table,
        threshold: int = 50
    ) -> pa.Table:
        """
        Create binary labels for listen completion prediction.
        
        A track is considered "listened" (Listen+) if played_ratio_pct >= threshold.
        Default threshold of 50% matches the paper's definition.
        
        Args:
            table: PyArrow table with played_ratio_pct column
            threshold: Percentage threshold for positive label (default: 50)
            
        Returns:
            Table with added 'label' column
        """
        played_ratio = table.column('played_ratio_pct')
        
        # Create binary label: 1 if played >= threshold%, else 0
        label = pc.cast(
            pc.greater_equal(played_ratio, threshold),
            pa.int8()
        )
        
        # Add label column
        table = table.append_column('label', label)
        
        # Log class distribution
        label_array = label.to_numpy()
        pos_rate = label_array.mean() * 100
        logger.info(f"Label distribution: {pos_rate:.1f}% positive (played >= {threshold}%)")
        
        return table
    
    def to_pandas(
        self,
        table: pa.Table,
        categorical_columns: Optional[List[str]] = None
    ):
        """
        Convert PyArrow table to pandas DataFrame with optional categorical encoding.
        
        Args:
            table: PyArrow table
            categorical_columns: Columns to convert to pandas Categorical type
            
        Returns:
            pandas DataFrame
        """
        df = table.to_pandas()
        
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    
        return df
    
    def load_embeddings(
        self,
        item_ids: Optional[List[int]] = None
    ) -> pa.Table:
        """
        Load audio embeddings for tracks.
        
        Warning: The full embeddings file is ~13GB. Use item_ids filter
        to load only embeddings for tracks in your train/test set.
        
        Args:
            item_ids: Optional list of item_ids to filter to
            
        Returns:
            PyArrow table with item_id, embed, normalized_embed columns
        """
        embeddings_path = self.data_dir.parent / 'embeddings.parquet'
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
        logger.info("Loading embeddings...")
        
        if item_ids:
            # Read in chunks and filter to save memory
            table = pq.read_table(embeddings_path)
            mask = pc.is_in(table.column('item_id'), pa.array(item_ids))
            table = table.filter(mask)
            logger.info(f"Loaded embeddings for {table.num_rows:,} items")
        else:
            table = pq.read_table(embeddings_path)
            logger.warning(f"Loaded ALL {table.num_rows:,} embeddings - this may use significant memory")
            
        return table
    
    def load_item_metadata(self) -> Tuple[pa.Table, pa.Table]:
        """
        Load album and artist mappings for items.
        
        Returns:
            Tuple of (album_mapping, artist_mapping) as PyArrow Tables
        """
        album_path = self.data_dir.parent / 'album_item_mapping.parquet'
        artist_path = self.data_dir.parent / 'artist_item_mapping.parquet'
        
        album_table = None
        artist_table = None
        
        if album_path.exists():
            album_table = pq.read_table(album_path)
            logger.info(f"Loaded album mapping: {album_table.num_rows:,} entries")
            
        if artist_path.exists():
            artist_table = pq.read_table(artist_path)
            logger.info(f"Loaded artist mapping: {artist_table.num_rows:,} entries")
            
        return album_table, artist_table


def get_train_test_data(
    data_dir: str,
    task: str = 'listen_completion',
    gts_config: Optional[GTSConfig] = None,
    sample_frac: Optional[float] = None,
    as_pandas: bool = True
):
    """
    Convenience function to load train/test data for ranking.
    
    Args:
        data_dir: Path to Yandex/flat directory
        task: One of 'listen_completion', 'multi_task'
        gts_config: Global Temporal Split configuration (default: paper's protocol)
        sample_frac: Optional sampling fraction for development
        as_pandas: If True, return pandas DataFrames; else PyArrow Tables
        
    Returns:
        Tuple of (train_data, test_data)
        
    Example:
    -------
    >>> from data_loader import get_train_test_data
    >>> from config import GTSConfig
    >>> 
    >>> train_df, test_df = get_train_test_data(
    ...     data_dir='path/to/yandex/flat',
    ...     task='listen_completion',
    ...     gts_config=GTSConfig(train_days=300, gap_minutes=30, test_days=1),
    ...     sample_frac=0.1  # 10% sample for testing
    ... )
    """
    loader = YambdaDataLoader(data_dir, gts_config=gts_config)
    
    if task == 'listen_completion':
        train_table, test_table = loader.load_listens_with_gts(sample_frac=sample_frac)
        
        # Note: Labels are NOT added here anymore
        # The calling code should add labels based on their threshold
        # This keeps the data loader more general-purpose
        
    elif task == 'multi_task':
        train_table, test_table = loader.load_multi_event_with_gts(
            event_types=['listen', 'like', 'dislike']
        )
        
    else:
        raise ValueError(f"Unknown task: {task}. Use 'listen_completion' or 'multi_task'")
        
    if as_pandas:
        # Note: is_organic is NOT converted to categorical because:
        # 1. It's binary (0/1) - we need to compute mean() for aggregations
        # 2. Categorical dtype doesn't support mean() aggregation
        categorical_cols = ['uid', 'item_id']
        train_data = loader.to_pandas(train_table, categorical_cols)
        test_data = loader.to_pandas(test_table, categorical_cols)
    else:
        train_data = train_table
        test_data = test_table
        
    return train_data, test_data


if __name__ == '__main__':
    # Example usage and validation
    import sys
    import os
    
    # Get data directory from arg or environment
    DEFAULT_DATA_DIR = os.environ.get('YAMBDA_DATA_DIR', '')
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    
    if not data_dir:
        print("Error: Data directory not specified.")
        print("Usage: python data_loader.py /path/to/Yandex/flat")
        print("   Or: set YAMBDA_DATA_DIR environment variable")
        sys.exit(1)
    
    print("=" * 60)
    print("Yambda Data Loader - Chapter 6 Ranking")
    print("=" * 60)
    
    # Initialize loader
    loader = YambdaDataLoader(data_dir)
    
    # Get dataset info
    print("\n[1] Dataset Information:")
    info = loader.get_dataset_info()
    for fname, finfo in info.items():
        print(f"  {fname}:")
        print(f"    Rows: {finfo['num_rows']:,}")
        print(f"    Size: {finfo['size_mb']:.1f} MB")
        print(f"    Columns: {finfo['schema']}")
        
    # Test GTS split
    print("\n[2] Testing Global Temporal Split:")
    try:
        train_table, test_table = loader.load_listens_with_gts(sample_frac=0.01)
        
        # Add labels
        train_table = loader.create_listen_completion_labels(train_table)
        test_table = loader.create_listen_completion_labels(test_table)
        
        print(f"  Train shape: {train_table.num_rows:,} rows, {train_table.num_columns} cols")
        print(f"  Test shape: {test_table.num_rows:,} rows, {test_table.num_columns} cols")
        
        # Convert to pandas for inspection
        train_df = loader.to_pandas(train_table)
        print(f"\n  Train sample:")
        print(train_df.head())
        
    except Exception as e:
        print(f"  Error: {e}")
        
    print("\n" + "=" * 60)
    print("Data loader ready for Chapter 6!")
    print("=" * 60)

