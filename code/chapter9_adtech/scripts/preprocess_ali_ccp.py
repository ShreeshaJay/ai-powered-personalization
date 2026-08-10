"""
Ali-CCP Dataset Preprocessor
============================

This script preprocesses the raw Ali-CCP dataset into clean parquet files
that are easy to work with. Run this ONCE before training.

The raw Ali-CCP format has:
- Cryptic sparse feature encoding (using ASCII delimiters 0x01, 0x02, 0x03)
- Two separate files that need joining
- ~42.3M samples (~18GB total)

After preprocessing, you get:
- Clean parquet files with explicit columns
- Pre-parsed feature IDs (hashed to vocabulary)
- Configurable sample sizes for different hardware

MEMORY-OPTIMIZED VERSION:
    Uses chunked processing to stay within 16GB RAM even for large datasets.
    Peak memory usage: ~4-6GB for any dataset size.

Usage:
    # Full preprocessing (16GB RAM OK with chunked processing, takes ~45 mins)
    python preprocess_ali_ccp.py --pct 100
    
    # 20% sample (~8.4M samples, ~15 mins)
    python preprocess_ali_ccp.py --pct 20
    
    # 10% sample (~4.2M samples, ~8 mins)
    python preprocess_ali_ccp.py --pct 10
    
    # Quick test (0.1% = ~42K samples, for development)
    python preprocess_ali_ccp.py --pct 0.1
    
    # Exact sample count (alternative to percentage)
    python preprocess_ali_ccp.py --n_samples 2000000

Output files saved to: Dataset/Ali-CCP Entire State Model/processed/pct_<X>/
    - ali_ccp_train.parquet (90% of data)
    - ali_ccp_val.parquet (10% of data)
    - metadata.json (statistics and config)
    
    Each --pct value gets its own subdirectory (e.g., pct_10/, pct_20/)
    so different preprocessing runs don't overwrite each other.

Dataset size reference (full dataset = 42.3M samples):
    - 0.1% = ~42K samples (quick test, ~2 min)
    - 1%   = ~423K samples (~3 min)
    - 5%   = ~2.1M samples (~5 min)
    - 10%  = ~4.2M samples (~8 min)
    - 20%  = ~8.4M samples (~15 min)
    - 100% = ~42.3M samples (~45 min)
"""

import argparse
import json
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import polars as pl
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# Raw data paths (relative to this script's grandparent directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # RecSys ML Book Writing
RAW_DATA_DIR = PROJECT_ROOT / "Dataset" / "Ali-CCP Entire State Model"
PROCESSED_BASE_DIR = RAW_DATA_DIR / "processed"

# Feature configuration
VOCAB_SIZE = 100_000  # Hash bucket size for feature IDs
MAX_SAMPLE_FEATURES = 50  # Max features from sample_skeleton
MAX_USER_FEATURES = 50  # Max features from common_features
TOTAL_MAX_FEATURES = MAX_SAMPLE_FEATURES + MAX_USER_FEATURES

# Validation split
VAL_RATIO = 0.1
RANDOM_SEED = 42


# =============================================================================
# Sparse Feature Parsing
# =============================================================================

# Ali-CCP uses ASCII delimiters for feature structure
# See: https://tianchi.aliyun.com/dataset/408
FEATURE_SEP = chr(0x01)  # Separates features
FIELD_SEP = chr(0x02)    # Separates field_id from feature_id
VALUE_SEP = chr(0x03)    # Separates feature_id from value


def hash_feature_id(feature_str: str, vocab_size: int = VOCAB_SIZE) -> int:
    """Hash a feature ID string to a vocabulary bucket using MD5."""
    hash_bytes = hashlib.md5(feature_str.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    return hash_int % vocab_size


def parse_field_id(field_str: str) -> int:
    """Parse field ID string to integer.
    
    Field IDs like '109_14' are converted to integers by removing underscore.
    This preserves the field semantics (e.g., 109 = category behavior, 14 = count).
    """
    # Remove underscore for compound field IDs (e.g., '109_14' -> 10914)
    clean_str = field_str.replace('_', '')
    try:
        return int(clean_str)
    except ValueError:
        # Hash non-numeric field IDs
        return hash_feature_id(field_str, 10000)  # Smaller vocab for field IDs


def parse_sparse_features(
    sparse_str: str,
    max_features: int,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[List[int], List[int], List[float]]:
    """Parse Ali-CCP sparse feature string into field IDs, feature IDs, and values.
    
    Ali-CCP Format (from official documentation):
        Features separated by 0x01 (ASCII SOH)
        Each feature: field_id <0x02> feature_id <0x03> value
        
        Where:
        - field_id: Semantic field (e.g., 101=UserID, 205=ItemID, 124=Gender)
        - feature_id: Actual feature value (globally encoded)
        - value: Real number weight
    
    Example raw string:
        "101<0x02>12345<0x03>1.0<0x01>124<0x02>2<0x03>1.0<0x01>205<0x02>67890<0x03>1.0"
        
    Parsed:
        - (field=101, id=12345, val=1.0)  # User ID
        - (field=124, id=2, val=1.0)      # Gender
        - (field=205, id=67890, val=1.0)  # Item ID
    
    Returns:
        Tuple of (field_ids, feature_ids, feature_values)
    """
    if not sparse_str or sparse_str == '' or (isinstance(sparse_str, float) and np.isnan(sparse_str)):
        return [0] * max_features, [0] * max_features, [0.0] * max_features
    
    sparse_str = str(sparse_str)
    
    field_ids = []
    feature_ids = []
    feature_values = []
    
    # Split by feature separator (0x01)
    features = sparse_str.split(FEATURE_SEP)
    
    for feat in features:
        if not feat or len(field_ids) >= max_features:
            continue
        
        # Try to parse: field_id <0x02> feature_id <0x03> value
        if FIELD_SEP in feat and VALUE_SEP in feat:
            try:
                # Split by 0x02 to get field_id and rest
                field_part, rest = feat.split(FIELD_SEP, 1)
                # Split rest by 0x03 to get feature_id and value
                fid_part, val_part = rest.split(VALUE_SEP, 1)
                
                field_id = parse_field_id(field_part.strip())
                hashed_fid = hash_feature_id(fid_part.strip(), vocab_size)
                value = float(val_part.strip())
                
                field_ids.append(field_id)
                feature_ids.append(hashed_fid)
                feature_values.append(value)
            except (ValueError, IndexError):
                # Skip malformed features
                continue
        else:
            # Fallback: try old format (concatenated without proper delimiters)
            # This handles edge cases in the data
            pass
    
    # Pad to max_features
    n_found = len(field_ids)
    if n_found < max_features:
        field_ids.extend([0] * (max_features - n_found))
        feature_ids.extend([0] * (max_features - n_found))
        feature_values.extend([0.0] * (max_features - n_found))
    
    return field_ids[:max_features], feature_ids[:max_features], feature_values[:max_features]


# =============================================================================
# Data Loading
# =============================================================================

def load_user_features(verbose: bool = True) -> Dict[str, Tuple[List[int], List[int], List[float]]]:
    """Load and parse user-level features from common_features_train.csv.
    
    Returns dict mapping user_hash -> (field_ids, feature_ids, feature_values)
    """
    filepath = RAW_DATA_DIR / "common_features_train.csv"
    
    if verbose:
        print(f"Loading user features from {filepath}...")
    
    # Read CSV (no header in raw file)
    df = pl.read_csv(
        filepath,
        has_header=False,
        new_columns=['user_hash', 'feature_count', 'sparse_features'],
        infer_schema_length=10000,
    )
    
    if verbose:
        print(f"  Found {len(df):,} users")
    
    # Parse features for each user
    user_features = {}
    iterator = df.iter_rows(named=True)
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="  Parsing user features")
    
    for row in iterator:
        user_hash = str(row['user_hash'])
        fields, fids, fvals = parse_sparse_features(row['sparse_features'], MAX_USER_FEATURES)
        user_features[user_hash] = (fields, fids, fvals)
    
    return user_features


def load_skeleton_data(
    n_samples: Optional[int] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Load sample skeleton data."""
    filepath = RAW_DATA_DIR / "sample_skeleton_train.csv"
    
    if verbose:
        print(f"Loading skeleton data from {filepath}...")
        if n_samples:
            print(f"  Limiting to {n_samples:,} samples")
    
    df = pl.read_csv(
        filepath,
        has_header=False,
        new_columns=['sample_id', 'click', 'purchase', 'user_hash', 'context_id', 'sparse_features'],
        n_rows=n_samples,
        infer_schema_length=10000,
    )
    
    if verbose:
        print(f"  Loaded {len(df):,} samples")
        print(f"  Click rate: {df['click'].mean():.4f}")
        print(f"  Purchase rate: {df['purchase'].mean():.4f}")
        if df['click'].sum() > 0:
            clicked_df = df.filter(pl.col('click') == 1)
            print(f"  Post-click CVR: {clicked_df['purchase'].mean():.4f}")
    
    return df


# =============================================================================
# Main Preprocessing (Memory-Optimized with Chunked Processing)
# =============================================================================

# Chunk size for memory-efficient processing
# 500K samples per chunk uses ~2-3GB RAM
CHUNK_SIZE = 500_000


def process_chunk(
    rows: List[Dict],
    user_features: Dict[str, Tuple[List[int], List[int], List[float]]],
) -> Tuple[pl.DataFrame, int, int]:
    """Process a chunk of samples and return a DataFrame.
    
    Returns:
        (DataFrame, n_with_user_features, total_sample_features)
    """
    n_chunk = len(rows)
    
    # Allocate arrays for this chunk only
    sample_ids = np.zeros(n_chunk, dtype=np.int64)
    clicks = np.zeros(n_chunk, dtype=np.int8)
    purchases = np.zeros(n_chunk, dtype=np.int8)
    user_hashes = []
    context_ids = np.zeros(n_chunk, dtype=np.int64)
    
    field_ids = np.zeros((n_chunk, TOTAL_MAX_FEATURES), dtype=np.int16)
    feature_ids = np.zeros((n_chunk, TOTAL_MAX_FEATURES), dtype=np.int32)
    feature_values = np.zeros((n_chunk, TOTAL_MAX_FEATURES), dtype=np.float32)
    
    n_with_user_features = 0
    total_sample_features = 0
    
    for i, row in enumerate(rows):
        # Basic fields
        sample_ids[i] = row['sample_id']
        clicks[i] = row['click']
        purchases[i] = row['purchase']
        user_hashes.append(str(row['user_hash']))
        context_ids[i] = row['context_id']
        
        # Parse sample-level features
        sample_fields, sample_fids, sample_fvals = parse_sparse_features(
            row['sparse_features'], MAX_SAMPLE_FEATURES
        )
        total_sample_features += sum(1 for f in sample_fids if f != 0)
        
        # Get user-level features
        user_hash = str(row['user_hash'])
        if user_hash in user_features:
            user_fields, user_fids, user_fvals = user_features[user_hash]
            n_with_user_features += 1
        else:
            user_fields = [0] * MAX_USER_FEATURES
            user_fids = [0] * MAX_USER_FEATURES
            user_fvals = [0.0] * MAX_USER_FEATURES
        
        # Combine: [sample_features | user_features]
        field_ids[i, :MAX_SAMPLE_FEATURES] = sample_fields
        field_ids[i, MAX_SAMPLE_FEATURES:] = user_fields
        feature_ids[i, :MAX_SAMPLE_FEATURES] = sample_fids
        feature_ids[i, MAX_SAMPLE_FEATURES:] = user_fids
        feature_values[i, :MAX_SAMPLE_FEATURES] = sample_fvals
        feature_values[i, MAX_SAMPLE_FEATURES:] = user_fvals
    
    # Build DataFrame for this chunk
    data = {
        'sample_id': sample_ids,
        'click': clicks,
        'purchase': purchases,
        'user_hash': user_hashes,
        'context_id': context_ids,
    }
    
    # Add feature columns
    for j in range(TOTAL_MAX_FEATURES):
        data[f'field_{j}'] = field_ids[:, j]
        data[f'fid_{j}'] = feature_ids[:, j]
        data[f'fval_{j}'] = feature_values[:, j]
    
    df = pl.DataFrame(data)
    return df, n_with_user_features, total_sample_features


def preprocess_and_save(
    n_samples: Optional[int] = None,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Main preprocessing function (memory-optimized).
    
    Uses chunked processing to stay within 16GB RAM:
    1. Load user features (kept in memory, ~2GB)
    2. Stream skeleton data in chunks
    3. Process each chunk and append to temporary parquet files
    4. Shuffle indices and assign to train/val
    5. Write final parquet files
    
    Returns metadata dictionary.
    """
    import gc
    import tempfile
    import shutil
    
    if output_dir is None:
        output_dir = PROCESSED_BASE_DIR
    
    start_time = datetime.now()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix='aliccp_'))
    
    try:
        # Step 1: Load user features
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 1: Loading user features")
            print("=" * 60)
        
        user_features = load_user_features(verbose)
        
        # Step 2: Load skeleton data metadata (just count rows)
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 2: Loading skeleton data")
            print("=" * 60)
        
        skeleton_df = load_skeleton_data(n_samples, verbose)
        n_total = len(skeleton_df)
        
        # Step 3: Process in chunks
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 3: Processing in chunks (memory-optimized)")
            print("=" * 60)
            print(f"  Chunk size: {CHUNK_SIZE:,} samples")
            print(f"  Estimated chunks: {(n_total + CHUNK_SIZE - 1) // CHUNK_SIZE}")
        
        # Track statistics across chunks
        total_with_user_features = 0
        total_sample_features = 0
        chunk_files = []
        
        # Process in chunks
        rows_buffer = []
        chunk_idx = 0
        
        iterator = skeleton_df.iter_rows(named=True)
        if verbose:
            iterator = tqdm(iterator, total=n_total, desc="  Processing")
        
        for row in iterator:
            rows_buffer.append(row)
            
            if len(rows_buffer) >= CHUNK_SIZE:
                # Process this chunk
                chunk_df, n_user_feat, n_sample_feat = process_chunk(rows_buffer, user_features)
                total_with_user_features += n_user_feat
                total_sample_features += n_sample_feat
                
                # Save to temp file
                chunk_path = temp_dir / f"chunk_{chunk_idx:04d}.parquet"
                chunk_df.write_parquet(chunk_path)
                chunk_files.append(chunk_path)
                
                # Clear memory
                del chunk_df
                rows_buffer = []
                gc.collect()
                chunk_idx += 1
        
        # Process remaining rows
        if rows_buffer:
            chunk_df, n_user_feat, n_sample_feat = process_chunk(rows_buffer, user_features)
            total_with_user_features += n_user_feat
            total_sample_features += n_sample_feat
            
            chunk_path = temp_dir / f"chunk_{chunk_idx:04d}.parquet"
            chunk_df.write_parquet(chunk_path)
            chunk_files.append(chunk_path)
            del chunk_df
            gc.collect()
        
        # Free skeleton_df memory
        del skeleton_df
        gc.collect()
        
        avg_sample_features = total_sample_features / n_total if n_total > 0 else 0
        
        if verbose:
            print(f"\n  Processed {len(chunk_files)} chunks")
            print(f"  Samples with user features: {total_with_user_features:,} ({100*total_with_user_features/n_total:.1f}%)")
            print(f"  Avg sample features: {avg_sample_features:.1f}")
        
        # Step 4: Combine chunks, shuffle, and split
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 4: Combining, shuffling, and splitting")
            print("=" * 60)
        
        # Load all chunks (now one at a time for merging)
        if verbose:
            print("  Loading chunks...")
        
        all_data = pl.concat([pl.read_parquet(f) for f in chunk_files])
        
        if verbose:
            print(f"  Combined: {len(all_data):,} samples")
        
        # Shuffle
        np.random.seed(random_seed)
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices.tolist()]
        
        # Split
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        
        val_df = all_data[:n_val]
        train_df = all_data[n_val:]
        
        # Free all_data
        del all_data
        gc.collect()
        
        if verbose:
            print(f"  Train samples: {n_train:,}")
            print(f"  Val samples: {n_val:,}")
        
        # Step 5: Save final parquet files
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 5: Saving to parquet")
            print("=" * 60)
        
        train_path = output_dir / "ali_ccp_train.parquet"
        train_df.write_parquet(train_path)
        if verbose:
            print(f"  Saved train set: {train_path}")
            print(f"    Size: {train_path.stat().st_size / 1e6:.1f} MB")
        
        val_path = output_dir / "ali_ccp_val.parquet"
        val_df.write_parquet(val_path)
        if verbose:
            print(f"  Saved val set: {val_path}")
            print(f"    Size: {val_path.stat().st_size / 1e6:.1f} MB")
        
        # Compute final statistics
        train_click_rate = train_df['click'].mean()
        train_purchase_rate = train_df['purchase'].mean()
        val_click_rate = val_df['click'].mean()
        val_purchase_rate = val_df['purchase'].mean()
        
        # Compute post-click CVR
        train_clicked = train_df.filter(pl.col('click') == 1)
        train_post_click_cvr = train_clicked['purchase'].mean() if len(train_clicked) > 0 else 0.0
        
        val_clicked = val_df.filter(pl.col('click') == 1)
        val_post_click_cvr = val_clicked['purchase'].mean() if len(val_clicked) > 0 else 0.0
        
        # Free DataFrames
        del train_df, val_df
        gc.collect()
        
        # Save metadata
        elapsed = (datetime.now() - start_time).total_seconds()
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'processing_time_seconds': elapsed,
            'config': {
                'n_samples_requested': n_samples,
                'n_samples_actual': n_total,
                'val_ratio': val_ratio,
                'random_seed': random_seed,
                'vocab_size': VOCAB_SIZE,
                'max_sample_features': MAX_SAMPLE_FEATURES,
                'max_user_features': MAX_USER_FEATURES,
                'total_max_features': TOTAL_MAX_FEATURES,
                'chunk_size': CHUNK_SIZE,
            },
            'statistics': {
                'n_train': n_train,
                'n_val': n_val,
                'n_users_in_lookup': len(user_features),
                'n_samples_with_user_features': total_with_user_features,
                'pct_with_user_features': 100 * total_with_user_features / n_total,
                'avg_sample_features': float(avg_sample_features),
                'train_click_rate': float(train_click_rate),
                'train_purchase_rate': float(train_purchase_rate),
                'train_post_click_cvr': float(train_post_click_cvr),
                'val_click_rate': float(val_click_rate),
                'val_purchase_rate': float(val_purchase_rate),
                'val_post_click_cvr': float(val_post_click_cvr),
            },
            'files': {
                'train': str(train_path),
                'val': str(val_path),
            },
            'schema': {
                'sample_id': 'int64 - Unique sample identifier',
                'click': 'int8 - Click label (0 or 1)',
                'purchase': 'int8 - Conversion/purchase label (0 or 1)',
                'user_hash': 'string - Hashed user identifier',
                'context_id': 'int64 - Context identifier',
                'field_0 to field_99': 'int16 - Semantic field IDs (0-49: sample, 50-99: user)',
                'fid_0 to fid_99': 'int32 - Hashed feature IDs (0-49: sample, 50-99: user)',
                'fval_0 to fval_99': 'float32 - Feature values',
            },
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            print(f"  Saved metadata: {metadata_path}")
            print("\n" + "=" * 60)
            print("PREPROCESSING COMPLETE")
            print("=" * 60)
            print(f"  Total time: {elapsed:.1f} seconds")
            print(f"\n  Output directory: {output_dir}")
            print(f"  Files created:")
            print(f"    - ali_ccp_train.parquet ({n_train:,} samples)")
            print(f"    - ali_ccp_val.parquet ({n_val:,} samples)")
            print(f"    - metadata.json")
            print(f"\n  Statistics:")
            print(f"    Train CTR: {train_click_rate:.4f}, CVR: {train_purchase_rate:.4f}, Post-click CVR: {train_post_click_cvr:.4f}")
            print(f"    Val CTR: {val_click_rate:.4f}, CVR: {val_purchase_rate:.4f}, Post-click CVR: {val_post_click_cvr:.4f}")
        
        return metadata
    
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# =============================================================================
# CLI
# =============================================================================

# Known dataset size (for percentage calculation)
TOTAL_SAMPLES_IN_DATASET = 42_300_134  # From Ali-CCP documentation


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Ali-CCP dataset into clean parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (0.1% = ~42K samples, ~2 mins)
  python preprocess_ali_ccp.py --pct 0.1
  
  # Standard preprocessing (5% = ~2.1M samples, 16GB RAM friendly, ~10 mins)
  python preprocess_ali_ccp.py --pct 5
  
  # Larger sample (10% = ~4.2M samples)
  python preprocess_ali_ccp.py --pct 10
  
  # Full dataset (100%, requires 32GB+ RAM, ~30 mins)
  python preprocess_ali_ccp.py --pct 100
  
  # Exact sample count (alternative to percentage)
  python preprocess_ali_ccp.py --n_samples 2000000

Dataset size reference (full = 42.3M samples):
  --pct 0.1  →  ~42K samples (quick test)
  --pct 1    →  ~423K samples
  --pct 5    →  ~2.1M samples (recommended for 16GB RAM)
  --pct 10   →  ~4.2M samples
  --pct 100  →  ~42.3M samples (requires 32GB+ RAM)
        """
    )
    
    parser.add_argument(
        '--pct',
        type=float,
        default=5.0,
        help='Percentage of dataset to process (default: 5%% = ~2.1M samples)'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Override: exact number of samples to process (takes precedence over --pct)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=VAL_RATIO,
        help=f'Validation split ratio (default: {VAL_RATIO})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Determine n_samples from percentage or explicit count
    if args.n_samples is not None:
        n_samples = args.n_samples
        pct_used = 100.0 * n_samples / TOTAL_SAMPLES_IN_DATASET
    elif args.pct >= 100:
        n_samples = None  # Process all
        pct_used = 100.0
    else:
        n_samples = int(TOTAL_SAMPLES_IN_DATASET * args.pct / 100)
        pct_used = args.pct
    
    # Check that raw data exists
    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DATA_DIR}")
        print("\nPlease ensure the Ali-CCP dataset is downloaded to:")
        print(f"  {RAW_DATA_DIR}")
        print("\nDownload from: https://tianchi.aliyun.com/dataset/408")
        print("\nRequired files:")
        print("  - sample_skeleton_train.csv")
        print("  - common_features_train.csv")
        sys.exit(1)
    
    skeleton_file = RAW_DATA_DIR / "sample_skeleton_train.csv"
    features_file = RAW_DATA_DIR / "common_features_train.csv"
    
    if not skeleton_file.exists():
        print(f"ERROR: Skeleton file not found: {skeleton_file}")
        sys.exit(1)
    
    if not features_file.exists():
        print(f"ERROR: Features file not found: {features_file}")
        sys.exit(1)
    
    # Determine output directory: processed/pct_10/, processed/pct_20/, etc.
    # This keeps each preprocessing run separate so they don't overwrite each other.
    pct_label = f"pct_{pct_used:g}"  # e.g. "pct_10", "pct_0.1", "pct_100"
    output_dir = PROCESSED_BASE_DIR / pct_label
    
    # Run preprocessing
    print("=" * 60)
    print("ALI-CCP DATASET PREPROCESSOR")
    print("=" * 60)
    print(f"Percentage: {pct_used:.1f}%")
    print(f"Samples: {n_samples:,}" if n_samples else "Samples: ALL (~42.3M)")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {output_dir}")
    
    metadata = preprocess_and_save(
        n_samples=n_samples,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        verbose=not args.quiet,
        output_dir=output_dir,
    )
    
    print("\n✅ Preprocessing complete! You can now use the clean parquet files for training.")


if __name__ == "__main__":
    main()

