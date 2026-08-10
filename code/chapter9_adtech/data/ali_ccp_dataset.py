"""
Ali-CCP Dataset (Clean Version)
===============================

This module provides a clean interface to the preprocessed Ali-CCP dataset.

Prerequisites:
    Run the preprocessing script ONCE before using this module:
    
    cd RecSys book code/chapter9_adtech/scripts
    python preprocess_ali_ccp.py --mode sample  # For 2M samples (16GB RAM)
    
    This creates parquet files in: Dataset/Ali-CCP Entire State Model/processed/

Usage:
    from data.ali_ccp_dataset import load_ali_ccp_data, AliCCPDataset
    
    # Load preprocessed data
    train_df, val_df, metadata = load_ali_ccp_data()
    
    # Create PyTorch datasets
    train_dataset = AliCCPDataset(train_df)
    val_dataset = AliCCPDataset(val_df)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

Schema (after preprocessing):
    - sample_id: Unique sample identifier
    - click: Click label (0 or 1) - CTR target
    - purchase: Conversion label (0 or 1) - CVR target
    - user_hash: Hashed user identifier
    - context_id: Context identifier
    - fid_0 to fid_99: Hashed feature IDs (int32)
        - fid_0 to fid_49: Sample-level features
        - fid_50 to fid_99: User-level features
    - fval_0 to fval_99: Feature values (float32)
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # RecSys ML Book Writing
PROCESSED_BASE_DIR = PROJECT_ROOT / "Dataset" / "Ali-CCP Entire State Model" / "processed"
DEFAULT_DATA_PCT = 10  # Default to 10% preprocessed data

# Default configuration (should match preprocessing)
DEFAULT_VOCAB_SIZE = 100_000
DEFAULT_NUM_FEATURES = 100  # 50 sample + 50 user


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_ali_ccp_data(
    processed_dir: Optional[Path] = None,
    data_pct: Optional[float] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
    """Load preprocessed Ali-CCP train and validation data.
    
    Args:
        processed_dir: Explicit path to processed data directory. 
                       If provided, overrides data_pct.
        data_pct: Which preprocessed percentage to load (e.g., 10, 20).
                  Resolves to processed/pct_10/, processed/pct_20/, etc.
                  Defaults to DEFAULT_DATA_PCT if neither argument is provided.
    
    Returns:
        Tuple of (train_df, val_df, metadata)
        
    Raises:
        FileNotFoundError: If preprocessed files don't exist
    """
    if processed_dir is None:
        pct = data_pct if data_pct is not None else DEFAULT_DATA_PCT
        pct_label = f"pct_{pct:g}"
        processed_dir = PROCESSED_BASE_DIR / pct_label
        
        # Fallback: check the old flat directory for backward compatibility
        if not processed_dir.exists() and PROCESSED_BASE_DIR.exists():
            old_train = PROCESSED_BASE_DIR / "ali_ccp_train.parquet"
            if old_train.exists():
                print(f"  Note: Using legacy flat directory: {PROCESSED_BASE_DIR}")
                processed_dir = PROCESSED_BASE_DIR
    
    processed_dir = Path(processed_dir)
    
    # Check files exist
    train_path = processed_dir / "ali_ccp_train.parquet"
    val_path = processed_dir / "ali_ccp_val.parquet"
    metadata_path = processed_dir / "metadata.json"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Preprocessed train file not found: {train_path}\n\n"
            "Please run the preprocessing script first:\n"
            "  cd 'RecSys book code/chapter9_adtech/scripts'\n"
            f"  python preprocess_ali_ccp.py --pct {data_pct or DEFAULT_DATA_PCT}\n\n"
            "This only needs to be done once per percentage."
        )
    
    # Load data
    print(f"Loading preprocessed Ali-CCP data from {processed_dir}...")
    
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val: {len(val_df):,} samples")
    print(f"  CTR: {metadata['statistics']['train_click_rate']:.4f}")
    print(f"  CVR (overall): {metadata['statistics']['train_purchase_rate']:.4f}")
    print(f"  CVR (post-click): {metadata['statistics']['train_post_click_cvr']:.4f}")
    
    return train_df, val_df, metadata


def get_feature_columns(df: pl.DataFrame) -> Tuple[list, list, list]:
    """Get field ID, feature ID, and value column names from a DataFrame.
    
    Returns:
        Tuple of (field_cols, fid_cols, fval_cols)
        - field_cols: Semantic field IDs (e.g., 101=UserID, 205=ItemID)
        - fid_cols: Hashed feature IDs
        - fval_cols: Feature values
    """
    field_cols = [c for c in df.columns if c.startswith('field_')]
    fid_cols = [c for c in df.columns if c.startswith('fid_')]
    fval_cols = [c for c in df.columns if c.startswith('fval_')]
    
    return (
        sorted(field_cols, key=lambda x: int(x.split('_')[1])),
        sorted(fid_cols, key=lambda x: int(x.split('_')[1])),
        sorted(fval_cols, key=lambda x: int(x.split('_')[1]))
    )


# =============================================================================
# PyTorch Dataset
# =============================================================================

class AliCCPDataset(Dataset):
    """PyTorch Dataset for preprocessed Ali-CCP data.
    
    Each sample returns a dictionary with:
        - field_ids: (num_features,) int32 tensor of semantic field IDs
                     (e.g., 101=UserID, 205=ItemID, 124=Gender)
        - feature_ids: (num_features,) int64 tensor of hashed feature IDs
        - feature_values: (num_features,) float32 tensor of feature values
        - click: float32 scalar click label
        - purchase: float32 scalar conversion label
    
    The feature IDs can be used directly for embedding lookup.
    Field IDs can be used for interpretability and feature grouping.
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        num_features: int = DEFAULT_NUM_FEATURES,
        lazy_load: bool = True,  # Memory-efficient mode
    ):
        """Initialize the dataset.
        
        Args:
            df: Polars DataFrame with preprocessed Ali-CCP data
            vocab_size: Vocabulary size for embedding lookup (for reference)
            num_features: Number of features per sample
            lazy_load: DEPRECATED - now always preloads efficiently. Kept for API compatibility.
        """
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.lazy_load = False  # Always preload - lazy loading via df.row() is too slow
        
        # Get column names
        self.field_cols, self.fid_cols, self.fval_cols = get_feature_columns(df)
        
        self.df = None
        self.n_samples = len(df)
        
        # Pre-load everything efficiently (column by column to minimize peak memory)
        # This is MUCH faster than lazy df.row() access
        print(f"  Converting {len(df):,} samples to NumPy arrays...")
        
        # Labels first (small)
        self.clicks = df['click'].to_numpy().astype(np.float32)
        self.purchases = df['purchase'].to_numpy().astype(np.float32)
        
        # Store user_hash for analysis (not used in training)
        self.user_hashes = df['user_hash'].to_list()
        self.sample_ids = df['sample_id'].to_numpy()
        
        # field_ids: semantic field identifiers (optional)
        # Note: Using int32 because compound field IDs (e.g., 12714) can exceed int16 max (32767)
        if self.field_cols:
            self.field_ids = df.select(self.field_cols).to_numpy().astype(np.int32)
        else:
            # Backward compatibility: create zeros if field columns don't exist
            self.field_ids = np.zeros((len(df), num_features), dtype=np.int32)
        
        # Feature IDs and values
        self.feature_ids = df.select(self.fid_cols).to_numpy().astype(np.int64)
        self.feature_values = df.select(self.fval_cols).to_numpy().astype(np.float32)
        
        print(f"  Dataset initialized: {len(self):,} samples, {self.num_features} features")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample by index. Fast O(1) access from pre-loaded arrays."""
        return {
            'field_ids': torch.from_numpy(self.field_ids[idx].copy()),
            'feature_ids': torch.from_numpy(self.feature_ids[idx].copy()),
            'feature_values': torch.from_numpy(self.feature_values[idx].copy()),
            'click': torch.tensor(self.clicks[idx], dtype=torch.float32),
            'purchase': torch.tensor(self.purchases[idx], dtype=torch.float32),
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics."""
        click_rate = float(self.clicks.mean())
        purchase_rate = float(self.purchases.mean())
        
        # Post-click CVR
        clicked_mask = self.clicks == 1
        post_click_cvr = float(self.purchases[clicked_mask].mean()) if clicked_mask.sum() > 0 else 0.0
        
        return {
            'n_samples': len(self),
            'click_rate': click_rate,
            'purchase_rate': purchase_rate,
            'post_click_cvr': post_click_cvr,
            'n_clicked': int(clicked_mask.sum()),
            'n_converted': int(self.purchases.sum()),
        }


# =============================================================================
# DataLoader Creation
# =============================================================================

def create_data_loaders(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    batch_size: int = 4096,
    num_workers: int = 4,  # Use multiple workers for faster data loading
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from preprocessed DataFrames.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        batch_size: Batch size (default 4096, good for CTR/CVR models)
        num_workers: Number of worker processes (4 is good for most systems)
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = AliCCPDataset(train_df)
    val_dataset = AliCCPDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


# =============================================================================
# Convenience Function
# =============================================================================

def load_ali_ccp_loaders(
    batch_size: int = 4096,
    num_workers: int = 4,
    processed_dir: Optional[Path] = None,
    data_pct: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """One-liner to load Ali-CCP data and create DataLoaders.
    
    Args:
        batch_size: Batch size
        num_workers: Number of workers
        processed_dir: Explicit path to processed data (overrides data_pct)
        data_pct: Which preprocessed percentage to load (e.g., 10, 20)
        
    Returns:
        Tuple of (train_loader, val_loader, metadata)
        
    Usage:
        train_loader, val_loader, meta = load_ali_ccp_loaders(data_pct=10)
        
        for batch in train_loader:
            feature_ids = batch['feature_ids']  # (B, 100)
            feature_values = batch['feature_values']  # (B, 100)
            click = batch['click']  # (B,)
            purchase = batch['purchase']  # (B,)
    """
    train_df, val_df, metadata = load_ali_ccp_data(processed_dir, data_pct=data_pct)
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, metadata


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Ali-CCP Dataset (Clean Version)")
    print("=" * 50)
    
    try:
        # Load data
        train_df, val_df, metadata = load_ali_ccp_data()
        
        print("\nMetadata:")
        print(f"  Created: {metadata['created_at']}")
        print(f"  Config: {metadata['config']}")
        
        # Create datasets
        print("\nCreating datasets...")
        train_dataset = AliCCPDataset(train_df)
        val_dataset = AliCCPDataset(val_df)
        
        print("\nTrain statistics:", train_dataset.get_statistics())
        print("Val statistics:", val_dataset.get_statistics())
        
        # Test a sample
        print("\nSample item:")
        sample = train_dataset[0]
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")
        
        # Test data loader
        print("\nTesting DataLoader...")
        train_loader, val_loader = create_data_loaders(train_df, val_df, batch_size=1024)
        
        batch = next(iter(train_loader))
        print("\nBatch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")

