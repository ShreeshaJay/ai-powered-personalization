"""
FedAds (Alibaba Federated Advertising) Data Loader
===================================================

This module handles loading and preprocessing the FedAds dataset, designed
for vertical federated learning scenarios.

Dataset Overview:
    FedAds simulates a real-world scenario where:
    - PUBLISHER (LOCAL): Has user browsing data, page context, impressions
    - ADVERTISER (FEDERATED): Has purchase history, ad catalog, conversion labels
    
    Neither party wants to share raw data, but they can collaborate via
    federated learning techniques.

Dataset Files:
    sample_train_aligned.csv (2.6M rows, ~1.35GB):
        - Samples where BOTH parties have features
        - Contains all 22 features (17 local + 5 federated)
        - Used for joint training
        
    sample_train_unaligned.csv (10.4M rows, ~4GB):
        - Samples where only LOCAL party has features
        - Contains 17 local features only
        - Can be used for semi-supervised learning

Feature Groups:
    LOCAL (Publisher) - 17 features:
        l_i_fea_1 to l_i_fea_10: Item features (10)
        l_u_fea_1 to l_u_fea_6:  User features (6)
        l_c_fea:                  Context feature (1)
        
    FEDERATED (Advertiser) - 5 features:
        f_u_fea_1, f_u_fea_2:    User features (2)
        f_uc_fea_1, f_uc_fea_2:  User-context cross features (2)
        f_c:                      Context feature (1)

Data Format:
    - sample_id: Hashed sample identifier
    - label: CVR label in JSON format ['0'] or ['1']
    - Features: Hashed values in JSON array format ['hash_value']
    - All values are privacy-preserving hashed representations

Usage:
    from data.fedads_loader import load_fedads_data, FedAdsDataset
    
    # Load aligned data (both parties)
    aligned_df = load_fedads_data(aligned=True, n_samples=500_000)
    
    # Create dataset
    dataset = FedAdsDataset(aligned_df, party='both')  # or 'local', 'federated'
"""

import json
import hashlib
import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Literal
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    FEDADS_ALIGNED, FEDADS_UNALIGNED, CACHE_DIR,
    FEDADS_LOCAL_FEATURES, FEDADS_FEDERATED_FEATURES, FEDADS_ALL_FEATURES
)


# Vocabulary sizes for feature hashing
LOCAL_VOCAB_SIZE = 50_000
FEDERATED_VOCAB_SIZE = 50_000


def parse_json_value(value: str) -> str:
    """Parse JSON-formatted feature value.
    
    FedAds features are stored as JSON arrays: ['hash_value']
    We extract the actual hash value.
    
    Args:
        value: String like "['abc123']" or "['0']"
        
    Returns:
        The extracted value string
    """
    if not value or value == '' or (isinstance(value, float) and np.isnan(value)):
        return '0'
    
    try:
        # Try parsing as JSON
        parsed = json.loads(value.replace("'", '"'))
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0])
        return str(parsed)
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, use as-is
        return str(value)


def hash_to_bucket(value: str, vocab_size: int) -> int:
    """Hash a string value to a vocabulary bucket.
    
    Args:
        value: The feature value string
        vocab_size: Number of hash buckets
        
    Returns:
        Integer in range [1, vocab_size] (0 reserved for padding)
    """
    hash_bytes = hashlib.md5(value.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    return (hash_int % vocab_size) + 1  # +1 to reserve 0 for padding


def load_fedads_data(
    aligned: bool = True,
    n_samples: Optional[int] = None,
) -> pl.DataFrame:
    """Load FedAds data from CSV.
    
    Args:
        aligned: If True, load aligned samples (both parties have features)
                 If False, load unaligned samples (local features only)
        n_samples: Number of samples to load. None for all.
        
    Returns:
        Polars DataFrame with parsed features
    """
    filepath = FEDADS_ALIGNED if aligned else FEDADS_UNALIGNED
    cache_name = f"fedads_{'aligned' if aligned else 'unaligned'}_{n_samples or 'full'}.parquet"
    cache_file = CACHE_DIR / cache_name
    
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        return pl.read_parquet(cache_file)
    
    print(f"Loading FedAds {'aligned' if aligned else 'unaligned'} data from {filepath}...")
    
    # Determine columns based on aligned/unaligned
    if aligned:
        columns = ['sample_id', 'label'] + FEDADS_ALL_FEATURES
    else:
        columns = ['sample_id', 'label'] + FEDADS_LOCAL_FEATURES
    
    df = pl.read_csv(
        filepath,
        has_header=True,
        n_rows=n_samples,
        infer_schema_length=10000,
    )
    
    print(f"Loaded {len(df):,} samples")
    
    # Parse label from JSON format
    # Label is stored as ['0'] or ['1']
    df = df.with_columns([
        pl.col('label').map_elements(
            lambda x: int(parse_json_value(x)),
            return_dtype=pl.Int64
        ).alias('label_parsed')
    ])
    
    cvr = df['label_parsed'].mean()
    print(f"CVR: {cvr:.4%}")
    
    # Cache for future use
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_file)
    
    return df


class FedAdsDataset(Dataset):
    """PyTorch Dataset for FedAds data.
    
    Supports three modes:
        - 'both': Returns features from both parties (for centralized baseline)
        - 'local': Returns only local (publisher) features
        - 'federated': Returns only federated (advertiser) features
    
    This separation enables simulation of federated learning scenarios.
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        party: Literal['both', 'local', 'federated'] = 'both',
        local_vocab_size: int = LOCAL_VOCAB_SIZE,
        fed_vocab_size: int = FEDERATED_VOCAB_SIZE,
        parse_on_load: bool = True,
    ):
        """Initialize dataset.
        
        Args:
            df: Polars DataFrame with FedAds data
            party: Which party's features to return
            local_vocab_size: Hash bucket size for local features
            fed_vocab_size: Hash bucket size for federated features
            parse_on_load: If True, pre-parse all features for faster training
        """
        self.df = df
        self.party = party
        self.local_vocab_size = local_vocab_size
        self.fed_vocab_size = fed_vocab_size
        
        # Determine which features to use
        if party == 'both':
            self.local_features = FEDADS_LOCAL_FEATURES
            self.fed_features = FEDADS_FEDERATED_FEATURES
        elif party == 'local':
            self.local_features = FEDADS_LOCAL_FEATURES
            self.fed_features = []
        elif party == 'federated':
            self.local_features = []
            self.fed_features = FEDADS_FEDERATED_FEATURES
        
        # Check if columns exist
        available_cols = df.columns
        self.local_features = [f for f in self.local_features if f in available_cols]
        self.fed_features = [f for f in self.fed_features if f in available_cols]
        
        self._parsed_data = None
        if parse_on_load:
            self._preparse_features()
    
    def _preparse_features(self):
        """Pre-parse all features for faster training."""
        print(f"Pre-parsing FedAds features (party={self.party})...")
        
        n_samples = len(self.df)
        n_local = len(self.local_features)
        n_fed = len(self.fed_features)
        
        local_ids = np.zeros((n_samples, n_local), dtype=np.int64) if n_local > 0 else None
        fed_ids = np.zeros((n_samples, n_fed), dtype=np.int64) if n_fed > 0 else None
        labels = np.zeros(n_samples, dtype=np.float32)
        
        for i, row in tqdm(enumerate(self.df.iter_rows(named=True)), total=n_samples):
            # Parse label
            if 'label_parsed' in row:
                labels[i] = float(row['label_parsed'])
            else:
                labels[i] = float(int(parse_json_value(row['label'])))
            
            # Parse local features
            if local_ids is not None:
                for j, feat_name in enumerate(self.local_features):
                    value = parse_json_value(row[feat_name])
                    local_ids[i, j] = hash_to_bucket(value, self.local_vocab_size)
            
            # Parse federated features
            if fed_ids is not None:
                for j, feat_name in enumerate(self.fed_features):
                    value = parse_json_value(row[feat_name])
                    fed_ids[i, j] = hash_to_bucket(value, self.fed_vocab_size)
        
        self._parsed_data = {
            'local_ids': local_ids,
            'fed_ids': fed_ids,
            'labels': labels,
        }
        print("Feature parsing complete!")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        if self._parsed_data is not None:
            result = {
                'label': torch.tensor(self._parsed_data['labels'][idx], dtype=torch.float32),
            }
            
            if self._parsed_data['local_ids'] is not None:
                result['local_ids'] = torch.from_numpy(self._parsed_data['local_ids'][idx])
            
            if self._parsed_data['fed_ids'] is not None:
                result['fed_ids'] = torch.from_numpy(self._parsed_data['fed_ids'][idx])
            
            return result
        else:
            # Parse on-the-fly (slower)
            row = self.df.row(idx, named=True)
            
            result = {}
            
            if 'label_parsed' in row:
                result['label'] = torch.tensor(float(row['label_parsed']), dtype=torch.float32)
            else:
                result['label'] = torch.tensor(float(int(parse_json_value(row['label']))), dtype=torch.float32)
            
            if self.local_features:
                local_ids = []
                for feat_name in self.local_features:
                    value = parse_json_value(row[feat_name])
                    local_ids.append(hash_to_bucket(value, self.local_vocab_size))
                result['local_ids'] = torch.tensor(local_ids, dtype=torch.int64)
            
            if self.fed_features:
                fed_ids = []
                for feat_name in self.fed_features:
                    value = parse_json_value(row[feat_name])
                    fed_ids.append(hash_to_bucket(value, self.fed_vocab_size))
                result['fed_ids'] = torch.tensor(fed_ids, dtype=torch.int64)
            
            return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self._parsed_data is not None:
            cvr = float(self._parsed_data['labels'].mean())
        else:
            cvr = float(self.df['label_parsed'].mean()) if 'label_parsed' in self.df.columns else None
        
        return {
            'n_samples': len(self),
            'n_local_features': len(self.local_features),
            'n_fed_features': len(self.fed_features),
            'cvr': cvr,
            'party': self.party,
        }


def load_fedads_split(
    n_samples: Optional[int] = None,
    val_ratio: float = 0.115,
    party: Literal['both', 'local', 'federated'] = 'both',
) -> Tuple[FedAdsDataset, FedAdsDataset]:
    """Load FedAds data and create train/val datasets.
    
    Uses time-based (row-order) split following the FedAds paper:
    "The last week of data is used as the test set" (Wei et al., SIGIR 2023).
    Since timestamps are hashed in the public dataset, we use row order as a
    proxy, assuming the CSV preserves chronological ordering.
    
    Args:
        n_samples: Number of samples to load. None = all aligned data (~2.6M).
        val_ratio: Fraction for validation set. Default 0.115 matches paper
                   (~1.3M test / 11.3M total).
        party: Which party's features to include.
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load aligned data (both parties have features)
    df = load_fedads_data(aligned=True, n_samples=n_samples)
    
    n_total = len(df)
    n_val = int(n_total * val_ratio)
    
    # Row-order split: last portion = validation (time-based proxy)
    print(f"\nUsing TIME-BASED split (row-order proxy)")
    print(f"  Rationale: paper splits by click timestamp; public data has hashed")
    print(f"  timestamps so we use row order as proxy for chronological order.")
    train_df = df[:n_total - n_val]
    val_df = df[n_total - n_val:]
    
    print(f"Train set: {len(train_df):,} samples")
    print(f"Val set: {len(val_df):,} samples")
    
    train_dataset = FedAdsDataset(train_df, party=party)
    val_dataset = FedAdsDataset(val_df, party=party)
    
    return train_dataset, val_dataset


def create_federated_dataloaders(
    train_dataset: FedAdsDataset,
    val_dataset: FedAdsDataset,
    batch_size: int = 4096,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for federated training.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers (0 recommended on Windows)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing FedAds data loader...")
    
    # Test JSON parsing
    test_values = ["['abc123']", "['0']", "['1']", "['hash_value_xyz']"]
    print("\nJSON parsing test:")
    for v in test_values:
        print(f"  {v} -> {parse_json_value(v)}")
    
    print("\n" + "=" * 50)
    print("Loading sample data...")
    
    try:
        train_ds, val_ds = load_fedads_split(n_samples=10_000, party='both')
        print(f"\nTrain statistics: {train_ds.get_statistics()}")
        print(f"Val statistics: {val_ds.get_statistics()}")
        
        # Test a batch
        sample = train_ds[0]
        print(f"\nSample item:")
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")
        
        # Test different parties
        print("\n" + "=" * 50)
        for party in ['local', 'federated']:
            ds = FedAdsDataset(train_ds.df, party=party)
            print(f"\nParty '{party}': {ds.get_statistics()}")
            
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please ensure FedAds data is downloaded to the Dataset folder.")
