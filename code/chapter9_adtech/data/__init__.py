"""
Chapter 9: Data Loading Utilities for Adtech Datasets

Usage:
    
    # For Ali-CCP (ESMM training):
    # First run preprocessing ONCE: python scripts/preprocess_ali_ccp.py --pct 5
    from data.ali_ccp_dataset import load_ali_ccp_loaders
    train_loader, val_loader, metadata = load_ali_ccp_loaders()
    
    # For FedAds (Federated Learning):
    from data.fedads_loader import load_fedads_data
    train_aligned, train_unaligned, val_aligned = load_fedads_data()
"""

# Ali-CCP loader (requires preprocessing - see scripts/preprocess_ali_ccp.py)
from .ali_ccp_dataset import (
    AliCCPDataset, 
    load_ali_ccp_data,
    load_ali_ccp_loaders,
    create_data_loaders,
)

# FedAds loader
from .fedads_loader import FedAdsDataset, load_fedads_data

__all__ = [
    # Ali-CCP
    'AliCCPDataset',
    'load_ali_ccp_data',
    'load_ali_ccp_loaders',
    'create_data_loaders',
    
    # FedAds
    'FedAdsDataset', 
    'load_fedads_data',
]
