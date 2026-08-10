"""
Chapter 7: Advanced Ranking Models
==================================
Sophisticated neural ranking architectures for recommender systems.

Models implemented:
- DCN-V2: Deep & Cross Network V2 (Google, 2021)
- DLRM: Deep Learning Recommendation Model (Facebook, 2019)
- MMoE: Multi-gate Mixture-of-Experts (Google, 2018)

These models build on the foundation from Chapter 6 (XGBoost, DeepFM)
with more advanced feature interaction and multi-task learning capabilities.
"""

# Single-Task Models
from .dcn import DCNV2, DCNV2Config, DCNV2Dataset, collate_fn as dcn_collate_fn, get_yambda_dcnv2_config
from .dlrm import DLRM, DLRMConfig, DLRMDataset, collate_fn as dlrm_collate_fn, get_yambda_dlrm_config

# Multi-Task Models
from .mmoe import (
    MMoE, 
    MMoEConfig, 
    MMoEDataset, 
    MultiTaskLoss, 
    FocalLoss,
    collate_fn as mmoe_collate_fn,
    get_yambda_mmoe_config,
)

__all__ = [
    # DCN-V2 (Single-Task)
    'DCNV2',
    'DCNV2Config',
    'DCNV2Dataset',
    'dcn_collate_fn',
    'get_yambda_dcnv2_config',
    
    # DLRM (Single-Task)
    'DLRM',
    'DLRMConfig',
    'DLRMDataset',
    'dlrm_collate_fn',
    'get_yambda_dlrm_config',
    
    # MMoE (Multi-Task)
    'MMoE',
    'MMoEConfig',
    'MMoEDataset',
    'MultiTaskLoss',
    'FocalLoss',
    'mmoe_collate_fn',
    'get_yambda_mmoe_config',
]

