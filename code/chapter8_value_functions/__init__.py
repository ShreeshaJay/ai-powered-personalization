"""
Chapter 8: Value Functions and Diversity Optimization

This module implements the Ordering Stage of a recommender system:
1. Multi-objective value functions - blending multiple model outputs
2. MMR diversity optimization - using embeddings to avoid homogeneity  
3. Business rules - artist pacing, slotting, and page construction

Prerequisites:
- Chapter 7 multi-task model artifacts
- Yambda audio embeddings (embeddings.parquet)
- Artist/album mappings (artist_item_mapping.parquet)

Author: Chapter 8 - Value Functions
"""

from .value_function import (
    ValueFunction, 
    ValueFunctionConfig,
    CalibrationFactors,
    compute_calibrated_weights,
)
from .mmr_diversity import MMRReranker, DiversityConfig
from .business_rules import ArtistPacer, SlotAllocator
from .pipeline import OrderingPipeline

__all__ = [
    'ValueFunction',
    'ValueFunctionConfig',
    'CalibrationFactors',
    'compute_calibrated_weights',
    'MMRReranker',
    'DiversityConfig',
    'ArtistPacer',
    'SlotAllocator',
    'OrderingPipeline',
]

