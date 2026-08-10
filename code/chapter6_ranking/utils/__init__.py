"""
Chapter 6: Utility Modules
==========================
Optional utilities for advanced use cases.

- memory_efficient: Memory-optimized feature processing for large datasets (pandas)
- polars_pipeline: Polars-based pipeline for memory-constrained environments
"""

from .memory_efficient import (
    optimize_dtypes,
    downcast_numeric,
    memory_usage_report,
    estimate_memory_for_rows,
    MemoryEfficientHistoricalFeatures,
    MemoryEfficientPipeline,
    process_in_chunks,
    OPTIMAL_DTYPES,
)

from .polars_pipeline import (
    load_yambda_polars,
    PolarsPipeline,
    PolarsFeatureEncoder,
    PolarsHistoricalFeatures,
)

__all__ = [
    # Memory efficient (pandas)
    'optimize_dtypes',
    'downcast_numeric', 
    'memory_usage_report',
    'estimate_memory_for_rows',
    'MemoryEfficientHistoricalFeatures',
    'MemoryEfficientPipeline',
    'process_in_chunks',
    'OPTIMAL_DTYPES',
    # Polars
    'load_yambda_polars',
    'PolarsPipeline',
    'PolarsFeatureEncoder',
    'PolarsHistoricalFeatures',
]

