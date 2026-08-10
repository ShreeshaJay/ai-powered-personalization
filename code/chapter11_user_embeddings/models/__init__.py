"""
Models for Chapter 11: User/Customer Embeddings.

Section 11.1: Aggregation-based user embeddings (no training).
Section 11.2: Sequence models (SASRec / GRU4Rec) with frozen SBERT embeddings.
"""

from .aggregators import (
    UserEmbeddingAggregator,
    SimpleMeanAggregator,
    LastKMeanAggregator,
    ExponentialDecayAggregator,
    TFIDFWeightedAggregator,
    TFIDFRecencyAggregator,
)

from .sequence_models import (
    SASRec,
    GRU4Rec,
    create_sequence_model,
    count_parameters,
)

from .sequence_models_itemid import (
    SASRecItemID,
    GRU4RecItemID,
    create_itemid_model,
)

from .lightgcn import (
    LightGCN,
    create_lightgcn,
)

__all__ = [
    # Section 11.1
    "UserEmbeddingAggregator",
    "SimpleMeanAggregator",
    "LastKMeanAggregator",
    "ExponentialDecayAggregator",
    "TFIDFWeightedAggregator",
    "TFIDFRecencyAggregator",
    # Section 11.2 (frozen embeddings)
    "SASRec",
    "GRU4Rec",
    "create_sequence_model",
    "count_parameters",
    # Section 11.2 (learnable Item IDs)
    "SASRecItemID",
    "GRU4RecItemID",
    "create_itemid_model",
    # Section 11.3 (LightGCN)
    "LightGCN",
    "create_lightgcn",
]
