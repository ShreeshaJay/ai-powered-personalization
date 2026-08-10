"""
Data loaders for Chapter 11: User/Customer Embeddings.

Provides dataset loaders that combine raw interaction data with cached item
embeddings from Chapter 10 to produce user-level (or session-level) data
ready for aggregation and evaluation.

Section 11.3 adds bipartite graph construction for LightGCN (MIND-only).
"""

from .amazon_session_loader import AmazonSessionDataset
from .mind_user_loader import MINDUserDataset
from .mind_graph_builder import get_graph_data, build_bipartite_graph

__all__ = [
    "AmazonSessionDataset",
    "MINDUserDataset",
    "get_graph_data",
    "build_bipartite_graph",
]
