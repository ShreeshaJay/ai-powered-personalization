"""
Bipartite Graph Construction for LightGCN (Section 11.3)

Builds the user-item bipartite interaction graph from MIND user behaviors,
computes the symmetrically normalized adjacency matrix, and provides it as
a PyTorch sparse tensor for efficient GCN propagation.

MIND-only:  Amazon KDD has no persistent user IDs (anonymous sessions),
so a bipartite graph cannot be constructed.

Graph Structure:
    Nodes:  ~40K training users  +  ~51K news articles  =  ~91K total
    Edges:  user u clicked article i  ⟹  undirected edge (u, i)

    IMPORTANT: No content features are used here — no category, title,
    abstract, or SBERT embeddings.  The graph captures ONLY behavioral
    signal: "who clicked what."  LightGCN learns embeddings purely from
    this structural information.

Adjacency Matrix (bipartite, symmetric):

    A = | 0    R   |   where R is (num_users × num_items) binary interaction
        | R^T  0   |   matrix, and A is (N × N) with N = num_users + num_items.

    The upper-right block R encodes user-to-item edges.
    The lower-left block R^T encodes item-to-user edges (same edges, reversed).
    The diagonal blocks are zero (no user-user or item-item edges in base LightGCN).

Normalization (symmetric):

    A_norm = D^{-1/2} A D^{-1/2}
    where D is the diagonal degree matrix.

    This normalizes each edge by the geometric mean of its endpoint degrees:
        a_{ij} = 1 / sqrt(degree(i) * degree(j))

    Intuition: a user who reads 100 articles sends weaker signal per edge
    than a user who reads 10.  Similarly, a popular article clicked by 1000
    users contributes less per edge than a niche article clicked by 10.

This normalized adjacency is used directly in LightGCN's propagation:
    E^{(k+1)} = A_norm @ E^{(k)}
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_bipartite_graph(
    mind_config,
    eval_config,
    exclude_eval_users: bool = True,
) -> Dict:
    """Build bipartite user-item graph from MIND behavior data.

    Walks through every non-eval user's timeline, collects all articles
    they ever clicked (across ALL impressions/page-loads), and creates
    one undirected edge per (user, article) pair.

    No content features (category, title, SBERT) are used — only the
    binary "user clicked article" signal.  The valid_news filter on
    line 118 is purely to ensure the article exists in our item catalog
    (has an embedding from Chapter 10), NOT to use the embedding content.

    Args:
        mind_config: MINDUserConfig with data paths.
        eval_config: EvaluationConfig with eval user count and seed.
        exclude_eval_users: If True, exclude eval users from graph (default).
            Eval users are held out so we can test cold-start retrieval.

    Returns:
        Dict with keys:
            - 'user_item_edges': List[(user_idx, item_idx)] — each pair is
              one edge in the bipartite graph
            - 'num_users': int — count of training users (eval excluded)
            - 'num_items': int — count of articles with valid embeddings
            - 'train_user_id_to_idx': Dict[str, int] — maps user_id string
              (e.g., "U12345") to a 0-based integer index for nn.Embedding
            - 'item_id_to_idx': Dict[str, int] — maps article_id string
              (e.g., "N67890") to a 0-based integer index (same mapping used
              across all sections 11.1/11.2/11.3 for consistency)
            - 'item_ids': List[str] — ordered list of article ID strings
            - 'eval_user_ids': Set[str] — user IDs held out for evaluation
            - 'num_edges': int — total edges in graph
            - 'user_positive_items': Dict[int, Set[int]] — for each user_idx,
              the set of item_idx values they clicked (used for BPR negative
              sampling: we sample negatives NOT in this set)
    """
    from .mind_user_loader import MINDUserDataset

    logger.info("Building bipartite user-item graph from MIND...")
    t0 = time.time()

    # ---------------------------------------------------------------
    # Load the MIND dataset (same pattern as train_sequence.py)
    # This loads: article embeddings, news metadata, user behaviors
    # ---------------------------------------------------------------
    dataset = MINDUserDataset(
        data_dir=mind_config.data_dir,
        news_file=mind_config.news_file,
        behaviors_file=mind_config.behaviors_file,
        min_history_length=mind_config.min_history_length,
        item_embedding_file=mind_config.item_embedding_file,
        item_embedding_dim=mind_config.item_embedding_dim,
    )
    dataset.load()

    # item_id_to_idx: maps article string ID → 0-based integer index.
    # Example: {"N12345": 0, "N67890": 1, "N11111": 2, ...}
    # This is created by MINDUserDataset when loading SBERT embeddings
    # from Chapter 10, and is reused across ALL sections for consistency.
    item_id_to_idx = dataset.item_id_to_idx
    item_ids = dataset.item_ids

    # valid_news: set of article IDs that have SBERT embeddings.
    # Used ONLY as a filter — we discard articles not in our catalog.
    # LightGCN does NOT use the SBERT embedding values themselves.
    valid_news = dataset._valid_news_ids
    num_items = len(item_ids)

    # ---------------------------------------------------------------
    # Identify and exclude evaluation users (same seed=42 as 11.1/11.2)
    # These 10K users are held out for cold-start evaluation.
    # They will NOT appear as nodes in the training graph.
    # ---------------------------------------------------------------
    eval_user_ids: Set[str] = set()
    if exclude_eval_users:
        eval_users = dataset.get_evaluation_users(
            max_users=eval_config.max_eval_users_mind,
            seed=eval_config.random_seed,
        )
        eval_user_ids = set(u["user_id"] for u in eval_users)
        logger.info(f"  Excluding {len(eval_user_ids):,} eval users from graph")

    # ---------------------------------------------------------------
    # Build user timelines: Dict[user_id → List[impression_dicts]]
    # Each impression dict has:
    #   - "timestamp": datetime of the page load
    #   - "history_list": List[str] — cumulative click history BEFORE
    #     this page load (articles the user had clicked in the past)
    #   - "clicked_articles": List[str] — articles clicked DURING this
    #     page load (a user can click multiple articles per page load;
    #     in MIND, an "impression" shows ~20 headlines and the user
    #     may click 0, 1, or several)
    # ---------------------------------------------------------------
    user_timelines = dataset._build_user_timelines()

    # ---------------------------------------------------------------
    # Walk through each training user's timeline and collect all
    # articles they ever interacted with → create graph edges
    # ---------------------------------------------------------------
    train_user_id_to_idx: Dict[str, int] = {}
    user_item_edges: List[Tuple[int, int]] = []
    user_positive_items: Dict[int, Set[int]] = {}

    for uid, timeline in user_timelines.items():
        # Skip eval users — they're held out for cold-start testing
        if uid in eval_user_ids:
            continue

        # Collect ALL articles this user clicked across ALL impressions.
        # We merge history_list (past clicks) and clicked_articles (current
        # clicks) into one set.  Using a set deduplicates automatically:
        # if an article appears in both history and current clicks, or
        # across multiple impressions, it counts as one edge.
        #
        # NOTE: LightGCN's graph is STATIC and UNORDERED — we deliberately
        # discard temporal ordering here.  The graph only knows "user X
        # clicked article Y", not "when" or "in what order."
        user_articles: Set[str] = set()
        for impression in timeline:
            # history_list: articles the user had clicked BEFORE this page load
            if impression["history_list"]:
                user_articles.update(impression["history_list"])
            # clicked_articles: articles clicked DURING this page load
            if impression["clicked_articles"]:
                user_articles.update(impression["clicked_articles"])

        # Keep only articles that exist in our item catalog (have SBERT
        # embeddings from Chapter 10).  This is a catalog consistency
        # filter, NOT using the embedding values — LightGCN learns its
        # own embeddings from scratch via nn.Embedding.
        valid_articles = [nid for nid in user_articles if nid in valid_news]

        # Skip users with too few articles (same threshold as other sections)
        if len(valid_articles) < mind_config.min_history_length:
            continue

        # Assign a sequential 0-based integer index to this user.
        # This index will be used for nn.Embedding lookup in LightGCN.
        user_idx = len(train_user_id_to_idx)
        train_user_id_to_idx[uid] = user_idx

        # Create one edge per (user, article) pair.
        # item_indices tracks the set for BPR negative sampling later:
        # when training, we need to know which items are "positive" for
        # each user so we can sample negatives NOT in this set.
        item_indices: Set[int] = set()
        for nid in valid_articles:
            item_idx = item_id_to_idx[nid]
            item_indices.add(item_idx)
            user_item_edges.append((user_idx, item_idx))

        user_positive_items[user_idx] = item_indices

    num_users = len(train_user_id_to_idx)
    num_edges = len(user_item_edges)
    elapsed = time.time() - t0

    logger.info(f"  Graph construction complete ({elapsed:.1f}s):")
    logger.info(f"    Training users: {num_users:,}")
    logger.info(f"    Items (articles): {num_items:,}")
    logger.info(f"    Edges: {num_edges:,}")
    logger.info(f"    Avg edges/user: {num_edges / max(num_users, 1):.1f}")

    return {
        "user_item_edges": user_item_edges,
        "num_users": num_users,
        "num_items": num_items,
        "train_user_id_to_idx": train_user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "item_ids": item_ids,
        "eval_user_ids": eval_user_ids,
        "num_edges": num_edges,
        "user_positive_items": user_positive_items,
    }


def build_normalized_adjacency(
    user_item_edges: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
) -> torch.sparse.FloatTensor:
    """Build symmetrically normalized bipartite adjacency matrix.

    Constructs the full (N x N) adjacency matrix where N = num_users + num_items:

        A = | 0    R   |
            | R^T  0   |

    Then applies symmetric normalization:  A_norm = D^{-1/2} A D^{-1/2}

    Node indexing in the combined matrix:
        Users:  indices 0, 1, ..., num_users - 1
        Items:  indices num_users, num_users + 1, ..., N - 1

        So item with item_idx=0 gets global index = num_users + 0,
        item with item_idx=5 gets global index = num_users + 5, etc.

    Why symmetric normalization?
        Without normalization, high-degree nodes (popular articles, heavy
        readers) dominate the propagation.  D^{-1/2} A D^{-1/2} ensures
        each edge contributes inversely proportional to the geometric mean
        of its endpoint degrees.  This is the standard GCN normalization
        (Kipf & Welling, 2017) adopted by LightGCN.

    Args:
        user_item_edges: List of (user_idx, item_idx) tuples.
            user_idx is 0-based among users, item_idx is 0-based among items.
        num_users: Number of user nodes.
        num_items: Number of item nodes.

    Returns:
        Sparse FloatTensor of shape (N, N), coalesced.
        Can be moved to GPU with .to(device) and used with torch.sparse.mm().
    """
    N = num_users + num_items
    logger.info(f"  Building normalized adjacency: {N:,} nodes, {len(user_item_edges):,} edges...")

    # ---------------------------------------------------------------
    # Build symmetric edge index: for each (user, item) edge,
    # add BOTH directions to make the adjacency matrix symmetric.
    # ---------------------------------------------------------------
    row_indices = []
    col_indices = []

    for (u, i) in user_item_edges:
        # Convert item's local index to its global index in the
        # combined (users + items) node space
        item_global = num_users + i

        # User → Item edge (upper-right block R of the adjacency)
        row_indices.append(u)
        col_indices.append(item_global)

        # Item → User edge (lower-left block R^T — makes A symmetric)
        row_indices.append(item_global)
        col_indices.append(u)

    row = torch.tensor(row_indices, dtype=torch.long)
    col = torch.tensor(col_indices, dtype=torch.long)

    # ---------------------------------------------------------------
    # Compute degree for each node: degree(i) = number of edges on node i.
    # scatter_add_ counts how many times each node appears as a source.
    # Since the graph is symmetric, counting sources = counting all edges.
    # ---------------------------------------------------------------
    degree = torch.zeros(N, dtype=torch.float32)
    degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))

    # ---------------------------------------------------------------
    # D^{-1/2}: inverse square root of degree for normalization.
    # Nodes with degree 0 (isolated nodes) get d_inv_sqrt = 0 to avoid
    # division by zero.  These nodes won't participate in propagation.
    # ---------------------------------------------------------------
    d_inv_sqrt = torch.zeros(N, dtype=torch.float32)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / degree[mask].sqrt()

    # ---------------------------------------------------------------
    # Normalize each edge value:
    #   a_{ij} = 1 / sqrt(degree(i)) * 1 / sqrt(degree(j))
    #          = d_inv_sqrt[i] * d_inv_sqrt[j]
    #
    # Example: if user has degree 10 and item has degree 100,
    # edge value = 1/sqrt(10) * 1/sqrt(100) = 0.316 * 0.1 = 0.0316
    # Compare to equal-degree case: degree 10 each →
    # 1/sqrt(10) * 1/sqrt(10) = 0.1 (stronger signal)
    # ---------------------------------------------------------------
    values = d_inv_sqrt[row] * d_inv_sqrt[col]

    # ---------------------------------------------------------------
    # Construct sparse tensor in COO format and coalesce.
    # COO = Coordinate format: store (row, col, value) triples.
    # coalesce() merges duplicate entries (shouldn't have any here,
    # but ensures canonical form for torch.sparse.mm).
    # ---------------------------------------------------------------
    indices = torch.stack([row, col], dim=0)
    adj_norm = torch.sparse_coo_tensor(indices, values, size=(N, N))
    adj_norm = adj_norm.coalesce()

    logger.info(f"  Adjacency matrix: shape=({N}, {N}), nnz={adj_norm._nnz():,}")

    return adj_norm


def get_graph_data(
    mind_config=None,
    eval_config=None,
) -> Dict:
    """Convenience wrapper: build graph + normalized adjacency in one call.

    Returns dict with all graph data including 'adj_norm' sparse tensor.
    """
    # Import defaults lazily to avoid circular imports at module level
    if mind_config is None:
        from config import DEFAULT_MIND_USER_CONFIG
        mind_config = DEFAULT_MIND_USER_CONFIG
    if eval_config is None:
        from config import DEFAULT_EVALUATION_CONFIG
        eval_config = DEFAULT_EVALUATION_CONFIG

    graph_data = build_bipartite_graph(mind_config, eval_config)

    adj_norm = build_normalized_adjacency(
        graph_data["user_item_edges"],
        graph_data["num_users"],
        graph_data["num_items"],
    )
    graph_data["adj_norm"] = adj_norm

    return graph_data


if __name__ == "__main__":
    """Quick test of graph construction."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DEFAULT_MIND_USER_CONFIG, DEFAULT_EVALUATION_CONFIG

    graph_data = get_graph_data(DEFAULT_MIND_USER_CONFIG, DEFAULT_EVALUATION_CONFIG)

    print(f"\n{'=' * 60}")
    print(f"MIND Bipartite Graph Summary")
    print(f"{'=' * 60}")
    print(f"  Training users:  {graph_data['num_users']:>10,}")
    print(f"  Items:           {graph_data['num_items']:>10,}")
    print(f"  Edges:           {graph_data['num_edges']:>10,}")
    print(f"  Eval users:      {len(graph_data['eval_user_ids']):>10,}")
    print(f"  Adj shape:       {tuple(graph_data['adj_norm'].shape)}")
    print(f"  Adj nnz:         {graph_data['adj_norm']._nnz():>10,}")
    print(f"{'=' * 60}\n")
