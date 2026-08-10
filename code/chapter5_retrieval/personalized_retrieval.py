"""
Chapter 5: Retrieval - Personalized Two-Tower Model
====================================================

Use Case: Personalized Product Recommendations
----------------------------------------------
Given a user's browsing history, recommend items they might be interested in.

Architecture:
- User Tower: Average of item embeddings from user's interaction history
- Item Tower: Pre-computed description embeddings (same as similar items model)

This is a simple but effective baseline for personalized retrieval:
- No learned parameters (unsupervised)
- User representation = mean pooling of interacted item embeddings
- Captures user's average preference across categories/styles

Evaluation Strategy:
- For each session with 3+ interactions:
  - First N-1 items = user history (build user embedding)
  - Last item = ground truth (what user engaged with next)
- This mimics real-time personalization: given history, predict next item

Author: Shreesha Jagadeesh
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
from collections import defaultdict

# Efficient DataFrame processing with lazy evaluation
import polars as pl

# For compatibility with visualization
import pandas as pd

# Vector search
import faiss
from sklearn.preprocessing import normalize

# Visualization
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - Update these to match your local setup
BASE_PATH = Path('path/to/Dataset/SIGIR-ecom-data-challenge')

# Toggle between sampled (local dev) and full data (final results)
USE_SAMPLED_DATA = True

if USE_SAMPLED_DATA:
    DATA_PATH = BASE_PATH / 'sampled'
    BROWSING_FILE = 'browsing_train_sampled.csv'
else:
    DATA_PATH = BASE_PATH / 'train'
    BROWSING_FILE = 'browsing_train.csv'

# Model parameters
TOP_K_RETRIEVAL = [5, 10, 20, 50]
MIN_SESSION_LENGTH = 3  # Need at least 3 items: 2 for history, 1 for ground truth

# Ground truth configuration
# Using all meaningful interactions (excluding 'remove')
GROUND_TRUTH_ACTIONS = ['detail', 'add', 'purchase']


# =============================================================================
# DATA LOADING (Reusing efficient Polars-based functions)
# =============================================================================

def load_product_catalog(data_path: Path) -> pl.DataFrame:
    """Load product catalog with embeddings using Polars."""
    print("Loading product catalog...")
    df = pl.read_csv(data_path / 'sku_to_content.csv')
    print(f"  Total products: {len(df):,}")
    print(f"  Memory usage: {df.estimated_size('mb'):.2f} MB")
    return df


def load_browsing_sessions_lazy(data_path: Path, browsing_file: str) -> pl.LazyFrame:
    """Load browsing sessions in lazy mode."""
    print("\nLoading browsing sessions (lazy mode)...")
    lf = pl.scan_csv(data_path / browsing_file)
    return lf


def parse_embeddings_batch(embedding_series: pl.Series) -> Tuple[np.ndarray, int]:
    """Parse embeddings in batch - returns 2D numpy array."""
    embeddings = []
    embedding_dim = None
    
    for emb_str in embedding_series.to_list():
        if emb_str is None:
            if embedding_dim is not None:
                embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            else:
                embeddings.append(None)
        else:
            try:
                if emb_str.startswith('['):
                    emb = np.array(json.loads(emb_str), dtype=np.float32)
                else:
                    emb = np.array([float(x) for x in emb_str.split()], dtype=np.float32)
                
                if embedding_dim is None:
                    embedding_dim = len(emb)
                embeddings.append(emb)
            except:
                if embedding_dim is not None:
                    embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
                else:
                    embeddings.append(None)
    
    if embedding_dim is not None:
        embeddings = [e if e is not None else np.zeros(embedding_dim, dtype=np.float32) 
                      for e in embeddings]
        return np.vstack(embeddings), embedding_dim
    
    return None, None


# =============================================================================
# BUILD USER SESSIONS FOR PERSONALIZED EVALUATION
# =============================================================================

def build_user_sessions(
    browsing_lf: pl.LazyFrame,
    min_session_length: int = 3,
    ground_truth_actions: List[str] = None
) -> List[Dict]:
    """
    Build user sessions for personalized retrieval evaluation.
    
    For each session with min_session_length+ items:
    - history_items: First N-1 items (to build user embedding)
    - ground_truth_item: Last item (what user engaged with next)
    
    This evaluation setup mimics real-time personalization:
    "Given what the user has browsed so far, can we predict their next interest?"
    
    Args:
        browsing_lf: Polars LazyFrame of browsing events
        min_session_length: Minimum items in session (need at least 3: 2 history + 1 GT)
        ground_truth_actions: Product actions to include
        
    Returns:
        List of dicts with 'session_id', 'history_items', 'ground_truth_item'
    """
    if ground_truth_actions is None:
        ground_truth_actions = ['detail', 'add', 'purchase']
    
    print("\nBuilding user sessions for personalized evaluation...")
    print(f"  Ground truth actions: {ground_truth_actions}")
    print(f"  Minimum session length: {min_session_length}")
    
    # Build query
    session_query = (
        browsing_lf
        .filter(
            (pl.col('product_sku_hash').is_not_null()) & 
            (pl.col('product_action').is_in(ground_truth_actions))
        )
        .select(['session_id_hash', 'product_sku_hash', 'server_timestamp_epoch_ms'])
        .sort(['session_id_hash', 'server_timestamp_epoch_ms'])
        .group_by('session_id_hash')
        .agg([
            pl.col('product_sku_hash').alias('products'),
            pl.len().alias('session_length')
        ])
        .filter(pl.col('session_length') >= min_session_length)
    )
    
    print("  Executing query...")
    session_df = session_query.collect(engine="streaming")
    print(f"  Sessions with {min_session_length}+ products: {len(session_df):,}")
    
    # Build evaluation samples
    user_sessions = []
    
    for row in tqdm(session_df.iter_rows(named=True), 
                    total=len(session_df),
                    desc="  Processing sessions"):
        products = row['products']
        
        # Remove consecutive duplicates
        unique_products = []
        for p in products:
            if not unique_products or p != unique_products[-1]:
                unique_products.append(p)
        
        if len(unique_products) >= min_session_length:
            user_sessions.append({
                'session_id': row['session_id_hash'],
                'history_items': unique_products[:-1],  # All but last
                'ground_truth_item': unique_products[-1]  # Last item
            })
    
    print(f"  Valid evaluation sessions: {len(user_sessions):,}")
    
    # Statistics
    history_lengths = [len(s['history_items']) for s in user_sessions]
    print(f"  History lengths: min={min(history_lengths)}, "
          f"median={np.median(history_lengths):.0f}, max={max(history_lengths)}")
    
    return user_sessions


# =============================================================================
# ITEM EMBEDDING INDEX (Same as similar items model)
# =============================================================================

class ItemEmbeddingIndex:
    """
    Item tower: Index of all item embeddings for retrieval.
    Uses description embeddings only (based on ablation study results).
    """
    
    def __init__(self):
        self.sku_to_idx = {}
        self.idx_to_sku = {}
        self.embeddings = None
        self.embedding_dim = None
        self.index = None
        
    def build(self, catalog_df: pl.DataFrame) -> 'ItemEmbeddingIndex':
        """Build item embeddings and FAISS index."""
        print("\nBuilding item embedding index...")
        
        # Get SKUs
        skus = catalog_df['product_sku_hash'].to_list()
        
        # Parse description embeddings
        print("  Parsing description embeddings...")
        desc_series = catalog_df['description_vector']
        self.embeddings, self.embedding_dim = parse_embeddings_batch(desc_series)
        
        if self.embeddings is None:
            raise ValueError("No valid embeddings found")
        
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        
        # Normalize for cosine similarity
        print("  Normalizing embeddings (L2)...")
        self.embeddings = normalize(self.embeddings, norm='l2').astype(np.float32)
        
        # Build mappings
        self.sku_to_idx = {sku: idx for idx, sku in enumerate(skus)}
        self.idx_to_sku = {idx: sku for idx, sku in enumerate(skus)}
        
        # Build FAISS index
        print("  Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)
        
        print(f"  Items indexed: {self.index.ntotal:,}")
        
        return self
    
    def get_embedding(self, sku: str) -> Optional[np.ndarray]:
        """Get embedding for a single SKU."""
        if sku not in self.sku_to_idx:
            return None
        return self.embeddings[self.sku_to_idx[sku]]
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search for top-K similar items given a query embedding."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:  # Valid index
                sku = self.idx_to_sku.get(idx)
                if sku:
                    results.append((sku, float(score)))
        
        return results


# =============================================================================
# USER TOWER: Average Pooling of History
# =============================================================================

class UserTower:
    """
    User tower: Builds user representations from interaction history.
    
    Representation: Mean pooling of item embeddings from user's history.
    
    This is a simple but effective baseline:
    - No learned parameters
    - Captures user's "average" preference
    - Works well when user has diverse history
    
    Limitations:
    - Doesn't capture sequential patterns
    - Recent items weighted same as older items
    - Can be diluted by long histories
    
    Extensions (for later chapters):
    - Weighted average (recency weighting)
    - Attention pooling
    - Sequential models (GRU, Transformer)
    """
    
    def __init__(self, item_index: ItemEmbeddingIndex):
        self.item_index = item_index
        
    def get_user_embedding(
        self, 
        history_items: List[str],
        aggregation: str = 'mean'
    ) -> Optional[np.ndarray]:
        """
        Build user embedding from interaction history.
        
        Args:
            history_items: List of SKUs the user has interacted with
            aggregation: How to combine item embeddings ('mean', 'sum', 'last')
            
        Returns:
            User embedding vector or None if no valid items
        """
        # Get embeddings for all history items
        embeddings = []
        for sku in history_items:
            emb = self.item_index.get_embedding(sku)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        embeddings = np.vstack(embeddings)
        
        # Aggregate
        if aggregation == 'mean':
            user_emb = np.mean(embeddings, axis=0)
        elif aggregation == 'sum':
            user_emb = np.sum(embeddings, axis=0)
        elif aggregation == 'last':
            user_emb = embeddings[-1]  # Most recent item
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Normalize for cosine similarity
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        return user_emb.astype(np.float32)


# =============================================================================
# PERSONALIZED RETRIEVER
# =============================================================================

class PersonalizedRetriever:
    """
    Two-tower personalized retrieval system.
    
    Query: User embedding (from history)
    Candidates: All items in catalog
    Similarity: Cosine similarity (dot product of normalized vectors)
    """
    
    def __init__(self, item_index: ItemEmbeddingIndex, user_tower: UserTower):
        self.item_index = item_index
        self.user_tower = user_tower
        
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True,
        aggregation: str = 'mean'
    ) -> List[Tuple[str, float]]:
        """
        Retrieve personalized recommendations for a user.
        
        Args:
            history_items: User's interaction history (SKUs)
            top_k: Number of items to retrieve
            exclude_history: Whether to exclude already-interacted items
            aggregation: How to build user embedding
            
        Returns:
            List of (sku, score) tuples
        """
        # Build user embedding
        user_emb = self.user_tower.get_user_embedding(history_items, aggregation)
        if user_emb is None:
            return []
        
        # Retrieve candidates (get extra if we need to filter)
        k_search = top_k + len(history_items) if exclude_history else top_k
        candidates = self.item_index.search(user_emb, k_search)
        
        # Filter out history items if requested
        if exclude_history:
            history_set = set(history_items)
            candidates = [(sku, score) for sku, score in candidates 
                         if sku not in history_set]
        
        return candidates[:top_k]


# =============================================================================
# EVALUATION
# =============================================================================

def generate_predictions(
    retriever: PersonalizedRetriever,
    user_sessions: List[Dict],
    top_k: int = 50,
    max_users: int = None,
    aggregation: str = 'mean'
) -> Dict[str, List[str]]:
    """
    Generate predictions for evaluation.
    
    Args:
        retriever: PersonalizedRetriever
        user_sessions: List of session dicts with history and ground truth
        top_k: Number of items to retrieve
        max_users: Limit number of users for faster evaluation
        aggregation: User embedding aggregation method
        
    Returns:
        Dict mapping session_id -> list of predicted SKUs
    """
    print(f"\nGenerating predictions (aggregation={aggregation})...")
    
    if max_users:
        user_sessions = user_sessions[:max_users]
    
    predictions = {}
    
    for session in tqdm(user_sessions, desc="  Retrieving"):
        retrieved = retriever.retrieve_for_user(
            session['history_items'],
            top_k=top_k,
            exclude_history=True,
            aggregation=aggregation
        )
        predictions[session['session_id']] = [sku for sku, _ in retrieved]
    
    print(f"  Generated predictions for {len(predictions):,} users")
    return predictions


def evaluate_personalized_retrieval(
    predictions: Dict[str, List[str]],
    user_sessions: List[Dict],
    top_k_values: List[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Evaluate personalized retrieval quality.
    
    For personalized retrieval, ground truth is a SINGLE item (next item in session),
    so we use slightly different metrics than similar items retrieval.
    
    Metrics:
    - Hit Rate@K: Was the ground truth item in top-K predictions?
    - MRR@K: Reciprocal rank of the ground truth item
    """
    print("\nEvaluating personalized retrieval...")
    
    # Build session_id -> ground_truth mapping
    session_to_gt = {s['session_id']: s['ground_truth_item'] for s in user_sessions}
    
    # Filter to sessions with predictions
    valid_sessions = [s for s in user_sessions if s['session_id'] in predictions]
    print(f"  Evaluating {len(valid_sessions):,} sessions...")
    
    results = {k: {'hits': 0, 'mrr_sum': 0} for k in top_k_values}
    
    for session in tqdm(valid_sessions, desc="  Computing metrics"):
        session_id = session['session_id']
        gt_item = session['ground_truth_item']
        pred_items = predictions.get(session_id, [])
        
        for k in top_k_values:
            top_k_preds = pred_items[:k]
            
            # Hit Rate: Is ground truth in top-K?
            if gt_item in top_k_preds:
                results[k]['hits'] += 1
                
                # MRR: Reciprocal rank
                rank = top_k_preds.index(gt_item) + 1
                results[k]['mrr_sum'] += 1.0 / rank
    
    # Aggregate
    n_sessions = len(valid_sessions)
    metrics_df = pd.DataFrame([
        {
            'K': k,
            'Hit Rate@K': results[k]['hits'] / n_sessions,
            'MRR@K': results[k]['mrr_sum'] / n_sessions
        }
        for k in top_k_values
    ])
    
    print("\n" + "=" * 60)
    print("PERSONALIZED RETRIEVAL RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    
    return metrics_df


# =============================================================================
# BASELINES: Random and Popularity
# =============================================================================

class RandomBaseline:
    """
    Random Baseline: Randomly sample items from catalog.
    
    This is the simplest baseline - if our model can't beat random,
    something is fundamentally wrong.
    """
    
    def __init__(self, all_skus: List[str]):
        self.all_skus = all_skus
        self.all_skus_set = set(all_skus)
        
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Tuple[str, float]]:
        """Randomly sample items, excluding history if requested."""
        if exclude_history:
            candidates = list(self.all_skus_set - set(history_items))
        else:
            candidates = self.all_skus
        
        # Random sample
        k = min(top_k, len(candidates))
        sampled = np.random.choice(candidates, size=k, replace=False)
        
        # Return with dummy scores
        return [(sku, 0.0) for sku in sampled]


class PopularityBaseline:
    """
    Popularity Baseline: Recommend most popular items.
    
    Popularity = total count of (detail + add + purchase) events.
    No personalization - same recommendations for everyone.
    
    This is a strong baseline in practice because:
    - Popular items are popular for a reason
    - Works well for new users (cold start)
    - Often hard to beat without good personalization
    """
    
    def __init__(self, popularity_scores: Dict[str, int], all_skus: List[str]):
        """
        Args:
            popularity_scores: Dict mapping SKU -> engagement count
            all_skus: List of all SKUs in catalog
        """
        self.popularity_scores = popularity_scores
        self.all_skus = all_skus
        
        # Pre-compute sorted list by popularity (descending)
        self.sorted_by_popularity = sorted(
            all_skus,
            key=lambda x: popularity_scores.get(x, 0),
            reverse=True
        )
        
    def retrieve_for_user(
        self,
        history_items: List[str],
        top_k: int = 10,
        exclude_history: bool = True
    ) -> List[Tuple[str, float]]:
        """Return top-K most popular items, excluding history if requested."""
        history_set = set(history_items) if exclude_history else set()
        
        results = []
        for sku in self.sorted_by_popularity:
            if sku not in history_set:
                score = self.popularity_scores.get(sku, 0)
                results.append((sku, float(score)))
                if len(results) >= top_k:
                    break
        
        return results


def compute_popularity_scores(
    browsing_lf: pl.LazyFrame,
    actions: List[str] = None
) -> Dict[str, int]:
    """
    Compute popularity scores for all items.
    
    Popularity = count of engagement events (detail + add + purchase).
    No weighting - each event counts as 1.
    
    Args:
        browsing_lf: Polars LazyFrame of browsing events
        actions: Actions to count (default: detail, add, purchase)
        
    Returns:
        Dict mapping SKU -> total engagement count
    """
    if actions is None:
        actions = ['detail', 'add', 'purchase']
    
    print("\nComputing popularity scores...")
    print(f"  Actions counted: {actions}")
    
    popularity_query = (
        browsing_lf
        .filter(
            (pl.col('product_sku_hash').is_not_null()) &
            (pl.col('product_action').is_in(actions))
        )
        .group_by('product_sku_hash')
        .agg(pl.len().alias('engagement_count'))
    )
    
    popularity_df = popularity_query.collect(engine="streaming")
    
    # Convert to dict
    popularity_scores = {
        row['product_sku_hash']: row['engagement_count']
        for row in popularity_df.iter_rows(named=True)
    }
    
    # Statistics
    counts = list(popularity_scores.values())
    print(f"  Items with engagement: {len(popularity_scores):,}")
    print(f"  Engagement distribution: min={min(counts)}, median={np.median(counts):.0f}, max={max(counts)}")
    
    return popularity_scores


def generate_baseline_predictions(
    baseline,
    user_sessions: List[Dict],
    top_k: int = 50,
    max_users: int = None
) -> Dict[str, List[str]]:
    """Generate predictions from a baseline model."""
    if max_users:
        user_sessions = user_sessions[:max_users]
    
    predictions = {}
    for session in tqdm(user_sessions, desc="  Generating predictions"):
        retrieved = baseline.retrieve_for_user(
            session['history_items'],
            top_k=top_k,
            exclude_history=True
        )
        predictions[session['session_id']] = [sku for sku, _ in retrieved]
    
    return predictions


# =============================================================================
# ABLATION: Compare Aggregation Methods
# =============================================================================

def run_full_comparison(
    retriever: PersonalizedRetriever,
    random_baseline: RandomBaseline,
    popularity_baseline: PopularityBaseline,
    user_sessions: List[Dict],
    top_k_values: List[int] = [5, 10, 20, 50],
    max_users: int = 5000
) -> pd.DataFrame:
    """
    Compare all models: baselines + personalized retrieval variants.
    
    Models compared:
    1. Random Baseline: Random sampling from catalog
    2. Popularity Baseline: Most popular items globally
    3. Personalized (mean): User embedding = mean of history
    4. Personalized (last): User embedding = last item only
    """
    print("\n" + "=" * 70)
    print("FULL MODEL COMPARISON")
    print("=" * 70)
    
    all_results = []
    max_k = max(top_k_values)
    sessions_subset = user_sessions[:max_users]
    
    # 1. Random Baseline
    print(f"\n{'-' * 60}")
    print("Model: Random Baseline")
    print(f"{'-' * 60}")
    
    predictions = generate_baseline_predictions(
        random_baseline, sessions_subset, top_k=max_k
    )
    metrics_df = evaluate_personalized_retrieval(
        predictions, sessions_subset, top_k_values
    )
    metrics_df['Model'] = 'Random'
    all_results.append(metrics_df)
    
    # 2. Popularity Baseline
    print(f"\n{'-' * 60}")
    print("Model: Popularity Baseline")
    print(f"{'-' * 60}")
    
    predictions = generate_baseline_predictions(
        popularity_baseline, sessions_subset, top_k=max_k
    )
    metrics_df = evaluate_personalized_retrieval(
        predictions, sessions_subset, top_k_values
    )
    metrics_df['Model'] = 'Popularity'
    all_results.append(metrics_df)
    
    # 3. Personalized (mean aggregation)
    print(f"\n{'-' * 60}")
    print("Model: Personalized (mean)")
    print(f"{'-' * 60}")
    
    predictions = generate_predictions(
        retriever, sessions_subset, top_k=max_k, aggregation='mean'
    )
    metrics_df = evaluate_personalized_retrieval(
        predictions, sessions_subset, top_k_values
    )
    metrics_df['Model'] = 'Personalized (mean)'
    all_results.append(metrics_df)
    
    # 4. Personalized (last item only)
    print(f"\n{'-' * 60}")
    print("Model: Personalized (last)")
    print(f"{'-' * 60}")
    
    predictions = generate_predictions(
        retriever, sessions_subset, top_k=max_k, aggregation='last'
    )
    metrics_df = evaluate_personalized_retrieval(
        predictions, sessions_subset, top_k_values
    )
    metrics_df['Model'] = 'Personalized (last)'
    all_results.append(metrics_df)
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY (K=10)")
    print("=" * 70)
    
    summary = combined_df[combined_df['K'] == 10][['Model', 'Hit Rate@K', 'MRR@K']].copy()
    
    # Calculate lift over random
    random_hr = summary[summary['Model'] == 'Random']['Hit Rate@K'].values[0]
    summary['Lift vs Random'] = summary['Hit Rate@K'].apply(
        lambda x: f"+{(x/random_hr - 1)*100:.1f}%" if x > random_hr else f"{(x/random_hr - 1)*100:.1f}%"
    )
    
    # Format percentages
    summary['Hit Rate@K'] = summary['Hit Rate@K'].apply(lambda x: f"{x*100:.2f}%")
    summary['MRR@K'] = summary['MRR@K'].apply(lambda x: f"{x:.4f}")
    
    print(summary.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print("- Random: Lower bound - if model is worse, something is wrong")
    print("- Popularity: Strong baseline - popular items are often relevant")
    print("- Personalized (mean): Uses full history, may dilute recent intent")
    print("- Personalized (last): Uses only most recent item, captures current intent")
    
    return combined_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_comparison(
    similar_items_metrics: pd.DataFrame,
    personalized_metrics: pd.DataFrame,
    output_path: Path = None
):
    """
    Compare similar items vs personalized retrieval.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hit Rate comparison
    axes[0].plot(similar_items_metrics['K'], similar_items_metrics['Hit Rate@K'], 
                 'o-', label='Similar Items (Item-to-Item)', linewidth=2, markersize=8)
    axes[0].plot(personalized_metrics['K'], personalized_metrics['Hit Rate@K'],
                 's-', label='Personalized (User-to-Item)', linewidth=2, markersize=8)
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Hit Rate@K', fontsize=12)
    axes[0].set_title('Hit Rate: Similar Items vs Personalized', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MRR comparison
    axes[1].plot(similar_items_metrics['K'], similar_items_metrics['MRR@K'],
                 'o-', label='Similar Items', linewidth=2, markersize=8)
    axes[1].plot(personalized_metrics['K'], personalized_metrics['MRR@K'],
                 's-', label='Personalized', linewidth=2, markersize=8)
    axes[1].set_xlabel('K', fontsize=12)
    axes[1].set_ylabel('MRR@K', fontsize=12)
    axes[1].set_title('MRR: Similar Items vs Personalized', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison chart saved to: {output_path}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution for personalized retrieval."""
    print("=" * 70)
    print("Chapter 5: Personalized Two-Tower Retrieval")
    print("=" * 70)
    print("\nModel Architecture:")
    print("  User Tower: Mean pooling of history item embeddings")
    print("  Item Tower: Description embeddings (from ablation study)")
    print("\nBaselines:")
    print("  Random: Sample randomly from catalog")
    print("  Popularity: Recommend most engaged items globally")
    
    # 1. Load data
    catalog_df = load_product_catalog(DATA_PATH)
    browsing_lf = load_browsing_sessions_lazy(DATA_PATH, BROWSING_FILE)
    
    # 2. Build user sessions for evaluation
    user_sessions = build_user_sessions(
        browsing_lf,
        min_session_length=MIN_SESSION_LENGTH,
        ground_truth_actions=GROUND_TRUTH_ACTIONS
    )
    
    # 3. Build item index
    item_index = ItemEmbeddingIndex()
    item_index.build(catalog_df)
    
    # 4. Build user tower and retriever
    user_tower = UserTower(item_index)
    retriever = PersonalizedRetriever(item_index, user_tower)
    
    # 5. Build baselines
    all_skus = catalog_df['product_sku_hash'].to_list()
    
    # Random baseline
    random_baseline = RandomBaseline(all_skus)
    
    # Popularity baseline - need to reload lazy frame for fresh scan
    browsing_lf_pop = load_browsing_sessions_lazy(DATA_PATH, BROWSING_FILE)
    popularity_scores = compute_popularity_scores(browsing_lf_pop, GROUND_TRUTH_ACTIONS)
    popularity_baseline = PopularityBaseline(popularity_scores, all_skus)
    
    # 6. Run full comparison (baselines + personalized models)
    comparison_df = run_full_comparison(
        retriever=retriever,
        random_baseline=random_baseline,
        popularity_baseline=popularity_baseline,
        user_sessions=user_sessions,
        top_k_values=TOP_K_RETRIEVAL,
        max_users=5000
    )
    
    # 7. Save results
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / 'personalized_comparison.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'personalized_comparison.csv'}")
    
    # 8. Show example
    print("\n" + "=" * 60)
    print("EXAMPLE: Personalized Recommendations")
    print("=" * 60)
    
    example_session = user_sessions[0]
    print(f"\nSession: {example_session['session_id']}")
    print(f"History ({len(example_session['history_items'])} items):")
    for i, sku in enumerate(example_session['history_items'][:5], 1):
        item_info = catalog_df.filter(pl.col('product_sku_hash') == sku)
        if len(item_info) > 0:
            print(f"  {i}. {sku[:20]}... (Category: {item_info['category_hash'][0][:30]}...)")
    if len(example_session['history_items']) > 5:
        print(f"  ... and {len(example_session['history_items']) - 5} more")
    
    print(f"\nGround Truth (next item): {example_session['ground_truth_item'][:40]}...")
    
    print("\nTop 5 Recommendations by Model:")
    print("-" * 50)
    
    # Random
    print("\nRandom Baseline:")
    recs = random_baseline.retrieve_for_user(example_session['history_items'], top_k=5)
    for i, (sku, _) in enumerate(recs, 1):
        match = " <-- MATCH!" if sku == example_session['ground_truth_item'] else ""
        print(f"  {i}. {sku[:50]}...{match}")
    
    # Popularity
    print("\nPopularity Baseline:")
    recs = popularity_baseline.retrieve_for_user(example_session['history_items'], top_k=5)
    for i, (sku, score) in enumerate(recs, 1):
        match = " <-- MATCH!" if sku == example_session['ground_truth_item'] else ""
        print(f"  {i}. {sku[:40]}... (engagements: {int(score):,}){match}")
    
    # Personalized
    print("\nPersonalized (mean):")
    recs = retriever.retrieve_for_user(example_session['history_items'], top_k=5, aggregation='mean')
    for i, (sku, score) in enumerate(recs, 1):
        match = " <-- MATCH!" if sku == example_session['ground_truth_item'] else ""
        print(f"  {i}. {sku[:40]}... (similarity: {score:.4f}){match}")
    
    return comparison_df, retriever, user_sessions


if __name__ == '__main__':
    comparison_df, retriever, user_sessions = main()

