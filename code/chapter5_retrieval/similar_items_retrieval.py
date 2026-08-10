"""
Chapter 5: Retrieval - Similar Items Recommendation
====================================================

Use Case: "Items Similar to This" Product Recommendations
---------------------------------------------------------
When a user lands on a product page, we want to show a carousel of similar items.
This is a non-personalized, content-based retrieval problem where:
- Query: The item the user is currently viewing (the "hero" item)
- Candidates: All other items in the catalog
- Retrieval: Find top-K most similar items using nearest neighbor search

Dataset: SIGIR E-commerce Data Challenge (Coveo)
- Pre-computed text embeddings (description_vector) 
- Pre-computed image embeddings (image_vector)
- User browsing sessions for offline evaluation

Evaluation Strategy:
- Sessions with 2+ item interactions provide ground truth
- First item in session = hero item (query)
- Subsequent items = ground truth similar items
- Metric: If we retrieve similar items, do they match what users actually browsed next?

Author: Shreesha Jagadeesh
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Generator
import json
from tqdm import tqdm

# Efficient DataFrame processing with lazy evaluation
import polars as pl

# For compatibility with visualization
import pandas as pd

# Vector search
import faiss
from sklearn.preprocessing import normalize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - Update these to match your local setup
BASE_PATH = Path('path/to/Dataset/SIGIR-ecom-data-challenge')

# Toggle between sampled (local dev) and full data (final results)
USE_SAMPLED_DATA = True  # Set to False for full dataset

if USE_SAMPLED_DATA:
    DATA_PATH = BASE_PATH / 'sampled'
    BROWSING_FILE = 'browsing_train_sampled.csv'
else:
    DATA_PATH = BASE_PATH / 'train'
    BROWSING_FILE = 'browsing_train.csv'

# Model parameters
EMBEDDING_DIM = 50  # Dimension of the pre-computed embeddings
TOP_K_RETRIEVAL = [5, 10, 20, 50]  # Evaluate at multiple K values
MIN_SESSION_LENGTH = 2  # Minimum items in session for evaluation

# Ground truth configuration
# Which product actions count as "user found this item relevant"?
# Options: 'detail' (viewed), 'add' (added to cart), 'purchase' (bought), 'remove' (removed - negative!)
GROUND_TRUTH_ACTIONS = ['detail', 'add', 'purchase']  # Excludes 'remove' by default
# For stricter evaluation, use: ['add', 'purchase'] or just ['purchase']


# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION (Memory-Efficient with Polars)
# =============================================================================

def load_product_catalog(data_path: Path) -> pl.DataFrame:
    """
    Load the product catalog with pre-computed embeddings using Polars.
    
    Polars advantages over Pandas:
    - 10-100x faster for many operations
    - More memory efficient (Apache Arrow backend)
    - Lazy evaluation support
    - Better parallelization
    
    The sku_to_content.csv contains:
    - product_sku_hash: Unique product identifier
    - description_vector: 50-dim text embedding from product description
    - image_vector: Visual embedding from product image
    - category_hash: Product category (anonymized)
    - price_bucket: Price range (1-10, where 1=cheapest)
    """
    print("Loading product catalog with Polars...")
    
    df = pl.read_csv(data_path / 'sku_to_content.csv')
    
    print(f"  Total products: {len(df):,}")
    print(f"  Columns: {df.columns}")
    print(f"  Memory usage: {df.estimated_size('mb'):.2f} MB")
    
    # Check for missing embeddings using Polars expressions
    has_desc = df.select(pl.col('description_vector').is_not_null().sum()).item()
    has_img = df.select(pl.col('image_vector').is_not_null().sum()).item()
    
    print(f"  Products with description embedding: {has_desc:,} ({100*has_desc/len(df):.1f}%)")
    print(f"  Products with image embedding: {has_img:,} ({100*has_img/len(df):.1f}%)")
    
    return df


def load_browsing_sessions_lazy(data_path: Path, browsing_file: str = None) -> pl.LazyFrame:
    """
    Load user browsing sessions using Polars LAZY mode.
    
    Lazy evaluation benefits:
    - Query optimization before execution
    - Only reads columns that are actually needed
    - Predicate pushdown (filters applied during scan)
    - Memory efficient for large files
    
    Returns a LazyFrame - operations are not executed until .collect() is called.
    """
    print("\nLoading browsing sessions (lazy mode)...")
    
    if browsing_file is None:
        browsing_file = BROWSING_FILE
    
    filepath = data_path / browsing_file
    
    # Create lazy frame - doesn't load data yet!
    lf = pl.scan_csv(filepath)
    
    # Get schema without loading data
    schema = lf.collect_schema()
    print(f"  Columns: {list(schema.names())}")
    print(f"  Note: Data not loaded yet (lazy evaluation)")
    
    return lf


def get_browsing_stats(lf: pl.LazyFrame) -> Dict:
    """
    Get statistics from browsing data efficiently using lazy evaluation.
    Only computes what's needed, doesn't load full dataset.
    """
    print("  Computing statistics (streaming)...")
    
    stats = (
        lf.select([
            pl.len().alias('total_events'),
            pl.col('session_id_hash').n_unique().alias('unique_sessions'),
            pl.col('product_sku_hash').drop_nulls().n_unique().alias('unique_products')
        ])
        .collect(engine="streaming")  # Stream through data, don't load all at once
    )
    
    result = {
        'total_events': stats['total_events'][0],
        'unique_sessions': stats['unique_sessions'][0],
        'unique_products': stats['unique_products'][0]
    }
    
    print(f"  Total events: {result['total_events']:,}")
    print(f"  Unique sessions: {result['unique_sessions']:,}")
    print(f"  Unique products: {result['unique_products']:,}")
    
    return result


def parse_embedding_string(embedding_str: str) -> np.ndarray:
    """
    Parse embedding from string representation to numpy array.
    Embeddings are stored as string lists like "[0.1, 0.2, ...]"
    """
    if embedding_str is None:
        return None
    try:
        # Handle both Python list format and space-separated format
        if embedding_str.startswith('['):
            # Use json.loads - faster than ast.literal_eval for simple lists
            return np.array(json.loads(embedding_str), dtype=np.float32)
        else:
            return np.array([float(x) for x in embedding_str.split()], dtype=np.float32)
    except:
        return None


def parse_embeddings_batch(embedding_series: pl.Series) -> np.ndarray:
    """
    Parse embeddings in batch - much faster than row-by-row.
    
    Returns a 2D numpy array of shape (n_items, embedding_dim).
    """
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
    
    # Fill in any None values now that we know the dimension
    if embedding_dim is not None:
        embeddings = [e if e is not None else np.zeros(embedding_dim, dtype=np.float32) 
                      for e in embeddings]
        return np.vstack(embeddings), embedding_dim
    
    return None, None


# =============================================================================
# PART 2: BUILD SESSION-BASED GROUND TRUTH (Memory-Efficient)
# =============================================================================

def build_session_ground_truth(
    browsing_lf: pl.LazyFrame,
    min_session_length: int = 2,
    ground_truth_actions: List[str] = None
) -> Dict[str, Dict]:
    """
    Build ground truth for "similar items" evaluation from browsing sessions.
    
    Uses Polars lazy evaluation and streaming for memory efficiency:
    - Only loads necessary columns
    - Filters and aggregates during streaming (not after loading)
    - Processes data in chunks, not all at once
    
    Logic:
    - Filter to sessions with at least min_session_length product interactions
    - For each qualifying session:
      - First item viewed = "hero" item (the query)
      - Subsequent items = ground truth similar items
    
    This simulates: User lands on product A, then browses products B, C, D
    → B, C, D are items the user found relevant/similar to A
    
    Ground Truth Action Types:
    - 'detail': User viewed the product page (weakest signal)
    - 'add': User added to cart (strong positive signal)
    - 'purchase': User bought the item (strongest signal)
    - 'remove': User removed from cart (NEGATIVE signal - exclude!)
    
    Args:
        browsing_lf: Polars LazyFrame of browsing events
        min_session_length: Minimum products in session to be included
        ground_truth_actions: List of product_action values to include.
                             Default: ['detail', 'add', 'purchase'] (excludes 'remove')
                             For stricter evaluation: ['add', 'purchase'] or ['purchase']
    
    Returns:
        Dict mapping hero_sku -> {
            'ground_truth': List of SKUs viewed after the hero item,
            'session_id': Original session ID,
            'num_sessions': Count of sessions with this hero item
        }
    """
    # Default: include views, adds, purchases but NOT removes
    if ground_truth_actions is None:
        ground_truth_actions = ['detail', 'add', 'purchase']
    
    print("\nBuilding session-based ground truth (streaming)...")
    print(f"  Ground truth actions: {ground_truth_actions}")
    
    # Build query with lazy evaluation - nothing executes yet!
    session_products_query = (
        browsing_lf
        # Step 1: Only keep rows with product interactions AND valid actions
        .filter(
            (pl.col('product_sku_hash').is_not_null()) & 
            (pl.col('product_action').is_in(ground_truth_actions))
        )
        # Step 2: Only select columns we need (reduces memory)
        .select(['session_id_hash', 'product_sku_hash', 'server_timestamp_epoch_ms'])
        # Step 3: Sort by session and time
        .sort(['session_id_hash', 'server_timestamp_epoch_ms'])
        # Step 4: Group by session and collect products as list
        .group_by('session_id_hash')
        .agg([
            pl.col('product_sku_hash').alias('products'),
            pl.len().alias('session_length')
        ])
        # Step 5: Filter to sessions with minimum length
        .filter(pl.col('session_length') >= min_session_length)
    )
    
    print("  Executing query with streaming...")
    
    # Execute with streaming - processes in chunks, not all at once
    session_products_df = session_products_query.collect(engine="streaming")
    
    print(f"  Sessions with {min_session_length}+ products: {len(session_products_df):,}")
    
    # Build ground truth dictionary
    print("  Building ground truth dictionary...")
    ground_truth = {}
    
    for row in tqdm(session_products_df.iter_rows(named=True), 
                    total=len(session_products_df), 
                    desc="  Processing sessions"):
        session_id = row['session_id_hash']
        products = row['products']
        
        # Remove consecutive duplicates (user refreshing same page)
        unique_products = []
        for p in products:
            if not unique_products or p != unique_products[-1]:
                unique_products.append(p)
        
        if len(unique_products) >= min_session_length:
            hero_item = unique_products[0]
            subsequent_items = unique_products[1:]
            
            # If hero item already exists, merge ground truth sets
            if hero_item in ground_truth:
                existing = set(ground_truth[hero_item]['ground_truth'])
                existing.update(subsequent_items)
                ground_truth[hero_item]['ground_truth'] = list(existing)
                ground_truth[hero_item]['num_sessions'] += 1
            else:
                ground_truth[hero_item] = {
                    'ground_truth': subsequent_items,
                    'session_id': session_id,
                    'num_sessions': 1
                }
    
    # Free memory
    del session_products_df
    
    print(f"  Unique hero items with ground truth: {len(ground_truth):,}")
    
    # Statistics on ground truth size
    if ground_truth:
        gt_sizes = [len(v['ground_truth']) for v in ground_truth.values()]
        print(f"  Ground truth set sizes: min={min(gt_sizes)}, median={np.median(gt_sizes):.0f}, max={max(gt_sizes)}")
    
    return ground_truth


# =============================================================================
# PART 3: TWO-TOWER EMBEDDING MODEL
# =============================================================================

class ItemEmbeddingModel:
    """
    Two-Tower style item embedding model for similar items retrieval.
    
    In this "unsupervised" version, we use pre-computed embeddings:
    - Description embedding: Captures textual/semantic similarity
    - Image embedding: Captures visual similarity
    
    IMPORTANT: Embedding Combination Strategy
    -----------------------------------------
    We CONCATENATE embeddings rather than taking a weighted sum because:
    
    1. Different Embedding Spaces: The text embedding (from NLP model) and image 
       embedding (from CNN) come from completely different models. Adding vectors 
       from different spaces doesn't produce meaningful results.
    
    2. Different Scales: Each embedding may have different magnitudes. A weighted 
       sum would allow one modality to dominate if not carefully normalized.
    
    3. Information Preservation: Concatenation preserves all information from both 
       modalities (d_text + d_image dimensions), while addition compresses them 
       into max(d_text, d_image) dimensions.
    
    4. Let Similarity Decide: With concatenation, the cosine similarity naturally 
       considers both modalities. Items must be similar in BOTH text AND visual 
       space to have high similarity.
    
    Architecture (conceptual):
    
    Query Tower (Hero Item):          Candidate Tower (Catalog Items):
    ┌─────────────────────┐           ┌─────────────────────┐
    │  Item Metadata      │           │  Item Metadata      │
    │  - Description      │           │  - Description      │
    │  - Image            │           │  - Image            │
    └─────────┬───────────┘           └─────────┬───────────┘
              │                                 │
              ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │  Text Embedding     │           │  Text Embedding     │
    │  (d_text dim)       │           │  (d_text dim)       │
    └─────────┬───────────┘           └─────────┬───────────┘
              │                                 │
    ┌─────────────────────┐           ┌─────────────────────┐
    │  Image Embedding    │           │  Image Embedding    │
    │  (d_image dim)      │           │  (d_image dim)      │
    └─────────┬───────────┘           └─────────┬───────────┘
              │                                 │
              ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │  CONCATENATE        │           │  CONCATENATE        │
    │  [text ; image]     │           │  [text ; image]     │
    │  (d_text + d_image) │           │  (d_text + d_image) │
    └─────────┬───────────┘           └─────────┬───────────┘
              │                                 │
              └────────────┬────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Cosine          │
                  │ Similarity      │
                  └─────────────────┘
                           │
                           ▼
                    Similarity Score
    """
    
    def __init__(
        self,
        use_description: bool = True,
        use_image: bool = True,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding model.
        
        Args:
            use_description: Whether to include text/description embeddings
            use_image: Whether to include visual/image embeddings
            normalize_embeddings: Whether to L2-normalize final embeddings
                                  (required for cosine similarity via dot product)
        """
        self.use_description = use_description
        self.use_image = use_image
        self.normalize_embeddings = normalize_embeddings
        
        if not use_description and not use_image:
            raise ValueError("At least one of use_description or use_image must be True")
        
        # Will be populated by fit()
        self.sku_to_idx = {}
        self.idx_to_sku = {}
        self.embeddings = None
        self.embedding_dim = None
        self.desc_dim = None
        self.img_dim = None
        
    def fit(self, catalog_df: pl.DataFrame) -> 'ItemEmbeddingModel':
        """
        Build item embeddings from the product catalog.
        
        Embeddings are CONCATENATED (not summed) to preserve information from
        both modalities without assuming they share the same embedding space.
        
        Uses batch processing instead of row-by-row iteration for efficiency.
        
        Args:
            catalog_df: Polars DataFrame with product_sku_hash, description_vector, image_vector
            
        Returns:
            self (for method chaining)
        """
        print("\nBuilding item embeddings (batch mode)...")
        print(f"  Using description embeddings: {self.use_description}")
        print(f"  Using image embeddings: {self.use_image}")
        
        # Get SKUs as list
        skus = catalog_df['product_sku_hash'].to_list()
        
        # Parse embeddings in batch - much faster than row-by-row
        desc_matrix = None
        img_matrix = None
        
        if self.use_description:
            print("  Parsing description embeddings (batch)...")
            desc_series = catalog_df['description_vector']
            desc_matrix, self.desc_dim = parse_embeddings_batch(desc_series)
            
            if desc_matrix is not None:
                print(f"    Description embedding dimension: {self.desc_dim}")
                print(f"    Description matrix shape: {desc_matrix.shape}")
            else:
                print("    WARNING: No valid description embeddings found")
                self.use_description = False
        
        if self.use_image:
            print("  Parsing image embeddings (batch)...")
            img_series = catalog_df['image_vector']
            img_matrix, self.img_dim = parse_embeddings_batch(img_series)
            
            if img_matrix is not None:
                print(f"    Image embedding dimension: {self.img_dim}")
                print(f"    Image matrix shape: {img_matrix.shape}")
            else:
                print("    WARNING: No valid image embeddings found")
                self.use_image = False
        
        if not self.use_description and not self.use_image:
            raise ValueError("No valid embeddings found in catalog")
        
        # Calculate total embedding dimension
        self.embedding_dim = (self.desc_dim or 0) + (self.img_dim or 0)
        print(f"  Combined embedding dimension: {self.embedding_dim}")
        
        # Concatenate embeddings (vectorized, no loops!)
        print("  Concatenating embeddings...")
        if self.use_description and self.use_image:
            self.embeddings = np.hstack([desc_matrix, img_matrix])
        elif self.use_description:
            self.embeddings = desc_matrix
        else:
            self.embeddings = img_matrix
        
        self.embeddings = self.embeddings.astype(np.float32)
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Embeddings memory: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        # Normalize if requested (important for cosine similarity via dot product)
        if self.normalize_embeddings:
            print("  Normalizing embeddings (L2)...")
            self.embeddings = normalize(self.embeddings, norm='l2')
        
        # Build SKU <-> index mappings
        self.sku_to_idx = {sku: idx for idx, sku in enumerate(skus)}
        self.idx_to_sku = {idx: sku for idx, sku in enumerate(skus)}
        
        print(f"  Total items indexed: {len(self.sku_to_idx):,}")
        
        return self
    
    def get_embedding(self, sku: str) -> np.ndarray:
        """Get embedding for a single SKU."""
        if sku not in self.sku_to_idx:
            return None
        return self.embeddings[self.sku_to_idx[sku]]
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings as a matrix."""
        return self.embeddings


# =============================================================================
# PART 4: FAISS INDEX FOR NEAREST NEIGHBOR RETRIEVAL
# =============================================================================

class SimilarItemsRetriever:
    """
    Retrieval system using FAISS for efficient nearest neighbor search.
    
    FAISS (Facebook AI Similarity Search) provides:
    - Fast approximate nearest neighbor search
    - GPU acceleration (optional)
    - Multiple index types for different tradeoffs
    
    For this example, we use IndexFlatIP (exact inner product search)
    which is suitable for catalogs up to ~1M items on CPU.
    """
    
    def __init__(self, embedding_model: ItemEmbeddingModel):
        """
        Initialize the retriever with an embedding model.
        
        Args:
            embedding_model: Fitted ItemEmbeddingModel
        """
        self.embedding_model = embedding_model
        self.index = None
        
    def build_index(self) -> 'SimilarItemsRetriever':
        """
        Build FAISS index from item embeddings.
        
        Index types:
        - IndexFlatIP: Exact search using inner product (cosine sim for normalized vectors)
        - IndexIVFFlat: Approximate search with inverted file index
        - IndexHNSW: Approximate search with hierarchical navigable small world graphs
        
        For catalogs < 1M items, IndexFlatIP is fast enough and gives exact results.
        """
        print("\nBuilding FAISS index...")
        
        embeddings = self.embedding_model.get_all_embeddings()
        embedding_dim = embeddings.shape[1]
        
        # Create index for inner product (equivalent to cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Add all embeddings to the index
        self.index.add(embeddings)
        
        print(f"  Index type: IndexFlatIP (exact search)")
        print(f"  Vectors indexed: {self.index.ntotal:,}")
        print(f"  Embedding dimension: {embedding_dim}")
        
        return self
    
    def retrieve(
        self, 
        query_sku: str, 
        top_k: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K similar items for a query item.
        
        Args:
            query_sku: SKU of the query (hero) item
            top_k: Number of similar items to retrieve
            exclude_query: Whether to exclude the query item from results
            
        Returns:
            List of (sku, similarity_score) tuples, sorted by similarity descending
        """
        # Get query embedding
        query_emb = self.embedding_model.get_embedding(query_sku)
        if query_emb is None:
            return []
        
        # Reshape for FAISS (expects 2D array)
        query_emb = query_emb.reshape(1, -1)
        
        # Search (retrieve extra if we need to exclude query)
        k_search = top_k + 1 if exclude_query else top_k
        distances, indices = self.index.search(query_emb, k_search)
        
        # Convert to SKU and score tuples
        results = []
        for idx, score in zip(indices[0], distances[0]):
            sku = self.embedding_model.idx_to_sku.get(idx)
            if sku and (not exclude_query or sku != query_sku):
                results.append((sku, float(score)))
                if len(results) >= top_k:
                    break
        
        return results
    
    def batch_retrieve(
        self,
        query_skus: List[str],
        top_k: int = 10,
        exclude_query: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve similar items for multiple query items efficiently.
        
        Args:
            query_skus: List of query SKUs
            top_k: Number of similar items per query
            exclude_query: Whether to exclude query items from results
            
        Returns:
            Dict mapping query_sku -> List of (sku, score) tuples
        """
        # Build query matrix
        query_indices = []
        valid_skus = []
        
        for sku in query_skus:
            if sku in self.embedding_model.sku_to_idx:
                query_indices.append(self.embedding_model.sku_to_idx[sku])
                valid_skus.append(sku)
        
        if not query_indices:
            return {}
        
        # Get embeddings for all queries
        query_embeddings = self.embedding_model.embeddings[query_indices]
        
        # Batch search
        k_search = top_k + 1 if exclude_query else top_k
        distances, indices = self.index.search(query_embeddings, k_search)
        
        # Convert to results dict
        results = {}
        for i, query_sku in enumerate(valid_skus):
            sku_results = []
            for idx, score in zip(indices[i], distances[i]):
                sku = self.embedding_model.idx_to_sku.get(idx)
                if sku and (not exclude_query or sku != query_sku):
                    sku_results.append((sku, float(score)))
                    if len(sku_results) >= top_k:
                        break
            results[query_sku] = sku_results
        
        return results


# =============================================================================
# PART 5: EVALUATION METRICS
# =============================================================================

def evaluate_retrieval(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Dict],
    top_k_values: List[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Evaluate retrieval quality using session-based ground truth.
    
    This function is decoupled from the retriever - it takes predictions directly,
    making it reusable for any retrieval method (embedding-based, collaborative 
    filtering, rules-based, etc.)
    
    Metrics:
    - Hit Rate@K: Fraction of queries where at least one ground truth item is in top-K
    - Recall@K: Average fraction of ground truth items found in top-K
    - MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant item
    
    Args:
        predictions: Dict mapping query_sku -> List of retrieved SKUs (ordered by rank)
                    Example: {'sku_123': ['sku_456', 'sku_789', ...], ...}
        ground_truth: Dict from build_session_ground_truth()
                     Example: {'sku_123': {'ground_truth': ['sku_456', ...], ...}, ...}
        top_k_values: List of K values to evaluate
        
    Returns:
        DataFrame with metrics for each K value
    """
    print("\nEvaluating retrieval quality...")
    
    # Get queries that have both predictions and ground truth
    query_skus = [
        sku for sku in predictions.keys() 
        if sku in ground_truth
    ]
    
    print(f"  Evaluating {len(query_skus):,} queries...")
    
    # Compute metrics
    results = {k: {'hits': 0, 'recall_sum': 0, 'mrr_sum': 0} for k in top_k_values}
    
    for query_sku in tqdm(query_skus, desc="  Computing metrics"):
        gt_items = set(ground_truth[query_sku]['ground_truth'])
        retrieved_skus = predictions.get(query_sku, [])
        
        for k in top_k_values:
            top_k_retrieved = set(retrieved_skus[:k])
            
            # Hit Rate: Is there any overlap?
            hits = len(top_k_retrieved & gt_items)
            if hits > 0:
                results[k]['hits'] += 1
            
            # Recall: What fraction of ground truth did we find?
            recall = hits / len(gt_items) if gt_items else 0
            results[k]['recall_sum'] += recall
            
            # MRR: Reciprocal rank of first relevant item
            mrr = 0
            for rank, sku in enumerate(retrieved_skus[:k], 1):
                if sku in gt_items:
                    mrr = 1.0 / rank
                    break
            results[k]['mrr_sum'] += mrr
    
    # Aggregate metrics
    n_queries = len(query_skus)
    metrics_df = pd.DataFrame([
        {
            'K': k,
            'Hit Rate@K': results[k]['hits'] / n_queries,
            'Recall@K': results[k]['recall_sum'] / n_queries,
            'MRR@K': results[k]['mrr_sum'] / n_queries
        }
        for k in top_k_values
    ])
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    
    return metrics_df


def generate_predictions(
    retriever: SimilarItemsRetriever,
    query_skus: List[str],
    top_k: int = 50
) -> Dict[str, List[str]]:
    """
    Generate predictions from a retriever for evaluation.
    
    This helper function bridges the retriever to the evaluate_retrieval function.
    
    Args:
        retriever: Fitted SimilarItemsRetriever
        query_skus: List of query SKUs to generate predictions for
        top_k: Number of similar items to retrieve per query
        
    Returns:
        Dict mapping query_sku -> List of retrieved SKUs (ordered by similarity)
    """
    print(f"\nGenerating predictions for {len(query_skus):,} queries...")
    
    # Filter to queries that exist in the index
    valid_skus = [
        sku for sku in query_skus 
        if sku in retriever.embedding_model.sku_to_idx
    ]
    
    if len(valid_skus) < len(query_skus):
        print(f"  Note: {len(query_skus) - len(valid_skus)} queries not in index, skipped")
    
    # Batch retrieve
    retrievals = retriever.batch_retrieve(valid_skus, top_k=top_k)
    
    # Convert to simple format: query -> list of retrieved SKUs
    predictions = {
        query_sku: [sku for sku, score in retrieved_items]
        for query_sku, retrieved_items in retrievals.items()
    }
    
    print(f"  Generated predictions for {len(predictions):,} queries")
    
    return predictions


# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

def visualize_results(metrics_df: pd.DataFrame, output_path: Path = None):
    """
    Create visualization of retrieval metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Hit Rate
    axes[0].plot(metrics_df['K'], metrics_df['Hit Rate@K'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('K (Retrieved Items)', fontsize=12)
    axes[0].set_ylabel('Hit Rate@K', fontsize=12)
    axes[0].set_title('Hit Rate: At Least One Relevant Item', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Recall
    axes[1].plot(metrics_df['K'], metrics_df['Recall@K'], 's-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('K (Retrieved Items)', fontsize=12)
    axes[1].set_ylabel('Recall@K', fontsize=12)
    axes[1].set_title('Recall: Fraction of Relevant Items Found', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # MRR
    axes[2].plot(metrics_df['K'], metrics_df['MRR@K'], '^-', linewidth=2, markersize=8, color='orange')
    axes[2].set_xlabel('K (Retrieved Items)', fontsize=12)
    axes[2].set_ylabel('MRR@K', fontsize=12)
    axes[2].set_title('MRR: Rank of First Relevant Item', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


def show_similar_items_example(
    retriever: SimilarItemsRetriever,
    catalog_df: pl.DataFrame,
    query_sku: str,
    top_k: int = 5
):
    """
    Display an example of similar items retrieval for a given product.
    """
    print(f"\n{'='*60}")
    print(f"SIMILAR ITEMS EXAMPLE")
    print(f"{'='*60}")
    
    # Get query item info using Polars filter
    query_info = catalog_df.filter(pl.col('product_sku_hash') == query_sku)
    
    if len(query_info) > 0:
        print(f"\nQuery Item (Hero):")
        print(f"  SKU: {query_sku}")
        print(f"  Category: {query_info['category_hash'][0]}")
        print(f"  Price Bucket: {query_info['price_bucket'][0]}")
    else:
        print(f"\nQuery Item (Hero): {query_sku}")
    
    # Retrieve similar items
    similar = retriever.retrieve(query_sku, top_k=top_k)
    
    print(f"\nTop {top_k} Similar Items:")
    print("-" * 50)
    for rank, (sku, score) in enumerate(similar, 1):
        item_info = catalog_df.filter(pl.col('product_sku_hash') == sku)
        if len(item_info) > 0:
            print(f"  {rank}. SKU: {sku}")
            print(f"     Category: {item_info['category_hash'][0]}")
            print(f"     Price Bucket: {item_info['price_bucket'][0]}")
            print(f"     Similarity: {score:.4f}")
        else:
            print(f"  {rank}. SKU: {sku}, Similarity: {score:.4f}")


# =============================================================================
# PART 7: ABLATION STUDY
# =============================================================================

def run_ablation_study(
    catalog_df: pl.DataFrame,
    ground_truth: Dict[str, Dict],
    top_k_values: List[int] = [5, 10, 20, 50],
    max_queries: int = 5000
) -> pd.DataFrame:
    """
    Run ablation study comparing different embedding configurations.
    
    Configurations tested:
    1. Description-only: Uses only text embeddings
    2. Image-only: Uses only visual embeddings  
    3. Combined (Description + Image): Concatenates both modalities
    
    This helps understand:
    - Which modality contributes more to retrieval quality
    - Whether multi-modal embeddings outperform single-modality
    - The complementary nature of text and visual features
    
    Args:
        catalog_df: Product catalog with embeddings
        ground_truth: Session-based ground truth
        top_k_values: K values to evaluate
        max_queries: Limit queries for faster evaluation
        
    Returns:
        DataFrame with metrics for each configuration
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: Comparing Embedding Configurations")
    print("=" * 70)
    
    configurations = [
        {
            'name': 'Description Only',
            'use_description': True,
            'use_image': False
        },
        {
            'name': 'Image Only',
            'use_description': False,
            'use_image': True
        },
        {
            'name': 'Combined (Desc + Image)',
            'use_description': True,
            'use_image': True
        }
    ]
    
    all_results = []
    
    # Get query SKUs (hero items with ground truth)
    query_skus = list(ground_truth.keys())
    if max_queries:
        query_skus = query_skus[:max_queries]
    
    max_k = max(top_k_values)
    
    for config in configurations:
        print(f"\n{'-' * 60}")
        print(f"Configuration: {config['name']}")
        print(f"{'-' * 60}")
        
        try:
            # Build embedding model with this configuration
            embedding_model = ItemEmbeddingModel(
                use_description=config['use_description'],
                use_image=config['use_image'],
                normalize_embeddings=True
            )
            embedding_model.fit(catalog_df)
            
            # Build retriever
            retriever = SimilarItemsRetriever(embedding_model)
            retriever.build_index()
            
            # Generate predictions
            predictions = generate_predictions(retriever, query_skus, top_k=max_k)
            
            # Evaluate predictions against ground truth
            metrics_df = evaluate_retrieval(
                predictions=predictions,
                ground_truth=ground_truth,
                top_k_values=top_k_values
            )
            
            # Add configuration name to results
            metrics_df['Configuration'] = config['name']
            metrics_df['Embedding Dim'] = embedding_model.embedding_dim
            all_results.append(metrics_df)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    return combined_df


def visualize_ablation_results(ablation_df: pd.DataFrame, output_path: Path = None):
    """
    Create visualization comparing ablation study results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    configurations = ablation_df['Configuration'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']
    
    metrics = ['Hit Rate@K', 'Recall@K', 'MRR@K']
    titles = [
        'Hit Rate: At Least One Relevant Item',
        'Recall: Fraction of Relevant Items Found',
        'MRR: Rank of First Relevant Item'
    ]
    
    for ax, metric, title in zip(axes, metrics, titles):
        for i, config in enumerate(configurations):
            config_data = ablation_df[ablation_df['Configuration'] == config]
            ax.plot(
                config_data['K'], 
                config_data[metric], 
                marker=markers[i],
                color=colors[i],
                linewidth=2, 
                markersize=8,
                label=config
            )
        
        ax.set_xlabel('K (Retrieved Items)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)  # Start from 0, auto-scale top
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nAblation visualization saved to: {output_path}")
    
    plt.show()


def print_ablation_summary(ablation_df: pd.DataFrame):
    """
    Print a summary table comparing configurations at K=10.
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY (K=10)")
    print("=" * 70)
    
    # Filter to K=10 for summary
    summary = ablation_df[ablation_df['K'] == 10][
        ['Configuration', 'Embedding Dim', 'Hit Rate@K', 'Recall@K', 'MRR@K']
    ].copy()
    
    # Calculate improvement over description-only baseline
    baseline = summary[summary['Configuration'] == 'Description Only']
    if len(baseline) > 0:
        baseline_hit = baseline['Hit Rate@K'].values[0]
        baseline_recall = baseline['Recall@K'].values[0]
        baseline_mrr = baseline['MRR@K'].values[0]
        
        summary['Hit Rate Delta'] = summary['Hit Rate@K'].apply(
            lambda x: f"+{(x - baseline_hit) / baseline_hit * 100:.1f}%" if x > baseline_hit 
            else f"{(x - baseline_hit) / baseline_hit * 100:.1f}%"
        )
        summary['Recall Delta'] = summary['Recall@K'].apply(
            lambda x: f"+{(x - baseline_recall) / baseline_recall * 100:.1f}%" if baseline_recall > 0 and x > baseline_recall
            else f"{(x - baseline_recall) / baseline_recall * 100:.1f}%" if baseline_recall > 0
            else "N/A"
        )
    
    # Format metrics as percentages
    summary['Hit Rate@K'] = summary['Hit Rate@K'].apply(lambda x: f"{x*100:.2f}%")
    summary['Recall@K'] = summary['Recall@K'].apply(lambda x: f"{x*100:.2f}%")
    summary['MRR@K'] = summary['MRR@K'].apply(lambda x: f"{x:.4f}")
    
    print(summary.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print("• Description embeddings capture semantic/textual similarity")
    print("• Image embeddings capture visual similarity (color, style, shape)")
    print("• Combined embeddings leverage complementary information from both")
    print("• Improvement shows the value of multi-modal representations")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(run_ablation: bool = True):
    """
    Main execution flow demonstrating the complete retrieval pipeline.
    
    Uses memory-efficient Polars with lazy evaluation for large datasets.
    
    Args:
        run_ablation: If True, runs ablation study comparing embedding configs
    """
    print("=" * 70)
    print("Chapter 5: Similar Items Retrieval with Two-Tower Model")
    print("=" * 70)
    
    # 1. Load data (memory-efficient)
    catalog_df = load_product_catalog(DATA_PATH)
    
    # Load browsing data lazily - doesn't load into memory yet
    browsing_lf = load_browsing_sessions_lazy(DATA_PATH, BROWSING_FILE)
    
    # Optional: Get stats (uses streaming, doesn't load all data)
    get_browsing_stats(browsing_lf)
    
    # 2. Build ground truth (processes data in streaming mode)
    # Note: We filter by product_action to only include meaningful interactions
    ground_truth = build_session_ground_truth(
        browsing_lf, 
        min_session_length=MIN_SESSION_LENGTH,
        ground_truth_actions=GROUND_TRUTH_ACTIONS
    )
    
    # Create output directory
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # 3. Run ablation study (compares description-only vs combined)
    if run_ablation:
        ablation_df = run_ablation_study(
            catalog_df,
            ground_truth,
            top_k_values=TOP_K_RETRIEVAL,
            max_queries=5000
        )
        
        # Visualize ablation results
        visualize_ablation_results(ablation_df, output_dir / 'ablation_study.png')
        
        # Print summary
        print_ablation_summary(ablation_df)
        
        # Save ablation results
        ablation_df.to_csv(output_dir / 'ablation_results.csv', index=False)
        print(f"\nAblation results saved to: {output_dir / 'ablation_results.csv'}")
    
    # 4. Build final model with combined embeddings
    print("\n" + "=" * 70)
    print("FINAL MODEL: Combined Embeddings")
    print("=" * 70)
    
    embedding_model = ItemEmbeddingModel(
        use_description=True,
        use_image=True,
        normalize_embeddings=True
    )
    embedding_model.fit(catalog_df)
    
    # 5. Build retrieval index
    retriever = SimilarItemsRetriever(embedding_model)
    retriever.build_index()
    
    # 6. Generate predictions and evaluate
    query_skus = list(ground_truth.keys())[:5000]  # Limit for faster evaluation
    max_k = max(TOP_K_RETRIEVAL)
    
    predictions = generate_predictions(retriever, query_skus, top_k=max_k)
    
    metrics_df = evaluate_retrieval(
        predictions=predictions,
        ground_truth=ground_truth,
        top_k_values=TOP_K_RETRIEVAL
    )
    
    # 7. Show example
    example_sku = list(ground_truth.keys())[0]
    show_similar_items_example(retriever, catalog_df, example_sku, top_k=5)
    
    # 8. Save final metrics
    metrics_df.to_csv(output_dir / 'retrieval_metrics.csv', index=False)
    print(f"\nFinal metrics saved to: {output_dir / 'retrieval_metrics.csv'}")
    
    return metrics_df, retriever, ground_truth, ablation_df if run_ablation else None


if __name__ == '__main__':
    results = main(run_ablation=True)
    metrics_df, retriever, ground_truth, ablation_df = results

