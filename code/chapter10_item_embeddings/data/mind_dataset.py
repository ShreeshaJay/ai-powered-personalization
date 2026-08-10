"""
MIND (Microsoft News) Dataset Loader for Item Embeddings

Loads news articles with rich text metadata: title, abstract, category, subcategory,
and entity annotations. Used for Section 10.1 (text encoder baselines) and 10.2 
(fine-tuning).

Dataset Schema:
- news.tsv: news_id, category, subcategory, title, abstract, url, 
            title_entities, abstract_entities
- behaviors.tsv: impression_id, user_id, time, history, impressions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MINDDataset:
    """
    MIND dataset loader for news article embeddings.
    
    Provides:
    - News article metadata with rich text fields
    - User click histories for co-click analysis
    - Entity embeddings for comparison
    """
    
    def __init__(
        self,
        data_dir: Path,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize MIND dataset loader.
        
        Args:
            data_dir: Path to MINDsmall_train directory
            sample_size: Number of news articles to sample (None = all)
            random_seed: Random seed for sampling
        """
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Data containers
        self.news_df: Optional[pd.DataFrame] = None
        self.behaviors_df: Optional[pd.DataFrame] = None
        self.entity_embeddings: Optional[Dict[str, np.ndarray]] = None
        
        # Statistics
        self.stats = {}
    
    def load_news(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load news articles with text metadata.
        
        Returns:
            DataFrame with columns: news_id, category, subcategory, title, 
                                   abstract, title_entities, abstract_entities
        """
        news_path = self.data_dir / "news.tsv"
        if not news_path.exists():
            raise FileNotFoundError(f"News file not found: {news_path}")
        
        logger.info(f"Loading news from {news_path}")
        
        # MIND news.tsv schema
        columns = [
            "news_id",
            "category", 
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities"
        ]
        
        self.news_df = pd.read_csv(
            news_path,
            sep="\t",
            names=columns,
            usecols=["news_id", "category", "subcategory", "title", "abstract",
                     "title_entities", "abstract_entities"]
        )
        
        # Handle missing abstracts (some news have no abstract)
        self.news_df["abstract"] = self.news_df["abstract"].fillna("")
        
        # Sample if requested
        if self.sample_size and self.sample_size < len(self.news_df):
            logger.info(f"Sampling {self.sample_size} articles from {len(self.news_df)}")
            self.news_df = self.news_df.sample(
                n=self.sample_size,
                random_state=self.random_seed
            ).reset_index(drop=True)
        
        # Collect statistics
        self.stats["total_news"] = len(self.news_df)
        self.stats["unique_categories"] = self.news_df["category"].nunique()
        self.stats["unique_subcategories"] = self.news_df["subcategory"].nunique()
        self.stats["avg_title_length"] = self.news_df["title"].str.len().mean()
        self.stats["avg_abstract_length"] = self.news_df["abstract"].str.len().mean()
        self.stats["missing_abstracts"] = (self.news_df["abstract"] == "").sum()
        
        logger.info(f"Loaded {len(self.news_df)} news articles")
        logger.info(f"  Categories: {self.stats['unique_categories']}")
        logger.info(f"  Subcategories: {self.stats['unique_subcategories']}")
        logger.info(f"  Missing abstracts: {self.stats['missing_abstracts']}")
        
        return self.news_df
    
    def load_behaviors(self) -> pd.DataFrame:
        """
        Load user behavior data (click histories and impressions).
        
        Returns:
            DataFrame with columns: impression_id, user_id, time, history, impressions
        """
        behaviors_path = self.data_dir / "behaviors.tsv"
        if not behaviors_path.exists():
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_path}")
        
        logger.info(f"Loading behaviors from {behaviors_path}")
        
        columns = ["impression_id", "user_id", "time", "history", "impressions"]
        
        self.behaviors_df = pd.read_csv(
            behaviors_path,
            sep="\t",
            names=columns
        )
        
        # Parse history (space-separated news IDs)
        self.behaviors_df["history_list"] = self.behaviors_df["history"].apply(
            lambda x: x.split() if pd.notna(x) and x else []
        )
        
        # Parse impressions (format: "news_id-label news_id-label ...")
        def parse_impressions(imp_str):
            if pd.isna(imp_str) or not imp_str:
                return [], []
            
            news_ids, labels = [], []
            for item in imp_str.split():
                news_id, label = item.split("-")
                news_ids.append(news_id)
                labels.append(int(label))
            return news_ids, labels
        
        parsed = self.behaviors_df["impressions"].apply(parse_impressions)
        self.behaviors_df["impression_news"] = parsed.apply(lambda x: x[0])
        self.behaviors_df["impression_labels"] = parsed.apply(lambda x: x[1])
        
        # Statistics
        self.stats["total_impressions"] = len(self.behaviors_df)
        self.stats["unique_users"] = self.behaviors_df["user_id"].nunique()
        self.stats["avg_history_length"] = self.behaviors_df["history_list"].apply(len).mean()
        
        logger.info(f"Loaded {len(self.behaviors_df)} impression logs")
        logger.info(f"  Unique users: {self.stats['unique_users']}")
        logger.info(f"  Avg history length: {self.stats['avg_history_length']:.2f}")
        
        return self.behaviors_df
    
    def get_coclick_pairs(self, min_support: int = 2) -> List[Tuple[str, str]]:
        """
        Extract co-clicked article pairs from user histories.
        
        NOTE: This method is NOT used in Section 10.1 (zero-shot baselines).
        It is pre-built for Section 10.3 (contrastive fine-tuning), where
        co-clicked articles form positive training pairs.
        
        Args:
            min_support: Minimum number of co-occurrences to include pair
        
        Returns:
            List of (news_id_1, news_id_2) tuples
        """
        if self.behaviors_df is None:
            self.load_behaviors()
        
        logger.info("Extracting co-clicked article pairs...")
        
        coclick_counts = defaultdict(int)
        
        for history in self.behaviors_df["history_list"]:
            if len(history) < 2:
                continue
            
            # All pairs in this user's history
            for i in range(len(history)):
                for j in range(i + 1, len(history)):
                    pair = tuple(sorted([history[i], history[j]]))
                    coclick_counts[pair] += 1
        
        # Filter by minimum support
        coclick_pairs = [
            pair for pair, count in coclick_counts.items()
            if count >= min_support
        ]
        
        logger.info(f"Found {len(coclick_pairs)} co-clicked pairs (min_support={min_support})")
        
        return coclick_pairs
    
    def load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load pre-trained WikiData entity embeddings (100-dim).
        
        NOTE: The MIND dataset also ships relation_embedding.vec (100-dim
        WikiData relation embeddings). Concatenating entity + relation 
        embeddings with text embeddings is left as an exercise for the reader.
        See: docs/Chapter10_Design_Notes.md for guidance.
        
        Returns:
            Dictionary mapping entity_id -> embedding vector
        """
        entity_path = self.data_dir / "entity_embedding.vec"
        if not entity_path.exists():
            logger.warning(f"Entity embeddings not found: {entity_path}")
            return {}
        
        logger.info(f"Loading entity embeddings from {entity_path}")
        
        embeddings = {}
        with open(entity_path, 'r', encoding='utf-8') as f:
            # First line: num_entities embedding_dim
            first_line = f.readline().strip().split()
            num_entities, emb_dim = int(first_line[0]), int(first_line[1])
            
            logger.info(f"  {num_entities} entities, {emb_dim}-dim")
            
            for line in f:
                parts = line.strip().split()
                entity_id = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[entity_id] = vector
        
        self.entity_embeddings = embeddings
        logger.info(f"Loaded {len(embeddings)} entity embeddings")
        
        return embeddings
    
    def get_item_texts(
        self,
        use_title: bool = True,
        use_abstract: bool = True,
        use_category: bool = True,
        use_subcategory: bool = True,
        template: str = "{title} [SEP] {abstract} [SEP] {category} {subcategory}"
    ) -> Dict[str, str]:
        """
        Generate text representations for each news article.
        
        Args:
            use_title: Include title
            use_abstract: Include abstract
            use_category: Include category
            use_subcategory: Include subcategory
            template: String template for combining fields
        
        Returns:
            Dictionary mapping news_id -> combined text
        """
        if self.news_df is None:
            self.load_news()
        
        logger.info("Generating item text representations...")
        
        item_texts = {}
        
        for _, row in self.news_df.iterrows():
            fields = {
                "title": row["title"] if use_title else "",
                "abstract": row["abstract"] if use_abstract else "",
                "category": row["category"] if use_category else "",
                "subcategory": row["subcategory"] if use_subcategory else "",
            }
            
            # Format using template
            text = template.format(**fields)
            
            # Clean up extra whitespace
            text = " ".join(text.split())
            
            item_texts[row["news_id"]] = text
        
        logger.info(f"Generated text for {len(item_texts)} items")
        
        return item_texts
    
    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "="*80)
        print("MIND Dataset Statistics")
        print("="*80)
        for key, value in self.stats.items():
            print(f"  {key:.<50} {value}")
        print("="*80 + "\n")


# ============================================================================
# Convenience Functions
# ============================================================================

def load_mind_news(
    data_dir: Path,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Convenience function to load MIND news data."""
    dataset = MINDDataset(data_dir, sample_size=sample_size)
    return dataset.load_news()


def load_mind_behaviors(
    data_dir: Path
) -> pd.DataFrame:
    """Convenience function to load MIND behavior data."""
    dataset = MINDDataset(data_dir)
    return dataset.load_behaviors()


# ============================================================================
# Main: Test Data Loading
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add parent directory to path for config import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import MIND_DATA_PATH
    
    print("Testing MIND Dataset Loader")
    print("="*80)
    
    # Initialize dataset
    dataset = MINDDataset(
        data_dir=MIND_DATA_PATH,
        sample_size=1000  # Sample for testing
    )
    
    # Load news
    news_df = dataset.load_news()
    print(f"\nNews DataFrame shape: {news_df.shape}")
    print("\nSample news articles:")
    print(news_df.head())
    
    # Load behaviors
    behaviors_df = dataset.load_behaviors()
    print(f"\nBehaviors DataFrame shape: {behaviors_df.shape}")
    print("\nSample behaviors:")
    print(behaviors_df[["user_id", "history_list", "impression_news"]].head())
    
    # Get item texts
    item_texts = dataset.get_item_texts()
    print(f"\nGenerated texts for {len(item_texts)} items")
    print("\nSample texts:")
    for i, (news_id, text) in enumerate(list(item_texts.items())[:3]):
        print(f"\n[{i+1}] {news_id}")
        print(f"  {text[:200]}...")
    
    # Get co-click pairs
    coclick_pairs = dataset.get_coclick_pairs(min_support=2)
    print(f"\nCo-clicked pairs: {len(coclick_pairs)}")
    print(f"Sample pairs: {coclick_pairs[:5]}")
    
    # Print statistics
    dataset.print_stats()
