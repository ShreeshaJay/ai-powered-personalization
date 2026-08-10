"""
Alibaba IJCAI-18 CVR Dataset Loader for Item Embeddings

Loads sponsored search conversion data with structured item metadata.
Used for Section 10.1 to demonstrate text serialization from structured fields
and bi-encoder evaluation (query intent vs. item).

Dataset Schema:
- Clicked samples: instance_id, is_trade, item_id, user_id, context_id, shop_id
- Items: item_id, item_category_list, item_property_list, item_brand_id, 
         item_city_id, item_price_level, item_sales_level, item_collected_level, 
         item_pv_level
- Context: context_id, context_timestamp, context_page_id, predict_category_property
- Users: user demographics
- Shops: shop metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IJCAIDataset:
    """
    IJCAI-18 CVR dataset loader for item embeddings from structured metadata.
    
    Demonstrates:
    - Serializing structured metadata into text for encoding
    - Bi-encoder evaluation: query intent vs. item embeddings
    - Precision@K on conversion prediction
    """
    
    def __init__(
        self,
        data_dir: Path,
        sample_size: Optional[int] = None,
        min_interactions: int = 1,
        random_seed: int = 42
    ):
        """
        Initialize IJCAI dataset loader.
        
        Args:
            data_dir: Path to IJCAI dataset directory
            sample_size: Number of samples to load (None = all)
            min_interactions: Filter items with < N interactions
            random_seed: Random seed for sampling
        """
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.min_interactions = min_interactions
        self.random_seed = random_seed
        
        # Data containers
        self.samples_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.context_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.shops_df: Optional[pd.DataFrame] = None
        
        # Statistics
        self.stats = {}
    
    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Load clicked samples with all features joined.
        
        Args:
            split: "train", "test_a", or "test_b"
        
        Returns:
            DataFrame with all features joined
        """
        # Determine file name
        if split == "train":
            filename = "round1_ijcai_18_train_20180301.txt"
        elif split == "test_a":
            filename = "round1_ijcai_18_test_a_20180301.txt"
        elif split == "test_b":
            filename = "round1_ijcai_18_test_b_20180418.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading {split} data from {filepath}")
        
        # Load the main data file (space-separated, has header row).
        # Actual column order from file:
        #   instance_id item_id item_category_list item_property_list
        #   item_brand_id item_city_id item_price_level item_sales_level
        #   item_collected_level item_pv_level user_id user_gender_id
        #   user_age_level user_occupation_id user_star_level context_id
        #   context_timestamp context_page_id predict_category_property
        #   shop_id shop_review_num_level shop_review_positive_rate
        #   shop_star_level shop_score_service shop_score_delivery
        #   shop_score_description is_trade
        # NOTE: is_trade is the LAST column, not second.
        
        self.samples_df = pd.read_csv(
            filepath,
            sep=" ",
            header=0,
            low_memory=False
        )
        
        logger.info(f"  Columns in file: {list(self.samples_df.columns)}")
        
        # Ensure numeric columns are properly typed
        numeric_cols = [
            "instance_id", "is_trade", "item_id", "item_brand_id", "item_city_id",
            "item_price_level", "item_sales_level", "item_collected_level", "item_pv_level",
            "user_id", "user_gender_id", "user_age_level", "user_occupation_id",
            "user_star_level", "context_id", "context_timestamp", "context_page_id",
            "shop_id", "shop_review_num_level", "shop_star_level"
        ]
        for col in numeric_cols:
            if col in self.samples_df.columns:
                self.samples_df[col] = pd.to_numeric(self.samples_df[col], errors="coerce")
        
        float_cols = ["shop_review_positive_rate", "shop_score_service",
                      "shop_score_delivery", "shop_score_description"]
        for col in float_cols:
            if col in self.samples_df.columns:
                self.samples_df[col] = pd.to_numeric(self.samples_df[col], errors="coerce")
        
        # Handle missing values (marked as -1)
        self.samples_df = self.samples_df.replace(-1, np.nan)
        
        # Sample if requested
        if self.sample_size and self.sample_size < len(self.samples_df):
            logger.info(f"Sampling {self.sample_size} records from {len(self.samples_df)}")
            self.samples_df = self.samples_df.sample(
                n=self.sample_size,
                random_state=self.random_seed
            ).reset_index(drop=True)
        
        # Filter items by minimum interactions if requested
        if self.min_interactions > 1:
            item_counts = self.samples_df["item_id"].value_counts()
            valid_items = item_counts[item_counts >= self.min_interactions].index
            
            before_count = len(self.samples_df)
            self.samples_df = self.samples_df[self.samples_df["item_id"].isin(valid_items)]
            after_count = len(self.samples_df)
            
            logger.info(f"Filtered items with < {self.min_interactions} interactions: "
                       f"{before_count} -> {after_count} samples")
        
        # Collect statistics
        self.stats["total_samples"] = len(self.samples_df)
        self.stats["unique_items"] = self.samples_df["item_id"].nunique()
        self.stats["unique_users"] = self.samples_df["user_id"].nunique()
        self.stats["unique_shops"] = self.samples_df["shop_id"].nunique()
        self.stats["conversion_rate"] = self.samples_df["is_trade"].mean()
        self.stats["missing_category"] = self.samples_df["item_category_list"].isna().sum()
        self.stats["missing_property"] = self.samples_df["item_property_list"].isna().sum()
        
        logger.info(f"Loaded {len(self.samples_df)} samples")
        logger.info(f"  Unique items: {self.stats['unique_items']}")
        logger.info(f"  Unique users: {self.stats['unique_users']}")
        logger.info(f"  Conversion rate: {self.stats['conversion_rate']:.2%}")
        
        return self.samples_df
    
    def get_item_texts(
        self,
        template: str = (
            "Category: {categories} | "
            "Properties: {properties} | "
            "Brand: {brand} | "
            "Price: {price_level} | "
            "Sales: {sales_level} | "
            "City: {city}"
        ),
        include_shop_features: bool = False
    ) -> Dict[int, str]:
        """
        Generate text representations from structured item metadata.
        
        This is a key pedagogical example: most e-commerce catalogs have
        structured metadata, not free-text descriptions. We serialize the
        structure into pseudo-text for encoding.
        
        Args:
            template: String template for combining fields
            include_shop_features: Whether to include shop-level features
        
        Returns:
            Dictionary mapping item_id -> text representation
        """
        if self.samples_df is None:
            raise ValueError("Must call load_data() first")
        
        logger.info("Generating item text representations from structured metadata...")
        
        # Get unique items with their features
        item_features = self.samples_df.groupby("item_id").first().reset_index()
        
        item_texts = {}
        
        for _, row in item_features.iterrows():
            # Parse category list (semicolon-separated hierarchy)
            categories = row["item_category_list"]
            if pd.notna(categories):
                # Format: "cat0;cat1;cat2" -> "cat0 > cat1 > cat2"
                categories = " > ".join(str(categories).split(";"))
            else:
                categories = "unknown"
            
            # Parse property list (semicolon-separated)
            properties = row["item_property_list"]
            if pd.notna(properties):
                properties = ", ".join(str(properties).split(";"))
            else:
                properties = "none"
            
            # Map numeric levels to descriptive text
            price_map = {0: "very low", 1: "low", 2: "medium-low", 3: "medium",
                        4: "medium-high", 5: "high", 6: "very high"}
            sales_map = {0: "new", 1: "low", 2: "moderate", 3: "high", 4: "very high"}
            
            price_level = price_map.get(int(row["item_price_level"]), "unknown") if pd.notna(row["item_price_level"]) else "unknown"
            sales_level = sales_map.get(int(row["item_sales_level"] // 4), "unknown") if pd.notna(row["item_sales_level"]) else "unknown"
            
            # Brand and city
            brand = f"brand_{row['item_brand_id']}" if pd.notna(row["item_brand_id"]) else "unknown"
            city = f"city_{row['item_city_id']}" if pd.notna(row["item_city_id"]) else "unknown"
            
            # Format text
            text = template.format(
                categories=categories,
                properties=properties,
                brand=brand,
                price_level=price_level,
                sales_level=sales_level,
                city=city
            )
            
            # Add shop features if requested
            if include_shop_features:
                shop_text = (
                    f" | Shop rating: {row['shop_star_level']:.0f} stars | "
                    f"Reviews: {row['shop_review_positive_rate']:.1%} positive"
                )
                text += shop_text
            
            # Clean up whitespace
            text = " ".join(text.split())
            
            item_texts[row["item_id"]] = text
        
        logger.info(f"Generated text for {len(item_texts)} items")
        
        return item_texts
    
    def get_query_texts(
        self,
        template: str = "User interested in: {categories_properties}"
    ) -> Dict[int, str]:
        """
        Generate query-side text from predicted category/property intent.
        
        The `predict_category_property` field contains the system's understanding
        of the user's search intent in structured form. We serialize this into
        query text for bi-encoder evaluation.
        
        Format: "category_A:property_A_1,property_A_2;category_B:-1;..."
        
        Args:
            template: String template for query text
        
        Returns:
            Dictionary mapping context_id -> query text
        """
        if self.samples_df is None:
            raise ValueError("Must call load_data() first")
        
        logger.info("Generating query texts from predicted intent...")
        
        # Get unique contexts with predicted intent
        context_features = self.samples_df.groupby("context_id")["predict_category_property"].first()
        
        query_texts = {}
        
        for context_id, intent_str in context_features.items():
            if pd.isna(intent_str):
                query_texts[context_id] = "general shopping"
                continue
            
            # Parse: "cat_A:prop1,prop2;cat_B:-1;..."
            try:
                categories_properties = []
                
                for cat_prop in str(intent_str).split(";"):
                    if ":" in cat_prop:
                        cat, props = cat_prop.split(":", 1)
                        
                        # Format category
                        cat_text = f"category {cat}"
                        
                        # Format properties
                        if props and props != "-1":
                            prop_list = props.split(",")
                            prop_text = " with " + ", ".join(prop_list)
                            cat_text += prop_text
                        
                        categories_properties.append(cat_text)
                
                # Combine all categories
                combined = " or ".join(categories_properties) if categories_properties else "general shopping"
                
            except Exception as e:
                logger.warning(f"Failed to parse intent '{intent_str}': {e}")
                combined = "general shopping"
            
            # Format using template
            query_text = template.format(categories_properties=combined)
            query_texts[context_id] = query_text
        
        logger.info(f"Generated {len(query_texts)} query texts")
        
        return query_texts
    
    def get_biencoder_eval_data(
        self,
        relevance_threshold: str = "conversion"
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Prepare data for bi-encoder evaluation.
        
        Returns query-item pairs with relevance labels for Precision@K evaluation.
        
        Args:
            relevance_threshold: "conversion" (is_trade=1) or "click" (all samples)
        
        Returns:
            Tuple of (query_texts, item_texts, relevance_labels)
        """
        if self.samples_df is None:
            raise ValueError("Must call load_data() first")
        
        logger.info(f"Preparing bi-encoder evaluation data (relevance={relevance_threshold})...")
        
        # Get text representations
        item_text_map = self.get_item_texts()
        query_text_map = self.get_query_texts()
        
        query_texts = []
        item_texts = []
        relevance_labels = []
        
        for _, row in self.samples_df.iterrows():
            context_id = row["context_id"]
            item_id = row["item_id"]
            
            # Get texts
            if context_id not in query_text_map or item_id not in item_text_map:
                continue
            
            query_text = query_text_map[context_id]
            item_text = item_text_map[item_id]
            
            # Determine relevance
            if relevance_threshold == "conversion":
                relevance = int(row["is_trade"])
            elif relevance_threshold == "click":
                relevance = 1  # All samples are clicked in this dataset
            else:
                raise ValueError(f"Unknown relevance threshold: {relevance_threshold}")
            
            query_texts.append(query_text)
            item_texts.append(item_text)
            relevance_labels.append(relevance)
        
        logger.info(f"Prepared {len(query_texts)} query-item pairs")
        logger.info(f"  Relevant pairs: {sum(relevance_labels)}")
        logger.info(f"  Relevance rate: {np.mean(relevance_labels):.2%}")
        
        return query_texts, item_texts, relevance_labels
    
    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "="*80)
        print("IJCAI CVR Dataset Statistics")
        print("="*80)
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.4f}")
            else:
                print(f"  {key:.<50} {value}")
        print("="*80 + "\n")


# ============================================================================
# Convenience Functions
# ============================================================================

def load_ijcai_data(
    data_dir: Path,
    split: str = "train",
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Convenience function to load IJCAI data."""
    dataset = IJCAIDataset(data_dir, sample_size=sample_size)
    return dataset.load_data(split=split)


# ============================================================================
# Main: Test Data Loading
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add parent directory to path for config import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import IJCAI_DATA_PATH
    
    print("Testing IJCAI CVR Dataset Loader")
    print("="*80)
    
    # Initialize dataset
    dataset = IJCAIDataset(
        data_dir=IJCAI_DATA_PATH,
        sample_size=10000  # Sample for testing
    )
    
    # Load data
    samples_df = dataset.load_data(split="train")
    print(f"\nSamples DataFrame shape: {samples_df.shape}")
    print("\nSample records:")
    print(samples_df.head())
    
    # Get item texts
    item_texts = dataset.get_item_texts()
    print(f"\nGenerated texts for {len(item_texts)} items")
    print("\nSample item texts:")
    for i, (item_id, text) in enumerate(list(item_texts.items())[:5]):
        print(f"\n[{i+1}] Item {item_id}")
        print(f"  {text}")
    
    # Get query texts
    query_texts = dataset.get_query_texts()
    print(f"\nGenerated texts for {len(query_texts)} queries")
    print("\nSample query texts:")
    for i, (context_id, text) in enumerate(list(query_texts.items())[:5]):
        print(f"\n[{i+1}] Context {context_id}")
        print(f"  {text}")
    
    # Get bi-encoder evaluation data
    queries, items, labels = dataset.get_biencoder_eval_data()
    print(f"\nBi-encoder eval data:")
    print(f"  Total pairs: {len(queries)}")
    print(f"  Relevant pairs: {sum(labels)}")
    print(f"  Relevance rate: {np.mean(labels):.2%}")
    
    # Print statistics
    dataset.print_stats()
