"""
Amazon KDD Cup 2023 Dataset Loader for Item Embeddings

Loads product metadata (title, brand, description, color, material) and
user session data from the Multilingual Shopping Session Dataset.
We filter to English (UK locale) products only.

Dataset reference:
    https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge

Schema — products_train.csv:
    id, locale, title, price, brand, color, size, model, material, author, desc

Schema — sessions_train.csv:
    prev_items, next_item, locale
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazonKDDDataset:
    """
    Amazon KDD Cup 2023 dataset loader for product embeddings.

    Provides:
    - Product metadata with rich text (title, brand, description)
    - Next-item retrieval pairs from sequential session data
    """

    def __init__(
        self,
        data_dir: Path,
        locale: str = "UK",
    ):
        self.data_dir = Path(data_dir)
        self.locale = locale

        self.products_df: Optional[pd.DataFrame] = None
        self.sessions_df: Optional[pd.DataFrame] = None
        self.stats = {}

    def load_products(self, chunk_size: int = 50000) -> pd.DataFrame:
        """
        Load product catalog, filtered to the target locale.

        Uses chunked reading to stay under ~2GB peak memory even for
        the full 1.55M-row CSV on a 16GB machine.

        Returns:
            DataFrame with product metadata
        """
        products_path = self.data_dir / "products_train.csv"
        if not products_path.exists():
            raise FileNotFoundError(f"Products file not found: {products_path}")

        logger.info(f"Loading products from {products_path} (locale={self.locale})")

        # Only load columns we actually need
        use_cols = ["id", "locale", "title", "price", "brand", "desc",
                    "color", "material"]

        chunks = []
        for chunk in pd.read_csv(products_path, encoding="utf-8",
                                 usecols=use_cols, chunksize=chunk_size):
            filtered = chunk[chunk["locale"] == self.locale]
            if len(filtered) > 0:
                chunks.append(filtered)

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  {self.locale} products: {len(df)}")

        # Clean up missing values
        for col in ["title", "brand", "desc", "color", "material"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

        self.products_df = df

        self.stats["total_products"] = len(df)
        self.stats["unique_brands"] = df["brand"].replace("", np.nan).nunique()
        self.stats["has_description"] = (df["desc"] != "").sum()
        self.stats["has_brand"] = (df["brand"] != "").sum()
        self.stats["has_color"] = (df["color"] != "").sum()
        self.stats["has_material"] = (df["material"] != "").sum()
        self.stats["avg_title_length"] = df["title"].str.len().mean()

        logger.info(f"Loaded {len(df)} products")
        logger.info(f"  Unique brands: {self.stats['unique_brands']}")
        logger.info(f"  Has description: {self.stats['has_description']} "
                     f"({self.stats['has_description']/len(df)*100:.1f}%)")

        return self.products_df

    def load_sessions(
        self,
        product_ids: Optional[Set[str]] = None,
        chunk_size: int = 100000
    ) -> pd.DataFrame:
        """
        Load session data, filtered to the target locale.

        Uses chunked reading to keep memory usage low (the sessions file
        has 3.6M+ rows).

        Args:
            product_ids: If provided, only keep sessions where ALL items
                         are in this set (ensures we have embeddings for them)
            chunk_size: Rows per chunk when reading CSV

        Returns:
            DataFrame with parsed session data
        """
        sessions_path = self.data_dir / "sessions_train.csv"
        if not sessions_path.exists():
            raise FileNotFoundError(f"Sessions file not found: {sessions_path}")

        logger.info(f"Loading sessions from {sessions_path} (locale={self.locale})")

        chunks = []
        for chunk in pd.read_csv(sessions_path, encoding="utf-8",
                                 chunksize=chunk_size):
            filtered = chunk[chunk["locale"] == self.locale]
            if len(filtered) > 0:
                chunks.append(filtered)

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  {self.locale} sessions: {len(df)}")

        # Parse prev_items: format is space-separated ASINs in brackets
        # e.g.  ['B09W9FND7K' 'B09JSPLN1M']  (no commas)
        def parse_items(val):
            if pd.isna(val):
                return []
            s = str(val).strip()
            if s.startswith("["):
                s = s[1:]
            if s.endswith("]"):
                s = s[:-1]
            items = [tok.strip("'\" \t\n") for tok in s.split()]
            return [item for item in items if item]

        df["prev_items_list"] = df["prev_items"].apply(parse_items)
        df["session_length"] = df["prev_items_list"].apply(len)

        # Only keep sessions where the query (last prev item) and the
        # target (next_item) are both in the product catalog.
        if product_ids is not None:
            before = len(df)
            has_query = df["prev_items_list"].apply(
                lambda items: len(items) > 0 and items[-1] in product_ids
            )
            has_target = df["next_item"].apply(
                lambda x: pd.notna(x) and x in product_ids
            )
            df = df[has_query & has_target].reset_index(drop=True)
            logger.info(f"  Filtered (query & target in catalog): "
                        f"{before} -> {len(df)}")

        self.sessions_df = df

        self.stats["total_sessions"] = len(df)
        self.stats["avg_session_length"] = df["session_length"].mean()
        self.stats["median_session_length"] = df["session_length"].median()

        logger.info(f"Loaded {len(df)} sessions")
        logger.info(f"  Avg session length: {self.stats['avg_session_length']:.1f}")

        return self.sessions_df

    def get_item_texts(
        self,
        use_title: bool = True,
        use_brand: bool = True,
        use_description: bool = True,
        use_color: bool = False,
        use_material: bool = False,
        template: str = "{title} [SEP] {brand} [SEP] {description}"
    ) -> Dict[str, str]:
        """
        Generate text representations for each product.

        Args:
            use_title: Include product title
            use_brand: Include brand name
            use_description: Include product description
            use_color: Include color
            use_material: Include material
            template: String template for combining fields

        Returns:
            Dictionary mapping product_id (ASIN) -> text representation
        """
        if self.products_df is None:
            self.load_products()

        logger.info("Generating product text representations...")

        item_texts = {}

        for _, row in self.products_df.iterrows():
            fields = {
                "title": row["title"] if use_title else "",
                "brand": row["brand"] if use_brand else "",
                "description": row["desc"] if use_description else "",
                "color": row["color"] if use_color else "",
                "material": row["material"] if use_material else "",
            }

            text = template.format(**fields)
            text = " ".join(text.split())  # collapse whitespace

            item_texts[row["id"]] = text

        logger.info(f"Generated text for {len(item_texts)} products")

        return item_texts

    def get_next_item_pairs(
        self,
        min_prev_items: int = 1,
    ) -> Dict[str, set]:
        """
        Build next-item ground truth that respects temporal ordering.

        For each session, the last item in prev_items is treated as the
        query and next_item is the target. This mirrors a realistic
        retrieval scenario: "given the item a user just viewed, which
        item do they engage with next?"

        Args:
            min_prev_items: Skip sessions with fewer prior items

        Returns:
            Dict mapping query_product_id -> set of valid next_product_ids
            (aggregated across all sessions where the query appeared last)
        """
        if self.sessions_df is None:
            raise ValueError("Must call load_sessions() first")

        logger.info("Building next-item ground truth (temporal ordering)...")

        next_item_map: Dict[str, set] = defaultdict(set)
        total_pairs = 0

        for _, row in self.sessions_df.iterrows():
            prev_items = row["prev_items_list"]
            next_item = row.get("next_item")

            if len(prev_items) < min_prev_items:
                continue
            if pd.isna(next_item):
                continue

            query_item = prev_items[-1]
            if query_item != next_item:
                next_item_map[query_item].add(next_item)
                total_pairs += 1

        logger.info(
            f"Next-item ground truth: {len(next_item_map)} unique query items, "
            f"{total_pairs} total (query, next) pairs"
        )

        return next_item_map

    def get_sessions_as_sequences(self) -> List[List[str]]:
        """
        Return sessions as item-ID sequences for Item2Vec training.

        Each session becomes [prev_items..., next_item], forming
        a complete sequence of items a user engaged with.

        Returns:
            List of sessions, each a list of product-ID strings.
        """
        if self.sessions_df is None:
            raise ValueError("Must call load_sessions() first")

        sessions = []
        for _, row in self.sessions_df.iterrows():
            prev_items = row["prev_items_list"]
            next_item = row.get("next_item")

            seq = list(prev_items)
            if pd.notna(next_item) and next_item:
                seq.append(str(next_item))

            if len(seq) >= 2:
                sessions.append(seq)

        logger.info(f"Extracted {len(sessions):,} sequences for Item2Vec "
                     f"(avg length: {np.mean([len(s) for s in sessions]):.1f})")
        return sessions

    def print_stats(self):
        """Print dataset statistics."""
        print("\n" + "=" * 80)
        print(f"Amazon KDD 2023 Dataset Statistics ({self.locale})")
        print("=" * 80)
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.2f}")
            else:
                print(f"  {key:.<50} {value}")
        print("=" * 80 + "\n")


# ============================================================================
# Main: Test Data Loading
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import AMAZON_KDD_DATA_PATH

    print("Testing Amazon KDD Dataset Loader")
    print("=" * 80)

    dataset = AmazonKDDDataset(
        data_dir=AMAZON_KDD_DATA_PATH,
        locale="UK",
    )

    products_df = dataset.load_products()
    print(f"\nProducts shape: {products_df.shape}")
    print("\nSample products:")
    for _, row in products_df.head(3).iterrows():
        safe = lambda s: s.encode("ascii", errors="replace").decode() if s else "N/A"
        print(f"  {row['id']}: {safe(str(row['title'])[:80])}")

    item_texts = dataset.get_item_texts()
    sample_id = list(item_texts.keys())[0]
    safe_text = item_texts[sample_id].encode("ascii", errors="replace").decode()
    print(f"\nSample text ({sample_id}): {safe_text[:150]}...")

    product_ids = set(products_df["id"])
    sessions_df = dataset.load_sessions(product_ids=product_ids)
    print(f"\nSessions shape: {sessions_df.shape}")

    next_pairs = dataset.get_next_item_pairs(min_prev_items=1)
    print(f"Next-item query items: {len(next_pairs)}")

    dataset.print_stats()
