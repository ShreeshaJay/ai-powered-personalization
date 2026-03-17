"""
Data loader for Amazon ESCI (Shopping Queries Dataset).

Loads the US-English subset of the ESCI dataset, providing:
  - Product catalog with serialized text representations
  - Query-product relevance judgments with ESCI labels (E, S, C, I)
  - Per-query candidate structures for evaluation

Follows the dataset loader pattern from Chapter 10 (AmazonKDDDataset).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from config import ESCIDataConfig, ESCI_GAINS, DEFAULT_DATA_CONFIG

logger = logging.getLogger(__name__)


class ESCIDataset:
    """Amazon ESCI Shopping Queries Dataset loader.

    The ESCI dataset contains real Amazon shopper queries matched with
    products and annotated with 4-level relevance labels:
      E (Exact)      - Product is relevant and satisfies the query
      S (Substitute) - Product is somewhat relevant, close but not exact
      C (Complement) - Complementary product (e.g., case for a phone query)
      I (Irrelevant)  - Not relevant to the query

    This loader filters to US-English locale and provides two views:
      1. Full product catalog for indexing (1.2M products)
      2. Per-query candidate sets for evaluation (~16 labeled products per query)
    """

    def __init__(self, config: ESCIDataConfig = DEFAULT_DATA_CONFIG):
        self.config = config
        self.products_df: Optional[pd.DataFrame] = None
        self.examples_df: Optional[pd.DataFrame] = None
        self._product_texts: Optional[Dict[str, str]] = None
        self._product_id_list: Optional[List[str]] = None

    def load_products(self) -> pd.DataFrame:
        """Load US product catalog from parquet."""
        logger.info("Loading product catalog...")
        products_path = self.config.data_dir / self.config.products_file

        self.products_df = pd.read_parquet(products_path)
        self.products_df = self.products_df[
            self.products_df["product_locale"] == self.config.locale
        ].copy()

        text_cols = ["product_title", "product_bullet_point",
                     "product_description", "product_brand", "product_color"]
        for col in text_cols:
            self.products_df[col] = self.products_df[col].fillna("")

        if self.config.max_products > 0 and len(self.products_df) > self.config.max_products:
            if self.examples_df is None:
                self.load_examples()
            required_ids = set(self.examples_df["product_id"].unique())
            other_products = self.products_df[
                ~self.products_df["product_id"].isin(required_ids)
            ]
            sampled_other = other_products.sample(
                n=min(self.config.max_products - len(required_ids), len(other_products)),
                random_state=self.config.random_seed
            )
            required_products = self.products_df[
                self.products_df["product_id"].isin(required_ids)
            ]
            self.products_df = pd.concat([required_products, sampled_other], ignore_index=True)

        self.products_df = self.products_df.drop(columns=["product_locale"])
        self._product_texts = None
        self._product_id_list = None

        logger.info(f"  Products loaded: {len(self.products_df):,}")
        return self.products_df

    def load_examples(self) -> pd.DataFrame:
        """Load query-product relevance judgments from parquet."""
        logger.info("Loading query-product judgments...")
        examples_path = self.config.data_dir / self.config.examples_file

        self.examples_df = pd.read_parquet(examples_path)
        self.examples_df = self.examples_df[
            (self.examples_df["product_locale"] == self.config.locale) &
            (self.examples_df["split"] == self.config.split)
        ].copy()

        self.examples_df = self.examples_df[
            ["query_id", "query", "product_id", "esci_label"]
        ].reset_index(drop=True)

        logger.info(f"  Judgments loaded: {len(self.examples_df):,}")
        logger.info(f"  Unique queries: {self.examples_df['query_id'].nunique():,}")
        logger.info(f"  Unique products: {self.examples_df['product_id'].nunique():,}")
        return self.examples_df

    def get_product_texts(self) -> Dict[str, str]:
        """Generate text representation for each product using config template."""
        if self._product_texts is not None:
            return self._product_texts

        if self.products_df is None:
            self.load_products()

        logger.info("Serializing product texts...")
        product_texts = {}
        for _, row in self.products_df.iterrows():
            text = self.config.text_template.format(
                title=row["product_title"],
                bullet_point=row["product_bullet_point"],
                brand=row["product_brand"],
                color=row["product_color"],
                description=row["product_description"],
            )
            text = " ".join(text.split())
            product_texts[row["product_id"]] = text

        self._product_texts = product_texts
        logger.info(f"  Product texts serialized: {len(product_texts):,}")

        sample_id = list(product_texts.keys())[0]
        sample_text = product_texts[sample_id][:200]
        logger.info(f"  Sample [{sample_id}]: {sample_text}...")
        return product_texts

    def get_product_texts_list(self) -> Tuple[List[str], List[str]]:
        """Return parallel lists of (product_ids, texts) for batch encoding."""
        product_texts = self.get_product_texts()
        if self._product_id_list is None:
            self._product_id_list = sorted(product_texts.keys())
        product_ids = self._product_id_list
        texts = [product_texts[pid] for pid in product_ids]
        return product_ids, texts

    def get_query_candidates(self, max_queries: int = 0) -> Dict[int, Dict[str, Any]]:
        """Build per-query candidate structures for score-based evaluation.

        Returns dict: query_id -> {query_text, candidate_ids, labels, gains}
        """
        if self.examples_df is None:
            self.load_examples()

        query_groups = self.examples_df.groupby("query_id")
        all_query_ids = list(query_groups.groups.keys())

        if max_queries > 0 and len(all_query_ids) > max_queries:
            rng = np.random.RandomState(self.config.random_seed)
            selected_ids = rng.choice(all_query_ids, size=max_queries, replace=False)
        else:
            selected_ids = all_query_ids

        query_candidates = {}
        for qid in selected_ids:
            group = query_groups.get_group(qid)
            query_text = group["query"].iloc[0]
            candidate_ids = group["product_id"].tolist()
            labels = group["esci_label"].tolist()
            gains = np.array([ESCI_GAINS.get(l, 0) for l in labels], dtype=np.float32)
            query_candidates[qid] = {
                "query_text": query_text,
                "candidate_ids": candidate_ids,
                "labels": labels,
                "gains": gains,
            }

        logger.info(f"  Query candidates built: {len(query_candidates):,} queries")
        avg_candidates = np.mean([len(v["candidate_ids"]) for v in query_candidates.values()])
        logger.info(f"  Avg candidates per query: {avg_candidates:.1f}")
        return query_candidates

    def get_evaluation_queries(self, max_queries: int = 0) -> Dict[int, Dict[str, Any]]:
        """Build per-query ground truth for full-catalog retrieval evaluation.

        Returns dict: query_id -> {query_text, exact_ids, exact_substitute_ids, all_labeled_ids}
        """
        if self.examples_df is None:
            self.load_examples()

        query_groups = self.examples_df.groupby("query_id")
        all_query_ids = list(query_groups.groups.keys())

        if max_queries > 0 and len(all_query_ids) > max_queries:
            rng = np.random.RandomState(self.config.random_seed)
            selected_ids = rng.choice(all_query_ids, size=max_queries, replace=False)
        else:
            selected_ids = all_query_ids

        eval_queries = {}
        for qid in selected_ids:
            group = query_groups.get_group(qid)
            query_text = group["query"].iloc[0]
            exact_ids = set(group[group["esci_label"] == "E"]["product_id"])
            substitute_ids = set(group[group["esci_label"] == "S"]["product_id"])
            all_labeled = dict(zip(group["product_id"], group["esci_label"]))
            eval_queries[qid] = {
                "query_text": query_text,
                "exact_ids": exact_ids,
                "exact_substitute_ids": exact_ids | substitute_ids,
                "all_labeled_ids": all_labeled,
            }

        logger.info(f"  Evaluation queries built: {len(eval_queries):,}")
        avg_exact = np.mean([len(v["exact_ids"]) for v in eval_queries.values()])
        logger.info(f"  Avg Exact products per query: {avg_exact:.1f}")
        return eval_queries

    def get_curated_pool_data(
        self,
        max_queries: int = 2000,
        split: str = "test",
    ) -> Tuple[Dict[int, Dict[str, Any]], set]:
        """Build a curated evaluation pool (Qdrant-style).

        Instead of evaluating against the full 1.2M catalog, this creates a
        small pool of products that have ESCI labels for the selected queries.
        This makes label coverage much higher, enabling meaningful NDCG/MRR.

        Args:
            max_queries: Number of test queries to include
            split: Dataset split to use ("test" recommended for evaluation)

        Returns:
            Tuple of:
              - query_data: {query_id: {query_text, labeled_products: {pid: label}}}
              - pool_product_ids: set of all product IDs in the curated pool
        """
        logger.info(f"Building curated pool from split='{split}'...")

        # Load examples for the specified split
        examples_path = self.config.data_dir / self.config.examples_file
        all_examples = pd.read_parquet(examples_path)
        pool_examples = all_examples[
            (all_examples["product_locale"] == self.config.locale) &
            (all_examples["split"] == split)
        ][["query_id", "query", "product_id", "esci_label"]].copy()

        logger.info(f"  Split='{split}' judgments: {len(pool_examples):,}")

        # Select queries
        query_groups = pool_examples.groupby("query_id")
        all_query_ids = list(query_groups.groups.keys())

        if max_queries > 0 and len(all_query_ids) > max_queries:
            rng = np.random.RandomState(self.config.random_seed)
            selected_ids = rng.choice(all_query_ids, size=max_queries, replace=False)
        else:
            selected_ids = all_query_ids

        # Build per-query structures and collect pool products
        query_data = {}
        pool_product_ids = set()
        for qid in selected_ids:
            group = query_groups.get_group(qid)
            query_text = group["query"].iloc[0]
            labeled_products = dict(zip(group["product_id"], group["esci_label"]))
            query_data[qid] = {
                "query_text": query_text,
                "labeled_products": labeled_products,
            }
            pool_product_ids.update(labeled_products.keys())

        logger.info(f"  Curated pool: {len(query_data):,} queries, "
                    f"{len(pool_product_ids):,} unique products")

        avg_labeled = np.mean([len(v["labeled_products"]) for v in query_data.values()])
        logger.info(f"  Avg labeled products per query: {avg_labeled:.1f}")

        return query_data, pool_product_ids

    def get_product_id_to_index(self) -> Dict[str, int]:
        """Return mapping from product_id to index in the sorted product list."""
        product_ids, _ = self.get_product_texts_list()
        return {pid: idx for idx, pid in enumerate(product_ids)}

    def print_stats(self) -> None:
        """Print dataset statistics."""
        if self.products_df is None:
            self.load_products()
        if self.examples_df is None:
            self.load_examples()

        print(f"\n{'='*60}")
        print(f"  Amazon ESCI Dataset Statistics (locale={self.config.locale})")
        print(f"{'='*60}")
        print(f"  Products: {len(self.products_df):,}")
        print(f"  Judgments: {len(self.examples_df):,}")
        print(f"  Unique queries: {self.examples_df['query_id'].nunique():,}")

        print(f"\n  Label distribution:")
        label_counts = self.examples_df["esci_label"].value_counts()
        for label in ["E", "S", "C", "I"]:
            count = label_counts.get(label, 0)
            pct = 100 * count / len(self.examples_df)
            print(f"    {label} (gain={ESCI_GAINS[label]}): {count:>10,}  ({pct:5.1f}%)")

        qlen = self.examples_df.groupby("query_id")["query"].first().str.split().str.len()
        print(f"\n  Query word count: mean={qlen.mean():.1f}, "
              f"median={qlen.median():.0f}, "
              f"p5={qlen.quantile(0.05):.0f}, "
              f"p95={qlen.quantile(0.95):.0f}")

        ppq = self.examples_df.groupby("query_id").size()
        print(f"  Products per query: mean={ppq.mean():.1f}, "
              f"median={ppq.median():.0f}, "
              f"p95={ppq.quantile(0.95):.0f}")

        print(f"\n  Product metadata coverage:")
        for col in ["product_title", "product_bullet_point",
                     "product_description", "product_brand", "product_color"]:
            non_empty = (self.products_df[col].str.len() > 0).sum()
            pct = 100 * non_empty / len(self.products_df)
            print(f"    {col}: {pct:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    dataset = ESCIDataset()
    dataset.load_products()
    dataset.load_examples()
    dataset.print_stats()

    qc = dataset.get_query_candidates(max_queries=100)
    print(f"\nSample query candidates: {len(qc)} queries")
    sample_qid = list(qc.keys())[0]
    sample = qc[sample_qid]
    print(f"  Query: \"{sample['query_text']}\"")
    print(f"  Candidates: {len(sample['candidate_ids'])}")
    print(f"  Labels: {sample['labels']}")

    product_ids, texts = dataset.get_product_texts_list()
    print(f"\nProduct texts: {len(product_ids)} products")
    print(f"  Sample text: {texts[0][:150]}...")
