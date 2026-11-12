"""BM25 with bm25s, Lucene variant, English stopwords, and no stemming.

Judged products are scored by a direct lookup into the BM25 score array.
The search does not ask the library to rank the entire catalog and then
throw the ranking away.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import BM25Settings
from fusion import HitList, ranking_from_scores

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, settings: BM25Settings | None = None) -> None:
        self.settings = settings or BM25Settings()
        self.bm25 = None
        self.product_ids: list[str] = []
        self.id_to_row: dict[str, int] = {}

    def build(self, product_ids: list[str], texts: list[str]) -> None:
        import bm25s

        if len(product_ids) != len(texts):
            raise ValueError("product ids and texts have different lengths.")
        self.product_ids = list(product_ids)
        self.id_to_row = {product_id: row for row, product_id in enumerate(self.product_ids)}
        tokens = bm25s.tokenize(
            texts,
            stopwords=self.settings.stopwords,
            stemmer=None,
            return_ids=True,
            show_progress=False,
        )
        self.bm25 = bm25s.BM25(method=self.settings.method, k1=self.settings.k1, b=self.settings.b)
        self.bm25.index(tokens, show_progress=False)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(directory), corpus=None)
        np.save(directory / "product_ids.npy", np.array(self.product_ids))

    def load(self, directory: Path) -> None:
        import bm25s

        self.bm25 = bm25s.BM25.load(str(directory), load_corpus=False)
        self.product_ids = np.load(directory / "product_ids.npy", allow_pickle=True).tolist()
        self.id_to_row = {product_id: row for row, product_id in enumerate(self.product_ids)}

    def _query_tokens(self, query: str) -> list[str]:
        import bm25s

        tokenized = bm25s.tokenize(
            [query],
            stopwords=self.settings.stopwords,
            stemmer=None,
            return_ids=False,
            show_progress=False,
        )
        if not tokenized or not tokenized[0]:
            return []
        return list(tokenized[0])

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every indexed product. Empty queries score as zeros."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not built.")
        tokens = self._query_tokens(query)
        if not tokens:
            return np.zeros(len(self.product_ids), dtype=np.float64)
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float64).ravel()

    def score_products(self, query: str, product_ids: list[str]) -> np.ndarray:
        all_scores = self.scores(query)
        found = np.zeros(len(product_ids), dtype=np.float64)
        for position, product_id in enumerate(product_ids):
            row = self.id_to_row.get(product_id)
            if row is not None:
                found[position] = float(all_scores[row])
        return found

    def search(self, query: str, top_k: int) -> HitList:
        return ranking_from_scores(self.product_ids, self.scores(query), depth=top_k)
