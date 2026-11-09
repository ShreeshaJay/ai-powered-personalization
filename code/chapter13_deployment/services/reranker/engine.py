"""
Value function and re-ranker (Section 13.6): Chapter 8's ordering steps.

    1. ValueFunction.compute_value with Chapter 8's task mapping:
       P(listen) = engagement, E[engagement] = play_ratio, P(like) = like,
       P(dislike) = dislike. If any candidate lacks task scores (the heavy
       ranker was skipped), the incoming model score is used unchanged.
    2. MMR over content embeddings, selecting num_results x multiplier items.
    3. ArtistPacer (max per artist and album, deferring violations).
    4. Truncate to num_results.

Chapter 8's MMRReranker compares each candidate with each selected item one
pair at a time through scikit-learn, which is too slow for a request path.
VectorizedMMRReranker subclasses it and keeps the same selection rule, tie
order and handling of missing embeddings, but keeps a running maximum
similarity per candidate and updates it with one matrix-vector product per
step. tests/test_reranker.py checks that both produce the same order.
"""

from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import published, warmup
from common.artifacts import read_json, require_file, verify_files
from contracts import RerankCandidate, TaskScores

_mmr = published.chapter8("mmr_diversity")
_rules = published.chapter8("business_rules")


class _RowMapping(Mapping):
    """item_id -> embedding row, without copying the matrix into a dict."""

    def __init__(self, item_ids: np.ndarray, matrix: np.ndarray):
        self._rows = {int(item): row for row, item in enumerate(item_ids)}
        self._matrix = matrix

    def __getitem__(self, item_id):
        return self._matrix[self._rows[item_id]]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def row(self, item_id) -> Optional[int]:
        return self._rows.get(item_id)


class VectorizedMMRReranker(_mmr.MMRReranker):
    def __init__(self, item_ids: np.ndarray, embeddings: np.ndarray, config=None):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        super().__init__(_RowMapping(item_ids, embeddings), config)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._unit = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)

    def rerank(self, candidates: List[Tuple[int, float]], top_k: Optional[int] = None,
               lambda_param: Optional[float] = None) -> List[int]:
        if not candidates:
            return []
        if self.config.similarity_metric != "cosine":
            return super().rerank(candidates, top_k, lambda_param)
        top_k = top_k or self.config.top_k
        lam = lambda_param if lambda_param is not None else self.config.lambda_param

        item_ids = [item_id for item_id, _ in candidates]
        scores = np.array([score for _, score in candidates], dtype=np.float32)
        if self.config.normalize_relevance:
            scores = self._normalize_scores(scores)
        score_map = {item_id: score for item_id, score in zip(item_ids, scores)}
        # The published loop scans `set(item_ids)`; scanning in the same order
        # makes ties resolve to the same item.
        order = list(set(item_ids))
        relevance = np.array([score_map[i] for i in order], dtype=np.float32)
        rows = [self.embeddings.row(i) for i in order]
        has_embedding = np.array([r is not None for r in rows])
        unit = np.zeros((len(order), self._unit.shape[1]), dtype=np.float32)
        if has_embedding.any():
            unit[has_embedding] = self._unit[[r for r in rows if r is not None]]

        max_similarity = np.zeros(len(order), dtype=np.float32)
        available = np.ones(len(order), dtype=bool)
        selected: List[int] = []
        while len(selected) < top_k and available.any():
            mmr = np.float32(lam) * relevance - np.float32(1 - lam) * max_similarity
            mmr = np.where(available, mmr, -np.inf)
            best = int(np.argmax(mmr))
            if not mmr[best] > -np.inf:
                break
            selected.append(order[best])
            available[best] = False
            if has_embedding[best]:
                similarity = unit @ unit[best]
                similarity[~has_embedding] = 0.0
                np.maximum(max_similarity, similarity, out=max_similarity)
        return selected


class RerankerEngine:
    def __init__(self, item_ids: np.ndarray, embeddings: np.ndarray, manifest: dict, config_store):
        self.mmr = VectorizedMMRReranker(item_ids, embeddings, _mmr.DiversityConfig())
        self.manifest = manifest
        self.embedding_version = manifest["embedding_version"]
        self.config_store = config_store

    @classmethod
    def load(cls, bundle_dir: Path, config_store) -> "RerankerEngine":
        directory = Path(bundle_dir) / "reranker"
        manifest = read_json(directory / "reranker_manifest.json", "re-ranker manifest")
        verify_files(directory, manifest["files"], "re-ranker")
        item_ids = np.load(require_file(directory / "item_ids.npy", "re-ranker item IDs"))
        embeddings = np.load(require_file(directory / "content_embeddings.npy", "re-ranker embeddings"))
        config_store.load_initial()
        return cls(item_ids, embeddings, manifest, config_store)

    def rerank(self, candidates: Sequence, num_results: int, policy_version: Optional[str] = None) -> Dict:
        active, policy_pin = self.config_store.resolve(policy_version)
        config = active.config
        item_ids = [c.item_id for c in candidates]
        if all(c.task_scores is not None for c in candidates):
            values = active.value_function.compute_value(
                p_listen=np.array([c.task_scores.engagement for c in candidates]),
                p_like=np.array([c.task_scores.like for c in candidates]),
                e_engagement=np.array([c.task_scores.play_ratio for c in candidates]),
                p_dislike=np.array([c.task_scores.dislike for c in candidates]),
            ).astype(np.float64)
            value_function_applied = True
        else:
            values = np.array([c.score for c in candidates], dtype=np.float64)
            value_function_applied = False

        keep = np.isfinite(values)   # min_listen_threshold marks filtered items with -inf
        scored = [(i, float(v)) for i, v, k in zip(item_ids, values, keep) if k]
        value_of = dict(scored)
        pool = num_results * config.diversity.candidate_multiplier
        if config.diversity.enabled and scored:
            ranked = self.mmr.rerank(scored, top_k=pool, lambda_param=config.diversity.lambda_param)
        else:
            ranked = [i for i, _ in sorted(scored, key=lambda pair: -pair[1])[:pool]]

        if config.pacing.enabled:
            pacer = _rules.ArtistPacer(
                {c.item_id: c.artist_id for c in candidates if c.artist_id is not None},
                {c.item_id: c.album_id for c in candidates if c.album_id is not None},
                active.pacing,
            )
            ranked = pacer.apply_pacing(ranked)
        final = ranked[:num_results]
        return {
            "items": [(i, value_of[i]) for i in final],
            "config_version": config.config_version,
            "policy_pin": policy_pin,
            "value_function_applied": value_function_applied,
            "diversity_applied": bool(config.diversity.enabled and scored),
            "pacing_applied": bool(config.pacing.enabled),
        }

    def warm_up(self, num_candidates: int = 100, num_results: int = 20) -> float:
        neutral = TaskScores(engagement=0.5, completion=0.5, play_ratio=0.5, like=0.5, dislike=0.5)
        candidates = [RerankCandidate(item_id=item_id, score=1.0 / (rank + 1), task_scores=neutral)
                      for rank, item_id in enumerate(islice(self.mmr.embeddings, num_candidates))]
        return warmup.timed(lambda: self.rerank(candidates, num_results))
