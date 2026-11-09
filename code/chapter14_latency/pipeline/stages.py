"""
An instrumented, in-process multi-stage recommendation pipeline.

One request runs through the stages of the chapter's attribution framework:

    S1 queue            waiting for a free server worker (plus, labelled
                        separately, any time the load generator sent late)
    S2 context          parse the request, build the user vector from history
    S3 retrieval        two parallel sources, then a merge:
                          two-tower: simulated network round trip + FAISS IVF search
                          popularity: in-process, naive recompute or precomputed
    S4 features         candidate features from a simulated remote store,
                        fetched in parallel chunks (optional cache, optional hedging)
    S5 light ranker     XGBoost over all candidates
       heavy ranker     DCN-V2 over the light ranker's top candidates,
                        one call per item or one batched call
    S6 re-rank          maximal marginal relevance, naive loops or vectorized
    S7 serialize        JSON response
    R  residual         end-to-end time not covered by any span

Model inference, search, re-ranking, and serialization are real computation.
Only the two remote calls (retrieval service round trip and feature store)
are simulated with calibrated sleeps, which is why Part B runs a timer
granularity check first.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch

from latency.timing import GENERATOR_LAG, SERVICE, WAIT, Trace, now_ns
from pipeline.build_artifacts import (Artifacts, UID_BUCKETS, rank_features, user_features)


@dataclass(frozen=True)
class PipelineConfig:
    candidates: int = 500             # two-tower candidates (--candidates)
    popularity_k: int = 50            # extra unseen items from the popularity source
    heavy_k: int = 100                # candidates passed to the heavy ranker
    slate_size: int = 20
    feature_cache: bool = False       # cache features of the most popular items
    cache_fraction: float = 0.2
    heavy_batching: str = "batched"   # batched | per-item
    mmr: str = "vectorized"           # vectorized | naive
    popularity: str = "naive"         # naive | precomputed
    hedge_feature_fetch: bool = False
    hedge_quantile: float = 95.0      # hedge after this percentile of store delay
    chunk_size: int = 100             # keys per feature-store call
    mmr_lambda: float = 0.7
    # Simulated remote calls, in milliseconds (lognormal median and log-sd).
    retrieval_rtt_median_ms: float = 1.5
    retrieval_rtt_sigma: float = 0.3
    store_median_ms: float = 2.0
    store_sigma: float = 0.35
    store_slow_prob: float = 0.02     # rare slow replica
    store_slow_factor: float = 8.0
    seed: int = 14

    def label(self) -> str:
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                        if getattr(self, f.name) != getattr(DEFAULT_CONFIG, f.name)) or "baseline"


DEFAULT_CONFIG = PipelineConfig()

# Command-line lever names -> config field and parser.
LEVERS = {
    "candidates": ("candidates", int),
    "feature-cache": ("feature_cache", lambda v: v.lower() in ("1", "on", "true", "yes")),
    "heavy-batching": ("heavy_batching", str),
    "mmr": ("mmr", str),
    "popularity": ("popularity", str),
    "hedge-feature-fetch": ("hedge_feature_fetch", lambda v: v.lower() in ("1", "on", "true", "yes")),
}
CHOICES = {"heavy_batching": ("per-item", "batched"), "mmr": ("naive", "vectorized"),
           "popularity": ("naive", "precomputed")}

# Spans (with their descendants) each lever acts on. The bound on a lever's
# gain is the baseline's tail critical-path time inside these spans.
LEVER_SPANS = {
    "candidates": ("retrieval", "features", "light_ranker"),
    "feature_cache": ("features",),
    "hedge_feature_fetch": ("features",),
    "heavy_batching": ("heavy_ranker",),
    "mmr": ("rerank",),
    "popularity": ("popularity",),
}


def apply_levers(base: PipelineConfig, spec: str) -> PipelineConfig:
    """Parse ``"mmr=vectorized,heavy-batching=batched"`` into a modified config."""
    changes = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        if "=" not in part:
            raise ValueError(f"lever must look like name=value, got {part!r}")
        name, value = (s.strip() for s in part.split("=", 1))
        if name not in LEVERS:
            raise ValueError(f"unknown lever {name!r}; choose from {sorted(LEVERS)}")
        field_name, parse = LEVERS[name]
        parsed = parse(value)
        if field_name in CHOICES and parsed not in CHOICES[field_name]:
            raise ValueError(f"{name} must be one of {CHOICES[field_name]}")
        changes[field_name] = parsed
    return replace(base, **changes)


def mmr_naive(scores: np.ndarray, vectors: np.ndarray, k: int, lam: float) -> List[int]:
    """Maximal marginal relevance with Python loops: O(k * n * k) scalar dot products."""
    selected: List[int] = []
    remaining = list(range(len(scores)))
    while remaining and len(selected) < k:
        best, best_val = -1, -np.inf
        for j in remaining:
            div = max((float(np.dot(vectors[j], vectors[s])) for s in selected), default=0.0)
            val = lam * float(scores[j]) - (1 - lam) * div
            if val > best_val:
                best, best_val = j, val
        selected.append(best)
        remaining.remove(best)
    return selected


def mmr_vectorized(scores: np.ndarray, vectors: np.ndarray, k: int, lam: float) -> List[int]:
    """Same selection as ``mmr_naive`` with one similarity matrix and running maxima."""
    n = len(scores)
    sims = vectors @ vectors.T
    max_sim = np.full(n, -np.inf)
    taken = np.zeros(n, dtype=bool)
    selected: List[int] = []
    for _ in range(min(k, n)):
        div = max_sim if selected else 0.0
        val = lam * scores - (1 - lam) * div
        val[taken] = -np.inf
        j = int(np.argmax(val))
        selected.append(j)
        taken[j] = True
        max_sim = np.maximum(max_sim, sims[j])
    return selected


class Pipeline:
    """Serves requests from prebuilt artifacts according to a ``PipelineConfig``."""

    def __init__(self, art: Artifacts, cfg: PipelineConfig, io_threads: int = 64):
        self.art, self.cfg = art, cfg
        self.io = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="io")
        self.vectors = art.catalog.vectors
        self.log_pop = art.item_features[:, 0]
        head = int(art.recent_events.max()) + 1
        counts = np.bincount(art.recent_events, minlength=head)
        self.pop_precomputed = art.popular_items[np.argsort(-counts, kind="stable")]
        n = art.catalog.n_items
        self.cached = np.zeros(n, dtype=bool)
        self.cached[art.popular_items[: int(cfg.cache_fraction * n)]] = True
        rng = np.random.default_rng(cfg.seed + 1)
        self.hedge_delay_ms = float(np.percentile(self._store_delays(rng, 200_000), cfg.hedge_quantile))

    def close(self) -> None:
        self.io.shutdown(wait=True)

    # ------------------------------------------------------------ simulation
    def _store_delays(self, rng: np.random.Generator, n: int) -> np.ndarray:
        c = self.cfg
        d = c.store_median_ms * np.exp(c.store_sigma * rng.standard_normal(n))
        return np.where(rng.random(n) < c.store_slow_prob, d * c.store_slow_factor, d)

    def _fetch_delays(self, rng: np.random.Generator, n_chunks: int) -> Tuple[np.ndarray, np.ndarray]:
        primary = self._store_delays(rng, n_chunks)
        backup = self._store_delays(rng, n_chunks)
        if not self.cfg.hedge_feature_fetch:
            return primary, np.zeros(n_chunks, dtype=bool)
        hedged = primary > self.hedge_delay_ms
        effective = np.where(hedged, np.minimum(primary, self.hedge_delay_ms + backup), primary)
        return effective, hedged

    # --------------------------------------------------------------- branches
    def _two_tower(self, tr: Trace, parent, uvec: np.ndarray, delay_ms: float, submit_ns: int):
        start = now_ns()
        grp = tr.add("two_tower", "S3", SERVICE, submit_ns, 0, parent)
        tr.add("two_tower.pool_wait", "S3", WAIT, submit_ns, start, grp)
        t = now_ns()
        time.sleep(delay_ms / 1000)
        tr.add("two_tower.network", "S3", WAIT, t, now_ns(), grp)
        t = now_ns()
        scores, ids = self.art.index.search(uvec[None, :], self.cfg.candidates)
        tr.add("two_tower.search", "S3", SERVICE, t, now_ns(), grp)
        grp.end_ns = now_ns()
        return scores[0], ids[0]

    def _popularity(self, tr: Trace, parent, history: np.ndarray, submit_ns: int):
        start = now_ns()
        grp = tr.add("popularity", "S3", SERVICE, submit_ns, 0, parent)
        tr.add("popularity.pool_wait", "S3", WAIT, submit_ns, start, grp)
        t = now_ns()
        want = self.cfg.popularity_k + len(history)
        if self.cfg.popularity == "naive":
            counts = np.bincount(self.art.recent_events)
            top = np.argpartition(-counts, min(want, len(counts) - 1))[:want]
            top = top[np.argsort(-counts[top], kind="stable")]
            ids = self.art.popular_items[top]
        else:
            ids = self.pop_precomputed[:want]
        ids = ids[~np.isin(ids, history)][: self.cfg.popularity_k]
        tr.add("popularity.compute", "S3", SERVICE, t, now_ns(), grp)
        grp.end_ns = now_ns()
        return ids

    def _fetch_chunk(self, tr: Trace, parent, keys: np.ndarray, delay_ms: float, hedged: bool,
                     submit_ns: int):
        start = now_ns()
        grp = tr.add("features.fetch", "S4", SERVICE, submit_ns, 0, parent)
        tr.add("features.fetch.pool_wait", "S4", WAIT, submit_ns, start, grp)
        t = now_ns()
        time.sleep(delay_ms / 1000)
        rows = self.art.item_features[keys]
        tr.add("features.fetch.store_hedged" if hedged else "features.fetch.store", "S4", WAIT,
               t, now_ns(), grp)
        grp.end_ns = now_ns()
        return rows

    # ---------------------------------------------------------------- handler
    def handle(self, i: int, intended_ns: int, submitted_ns: int, start_ns: int) -> Dict[str, object]:
        cfg, art = self.cfg, self.art
        tr = Trace(i)
        root = tr.add("request", "R", SERVICE, intended_ns, 0)
        submitted_ns = max(submitted_ns, intended_ns)
        if submitted_ns > intended_ns:
            tr.add("queue.generator_lag", "S1", GENERATOR_LAG, intended_ns, submitted_ns, root)
        tr.add("queue", "S1", WAIT, submitted_ns, start_ns, root)
        rng = np.random.default_rng([cfg.seed, i])

        with tr.span("context", "S2", parent=root):
            payload = json.loads(art.requests[i % len(art.requests)])
            history = np.asarray(payload["history"], dtype=np.int64)
            hv = self.vectors[history]
            uvec = hv.mean(axis=0)
            uvec = (uvec / max(np.linalg.norm(uvec), 1e-12)).astype(np.float32)
            ufeat = user_features(hv, len(history))
            rtt = cfg.retrieval_rtt_median_ms * float(np.exp(cfg.retrieval_rtt_sigma * rng.standard_normal()))

        ret = tr.open("retrieval", "S3", parent=root)
        f_tt = self.io.submit(self._two_tower, tr, ret, uvec, rtt, now_ns())
        f_pop = self.io.submit(self._popularity, tr, ret, history, now_ns())
        ann_scores, ann_ids = f_tt.result()
        pop_ids = f_pop.result()
        with tr.span("retrieval.merge", "S3", parent=ret):
            keep = (ann_ids >= 0) & ~np.isin(ann_ids, history)
            ann_ids, ann_scores = ann_ids[keep], ann_scores[keep]
            extra = pop_ids[~np.isin(pop_ids, ann_ids)]
            cands = np.concatenate([ann_ids, extra])
            scores = np.concatenate([ann_scores, self.vectors[extra] @ uvec]).astype(np.float32)
            from_pop = np.concatenate([np.zeros(len(ann_ids)), np.ones(len(extra))]).astype(np.float32)
        ret.end_ns = now_ns()

        feat = tr.open("features", "S4", parent=root)
        with tr.span("features.cache", "S4", parent=feat):
            miss = cands[~self.cached[cands]] if cfg.feature_cache else cands
            chunks = [miss[j:j + cfg.chunk_size] for j in range(0, len(miss), cfg.chunk_size)]
            delays, hedged = self._fetch_delays(rng, len(chunks))
        futures = [self.io.submit(self._fetch_chunk, tr, feat, ch, float(d), bool(h), now_ns())
                   for ch, d, h in zip(chunks, delays, hedged)]
        for f in futures:
            f.result()
        with tr.span("features.assemble", "S4", parent=feat):
            item_feats = art.item_features[cands]
        feat.end_ns = now_ns()

        with tr.span("light_ranker", "S5", parent=root):
            x = rank_features(scores, cands, from_pop, item_feats, ufeat, self.log_pop)
            light = art.light_ranker.inplace_predict(x)
            k = min(cfg.heavy_k, len(cands))
            top = np.argpartition(-light, k - 1)[:k]

        with tr.span("heavy_ranker", "S5", parent=root):
            uid = torch.full((k,), payload["user_id"] % UID_BUCKETS, dtype=torch.long)
            sparse = {"uid": uid, "item_id": torch.from_numpy(cands[top] + 1),
                      "hour_of_day": torch.full((k,), 12, dtype=torch.long),
                      "day_of_week": torch.full((k,), 3, dtype=torch.long)}
            dense = torch.from_numpy(x[top])
            with torch.inference_mode():
                if cfg.heavy_batching == "batched":
                    logits = art.heavy_ranker(sparse, dense).numpy().ravel()
                else:
                    logits = np.array([
                        art.heavy_ranker({n: t[j:j + 1] for n, t in sparse.items()}, dense[j:j + 1]).item()
                        for j in range(k)])
            heavy = 1 / (1 + np.exp(-logits))

        with tr.span("rerank", "S6", parent=root):
            vecs = self.vectors[cands[top]].astype(np.float64)
            pick = (mmr_vectorized if cfg.mmr == "vectorized" else mmr_naive)(
                heavy.astype(np.float64), vecs, cfg.slate_size, cfg.mmr_lambda)

        with tr.span("serialize", "S7", parent=root):
            body = json.dumps({"user_id": payload["user_id"],
                               "items": [int(cands[top][j]) for j in pick],
                               "scores": [round(float(heavy[j]), 6) for j in pick]}).encode()

        root.end_ns = now_ns()
        return {"trace": tr, "chunks": len(chunks), "hedged": int(hedged.sum()),
                "cache_hits": int(len(cands) - len(miss)), "candidates": int(len(cands)),
                "bytes": len(body)}


def configure_threads() -> None:
    """One compute thread per library call; concurrency comes from the worker pool."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # can only be set once per process
    faiss.omp_set_num_threads(1)
