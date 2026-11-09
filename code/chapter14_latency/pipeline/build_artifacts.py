"""
Build everything the Exercise 1 Part B pipeline serves from.

* an IVF index over the item vectors (Coveo description vectors or the
  synthetic catalog);
* a precomputed popularity ranking and a buffer of "recent events" from
  which the naive popularity source recomputes it per request;
* an item feature table, standing in for a remote feature store;
* an XGBoost light ranker trained on synthetic labels with planted structure;
* a DCN-V2 heavy ranker (the vendored Chapter 7 model) with random weights.
  Inference cost does not depend on weight values, so training it would add
  minutes without changing any latency measurement.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
import torch
import xgboost as xgb

from ann.data import Catalog
from ann.sweep import build_index, default_nlist
from ranker.model_dcn_vendored import DCNV2, DCNV2Config

N_ITEM_FEATURES = 12
N_USER_FEATURES = 4
# ann score, log popularity, from-popularity-source flag, item features, user features
N_RANK_FEATURES = 3 + N_ITEM_FEATURES + N_USER_FEATURES
UID_BUCKETS = 100_000


@dataclass
class Artifacts:
    catalog: Catalog
    index: faiss.Index
    popular_items: np.ndarray       # item ids, most popular first
    recent_events: np.ndarray       # ranks into popular_items, for the naive source
    item_features: np.ndarray       # (n_items, N_ITEM_FEATURES), the "remote" store
    light_ranker: xgb.Booster
    heavy_ranker: DCNV2
    requests: List[bytes]           # serialized request payloads


def item_feature_table(catalog: Catalog, rng: np.random.Generator) -> np.ndarray:
    n = catalog.n_items
    log_pop = np.log1p(catalog.popularity / catalog.popularity.max() * 1e4)
    proj = rng.standard_normal((catalog.dim, N_ITEM_FEATURES - 2)).astype(np.float32)
    feats = np.empty((n, N_ITEM_FEATURES), dtype=np.float32)
    feats[:, 0] = log_pop
    feats[:, 1] = rng.random(n)                         # e.g. price bucket, freshness
    feats[:, 2:] = catalog.vectors @ proj / np.sqrt(catalog.dim)
    return feats


def user_features(history_vectors: np.ndarray, history_len: int) -> np.ndarray:
    return np.array([history_len, float(np.linalg.norm(history_vectors.mean(axis=0))),
                     float(history_vectors.std()), 1.0], dtype=np.float32)


def rank_features(ann_scores: np.ndarray, items: np.ndarray, from_pop: np.ndarray,
                  item_feats: np.ndarray, user_feats: np.ndarray, log_pop: np.ndarray) -> np.ndarray:
    n = len(items)
    out = np.empty((n, N_RANK_FEATURES), dtype=np.float32)
    out[:, 0] = ann_scores
    out[:, 1] = log_pop[items]
    out[:, 2] = from_pop
    out[:, 3:3 + N_ITEM_FEATURES] = item_feats
    out[:, 3 + N_ITEM_FEATURES:] = user_feats
    return out


def _planted_label(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logit = 3.0 * x[:, 0] + 0.4 * x[:, 1] + 0.8 * x[:, 3] * x[:, 4] - 0.5 * x[:, 5] - 1.5
    return (rng.random(len(x)) < 1 / (1 + np.exp(-logit))).astype(np.float32)


def train_light_ranker(catalog: Catalog, index: faiss.Index, item_feats: np.ndarray,
                       rng: np.random.Generator, n_queries: int, k: int) -> xgb.Booster:
    log_pop = item_feats[:, 0]
    rows = []
    qi = rng.choice(len(catalog.queries), size=min(n_queries, len(catalog.queries)), replace=False)
    scores, ids = index.search(catalog.queries[qi], k)
    for row, q in enumerate(qi):
        items = ids[row][ids[row] >= 0]
        hv = catalog.vectors[catalog.histories[q]]
        uf = user_features(hv, len(catalog.histories[q]))
        rows.append(rank_features(scores[row][: len(items)], items, np.zeros(len(items)),
                                  item_feats[items], uf, log_pop))
    x = np.concatenate(rows)
    y = _planted_label(x, rng)
    dtrain = xgb.DMatrix(x, label=y)
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "nthread": 0,
              "seed": int(rng.integers(1 << 30)), "tree_method": "hist"}
    booster = xgb.train(params, dtrain, num_boost_round=100)
    booster.set_param({"nthread": 1})
    return booster


def heavy_ranker(n_items: int, seed: int) -> DCNV2:
    torch.manual_seed(seed)
    config = DCNV2Config(
        sparse_features={"uid": UID_BUCKETS, "item_id": n_items + 1, "hour_of_day": 24, "day_of_week": 7},
        dense_features=[f"f{i}" for i in range(N_RANK_FEATURES)],
        embed_dim=16, cross_layers=3, cross_rank=32, mlp_dims=[256, 128, 64],
        dropout=0.1, use_bn=True, structure="parallel",
    )
    model = DCNV2(config)
    model.eval()
    return model


def build(catalog: Catalog, seed: int, quick: bool, nprobe: int = 16, log=print) -> Artifacts:
    rng = np.random.default_rng(seed)
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    log(f"  building IVF index over {catalog.n_items:,} items (nlist={default_nlist(catalog.n_items)}) ...")
    index, _ = build_index("ivf_flat", catalog.vectors, seed=seed)
    index.nprobe = nprobe

    popular = np.argsort(-catalog.popularity, kind="stable").astype(np.int64)
    head = min(10_000, catalog.n_items)
    w = catalog.popularity[popular[:head]]
    recent = rng.choice(head, size=20_000 if quick else 100_000, p=w / w.sum()).astype(np.int64)

    feats = item_feature_table(catalog, rng)
    log("  training XGBoost light ranker on planted labels ...")
    light = train_light_ranker(catalog, index, feats, rng, n_queries=200 if quick else 500, k=200)
    log("  initializing DCN-V2 heavy ranker (random weights; cost does not depend on them) ...")
    heavy = heavy_ranker(catalog.n_items, seed)

    requests = []
    for i, hist in enumerate(catalog.histories):
        requests.append(json.dumps({"user_id": i, "history": [int(h) for h in hist], "slate_size": 20}).encode())
    return Artifacts(catalog, index, popular, recent, feats, light, heavy, requests)
