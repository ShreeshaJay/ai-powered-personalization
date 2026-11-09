"""
Lightweight ranker: Chapter 6's XGBoostRanker served as is (Section 13.4).

The service rebuilds the raw columns the Chapter 6 FeaturePipeline expects
(uid, item_id, is_organic, timestamp, track_length_seconds and the
historical aggregates) and calls the published predict_proba, so encoding
and derived features run through the same code as in training. Before
scoring it checks the request against the model's feature contract: every
required feature present, and the feature-set version the model was
trained on. Violations return 422 rather than a silently wrong score.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from common import published, warmup
from common.artifacts import read_json, verify_files
from common.feature_store import check_model_features, load_contract
from common.service import ContractViolation


class LightweightRankerEngine:
    def __init__(self, model, manifest: dict):
        self.model = model
        self.manifest = manifest
        self.model_version = manifest["model_version"]
        self.feature_set_version = manifest["feature_set_version"]
        self.score_name = manifest["score_name"]
        required = manifest["required_columns"]
        user_names = set(published.chapter6("historical_features").get_historical_feature_columns()["user_features"])
        feature_names = list(required["numerical"]) + list(required["historical"])
        self.user_features = [f for f in feature_names if f in user_names]
        self.candidate_features = [f for f in feature_names if f not in user_names]

    @classmethod
    def load(cls, bundle_dir: Path) -> "LightweightRankerEngine":
        directory = Path(bundle_dir) / "lightweight_ranker"
        manifest = read_json(directory / "serving_manifest.json", "lightweight ranker manifest")
        verify_files(directory, manifest["files"], "lightweight ranker")
        ch6 = published.chapter6("xgboost_ranker")
        model = ch6.XGBoostRanker.from_pretrained(str(directory))
        # One thread per request: with many concurrent requests, per-call
        # parallelism only adds contention (Chapter 14 measures the trade-off).
        model.model.set_params(n_jobs=int(os.environ.get("XGB_NUM_THREADS", "1")))
        engine = cls(model, manifest)
        check_model_features(load_contract(Path(bundle_dir) / "feature_store", verify=False),
                             engine.user_features + engine.candidate_features, engine.feature_set_version,
                             "lightweight ranker")
        return engine

    def check_contract(self, body) -> None:
        if body.feature_set_version and body.feature_set_version != self.feature_set_version:
            raise ContractViolation(
                f"feature_set_version {body.feature_set_version} does not match the model's "
                f"{self.feature_set_version}; features and model must come from the same build.")
        if body.context.timestamp is None:
            raise ContractViolation("context.timestamp is required by the ranker (time-of-day features).")
        missing_user = [f for f in self.user_features if f not in body.user_features]
        if missing_user:
            raise ContractViolation(f"user_features missing {missing_user}")
        for candidate in body.candidates:
            missing = [f for f in self.candidate_features if f not in candidate.features]
            if missing:
                raise ContractViolation(f"candidate {candidate.item_id} features missing {missing}")

    def frame(self, body) -> pd.DataFrame:
        n = len(body.candidates)
        columns = {
            "uid": np.full(n, body.user_id, dtype=np.int64),
            "item_id": np.array([c.item_id for c in body.candidates], dtype=np.int64),
            "is_organic": np.full(n, body.context.is_organic, dtype=np.int64),
            "timestamp": np.full(n, body.context.timestamp, dtype=np.int64),
        }
        for name in self.user_features:
            columns[name] = np.full(n, body.user_features[name], dtype=np.float64)
        for name in self.candidate_features:
            columns[name] = np.array([c.features[name] for c in body.candidates], dtype=np.float64)
        return pd.DataFrame(columns)

    def score(self, body) -> List[Tuple[int, float]]:
        self.check_contract(body)
        probabilities = self.model.predict_proba(self.frame(body))
        ranked = sorted(zip((c.item_id for c in body.candidates), probabilities.tolist()),
                        key=lambda pair: -pair[1])
        return ranked[:body.num_results] if body.num_results else ranked

    def warm_up(self, num_candidates: int = 300) -> float:
        body = warmup.ranking_request(self.user_features, self.candidate_features, list(range(num_candidates)),
                                      self.feature_set_version)
        return warmup.timed(lambda: self.score(body))
