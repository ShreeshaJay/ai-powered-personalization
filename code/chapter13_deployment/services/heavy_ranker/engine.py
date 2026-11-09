"""
Heavy ranker: Chapter 7's multi-task MMoE on CPU (Section 13.5).

Output contract: every ranked item carries `task_scores` for the five heads
(engagement, completion, play_ratio, like, dislike), each passed through the
sigmoid as in Chapter 7's predict_proba. play_ratio is a regression head
trained on sigmoid outputs, so its value is an expected play fraction rather
than a probability. The sort key `score` is P(completion), the same quantity
the lightweight ranker estimates, so skipping this stage leaves a score with
the same meaning.

Features are encoded exactly as Chapter 7's MMoEFeatureProcessor.transform
encodes them: IDs as strings through the fitted LabelEncoder classes (index
plus one, zero for an unseen value), dense columns through the fitted scaler.
That method rebuilds a lookup over every class on each call, which costs
hundreds of milliseconds per request once the item vocabulary reaches tens of
thousands, so the engine builds the same lookups once at load.
tests/test_services.py checks the two encodings agree. The published
transform skips dense columns it cannot find, which would shift every later
column, so the engine checks the full feature list first and rejects the
request (422).
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from common import published, warmup
from common.artifacts import ArtifactError, read_json, require_file, verify_files
from common.feature_store import check_model_features, load_contract
from common.service import ContractViolation


class HeavyRankerEngine:
    def __init__(self, model, processor, manifest: dict):
        self.model = model
        self.processor = processor
        self.manifest = manifest
        self.model_version = manifest["model_version"]
        self.feature_set_version = manifest["feature_set_version"]
        self.score_name = manifest["score_name"]
        self.task_names = list(manifest["task_names"])
        user_names = set(published.chapter6("historical_features").get_historical_feature_columns()["user_features"])
        context = {"is_organic"}
        self.user_features = [f for f in processor.dense_features if f in user_names]
        self.candidate_features = [f for f in processor.dense_features if f not in user_names and f not in context]
        self.add_derived = published.chapter6("feature_encoder").add_derived_features
        self.sparse_lookup = {name: {label: index + 1 for index, label in enumerate(encoder.classes_)}
                              for name, encoder in processor.sparse_encoders.items()
                              if name in processor.sparse_features}

    @classmethod
    def load(cls, bundle_dir: Path) -> "HeavyRankerEngine":
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        directory = Path(bundle_dir) / "heavy_ranker"
        manifest = read_json(directory / "serving_manifest.json", "heavy ranker manifest")
        verify_files(directory, manifest["files"], "heavy ranker")
        mm = published.chapter7_mmoe()
        train_mmoe = published.chapter7_train_mmoe()

        checkpoint = torch.load(require_file(directory / "mmoe.pt", "MMoE checkpoint"), map_location="cpu",
                                weights_only=True)
        for key in ("model_version", "feature_set_version"):
            if checkpoint.get(key) != manifest[key]:
                raise ArtifactError(f"MMoE checkpoint {key} {checkpoint.get(key)} does not match manifest {manifest[key]}")
        model = mm.MMoE(mm.MMoEConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        with open(require_file(directory / "feature_processor.pkl", "MMoE feature processor"), "rb") as handle:
            state = pickle.load(handle)
        processor = train_mmoe.MMoEFeatureProcessor(state["sparse_features"], state["dense_features"],
                                                    state["task_names"])
        processor.sparse_encoders = state["sparse_encoders"]
        processor.dense_scaler = state["dense_scaler"]
        processor.vocab_sizes = state["vocab_sizes"]
        processor._fitted = True
        if list(processor.sparse_features) != list(manifest["sparse_features"]) or \
                list(processor.dense_features) != list(manifest["dense_features"]):
            raise ArtifactError("MMoE feature processor does not match the serving manifest.")
        engine = cls(model, processor, manifest)
        check_model_features(load_contract(Path(bundle_dir) / "feature_store", verify=False),
                             engine.user_features + engine.candidate_features, engine.feature_set_version,
                             "heavy ranker")
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
            "timestamp": np.full(n, body.context.timestamp, dtype=np.int64),
            "is_organic": np.full(n, body.context.is_organic, dtype=np.int64),
        }
        for name in self.user_features:
            columns[name] = np.full(n, body.user_features[name], dtype=np.float64)
        for name in self.candidate_features:
            columns[name] = np.array([c.features[name] for c in body.candidates], dtype=np.float64)
        return self.add_derived(pd.DataFrame(columns))

    def encode(self, frame: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        sparse = {}
        for name in self.processor.sparse_features:
            if name in frame.columns and name in self.sparse_lookup:
                lookup = self.sparse_lookup[name]
                values = frame[name].fillna(-1).astype(str)
                sparse[name] = np.fromiter((lookup.get(value, 0) for value in values), dtype=np.int64,
                                           count=len(values))
        dense = frame[[c for c in self.processor.dense_features if c in frame.columns]].fillna(0).values
        dense = dense.astype(np.float32)
        if self.processor.dense_scaler is not None:
            dense = self.processor.dense_scaler.transform(dense)
        return sparse, dense

    def predict(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        sparse, dense = self.encode(frame)
        with torch.no_grad():
            outputs = self.model.predict_proba(
                {name: torch.from_numpy(values) for name, values in sparse.items()},
                torch.from_numpy(np.asarray(dense, dtype=np.float32)))
        return {name: outputs[name].squeeze(-1).numpy().astype(np.float64) for name in self.task_names}

    def score(self, body) -> List[Tuple[int, float, Dict[str, float]]]:
        self.check_contract(body)
        tasks = self.predict(self.frame(body))
        ranked = []
        for row, candidate in enumerate(body.candidates):
            task_scores = {name: float(tasks[name][row]) for name in self.task_names}
            ranked.append((candidate.item_id, task_scores["completion"], task_scores))
        ranked.sort(key=lambda entry: -entry[1])
        return ranked[:body.num_results] if body.num_results else ranked

    def warm_up(self, num_candidates: int = 100) -> float:
        body = warmup.ranking_request(self.user_features, self.candidate_features, list(range(num_candidates)),
                                      self.feature_set_version)
        return warmup.timed(lambda: self.score(body))
