"""
Stage services: health, contracts, and equivalence with the published code.

* retrieval returns what Chapter 5's SupervisedRetriever returns for the same history;
* the online ranking request built from the feature store reproduces the
  offline training features exactly (training/serving consistency), and both
  rankers score it as the published models score the offline rows.
"""

import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest
import torch

from common import published
from common.feature_store import open_feature_store

STAGE_PATHS = {"retrieval": "/v1/retrieve", "lightweight_ranker": "/v1/rank", "heavy_ranker": "/v1/rank",
               "reranker": "/v1/rerank"}


@pytest.mark.parametrize("stage", list(STAGE_PATHS))
def test_health_endpoints(stage_apps, stage):
    client = stage_apps["clients"][stage]
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    assert ready.json()["model_version"] not in ("", "unloaded")


@pytest.mark.parametrize("stage", list(STAGE_PATHS))
def test_contract_violation_is_422_with_trace_and_version(stage_apps, stage):
    client = stage_apps["clients"][stage]
    response = client.post(STAGE_PATHS[stage], json={"user_id": -1}, headers={"X-Trace-Id": "t-1"})
    body = response.json()
    assert response.status_code == 422
    assert body["error"] == "contract_violation"
    assert body["trace_id"] == "t-1" and body["model_version"]
    assert response.headers["X-Model-Version"] == body["model_version"]


@pytest.mark.parametrize("stage", list(STAGE_PATHS))
def test_expired_deadline_is_refused(stage_apps, stage):
    response = stage_apps["clients"][stage].post(STAGE_PATHS[stage], json={}, headers={"X-Request-Timeout-Ms": "0"})
    assert response.status_code == 504
    assert response.json()["error"] == "deadline_exceeded"


# -- retrieval ----------------------------------------------------------------

def test_retrieval_excludes_history_and_requested_items(stage_apps, sample_users):
    client = stage_apps["clients"]["retrieval"]
    user = sample_users["warm"][0]
    body = client.post("/v1/retrieve", json={"user_id": user, "num_candidates": 50}).json()
    assert body["source"] == "two_tower" and body["history_length"] > 0
    first = [c["item_id"] for c in body["candidates"]]
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)
    engine = stage_apps["apps"]["retrieval"].state.service.engine
    history = engine.store.get_user(user).recent_item_ids
    assert not set(first) & set(history)

    excluded = first[:5]
    body = client.post("/v1/retrieve", json={"user_id": user, "num_candidates": 50,
                                             "exclude_item_ids": excluded}).json()
    assert not set(c["item_id"] for c in body["candidates"]) & set(excluded)


def test_retrieval_matches_chapter5_supervised_retriever(stage_apps, sample_users):
    engine = stage_apps["apps"]["retrieval"].state.service.engine
    ch5 = published.chapter5_two_tower()
    with contextlib.redirect_stdout(io.StringIO()):
        reference = ch5.SupervisedRetriever(engine.model, engine.lookup, torch.device("cpu"))
    for user in sample_users["warm"][:10]:
        history = engine.store.get_user(user).recent_item_ids
        ours, source, _ = engine.retrieve(user, history, [], 50)
        theirs = reference.retrieve_for_user(history, top_k=50)
        assert source == "two_tower"
        assert [i for i, _ in ours] == [i for i, _ in theirs]
        np.testing.assert_allclose([s for _, s in ours], [s for _, s in theirs], rtol=0, atol=1e-5)


def test_cold_user_gets_popularity_and_the_source_is_counted(stage_apps, sample_users):
    client = stage_apps["clients"]["retrieval"]
    before = client.get("/health").json()["candidate_sources"].get("popularity_cold_start", 0)
    body = client.post("/v1/retrieve", json={"user_id": sample_users["cold"][0], "num_candidates": 20}).json()
    assert body["source"] == "popularity_cold_start" and body["history_length"] == 0
    assert len(body["candidates"]) == 20
    assert client.get("/health").json()["candidate_sources"]["popularity_cold_start"] == before + 1


# -- rankers: training/serving consistency ------------------------------------

@pytest.fixture(scope="module")
def consistency_rows(bundle_dir):
    rows = pd.read_parquet(bundle_dir / "lightweight_ranker" / "consistency_sample.parquet")
    return rows.head(60).reset_index(drop=True)


def online_request(store, row, feature_set_version):
    user = store.get_user(int(row.uid))
    item_id = int(row.item_id)
    item = store.get_items([item_id])[item_id]
    pair = store.get_user_items(int(row.uid), [item_id])[item_id]
    return {
        "user_id": int(row.uid),
        "context": {"timestamp": int(row.timestamp), "is_organic": int(row.is_organic)},
        "user_features": user.features,
        "candidates": [{"item_id": item_id, "features": {**item.features, **pair}}],
        "feature_set_version": feature_set_version,
    }


@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_online_features_equal_offline_training_features(bundle_dir, consistency_rows, backend):
    store = open_feature_store(bundle_dir / "feature_store", backend=backend)
    assert store.describe()["counts"]["items"] > 0
    manifest = store.manifest
    assert manifest["checks"]["items_with_varying_track_length"] == 0
    for row in consistency_rows.itertuples(index=False):
        request = online_request(store, row, store.feature_set_version)
        online = {**request["user_features"], **request["candidates"][0]["features"]}
        for name, value in online.items():
            assert value == pytest.approx(float(getattr(row, name)), abs=1e-9), name


def test_lightweight_ranker_scores_like_published_xgboost(stage_apps, bundle_dir, consistency_rows):
    client = stage_apps["clients"]["lightweight_ranker"]
    engine = stage_apps["apps"]["lightweight_ranker"].state.service.engine
    store = open_feature_store(bundle_dir / "feature_store")
    reference = engine.model.predict_proba(consistency_rows)
    for row, expected in zip(consistency_rows.itertuples(index=False), reference):
        response = client.post("/v1/rank", json=online_request(store, row, engine.feature_set_version))
        assert response.status_code == 200, response.text
        assert response.json()["ranked_items"][0]["score"] == pytest.approx(float(expected), abs=1e-6)


def published_mmoe_predict(engine, frame):
    sparse, dense, _ = engine.processor.transform(frame)
    with torch.no_grad():
        outputs = engine.model.predict_proba({name: torch.from_numpy(values) for name, values in sparse.items()},
                                             torch.from_numpy(np.asarray(dense, dtype=np.float32)))
    return {name: outputs[name].squeeze(-1).numpy() for name in engine.task_names}


def test_heavy_ranker_encoding_equals_published_transform(stage_apps, consistency_rows):
    engine = stage_apps["apps"]["heavy_ranker"].state.service.engine
    frame = published.chapter6("feature_encoder").add_derived_features(consistency_rows.copy())
    unseen = frame.copy()
    unseen["uid"] = unseen["uid"] + 10**9
    unseen.loc[unseen.index[::2], "item_id"] = unseen["item_id"].max() + 10**9
    for case in (frame, unseen):
        sparse, dense, _ = engine.processor.transform(case)
        fast_sparse, fast_dense = engine.encode(case)
        assert set(fast_sparse) == set(sparse)
        for name in sparse:
            np.testing.assert_array_equal(fast_sparse[name], sparse[name])
        np.testing.assert_array_equal(fast_dense, dense)
    assert (engine.encode(unseen)[0]["uid"] == 0).all()


def test_heavy_ranker_scores_like_published_mmoe(stage_apps, bundle_dir, consistency_rows):
    client = stage_apps["clients"]["heavy_ranker"]
    engine = stage_apps["apps"]["heavy_ranker"].state.service.engine
    store = open_feature_store(bundle_dir / "feature_store")
    offline = published.chapter6("feature_encoder").add_derived_features(consistency_rows.copy())
    reference = published_mmoe_predict(engine, offline)
    for index, row in enumerate(consistency_rows.itertuples(index=False)):
        response = client.post("/v1/rank", json=online_request(store, row, engine.feature_set_version))
        assert response.status_code == 200, response.text
        item = response.json()["ranked_items"][0]
        assert item["score"] == pytest.approx(float(reference["completion"][index]), abs=1e-5)
        for task, value in item["task_scores"].items():
            assert value == pytest.approx(float(reference[task][index]), abs=1e-5)


def test_heavy_ranker_returns_all_task_scores_sorted_by_completion(stage_apps, bundle_dir, sample_users):
    store = open_feature_store(bundle_dir / "feature_store")
    engine = stage_apps["apps"]["heavy_ranker"].state.service.engine
    user = store.get_user(sample_users["warm"][0])
    popular = json.loads((bundle_dir / "orchestrator" / "popularity.json").read_text())["item_ids"][:200]
    known = store.get_items(popular)
    item_ids = [i for i, record in known.items() if record.track_length_seconds is not None][:40]
    items, pairs = store.get_items(item_ids), store.get_user_items(user.user_id, item_ids)
    body = stage_apps["clients"]["heavy_ranker"].post("/v1/rank", json={
        "user_id": user.user_id, "context": {"timestamp": store.reference_timestamp},
        "user_features": user.features,
        "candidates": [{"item_id": i, "features": {**items[i].features, **pairs[i]}} for i in item_ids],
        "feature_set_version": engine.feature_set_version}).json()
    scores = [r["score"] for r in body["ranked_items"]]
    assert body["model_name"] == "mmoe" and len(scores) == 40
    assert scores == sorted(scores, reverse=True)
    for item in body["ranked_items"]:
        assert set(item["task_scores"]) == {"engagement", "completion", "play_ratio", "like", "dislike"}
        assert item["score"] == item["task_scores"]["completion"]


@pytest.mark.parametrize("stage", ["lightweight_ranker", "heavy_ranker"])
def test_rankers_reject_wrong_feature_set_and_missing_features(stage_apps, bundle_dir, consistency_rows, stage):
    client = stage_apps["clients"][stage]
    store = open_feature_store(bundle_dir / "feature_store")
    request = online_request(store, next(consistency_rows.itertuples(index=False)), "fs-some-other-build")
    response = client.post("/v1/rank", json=request)
    assert response.status_code == 422 and "feature_set_version" in response.text

    request = online_request(store, next(consistency_rows.itertuples(index=False)), None)
    request["candidates"][0]["features"].pop("track_length_seconds")
    response = client.post("/v1/rank", json=request)
    assert response.status_code == 422 and "missing" in response.text

    request = online_request(store, next(consistency_rows.itertuples(index=False)), None)
    request["context"].pop("timestamp")
    assert client.post("/v1/rank", json=request).status_code == 422
