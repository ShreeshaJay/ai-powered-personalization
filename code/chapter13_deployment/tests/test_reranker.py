"""Value function and re-ranker: MMR equivalence with Chapter 8, config hot reload and validation."""

import json
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from common import published
from services.reranker.engine import VectorizedMMRReranker
from tests.helpers import copy_config_dir, paced_order

mmr_module = published.chapter8("mmr_diversity")


@pytest.mark.parametrize("seed", range(4))
def test_vectorized_mmr_selects_like_chapter8(seed):
    # The Chapter 8 reference makes one cosine_similarity call per item pair, so sizes stay small.
    rng = np.random.default_rng(seed)
    item_ids = np.arange(100)
    embeddings = rng.normal(size=(100, 16)).astype(np.float32)
    embeddings[::7] = embeddings[0]                         # exact duplicates
    fast = VectorizedMMRReranker(item_ids, embeddings, mmr_module.DiversityConfig())
    pool = rng.choice(np.arange(110), size=50, replace=False)    # some IDs have no embedding
    scores = np.round(rng.random(len(pool)), 1)             # ties
    candidates = [(int(i), float(s)) for i, s in zip(pool, scores)]
    for lam in (0.0, 0.3, 0.7, 1.0):
        expected = mmr_module.MMRReranker.rerank(fast, candidates, top_k=15, lambda_param=lam)
        assert fast.rerank(candidates, top_k=15, lambda_param=lam) == expected


@pytest.fixture
def reranker(bundle_dir, tmp_path):
    from services.reranker.app import create_app

    config_dir = copy_config_dir(tmp_path / "config")
    with TestClient(create_app(bundle_dir=bundle_dir, config_dir=config_dir, poll_seconds=0)) as client:
        yield client, config_dir


def candidates(n=60, task_scores=True):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        c = {"item_id": i, "score": float(rng.random()), "artist_id": i % 7, "album_id": i % 13}
        if task_scores:
            c["task_scores"] = {k: float(rng.random()) for k in ("engagement", "completion", "play_ratio", "like",
                                                                  "dislike")}
        out.append(c)
    return out


def rerank(client, items, n=10):
    response = client.post("/v1/rerank", json={"user_id": 1, "candidates": items, "num_results": n})
    assert response.status_code == 200, response.text
    return response.json()


def test_value_function_applies_only_with_task_scores(reranker):
    client, _ = reranker
    with_tasks = rerank(client, candidates())
    assert with_tasks["value_function_applied"] and with_tasks["config_version"] == "vf-default-v1"
    assert with_tasks["model_version"] == "vf-default-v1"
    without = rerank(client, candidates(task_scores=False))
    assert not without["value_function_applied"]
    scores = {c["item_id"]: c["score"] for c in candidates(task_scores=False)}
    assert all(item["score"] == scores[item["item_id"]] for item in without["items"])


def test_pacing_defers_artist_and_album_repeats(reranker):
    client, _ = reranker
    body = rerank(client, candidates(), n=10)
    assert body["pacing_applied"] and body["diversity_applied"]
    ids = [item["item_id"] for item in body["items"]]
    assert paced_order([i % 7 for i in ids], [i % 13 for i in ids])


def test_hot_reload_switches_and_rejects_invalid_configs(reranker):
    client, config_dir = reranker
    (config_dir / "ACTIVE").write_text("vf-growth-v1\n", encoding="utf-8")
    response = client.post("/v1/config/reload")
    assert response.status_code == 200 and response.json()["config_version"] == "vf-growth-v1"
    assert rerank(client, candidates())["config_version"] == "vf-growth-v1"

    bad = json.loads((config_dir / "vf-growth-v1.json").read_text(encoding="utf-8"))
    bad["config_version"] = "vf-bad-v1"
    bad["value_function"]["w_like"] = -1.0
    (config_dir / "vf-bad-v1.json").write_text(json.dumps(bad), encoding="utf-8")
    (config_dir / "ACTIVE").write_text("vf-bad-v1\n", encoding="utf-8")
    response = client.post("/v1/config/reload")
    assert response.status_code == 422 and "w_like" in response.json()["last_error"]
    assert rerank(client, candidates())["config_version"] == "vf-growth-v1"     # last good config kept
    status = client.get("/v1/config").json()["status"]
    assert status["config_version"] == "vf-growth-v1" and status["last_error"]
    assert client.get("/health").json()["config"]["last_error"]

    misnamed = dict(bad, config_version="vf-other-v1")
    misnamed["value_function"]["w_like"] = 1.0
    (config_dir / "vf-bad-v1.json").write_text(json.dumps(misnamed), encoding="utf-8")
    assert client.post("/v1/config/reload").status_code == 422

    anonymous = {k: v for k, v in misnamed.items() if k != "author"}
    anonymous["config_version"] = "vf-bad-v1"
    (config_dir / "vf-bad-v1.json").write_text(json.dumps(anonymous), encoding="utf-8")
    response = client.post("/v1/config/reload")
    assert response.status_code == 422 and "author" in response.json()["last_error"]


def test_policy_version_pin_is_honoured_and_reported(reranker):
    client, config_dir = reranker
    items = candidates()

    def pinned(version):
        response = client.post("/v1/rerank", json={"user_id": 1, "candidates": items, "num_results": 10,
                                                   "policy_version": version})
        assert response.status_code == 200, response.text
        return response

    unpinned = rerank(client, items)
    assert unpinned["config_version"] == "vf-default-v1" and unpinned["policy_pin"] == "none"

    response = pinned("vf-growth-v1")
    body = response.json()
    assert body["config_version"] == body["model_version"] == "vf-growth-v1" and body["policy_pin"] == "honoured"
    assert response.headers["X-Model-Version"] == "vf-growth-v1"
    assert client.get("/v1/config").json()["status"]["config_version"] == "vf-default-v1"   # ACTIVE unchanged

    (config_dir / "ACTIVE").write_text("vf-growth-v1\n", encoding="utf-8")
    assert client.post("/v1/config/reload").status_code == 200
    assert rerank(client, items)["items"] == body["items"]          # the pin served exactly that policy

    (config_dir / "ACTIVE").write_text("vf-default-v1\n", encoding="utf-8")
    assert client.post("/v1/config/reload").status_code == 200
    assert pinned("vf-default-v1").json()["policy_pin"] == "honoured"


def test_unavailable_pin_serves_the_active_version_and_says_so(reranker):
    client, config_dir = reranker
    bad = json.loads((config_dir / "vf-growth-v1.json").read_text(encoding="utf-8"))
    bad["config_version"] = "vf-badpin-v1"
    bad["diversity"]["lambda_param"] = 1.5
    (config_dir / "vf-badpin-v1.json").write_text(json.dumps(bad), encoding="utf-8")
    for version in ("vf-missing-v9", "vf-badpin-v1"):
        body = client.post("/v1/rerank", json={"user_id": 1, "candidates": candidates(), "num_results": 10,
                                               "policy_version": version}).json()
        assert body["config_version"] == "vf-default-v1" and body["policy_pin"] == "unavailable", version
    response = client.post("/v1/rerank", json={"user_id": 1, "candidates": candidates(),
                                               "policy_version": "../ACTIVE"})
    assert response.status_code == 422                               # not a version name


def test_promote_config_validates_before_switching(tmp_path, monkeypatch, capsys):
    import importlib.util

    from common.artifacts import CHAPTER_DIR

    config_dir = copy_config_dir(tmp_path / "config")
    spec = importlib.util.spec_from_file_location("promote_config", CHAPTER_DIR / "scripts" / "promote_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (config_dir / "vf-broken-v1.json").write_text('{"config_version": "vf-broken-v1"}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["promote_config.py", "--config-dir", str(config_dir),
                                      "--activate", "vf-broken-v1"])
    assert module.main() == 1
    assert (config_dir / "ACTIVE").read_text(encoding="utf-8").strip() == "vf-default-v1"

    monkeypatch.setattr(sys, "argv", ["promote_config.py", "--config-dir", str(config_dir),
                                      "--activate", "vf-growth-v1"])
    assert module.main() == 0
    assert (config_dir / "ACTIVE").read_text(encoding="utf-8").strip() == "vf-growth-v1"
