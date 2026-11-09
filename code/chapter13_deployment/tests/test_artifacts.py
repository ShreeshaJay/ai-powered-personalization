"""Missing, altered or mismatched artifacts stop a service at startup with a clear message."""

import importlib.util
import json
import shutil

import faiss
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from common import published
from common.artifacts import CHAPTER_DIR, ArtifactError, resolve_bundle_dir, sha256_file


@pytest.fixture
def bundle_copy(bundle_dir, tmp_path):
    copy = tmp_path / bundle_dir.name
    shutil.copytree(bundle_dir, copy)
    return copy


def test_no_active_bundle_explains_how_to_build(tmp_path, monkeypatch):
    monkeypatch.delenv("CH13_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("CH13_ARTIFACT_ROOT", str(tmp_path))
    with pytest.raises(ArtifactError, match="build_artifacts"):
        resolve_bundle_dir()


def test_missing_index_stops_retrieval_startup(bundle_copy):
    from services.retrieval.app import create_app

    (bundle_copy / "retrieval" / "item_index.faiss").unlink()
    with pytest.raises(ArtifactError, match="item_index.faiss"):
        with TestClient(create_app(bundle_dir=bundle_copy)):
            pass


def test_index_manifest_for_another_tower_is_refused(bundle_copy):
    from services.retrieval.engine import RetrievalEngine

    path = bundle_copy / "retrieval" / "index_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_version"] = "two-tower-some-other-build"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="does not match the index"):
        RetrievalEngine.load(bundle_copy)


def test_index_vectors_from_another_tower_are_detected(bundle_copy):
    """A consistent-looking manifest whose index was built by different weights fails the spot check."""
    from services.retrieval.engine import RetrievalEngine

    directory = bundle_copy / "retrieval"
    checkpoint = torch.load(directory / "two_tower.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(123)
    other = published.chapter5_two_tower().TwoTowerModel(**checkpoint["model_args"]).eval()
    content = np.load(directory / "content_embeddings.npy")
    with torch.no_grad():
        vectors = np.ascontiguousarray(other.encode_items(torch.tensor(content)).numpy(), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(directory / "item_index.faiss"))
    manifest_path = directory / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["item_index.faiss"] = sha256_file(directory / "item_index.faiss")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="differ"):
        RetrievalEngine.load(bundle_copy)


def test_altered_mmoe_checkpoint_is_refused(bundle_copy):
    from services.heavy_ranker.engine import HeavyRankerEngine

    with open(bundle_copy / "heavy_ranker" / "mmoe.pt", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ArtifactError, match="mmoe.pt"):
        HeavyRankerEngine.load(bundle_copy)


def test_missing_xgboost_file_is_refused(bundle_copy):
    from services.lightweight_ranker.engine import LightweightRankerEngine

    directory = bundle_copy / "lightweight_ranker"
    manifest = json.loads((directory / "serving_manifest.json").read_text(encoding="utf-8"))
    victim = next(iter(manifest["files"]))
    (directory / victim).unlink()
    with pytest.raises(ArtifactError, match=victim.split("/")[-1]):
        LightweightRankerEngine.load(bundle_copy)


def test_missing_popularity_list_stops_the_orchestrator(bundle_copy):
    from services.orchestrator.app import create_app

    (bundle_copy / "orchestrator" / "popularity.json").unlink()
    with pytest.raises(ArtifactError, match="popularity"):
        with TestClient(create_app(bundle_dir=bundle_copy)):
            pass


def _edit_bundle_manifest(bundle, edit):
    path = bundle / "bundle_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_bundle_built_with_other_published_code_is_refused(bundle_copy):
    """Serving features computed by changed published code would differ from the training features."""
    from services.lightweight_ranker.app import create_app

    changed = "chapter6_ranking/models/xgboost_ranker.py"
    _edit_bundle_manifest(bundle_copy, lambda m: m["published_code"]["files"].update({changed: "0" * 64}))
    with pytest.raises(ArtifactError, match=changed):
        with TestClient(create_app(bundle_dir=bundle_copy)):
            pass


def test_bundle_without_published_code_record_is_refused(bundle_copy):
    from services.heavy_ranker.app import create_app

    _edit_bundle_manifest(bundle_copy, lambda m: m.pop("published_code"))
    with pytest.raises(ArtifactError, match="does not record the published code"):
        with TestClient(create_app(bundle_dir=bundle_copy)):
            pass


def test_every_service_records_the_published_code_it_ran(stage_apps):
    loaded = published.loaded_files()
    for path in ("chapter5_retrieval/supervised_two_tower.py", "chapter6_ranking/models/xgboost_ranker.py",
                 "chapter7_advanced_ranking/models/mmoe.py", "chapter7_advanced_ranking/train_mmoe.py",
                 "chapter8_value_functions/mmr_diversity.py"):
        assert path in loaded


def test_vendored_code_matches_its_recorded_hashes():
    spec = importlib.util.spec_from_file_location("check_vendor", CHAPTER_DIR / "scripts" / "check_vendor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check() == []
