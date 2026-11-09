"""
Shared fixtures.

The session builds one small synthetic bundle (profile `tiny`) in a temp
folder, which takes a minute or two on a laptop. To reuse an existing bundle
instead, set CH13_TEST_BUNDLE to its folder, for example artifacts/<bundle_version>.
"""

import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))
for name in ("TORCH_NUM_THREADS", "XGB_NUM_THREADS", "FAISS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from tests.helpers import FaultTransport, copy_config_dir, relaxed_settings  # noqa: E402


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory) -> Path:
    existing = os.environ.get("CH13_TEST_BUNDLE")
    if existing:
        return Path(existing).resolve()
    from pipeline_build.build_artifacts import build, parse_args

    root = tmp_path_factory.mktemp("artifacts")
    return build(parse_args(["--profile", "tiny", "--out", str(root), "--bundle-version", "test-v1",
                             "--no-activate"]))


@pytest.fixture(scope="session")
def sample_users(bundle_dir) -> dict:
    return json.loads((bundle_dir / "sample_users.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def stage_apps(bundle_dir, tmp_path_factory):
    """The four stage services with their lifespans running, keyed by stage name."""
    from fastapi.testclient import TestClient

    from services.heavy_ranker.app import create_app as heavy_app
    from services.lightweight_ranker.app import create_app as light_app
    from services.reranker.app import create_app as rerank_app
    from services.retrieval.app import create_app as retrieval_app

    config_dir = copy_config_dir(tmp_path_factory.mktemp("stack") / "reranker_config")
    apps = {
        "retrieval": retrieval_app(bundle_dir=bundle_dir),
        "lightweight_ranker": light_app(bundle_dir=bundle_dir),
        "heavy_ranker": heavy_app(bundle_dir=bundle_dir),
        "reranker": rerank_app(bundle_dir=bundle_dir, config_dir=config_dir, poll_seconds=0),
    }
    with ExitStack() as stack:
        clients = {name: stack.enter_context(TestClient(app)) for name, app in apps.items()}
        yield {"apps": apps, "clients": clients}


@pytest.fixture
def orchestrator(bundle_dir, stage_apps):
    """Factory: orchestrator(faults={stage: FaultTransport kwargs}, settings=..., clock=..., store=...)."""
    from fastapi.testclient import TestClient

    from services.orchestrator.app import create_app

    @contextmanager
    def make(faults=None, settings=None, clock=None, store=None):
        transports = {name: FaultTransport(app, **(faults or {}).get(name, {}))
                      for name, app in stage_apps["apps"].items()}
        kwargs = {"clock": clock} if clock is not None else {}
        app = create_app(bundle_dir=bundle_dir, settings=settings or relaxed_settings(), transports=transports,
                         store=store, **kwargs)
        with TestClient(app) as client:
            yield client, transports

    return make
