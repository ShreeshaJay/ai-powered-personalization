"""
Retrieval service: POST /v1/retrieve

    uvicorn services.retrieval.app:app --port 8001
"""

import threading
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import Request

from common.artifacts import resolve_bundle_dir
from common.feature_store import open_feature_store
from common.service import (ServiceState, check_deadline, configure_logging, create_service_app, get_trace_id,
                            require_engine)
from contracts import Candidate, RetrievalRequest, RetrievalResponse
from services.retrieval.engine import RetrievalEngine


def create_app(bundle_dir: Optional[Path] = None, store=None):
    configure_logging()
    state = ServiceState("retrieval")
    # How often each candidate source answered: a popularity fallback that
    # keeps responses looking normal can hide a broken retriever (Section 13.3.6).
    sources, sources_lock = Counter(), threading.Lock()
    state.details["candidate_sources"] = lambda: dict(sources)

    def load() -> Path:
        bundle = resolve_bundle_dir(bundle_dir)
        feature_store = store or open_feature_store(bundle / "feature_store")
        engine = RetrievalEngine.load(bundle, feature_store)
        state.engine = engine
        state.model_version = engine.model_version
        state.details.update({"index_version": engine.index_version, "num_items": int(engine.index.ntotal),
                              "feature_store": feature_store.describe, "warm_up_ms": engine.warm_up()})
        return bundle

    app = create_service_app("retrieval", state, load, title="Retrieval (Chapter 5 two-tower + FAISS)")

    @app.post("/v1/retrieve", response_model=RetrievalResponse)
    def retrieve(body: RetrievalRequest, request: Request) -> RetrievalResponse:
        engine = require_engine(state)
        check_deadline(request, "retrieval")
        candidates, source, history_length = engine.retrieve(
            body.user_id, body.recent_item_ids, body.exclude_item_ids, body.num_candidates)
        with sources_lock:
            sources[source] += 1
        return RetrievalResponse(
            trace_id=get_trace_id(request),
            model_version=engine.model_version,
            user_id=body.user_id,
            candidates=[Candidate(item_id=i, score=s) for i, s in candidates],
            source=source,
            index_version=engine.index_version,
            history_length=history_length,
        )

    return app


app = create_app()
