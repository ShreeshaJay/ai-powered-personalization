"""
Value function and re-ranker service.

    POST /v1/rerank          re-rank scored candidates (optionally pinned to a policy_version)
    GET  /v1/config          active value-function config and reload status
    POST /v1/config/reload   re-read the ACTIVE pointer now (validated)

    uvicorn services.reranker.app:app --port 8004
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from common.artifacts import resolve_bundle_dir
from common.service import (MODEL_VERSION_HEADER, ServiceState, check_deadline, configure_logging,
                            create_service_app, get_trace_id, require_engine)
from contracts import RerankedItem, RerankRequest, RerankResponse
from services.reranker.config_store import DEFAULT_CONFIG_DIR, ConfigStore
from services.reranker.engine import RerankerEngine


def create_app(bundle_dir: Optional[Path] = None, config_dir: Optional[Path] = None,
               poll_seconds: Optional[float] = None):
    configure_logging()
    state = ServiceState("reranker")
    store = ConfigStore(Path(config_dir or os.environ.get("RERANKER_CONFIG_DIR", DEFAULT_CONFIG_DIR)),
                        poll_seconds=poll_seconds)

    def load() -> Path:
        bundle = resolve_bundle_dir(bundle_dir)
        engine = RerankerEngine.load(bundle, store)
        state.engine = engine
        state.model_version = store.active.config.config_version
        state.details.update({"embedding_version": engine.embedding_version, "config": store.status,
                              "warm_up_ms": engine.warm_up()})
        store.start_polling()
        return bundle

    async def shutdown():
        store.stop()

    app = create_service_app("reranker", state, load, title="Value function and re-ranker (Chapter 8)",
                             on_shutdown=shutdown)

    @app.post("/v1/rerank", response_model=RerankResponse)
    def rerank(body: RerankRequest, request: Request, response: Response) -> RerankResponse:
        engine = require_engine(state)
        check_deadline(request, "re-ranking")
        result = engine.rerank(body.candidates, body.num_results, body.policy_version)
        response.headers[MODEL_VERSION_HEADER] = result["config_version"]
        return RerankResponse(
            trace_id=get_trace_id(request),
            model_version=result["config_version"],
            items=[RerankedItem(item_id=i, score=s) for i, s in result["items"]],
            config_version=result["config_version"],
            policy_pin=result["policy_pin"],
            value_function_applied=result["value_function_applied"],
            diversity_applied=result["diversity_applied"],
            pacing_applied=result["pacing_applied"],
        )

    @app.get("/v1/config")
    def get_config():
        require_engine(state)
        return {"active": store.active.config.model_dump(), "status": store.status()}

    @app.post("/v1/config/reload")
    def reload_config():
        require_engine(state)
        status = store.reload()
        state.model_version = status["config_version"]
        return JSONResponse(status_code=200 if status["last_error"] is None else 422, content=status)

    @app.middleware("http")
    async def track_config_version(request: Request, call_next):
        if store.active is not None:
            state.model_version = store.active.config.config_version
        return await call_next(request)

    return app


app = create_app()
