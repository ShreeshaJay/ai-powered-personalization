"""
Lightweight ranker service: POST /v1/rank

    uvicorn services.lightweight_ranker.app:app --port 8002
"""

from pathlib import Path
from typing import Optional

from fastapi import Request

from common.artifacts import resolve_bundle_dir
from common.service import (ServiceState, check_deadline, configure_logging, create_service_app, get_trace_id,
                            require_engine)
from contracts import RankedItem, RankingRequest, RankingResponse
from services.lightweight_ranker.engine import LightweightRankerEngine


def create_app(bundle_dir: Optional[Path] = None):
    configure_logging()
    state = ServiceState("lightweight_ranker")

    def load() -> Path:
        bundle = resolve_bundle_dir(bundle_dir)
        engine = LightweightRankerEngine.load(bundle)
        state.engine = engine
        state.model_version = engine.model_version
        state.details.update({"model_name": "xgboost", "score_name": engine.score_name,
                              "feature_set_version": engine.feature_set_version,
                              "warm_up_ms": engine.warm_up()})
        return bundle

    app = create_service_app("lightweight_ranker", state, load, title="Lightweight ranker (Chapter 6 XGBoost)")

    @app.post("/v1/rank", response_model=RankingResponse)
    def rank(body: RankingRequest, request: Request) -> RankingResponse:
        engine = require_engine(state)
        check_deadline(request, "lightweight ranking")
        ranked = engine.score(body)
        return RankingResponse(
            trace_id=get_trace_id(request),
            model_version=engine.model_version,
            model_name="xgboost",
            score_name=engine.score_name,
            ranked_items=[RankedItem(item_id=i, score=s) for i, s in ranked],
        )

    return app


app = create_app()
