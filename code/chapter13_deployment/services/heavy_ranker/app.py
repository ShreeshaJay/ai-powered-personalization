"""
Heavy ranker service: POST /v1/rank

    uvicorn services.heavy_ranker.app:app --port 8003
"""

from pathlib import Path
from typing import Optional

from fastapi import Request

from common.artifacts import resolve_bundle_dir
from common.service import (ServiceState, check_deadline, configure_logging, create_service_app, get_trace_id,
                            require_engine)
from contracts import RankedItem, RankingRequest, RankingResponse, TaskScores
from services.heavy_ranker.engine import HeavyRankerEngine


def create_app(bundle_dir: Optional[Path] = None):
    configure_logging()
    state = ServiceState("heavy_ranker")

    def load() -> Path:
        bundle = resolve_bundle_dir(bundle_dir)
        engine = HeavyRankerEngine.load(bundle)
        state.engine = engine
        state.model_version = engine.model_version
        state.details.update({"model_name": "mmoe", "score_name": engine.score_name,
                              "task_names": engine.task_names, "feature_set_version": engine.feature_set_version,
                              "warm_up_ms": engine.warm_up()})
        return bundle

    app = create_service_app("heavy_ranker", state, load, title="Heavy ranker (Chapter 7 MMoE, CPU)")

    @app.post("/v1/rank", response_model=RankingResponse)
    def rank(body: RankingRequest, request: Request) -> RankingResponse:
        engine = require_engine(state)
        check_deadline(request, "heavy ranking")
        ranked = engine.score(body)
        return RankingResponse(
            trace_id=get_trace_id(request),
            model_version=engine.model_version,
            model_name="mmoe",
            score_name=engine.score_name,
            ranked_items=[RankedItem(item_id=i, score=s, task_scores=TaskScores(**t)) for i, s, t in ranked],
        )

    return app


app = create_app()
