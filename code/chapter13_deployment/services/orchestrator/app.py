"""
Orchestrator service: POST /v1/recommend

    uvicorn services.orchestrator.app:app --port 8000

Readiness depends only on the orchestrator's own artifacts (feature store and
popularity list), never on the stage services: a slow ranker should make
requests degrade, not take the entry point out of the load balancer.
"""

import time
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import Request

from common.artifacts import read_json, resolve_bundle_dir
from common.feature_store import open_feature_store
from common.resilience import Deadline
from common.service import (DEADLINE_HEADER, ServiceState, configure_logging, create_service_app, get_deadline,
                            get_trace_id, require_engine)
from contracts import ClientRequest, ClientResponse
from services.orchestrator.auth import GcpIdTokenAuth
from services.orchestrator.pipeline import RecommendationPipeline
from services.orchestrator.settings import OrchestratorSettings


def create_app(bundle_dir: Optional[Path] = None, settings: Optional[OrchestratorSettings] = None,
               transports: Optional[Dict[str, httpx.AsyncBaseTransport]] = None, store=None,
               clock=time.monotonic):
    configure_logging()
    state = ServiceState("orchestrator")
    settings = settings or OrchestratorSettings.from_env()
    clients: Dict[str, httpx.AsyncClient] = {}

    def load() -> Path:
        bundle = resolve_bundle_dir(bundle_dir)
        manifest = read_json(bundle / "bundle_manifest.json", "bundle manifest")
        feature_store = store or open_feature_store(bundle / "feature_store")
        popularity = read_json(bundle / "orchestrator" / "popularity.json", "popularity list")["item_ids"]
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        for stage, url in settings.urls().items():
            auth = GcpIdTokenAuth(url) if settings.service_auth == "gcp_id_token" else None
            transport = (transports or {}).get(stage)
            clients[stage] = httpx.AsyncClient(base_url=url, transport=transport, auth=auth, limits=limits)
        pipeline = RecommendationPipeline(settings, feature_store, popularity, clients,
                                          manifest["bundle_version"], clock=clock)
        state.engine = pipeline
        state.model_version = manifest["bundle_version"]
        state.details.update({
            "feature_store": feature_store.describe,
            "downstream": settings.urls(),
            "circuit_breakers": pipeline.status,
            "retry_budget": pipeline.retry_budget.snapshot,
            "contract_rejections": lambda: dict(pipeline.rejections),
            "default_deadline_ms": settings.deadline_ms,
        })
        return bundle

    async def shutdown():
        for client in clients.values():
            await client.aclose()

    app = create_service_app("orchestrator", state, load, title="Orchestrator", on_shutdown=shutdown)

    @app.post("/v1/recommend", response_model=ClientResponse)
    async def recommend(body: ClientRequest, request: Request) -> ClientResponse:
        pipeline = require_engine(state)
        if request.headers.get(DEADLINE_HEADER) is None:
            deadline = Deadline(settings.deadline_ms)
        else:
            deadline = get_deadline(request)
            if deadline.remaining_ms() > settings.max_deadline_ms:
                deadline = Deadline(settings.max_deadline_ms)
        return await pipeline.recommend(body, get_trace_id(request), deadline)

    return app


app = create_app()
