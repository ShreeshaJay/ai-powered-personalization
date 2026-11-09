"""
Plumbing shared by the five services.

* Trace IDs: `X-Trace-Id` is read from the request or created, attached to
  every response body and header, and forwarded downstream.
* Deadlines: `X-Request-Timeout-Ms` carries the caller's remaining budget.
  Each service turns it into an absolute deadline on arrival and refuses work
  that would finish after it (HTTP 504), for example after waiting in a queue.
* Contract violations return HTTP 422 with a structured body.
* Health: /health (details), /health/live (process is responsive) and
  /health/ready (artifacts loaded and not draining).
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import published
from .artifacts import check_published_code
from .resilience import Deadline

TRACE_HEADER = "X-Trace-Id"
DEADLINE_HEADER = "X-Request-Timeout-Ms"
MODEL_VERSION_HEADER = "X-Model-Version"
DEFAULT_DEADLINE_MS = float(os.environ.get("SERVICE_DEFAULT_DEADLINE_MS", "2000"))

logger = logging.getLogger("ch13")


class ContractViolation(ValueError):
    """The request parsed but breaks a contract the schema cannot express (for example missing features)."""


class DeadlineExceeded(RuntimeError):
    pass


class NotReady(RuntimeError):
    pass


class ServiceState:
    """Readiness and version information for one service process."""

    def __init__(self, name: str):
        self.name = name
        self.ready = False
        self.draining = False
        self.model_version = "unloaded"
        self.load_error: Optional[str] = None
        self.started_at = time.time()
        self.details: dict = {}
        self.engine = None

    def health(self) -> dict:
        return {
            "service": self.name,
            "status": "ok" if self.ready and not self.draining else "not_ready",
            "ready": self.ready and not self.draining,
            "model_version": self.model_version,
            "uptime_s": round(time.time() - self.started_at, 1),
            "load_error": self.load_error,
            **{key: value() if callable(value) else value for key, value in self.details.items()},
        }


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", None) or uuid.uuid4().hex


def get_deadline(request: Request) -> Deadline:
    return getattr(request.state, "deadline", None) or Deadline(DEFAULT_DEADLINE_MS)


def check_deadline(request: Request, step: str) -> None:
    if get_deadline(request).expired():
        raise DeadlineExceeded(f"deadline passed before {step}")


def _error(status: int, error: str, detail, request: Request, state: ServiceState) -> JSONResponse:
    trace_id = get_trace_id(request)
    return JSONResponse(
        status_code=status,
        content={"error": error, "detail": detail, "trace_id": trace_id, "model_version": state.model_version},
        headers={TRACE_HEADER: trace_id, MODEL_VERSION_HEADER: state.model_version},
    )


def _validation_detail(exc: RequestValidationError) -> list:
    return [
        {"loc": [str(part) for part in err.get("loc", ())], "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]


def create_service_app(name: str, state: ServiceState, load: Callable[[], Path],
                       title: Optional[str] = None,
                       on_shutdown: Optional[Callable[[], Awaitable[None]]] = None) -> FastAPI:
    """
    Build a FastAPI app whose lifespan runs `load()` before the server accepts
    connections. `load()` returns the bundle directory it loaded, and the
    published code this process ran must be the code that bundle was built
    with. If either step fails the process exits with the error, so a missing
    or mismatched artifact is reported at startup and never served.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.perf_counter()
        try:
            check_published_code(load(), published.loaded_files())
        except Exception as exc:
            state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("%s failed to load artifacts: %s", name, exc)
            raise
        state.ready = True
        state.details["load_seconds"] = round(time.perf_counter() - started, 2)
        logger.info("%s ready (model_version=%s)", name, state.model_version)
        yield
        state.draining = True
        state.ready = False
        logger.info("%s draining and shutting down", name)
        if on_shutdown is not None:
            await on_shutdown()

    app = FastAPI(title=title or name, version="v1", lifespan=lifespan)
    app.state.service = state

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        trace_id = request.headers.get(TRACE_HEADER) or uuid.uuid4().hex
        request.state.trace_id = trace_id
        raw = request.headers.get(DEADLINE_HEADER)
        if raw is None:
            request.state.deadline = Deadline(DEFAULT_DEADLINE_MS)
        else:
            try:
                budget_ms = float(raw)
            except ValueError:
                return _error(422, "contract_violation",
                              [{"loc": ["header", DEADLINE_HEADER], "msg": "must be a number of milliseconds"}],
                              request, state)
            if budget_ms <= 0:
                return _error(504, "deadline_exceeded", "deadline already passed on arrival", request, state)
            request.state.deadline = Deadline(budget_ms)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[TRACE_HEADER] = trace_id
        response.headers.setdefault(MODEL_VERSION_HEADER, state.model_version)
        response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
        if request.method == "POST":
            logger.info("%s trace_id=%s path=%s status=%d dur_ms=%.1f model_version=%s", name, trace_id,
                        request.url.path, response.status_code, elapsed_ms, response.headers[MODEL_VERSION_HEADER])
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        return _error(422, "contract_violation", _validation_detail(exc), request, state)

    @app.exception_handler(ContractViolation)
    async def on_contract_violation(request: Request, exc: ContractViolation):
        return _error(422, "contract_violation", str(exc), request, state)

    @app.exception_handler(DeadlineExceeded)
    async def on_deadline(request: Request, exc: DeadlineExceeded):
        return _error(504, "deadline_exceeded", str(exc), request, state)

    @app.exception_handler(NotReady)
    async def on_not_ready(request: Request, exc: NotReady):
        return _error(503, "not_ready", str(exc), request, state)

    @app.get("/health")
    def health():
        return state.health()

    @app.get("/health/live")
    def live():
        return {"service": name, "status": "alive"}

    @app.get("/health/ready")
    def ready():
        body = state.health()
        return JSONResponse(status_code=200 if body["ready"] else 503, content=body)

    return app


def require_engine(state: ServiceState):
    if not state.ready or state.engine is None:
        raise NotReady(f"{state.name} is not ready")
    return state.engine


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # The published Chapter 6 and 8 classes log at INFO on every construction,
    # and httpx logs every downstream call.
    for noisy in ("ch6_models", "ch8_value_functions", "ch7_models", "ch5_supervised_two_tower", "faiss.loader",
                  "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))
