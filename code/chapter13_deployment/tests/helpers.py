"""Test doubles and checks shared by the test modules."""

import asyncio
import json
import shutil
from pathlib import Path

import httpx

from common.artifacts import CHAPTER_DIR
from common.resilience import StageBudget
from services.orchestrator.settings import STAGES, OrchestratorSettings


def copy_config_dir(destination: Path) -> Path:
    """A private copy of config/reranker, so tests never touch the real ACTIVE pointer."""
    shutil.copytree(CHAPTER_DIR / "config" / "reranker", destination)
    return destination


def paced_order(artists, albums, max_per_artist=2, max_per_album=1) -> bool:
    """Chapter 8 pacing with deferral: every compliant item comes before every deferred one."""
    per_artist, per_album, flags = {}, {}, []
    for artist, album in zip(artists, albums):
        flags.append(per_artist.get(artist, 0) >= max_per_artist or per_album.get(album, 0) >= max_per_album)
        per_artist[artist] = per_artist.get(artist, 0) + 1
        per_album[album] = per_album.get(album, 0) + 1
    return flags == sorted(flags)


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FaultTransport(httpx.AsyncBaseTransport):
    """
    Wraps a stage service and injects failures: connect_error, http_500,
    http_503, http_422, bad_body, decoding_error (an error httpx does not
    class as a transport failure), no_task_scores (the real response with
    every item's task_scores removed), or delay (seconds).
    `fail_times` limits the injected failures to the first N calls.
    """

    def __init__(self, app, mode=None, delay_s: float = 0.0, fail_times=None):
        self.inner = httpx.ASGITransport(app=app)
        self.mode = mode
        self.delay_s = delay_s
        self.fail_times = fail_times
        self.calls = 0
        self.headers = []

    async def handle_async_request(self, request):
        self.calls += 1
        self.headers.append(dict(request.headers))
        active = self.mode is not None and (self.fail_times is None or self.calls <= self.fail_times)
        if active:
            if self.mode == "connect_error":
                raise httpx.ConnectError("injected connection failure", request=request)
            if self.mode in ("http_500", "http_503", "http_422"):
                status = int(self.mode.split("_")[1])
                return httpx.Response(status, json={"error": "injected"}, request=request)
            if self.mode == "bad_body":
                return httpx.Response(200, json={"unexpected": True}, request=request)
            if self.mode == "decoding_error":
                raise httpx.DecodingError("injected undecodable body", request=request)
            if self.mode == "no_task_scores":
                response = await self.inner.handle_async_request(request)
                body = json.loads(await response.aread())
                for item in body.get("ranked_items", []):
                    item.pop("task_scores", None)
                return httpx.Response(response.status_code, json=body, request=request)
            if self.mode == "delay":
                await asyncio.sleep(self.delay_s)
        return await self.inner.handle_async_request(request)


def relaxed_settings(**overrides) -> OrchestratorSettings:
    """Wide budgets so in-process tests do not time out by accident on a slow machine."""
    settings = OrchestratorSettings()
    settings.deadline_ms = 30000.0
    settings.max_deadline_ms = 60000.0
    settings.budgets = {name: StageBudget(name, cap_ms=10000.0, min_ms=5.0) for name in STAGES}
    settings.breaker_reset_s = 30.0
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings
