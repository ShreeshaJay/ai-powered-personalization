"""
Orchestrator settings, read from environment variables.

The budgets are illustrative defaults for a laptop run, not measurements.
Chapter 14 shows how to derive them from load tests.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

from common.resilience import StageBudget

STAGES = ("retrieval", "lightweight_ranker", "heavy_ranker", "reranker")


def _env(name: str, default):
    raw = os.environ.get(name)
    return type(default)(raw) if raw is not None else default


@dataclass
class OrchestratorSettings:
    retrieval_url: str = "http://127.0.0.1:8001"
    lightweight_url: str = "http://127.0.0.1:8002"
    heavy_url: str = "http://127.0.0.1:8003"
    reranker_url: str = "http://127.0.0.1:8004"
    deadline_ms: float = 1000.0          # used when the client sends no X-Request-Timeout-Ms
    max_deadline_ms: float = 5000.0      # cap on a client-supplied deadline
    retrieval_candidates: int = 300      # retrieval -> lightweight ranker
    lightweight_keep: int = 100          # lightweight ranker -> heavy ranker
    budgets: Dict[str, StageBudget] = field(default_factory=lambda: {
        "retrieval": StageBudget("retrieval", cap_ms=250, min_ms=20),
        "lightweight_ranker": StageBudget("lightweight_ranker", cap_ms=200, min_ms=15),
        "heavy_ranker": StageBudget("heavy_ranker", cap_ms=300, min_ms=25),
        "reranker": StageBudget("reranker", cap_ms=150, min_ms=10),
    })
    response_reserve_ms: float = 10.0
    feature_lookup_cap_ms: float = 100.0  # timeout cap for a networked feature store (Redis)
    breaker_failure_rate: float = 0.5    # open above this share of failed calls ...
    breaker_min_calls: int = 20          # ... once the window holds at least this many calls
    breaker_window_s: float = 10.0
    breaker_reset_s: float = 10.0        # cool-down before the half-open trial call
    retry_max: int = 1
    retry_backoff_ms: float = 10.0
    retry_budget_per_s: float = 1.0      # process-wide retry tokens per second ...
    retry_budget_burst: float = 10.0     # ... and the most that can accumulate
    service_auth: str = "none"           # "gcp_id_token" on Cloud Run

    def urls(self) -> Dict[str, str]:
        return {"retrieval": self.retrieval_url, "lightweight_ranker": self.lightweight_url,
                "heavy_ranker": self.heavy_url, "reranker": self.reranker_url}

    @classmethod
    def from_env(cls) -> "OrchestratorSettings":
        s = cls()
        s.retrieval_url = _env("RETRIEVAL_URL", s.retrieval_url)
        s.lightweight_url = _env("LIGHTWEIGHT_RANKER_URL", s.lightweight_url)
        s.heavy_url = _env("HEAVY_RANKER_URL", s.heavy_url)
        s.reranker_url = _env("RERANKER_URL", s.reranker_url)
        s.deadline_ms = _env("ORCH_DEADLINE_MS", s.deadline_ms)
        s.max_deadline_ms = _env("ORCH_MAX_DEADLINE_MS", s.max_deadline_ms)
        s.retrieval_candidates = _env("ORCH_RETRIEVAL_CANDIDATES", s.retrieval_candidates)
        s.lightweight_keep = _env("ORCH_LIGHTWEIGHT_KEEP", s.lightweight_keep)
        for stage in STAGES:
            prefix = f"ORCH_{stage.upper()}"
            budget = s.budgets[stage]
            s.budgets[stage] = StageBudget(stage, _env(f"{prefix}_CAP_MS", float(budget.cap_ms)),
                                           _env(f"{prefix}_MIN_MS", float(budget.min_ms)))
        s.response_reserve_ms = _env("ORCH_RESPONSE_RESERVE_MS", s.response_reserve_ms)
        s.feature_lookup_cap_ms = _env("ORCH_FEATURE_LOOKUP_CAP_MS", s.feature_lookup_cap_ms)
        s.breaker_failure_rate = _env("ORCH_BREAKER_FAILURE_RATE", s.breaker_failure_rate)
        s.breaker_min_calls = _env("ORCH_BREAKER_MIN_CALLS", s.breaker_min_calls)
        s.breaker_window_s = _env("ORCH_BREAKER_WINDOW_S", s.breaker_window_s)
        s.breaker_reset_s = _env("ORCH_BREAKER_RESET_S", s.breaker_reset_s)
        s.retry_max = _env("ORCH_RETRY_MAX", s.retry_max)
        s.retry_backoff_ms = _env("ORCH_RETRY_BACKOFF_MS", s.retry_backoff_ms)
        s.retry_budget_per_s = _env("ORCH_RETRY_BUDGET_PER_S", s.retry_budget_per_s)
        s.retry_budget_burst = _env("ORCH_RETRY_BUDGET_BURST", s.retry_budget_burst)
        s.service_auth = _env("SERVICE_AUTH", s.service_auth)
        return s
