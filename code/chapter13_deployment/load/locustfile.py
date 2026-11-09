"""
Closed-loop load ramp for POST /v1/recommend.

Each simulated user sends a request, waits for the response, then sends the
next one with no think time. Offered load therefore rises with the number of
users and drops when the system slows down: the test shows where latency
bends as concurrency grows and which degradation rungs appear, but a fixed
arrival rate (open loop, Chapter 14) is needed to see queueing collapse.

    locust -f load/locustfile.py --headless --host http://127.0.0.1:8000
    RAMP="30:5,60:10,90:20" locust -f load/locustfile.py --headless --host http://127.0.0.1:8000

Environment:
    RAMP            end_second:users stages; the run stops after the last one
                    (default 20:2,40:5,60:10,80:20,100:30)
    DEADLINE_MS     X-Request-Timeout-Ms per request (default: the orchestrator's)
    NUM_RESULTS     slate size (default 20)
    USERS_FILE      sample_users.json (default: the active bundle's)
    COLD_FRACTION   share of requests from cold-start users (default 0.1)
    AUTH_TOKEN      bearer token, for Cloud Run services without public access
    LOCUST_SUMMARY  path for a JSON summary: latency, throughput, rungs and skipped
                    stages (stage:reason) per user level
"""

import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from locust import HttpUser, LoadTestShape, constant, events, task

CHAPTER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAMP = "20:2,40:5,60:10,80:20,100:30"


def parse_ramp(text: str):
    stages = []
    for part in text.split(","):
        end, users = part.split(":")
        stages.append((float(end), int(users)))
    if [end for end, _ in stages] != sorted(end for end, _ in stages):
        raise ValueError(f"RAMP stage end times must increase: {text}")
    return stages


def load_users():
    path = os.environ.get("USERS_FILE")
    if not path:
        root = Path(os.environ.get("CH13_ARTIFACT_ROOT", CHAPTER_DIR / "artifacts"))
        pointer = root / "ACTIVE"
        bundle = Path(os.environ["CH13_BUNDLE_DIR"]) if os.environ.get("CH13_BUNDLE_DIR") else \
            (root / pointer.read_text(encoding="utf-8").strip() if pointer.is_file() else None)
        path = bundle / "sample_users.json" if bundle else None
    if path and Path(path).is_file():
        users = json.loads(Path(path).read_text(encoding="utf-8"))
        return users["warm"], users["cold"]
    return list(range(100)), []


WARM, COLD = load_users()
COLD_FRACTION = float(os.environ.get("COLD_FRACTION", "0.1")) if COLD else 0.0
DEADLINE_MS = os.environ.get("DEADLINE_MS")
NUM_RESULTS = int(os.environ.get("NUM_RESULTS", "20"))
STAGES = parse_ramp(os.environ.get("RAMP", DEFAULT_RAMP))

by_level = defaultdict(lambda: {"latency_ms": [], "rungs": Counter(), "skipped": Counter(), "errors": 0,
                                "first": None, "last": None})
totals = {"rungs": Counter(), "skipped": Counter(), "score_sources": Counter(), "status": Counter()}


class RecommendUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.headers = {}
        if os.environ.get("AUTH_TOKEN"):
            self.headers["Authorization"] = f"Bearer {os.environ['AUTH_TOKEN']}"
        if DEADLINE_MS:
            self.headers["X-Request-Timeout-Ms"] = DEADLINE_MS

    @task
    def recommend(self):
        cold = COLD and random.random() < COLD_FRACTION
        user_id = random.choice(COLD if cold else WARM)
        level = by_level[self.environment.runner.user_count]
        started = time.perf_counter()
        with self.client.post("/v1/recommend", json={"user_id": user_id, "num_results": NUM_RESULTS},
                              headers=self.headers, name="/v1/recommend", catch_response=True) as response:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            now = time.time()
            level["first"] = level["first"] or now
            level["last"] = now
            totals["status"][response.status_code] += 1
            if response.status_code != 200:
                level["errors"] += 1
                response.failure(f"HTTP {response.status_code}")
                return
            body = response.json()
            rung = body["degradation"]["rung"]
            level["latency_ms"].append(elapsed_ms)
            level["rungs"][rung] += 1
            totals["rungs"][rung] += 1
            for fallback in body["degradation"]["fallbacks"]:
                key = f"{fallback['stage']}:{fallback['reason']}"
                level["skipped"][key] += 1
                totals["skipped"][key] += 1
            totals["score_sources"][body["score_source"]] += 1
            if body["items"]:
                response.success()
            else:
                response.failure("empty slate")


class ClosedLoopRamp(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()
        for end, users in STAGES:
            if run_time < end:
                return users, max(1, users)
        return None


def _percentile(values, q):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1) if ordered else None


@events.test_stop.add_listener
def write_summary(environment, **kwargs):
    stats = environment.stats.total
    levels = {}
    for users, level in sorted(by_level.items()):
        latencies = level["latency_ms"]
        span = (level["last"] - level["first"]) if level["first"] and level["last"] else 0.0
        levels[str(users)] = {
            "requests": len(latencies) + level["errors"], "errors": level["errors"],
            "throughput_rps": round((len(latencies) + level["errors"]) / span, 1) if span > 0 else None,
            "p50_ms": _percentile(latencies, 0.50), "p95_ms": _percentile(latencies, 0.95),
            "p99_ms": _percentile(latencies, 0.99),
            "mean_ms": round(statistics.fmean(latencies), 1) if latencies else None,
            "rungs": dict(sorted(level["rungs"].items())),
            "skipped": dict(level["skipped"].most_common()),
        }
    summary = {
        "host": environment.host, "ramp": STAGES, "deadline_ms": DEADLINE_MS, "num_results": NUM_RESULTS,
        "requests": stats.num_requests, "failures": stats.num_failures,
        "p50_ms": stats.get_response_time_percentile(0.50), "p95_ms": stats.get_response_time_percentile(0.95),
        "p99_ms": stats.get_response_time_percentile(0.99),
        "rungs": dict(sorted(totals["rungs"].items())), "skipped": dict(totals["skipped"].most_common()),
        "score_sources": dict(totals["score_sources"]),
        "status_codes": {str(k): v for k, v in totals["status"].items()},
        "by_user_level": levels,
    }
    print(json.dumps(summary, indent=2))
    path = os.environ.get("LOCUST_SUMMARY")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
