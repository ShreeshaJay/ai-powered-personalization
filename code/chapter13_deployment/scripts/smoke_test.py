"""
Smoke test for a running stack, local or on Cloud Run.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --url https://ch13-orchestrator-....run.app \
        --service retrieval=https://ch13-retrieval-....run.app --token-env AUTH_TOKEN

Checks:
  * every service listed is ready and reports a model_version;
  * warm users get a full, well-formed slate (unique items, ranks 1..k,
    trace_id echoed, model_version present) and at least one request runs
    the full pipeline (rung 1);
  * cold-start users are served from the popularity list by retrieval;
  * a contract violation returns 422 with trace_id and model_version, on the
    orchestrator and on each stage service listed;
  * an unparseable deadline header returns 422, and a 1 ms deadline still
    returns 200 from the bottom rung of the ladder (6, "Cached list").

Exit code 0 when every check passes; --out writes the report as JSON.
"""

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import httpx

CHAPTER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_URLS = {"retrieval": "http://127.0.0.1:8001", "lightweight_ranker": "http://127.0.0.1:8002",
                "heavy_ranker": "http://127.0.0.1:8003", "reranker": "http://127.0.0.1:8004"}
BAD_REQUESTS = {
    "retrieval": ("/v1/retrieve", {"user_id": 1, "num_candidates": 0}),
    "lightweight_ranker": ("/v1/rank", {"user_id": 1, "context": {"timestamp": 0}, "user_features": {},
                                        "candidates": [{"item_id": 1, "features": {}}]}),
    "heavy_ranker": ("/v1/rank", {"user_id": 1, "context": {"timestamp": 0}, "user_features": {},
                                  "candidates": [{"item_id": 1, "features": {}}]}),
    "reranker": ("/v1/rerank", {"user_id": 1, "candidates": [{"item_id": 1, "score": 0.5},
                                                             {"item_id": 1, "score": 0.4}]}),
}


def sample_users(path: Path, warm: int, cold: int) -> Dict[str, List[int]]:
    if path is not None and path.is_file():
        users = json.loads(path.read_text(encoding="utf-8"))
        return {"warm": users["warm"][:warm], "cold": users["cold"][:cold]}
    print(f"No sample_users.json at {path}; using user IDs 0..{warm - 1} and no cold users.")
    return {"warm": list(range(warm)), "cold": []}


def default_users_file() -> Path:
    root = Path(os.environ.get("CH13_ARTIFACT_ROOT", CHAPTER_DIR / "artifacts"))
    pointer = root / "ACTIVE"
    bundle = Path(os.environ["CH13_BUNDLE_DIR"]) if os.environ.get("CH13_BUNDLE_DIR") else \
        (root / pointer.read_text(encoding="utf-8").strip() if pointer.is_file() else None)
    return bundle / "sample_users.json" if bundle else None


class Smoke:
    def __init__(self, client: httpx.Client):
        self.client = client
        self.checks: List[dict] = []

    def check(self, name: str, ok: bool, detail="") -> bool:
        self.checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        return ok

    def ready(self, name: str, url: str) -> None:
        try:
            response = self.client.get(url.rstrip("/") + "/health/ready")
            body = response.json()
            self.check(f"{name} ready", response.status_code == 200 and body.get("model_version") not in
                       (None, "", "unloaded"), f"HTTP {response.status_code}, model_version={body.get('model_version')}")
        except (httpx.HTTPError, ValueError) as exc:
            self.check(f"{name} ready", False, type(exc).__name__)

    def contract_violation(self, name: str, url: str, path: str, payload: dict) -> None:
        response = self.client.post(url.rstrip("/") + path, json=payload)
        try:
            body = response.json()
        except ValueError:
            body = {}
        self.check(f"{name} returns 422 on a contract violation",
                   response.status_code == 422 and body.get("error") == "contract_violation"
                   and bool(body.get("trace_id")) and bool(body.get("model_version")),
                   f"HTTP {response.status_code}")


def recommend(client: httpx.Client, url: str, user_id: int, num_results: int, deadline_ms) -> tuple:
    trace_id = uuid.uuid4().hex
    headers = {"X-Trace-Id": trace_id}
    if deadline_ms is not None:
        headers["X-Request-Timeout-Ms"] = str(deadline_ms)
    started = time.perf_counter()
    response = client.post(url.rstrip("/") + "/v1/recommend", json={"user_id": user_id, "num_results": num_results},
                           headers=headers)
    return response, trace_id, (time.perf_counter() - started) * 1000.0


def slate_problems(response: httpx.Response, trace_id: str, num_results: int) -> List[str]:
    if response.status_code != 200:
        return [f"HTTP {response.status_code}"]
    body = response.json()
    problems = []
    if body.get("trace_id") != trace_id or response.headers.get("X-Trace-Id") != trace_id:
        problems.append("trace_id not echoed")
    if not body.get("model_version"):
        problems.append("no model_version")
    items = body.get("items", [])
    if not 0 < len(items) <= num_results:
        problems.append(f"{len(items)} items")
    if len({i["item_id"] for i in items}) != len(items):
        problems.append("duplicate items")
    if [i["rank"] for i in items] != list(range(1, len(items) + 1)):
        problems.append("ranks not 1..k")
    if body.get("degradation", {}).get("rung") not in (1, 2, 3, 4, 5, 6):
        problems.append("no degradation rung")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="orchestrator base URL")
    parser.add_argument("--service", action="append", default=None, metavar="NAME=URL",
                        help="stage service to check directly (repeatable); default: the four local ports")
    parser.add_argument("--no-services", action="store_true", help="check the orchestrator only")
    parser.add_argument("--users-file", type=Path, default=None, help="default: <active bundle>/sample_users.json")
    parser.add_argument("--warm", type=int, default=20)
    parser.add_argument("--cold", type=int, default=5)
    parser.add_argument("--num-results", type=int, default=20)
    parser.add_argument("--deadline-ms", type=float, default=None, help="X-Request-Timeout-Ms for slate requests")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP client timeout")
    parser.add_argument("--token-env", default="AUTH_TOKEN",
                        help="environment variable holding a bearer token (Cloud Run); unset means no auth")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.no_services:
        services = {}
    elif args.service:
        services = dict(item.split("=", 1) for item in args.service)
    else:
        services = dict(DEFAULT_URLS)
    headers = {}
    if os.environ.get(args.token_env):
        headers["Authorization"] = f"Bearer {os.environ[args.token_env]}"
    users = sample_users(args.users_file or default_users_file(), args.warm, args.cold)

    started = time.time()
    with httpx.Client(timeout=args.timeout_s, headers=headers) as client:
        smoke = Smoke(client)
        print("Readiness")
        smoke.ready("orchestrator", args.url)
        for name, url in services.items():
            smoke.ready(name, url)

        print("Slates")
        latencies, rungs, sources, failures = [], {}, {}, []
        for user_id in users["warm"]:
            response, trace_id, ms = recommend(client, args.url, user_id, args.num_results, args.deadline_ms)
            problems = slate_problems(response, trace_id, args.num_results)
            if problems:
                failures.append({"user_id": user_id, "problems": problems})
                continue
            body = response.json()
            latencies.append(ms)
            rung = body["degradation"]["rung"]
            rungs[rung] = rungs.get(rung, 0) + 1
            sources[body["score_source"]] = sources.get(body["score_source"], 0) + 1
        smoke.check(f"{len(users['warm'])} warm-user slates well formed", not failures and bool(latencies),
                    json.dumps(failures[:3]) if failures else f"rungs {rungs}, score sources {sources}")
        smoke.check("full pipeline (rung 1) reached", rungs.get(1, 0) > 0, f"{rungs.get(1, 0)} of {len(latencies)}")

        cold_ok, cold_detail = True, []
        for user_id in users["cold"]:
            response, trace_id, _ = recommend(client, args.url, user_id, args.num_results, args.deadline_ms)
            problems = slate_problems(response, trace_id, args.num_results)
            body = response.json() if response.status_code == 200 else {}
            source = body.get("stage_versions", {}).get("retrieval_source")
            rung = body.get("degradation", {}).get("rung")
            if problems or (rung is not None and rung < 5 and source != "popularity_cold_start"):
                cold_ok = False
                cold_detail.append({"user_id": user_id, "problems": problems, "retrieval_source": source})
        if users["cold"]:
            smoke.check(f"{len(users['cold'])} cold-start users served from popularity", cold_ok,
                        json.dumps(cold_detail[:3]) if cold_detail else "")

        print("Contracts and deadlines")
        smoke.contract_violation("orchestrator", args.url, "/v1/recommend", {"user_id": -1})
        for name, url in services.items():
            path, payload = BAD_REQUESTS[name]
            smoke.contract_violation(name, url, path, payload)
        response = client.post(args.url.rstrip("/") + "/v1/recommend", json={"user_id": users["warm"][0]},
                               headers={"X-Request-Timeout-Ms": "soon"})
        smoke.check("unparseable deadline header returns 422", response.status_code == 422,
                    f"HTTP {response.status_code}")
        response = client.post(args.url.rstrip("/") + "/v1/recommend", json={"user_id": users["warm"][0]},
                               headers={"X-Request-Timeout-Ms": "1"})
        degradation = response.json().get("degradation", {}) if response.status_code == 200 else {}
        rung, label = degradation.get("rung"), degradation.get("label")
        smoke.check("1 ms deadline degrades to the cached list instead of failing",
                    response.status_code == 200 and (rung, label) == (6, "Cached list"),
                    f"HTTP {response.status_code}, rung {rung} {label}")

    passed = all(c["passed"] for c in smoke.checks)
    report = {
        "passed": passed, "orchestrator_url": args.url, "services": services,
        "started_at_unix": round(started, 1), "duration_s": round(time.time() - started, 1),
        "warm_requests": len(latencies),
        "client_latency_ms": {
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "rungs": rungs, "score_sources": sources, "checks": smoke.checks,
    }
    print(f"\n{'PASSED' if passed else 'FAILED'}: {sum(c['passed'] for c in smoke.checks)}/{len(smoke.checks)} checks; "
          f"client latency p50 {report['client_latency_ms']['p50']} ms over {len(latencies)} warm requests")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
