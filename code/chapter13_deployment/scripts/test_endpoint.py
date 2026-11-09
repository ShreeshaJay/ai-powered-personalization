"""
Send one recommendation request and print the response, stage by stage.

    python scripts/test_endpoint.py                       # first warm sample user
    python scripts/test_endpoint.py --user-id 1234 --deadline-ms 300

The equivalent curl command (bash):

    curl -s -X POST http://127.0.0.1:8000/v1/recommend \
      -H "Content-Type: application/json" -H "X-Request-Timeout-Ms: 1000" \
      -d '{"user_id": 1234, "num_results": 10}'

Requires the services to be running (python scripts/run_local.py).
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test import default_users_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--deadline-ms", type=float, default=1000.0)
    parser.add_argument("--raw", action="store_true", help="print the raw JSON response")
    args = parser.parse_args()

    user_id = args.user_id
    if user_id is None:
        users_file = default_users_file()
        user_id = json.loads(users_file.read_text(encoding="utf-8"))["warm"][0] if users_file and \
            users_file.is_file() else 0
    payload = {"user_id": user_id, "num_results": args.num_results}
    try:
        response = httpx.post(args.url.rstrip("/") + "/v1/recommend", json=payload,
                              headers={"X-Request-Timeout-Ms": f"{args.deadline_ms:.0f}"}, timeout=30.0)
    except httpx.ConnectError:
        print(f"Could not connect to {args.url}. Start the services with: python scripts/run_local.py")
        return 1
    body = response.json()
    if args.raw or response.status_code != 200:
        print(f"HTTP {response.status_code}")
        print(json.dumps(body, indent=2))
        return 0 if response.status_code == 200 else 1

    degradation = body["degradation"]
    print(f"POST /v1/recommend {json.dumps(payload)}  ->  HTTP {response.status_code}")
    print(f"trace_id       {body['trace_id']}")
    print(f"model_version  {body['model_version']}   deadline {body['deadline_ms']:.0f} ms")
    print(f"rung           {degradation['rung']}: {degradation['label']}")
    for fallback in degradation["fallbacks"]:
        print(f"  skipped      rung {fallback['rung']} {fallback['label']}: {fallback['stage']} "
              f"({fallback['reason']}), {fallback['action']}")
    features = degradation.get("feature_defaults", {})
    if features.get("lookup_failed") or features.get("missing") or features.get("stale"):
        print(f"feature defaults  lookup failed {features.get('lookup_failed')}, "
              f"missing {features.get('missing')}, stale {features.get('stale')}")
    print(f"score_source   {body['score_source']}")
    print("stage versions " + ", ".join(f"{k}={v}" for k, v in body["stage_versions"].items()))
    print("timings (ms)   " + ", ".join(f"{k} {v:.1f}" for k, v in body["timings_ms"].items()))
    print(f"\n{'rank':>4}  {'item_id':>10}  {'score':>8}  {'artist_id':>9}  {'album_id':>9}")
    for item in body["items"]:
        print(f"{item['rank']:>4}  {item['item_id']:>10}  {item['score']:>8.4f}  "
              f"{str(item.get('artist_id')):>9}  {str(item.get('album_id')):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
