"""
The orchestrator's request path under injected faults (Figure 13.6).

Each test wires the real stage services through a transport that can fail,
delay or corrupt calls, and checks which rung of the ladder the response
reports, which fallbacks it lists, and how many calls each stage received.
"""

import json
import logging
import re
import time

import pytest

from common.artifacts import CHAPTER_DIR
from common.resilience import StageBudget
from contracts import ClientResponse
from services.orchestrator.pipeline import LADDER, RUNG_LABELS
from tests.helpers import FakeClock, paced_order, relaxed_settings

# Figure 13.6 (figures_src/ch13_figures.py, fig_13_6), top to bottom.
FIGURE_13_6 = [
    ("Full pipeline", "all five services answer"),
    ("Skip heavy ranker", "re-rank the lightweight scores"),
    ("Skip re-ranker", "sort by best score, minimal hard rules"),
    ("Skip lightweight ranker", "top candidates by similarity"),
    ("Popularity candidates", "in-memory list, ranked if possible"),
    ("Cached list", "precomputed, returned without ranking"),
]


def recommend(client, user_id, body=None, **headers):
    response = client.post("/v1/recommend", json={"user_id": user_id, "num_results": 10, **(body or {})},
                           headers={k.replace("_", "-"): v for k, v in headers.items()})
    assert response.status_code == 200, response.text
    return response.json()


def fallbacks(body):
    return {(f["stage"], f["reason"]) for f in body["degradation"]["fallbacks"]}


def skipped(body):
    return [(f["rung"], f["label"], f["stage"]) for f in body["degradation"]["fallbacks"]]


def test_ladder_matches_figure_13_6_in_order_and_wording():
    assert [(label, description) for _, label, description, _ in LADDER] == FIGURE_13_6
    assert [rung for rung, *_ in LADDER] == [1, 2, 3, 4, 5, 6]
    assert [stage for *_, stage in LADDER] == [None, "heavy_ranker", "reranker", "lightweight_ranker",
                                               "retrieval", "pipeline"]


def assert_slate(body, n=10):
    items = body["items"]
    assert 0 < len(items) <= n
    assert len({i["item_id"] for i in items}) == len(items)
    assert [i["rank"] for i in items] == list(range(1, len(items) + 1))


def test_full_pipeline_is_rung_1(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    user = sample_users["warm"][0]
    history = set(open_feature_store(bundle_dir / "feature_store").get_user(user).recent_item_ids)
    with orchestrator() as (client, transports):
        response = client.post("/v1/recommend", json={"user_id": user, "num_results": 10},
                               headers={"X-Trace-Id": "trace-abc"})
        body = response.json()
        assert response.status_code == 200
        assert body["trace_id"] == "trace-abc" and response.headers["X-Trace-Id"] == "trace-abc"
        degradation = body["degradation"]
        assert (degradation["rung"], degradation["label"], degradation["fallbacks"]) == (1, "Full pipeline", [])
        assert degradation["feature_defaults"]["lookup_failed"] == [] and degradation["feature_defaults"]["stale"] == {}
        assert body["score_source"] == "value_function"
        assert body["model_version"] == response.headers["X-Model-Version"]
        assert {"retrieval", "index", "lightweight_ranker", "heavy_ranker", "reranker_config",
                "feature_set"} <= set(body["stage_versions"])
        assert body["stage_versions"]["retrieval_source"] == "two_tower"
        assert_slate(body)
        assert history and not {i["item_id"] for i in body["items"]} & history
        assert all(t.calls == 1 for t in transports.values())
        for transport in transports.values():
            assert transport.headers[0]["x-trace-id"] == "trace-abc"


def test_cold_start_user_gets_popularity_from_retrieval(orchestrator, sample_users):
    with orchestrator() as (client, _):
        body = recommend(client, sample_users["cold"][0])
        assert body["degradation"]["rung"] == 1
        assert body["stage_versions"]["retrieval_source"] == "popularity_cold_start"
        assert_slate(body)


def test_heavy_ranker_timeout_is_rung_2_and_not_retried(orchestrator, sample_users):
    settings = relaxed_settings()
    settings.budgets["heavy_ranker"] = StageBudget("heavy_ranker", cap_ms=300, min_ms=5)
    with orchestrator(faults={"heavy_ranker": {"mode": "delay", "delay_s": 2.0}}, settings=settings) as (client, t):
        body = recommend(client, sample_users["warm"][1])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (2, "Skip heavy ranker")
        assert skipped(body) == [(2, "Skip heavy ranker", "heavy_ranker")]
        assert fallbacks(body) == {("heavy_ranker", "timeout")}
        # Without task scores the re-ranker keeps the lightweight score (no value function).
        assert body["score_source"] == "lightweight_ranker"
        assert t["heavy_ranker"].calls == 1
        assert body["timings_ms"]["heavy_ranker"] < 1500
        assert_slate(body)


def test_reranker_failure_is_rung_3_with_hard_rules(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    user = sample_users["warm"][3]
    history = set(open_feature_store(bundle_dir / "feature_store").get_user(user).recent_item_ids)
    with orchestrator(faults={"reranker": {"mode": "http_500"}}) as (client, _):
        body = recommend(client, user)
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (3, "Skip re-ranker")
        assert skipped(body) == [(3, "Skip re-ranker", "reranker")]
        assert body["score_source"] == "heavy_ranker"
        assert paced_order([i["artist_id"] for i in body["items"]], [i["album_id"] for i in body["items"]])
        assert not {i["item_id"] for i in body["items"]} & history


def test_lightweight_failure_is_rung_4_and_still_runs_heavy_ranker(orchestrator, sample_users):
    with orchestrator(faults={"lightweight_ranker": {"mode": "http_500"}}) as (client, t):
        body = recommend(client, sample_users["warm"][2])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (4, "Skip lightweight ranker")
        assert skipped(body) == [(4, "Skip lightweight ranker", "lightweight_ranker")]
        assert "by retrieval score" in body["degradation"]["fallbacks"][0]["action"]
        assert t["heavy_ranker"].calls == 1
        assert body["score_source"] == "value_function"


def test_retrieval_down_is_rung_5_popularity_candidates(orchestrator, sample_users):
    with orchestrator(faults={"retrieval": {"mode": "connect_error"}}) as (client, t):
        body = recommend(client, sample_users["warm"][4])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (5, "Popularity candidates")
        assert skipped(body) == [(5, "Popularity candidates", "retrieval")]
        assert fallbacks(body) == {("retrieval", "error")}
        assert t["retrieval"].calls == 2          # one bounded retry for a refused connection
        assert t["lightweight_ranker"].calls == 1  # the popularity candidates are still ranked
        assert_slate(body)


def test_everything_down_is_rung_6_cached_list_and_stays_ready(orchestrator, sample_users):
    faults = {name: {"mode": "connect_error"} for name in ("retrieval", "lightweight_ranker", "heavy_ranker",
                                                           "reranker")}
    with orchestrator(faults=faults) as (client, t):
        assert client.get("/health/ready").status_code == 200
        body = recommend(client, sample_users["warm"][5])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (6, "Cached list")
        assert body["score_source"] == "popularity"
        assert skipped(body) == [(2, "Skip heavy ranker", "heavy_ranker"), (3, "Skip re-ranker", "reranker"),
                                 (4, "Skip lightweight ranker", "lightweight_ranker"),
                                 (5, "Popularity candidates", "retrieval"), (6, "Cached list", "pipeline")]
        assert all(transport.calls == 2 for transport in t.values())
        assert_slate(body)


def test_combined_failures_report_the_lowest_rung_and_list_both_in_figure_order(orchestrator, sample_users):
    faults = {"lightweight_ranker": {"mode": "http_500"}, "heavy_ranker": {"mode": "http_500"}}
    with orchestrator(faults=faults) as (client, _):
        body = recommend(client, sample_users["warm"][6])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (4, "Skip lightweight ranker")
        assert skipped(body) == [(2, "Skip heavy ranker", "heavy_ranker"),
                                 (4, "Skip lightweight ranker", "lightweight_ranker")]
        assert body["score_source"] == "retrieval"


def test_tiny_deadline_returns_the_cached_list(orchestrator, sample_users):
    with orchestrator() as (client, _):
        body = recommend(client, sample_users["warm"][0], X_Request_Timeout_Ms="1")
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (6, RUNG_LABELS[6])
        assert skipped(body) == [(6, "Cached list", "pipeline")]
        assert body["score_source"] == "popularity"
        assert_slate(body)


def test_short_deadline_skips_stages_without_calling_them(orchestrator, sample_users):
    from services.orchestrator.settings import OrchestratorSettings

    with orchestrator(settings=OrchestratorSettings()) as (client, t):
        body = recommend(client, sample_users["warm"][0], X_Request_Timeout_Ms="40")
        assert body["degradation"]["rung"] >= 5
        assert t["retrieval"].calls == 0
        assert ("retrieval", "deadline_budget") in fallbacks(body) or body["degradation"]["rung"] == 6


def test_downstream_deadline_is_the_remaining_budget(orchestrator, sample_users):
    settings = relaxed_settings()
    with orchestrator(settings=settings) as (client, t):
        recommend(client, sample_users["warm"][0], X_Request_Timeout_Ms="20000")
        from common.resilience import BudgetPlanner
        from services.orchestrator.settings import STAGES

        planner = BudgetPlanner([settings.budgets[s] for s in STAGES], settings.response_reserve_ms)
        for stage in STAGES:
            sent = float(t[stage].headers[0]["x-request-timeout-ms"])
            assert 0 < sent <= min(10000, 20000 - planner.reserve_after(stage))


def test_client_deadline_is_capped(orchestrator, sample_users):
    with orchestrator() as (client, _):
        body = recommend(client, sample_users["warm"][0], X_Request_Timeout_Ms="999999")
        assert body["deadline_ms"] == 60000


def test_reranker_503_is_retried_once(orchestrator, sample_users):
    with orchestrator(faults={"reranker": {"mode": "http_503", "fail_times": 1}}) as (client, t):
        body = recommend(client, sample_users["warm"][0])
        assert body["degradation"]["rung"] == 1
        assert t["reranker"].calls == 2


def test_breaker_opens_on_failure_rate_then_trial_call_closes_it(orchestrator, sample_users):
    clock = FakeClock()
    settings = relaxed_settings(breaker_min_calls=4)
    with orchestrator(faults={"lightweight_ranker": {"mode": "http_500"}}, settings=settings,
                      clock=clock) as (client, t):
        user = sample_users["warm"][0]
        for _ in range(4):
            assert fallbacks(recommend(client, user)) == {("lightweight_ranker", "error")}
        assert t["lightweight_ranker"].calls == 4
        breaker = client.get("/health").json()["circuit_breakers"]["lightweight_ranker"]
        assert breaker["state"] == "open" and breaker["failure_rate"] == 1.0

        body = recommend(client, user)
        assert fallbacks(body) == {("lightweight_ranker", "circuit_open")}
        assert t["lightweight_ranker"].calls == 4          # no call while open

        t["lightweight_ranker"].mode = None                 # the service recovers
        clock.advance(31)
        body = recommend(client, user)
        assert body["degradation"]["rung"] == 1
        assert t["lightweight_ranker"].calls == 5
        assert client.get("/health").json()["circuit_breakers"]["lightweight_ranker"]["state"] == "closed"


def test_breaker_stays_closed_below_the_failure_rate(orchestrator, sample_users):
    clock = FakeClock()
    settings = relaxed_settings(breaker_min_calls=4)
    faults = {"lightweight_ranker": {"mode": "http_500", "fail_times": 2}}
    with orchestrator(faults=faults, settings=settings, clock=clock) as (client, t):
        for _ in range(4):                                  # two failures in a row, then two successes
            recommend(client, sample_users["warm"][0])
        breaker = client.get("/health").json()["circuit_breakers"]["lightweight_ranker"]
        assert breaker["state"] == "closed" and breaker["window_failures"] == 2 and breaker["window_calls"] == 4


def test_unexpected_call_error_still_settles_the_breaker(orchestrator, sample_users):
    """An error outside the expected kinds is a failure, including on the half-open trial call."""
    clock = FakeClock()
    settings = relaxed_settings(breaker_min_calls=4)
    with orchestrator(faults={"lightweight_ranker": {"mode": "decoding_error"}}, settings=settings,
                      clock=clock) as (client, t):
        user = sample_users["warm"][0]

        def breaker_state():
            return client.get("/health").json()["circuit_breakers"]["lightweight_ranker"]["state"]

        for _ in range(4):
            body = recommend(client, user)
            assert fallbacks(body) == {("lightweight_ranker", "error")} and body["degradation"]["rung"] == 4
        assert breaker_state() == "open"

        clock.advance(31)                                   # the trial call fails the same way
        assert fallbacks(recommend(client, user)) == {("lightweight_ranker", "error")}
        assert t["lightweight_ranker"].calls == 5 and breaker_state() == "open"

        t["lightweight_ranker"].mode = None
        clock.advance(31)                                   # the next trial call is allowed and closes it
        assert recommend(client, user)["degradation"]["rung"] == 1
        assert breaker_state() == "closed"


def test_retry_budget_exhaustion_drops_retries_not_requests(orchestrator, sample_users):
    clock = FakeClock()
    settings = relaxed_settings(retry_budget_per_s=0.0, retry_budget_burst=1.0)
    with orchestrator(faults={"retrieval": {"mode": "http_503"}}, settings=settings, clock=clock) as (client, t):
        first = recommend(client, sample_users["warm"][0])
        assert t["retrieval"].calls == 2                    # the one token pays for a retry
        second = recommend(client, sample_users["warm"][1])
        assert t["retrieval"].calls == 3                    # no token left: no retry
        for body in (first, second):
            assert body["degradation"]["rung"] == 5
            assert_slate(body)
        assert "retry budget exhausted" in second["degradation"]["fallbacks"][0]["detail"]
        budget = client.get("/health").json()["retry_budget"]
        assert budget["retries_spent"] == 1 and budget["retries_dropped"] == 1


def test_request_log_feeds_stage_latency_percentiles(orchestrator, sample_users, caplog, tmp_path):
    from scripts.stage_latency import read_records, stage_percentiles

    with orchestrator() as (client, _):
        with caplog.at_level(logging.INFO, logger="ch13"):
            bodies = [recommend(client, user) for user in sample_users["warm"][:3]]
    trace = bodies[0]["trace_id"]
    stage_lines = [r.getMessage() for r in caplog.records if r.name == "ch13" and f"trace_id={trace}" in r.getMessage()]
    assert {line.split()[0] for line in stage_lines} >= {"retrieval", "lightweight_ranker", "heavy_ranker", "reranker"}
    log = tmp_path / "orchestrator.log"
    log.write_text("".join(f"2026-01-01 INFO {r.name} {r.getMessage()}\n" for r in caplog.records),
                   encoding="utf-8")
    records = read_records(log)
    assert [r["trace_id"] for r in records] == [b["trace_id"] for b in bodies]
    table = stage_percentiles(records, rung=1)
    assert table["heavy_ranker"]["count"] == 3
    assert table["heavy_ranker"]["p50"] <= table["heavy_ranker"]["p99"] <= table["total"]["p99"]


def test_downstream_422_is_a_contract_violation_not_an_outage(orchestrator, sample_users, caplog):
    with orchestrator(faults={"lightweight_ranker": {"mode": "http_422"}}) as (client, transports):
        with caplog.at_level(logging.WARNING, logger="ch13.orchestrator"):
            body = recommend(client, sample_users["warm"][0])
        assert fallbacks(body) == {("lightweight_ranker", "contract_violation")}
        assert transports["lightweight_ranker"].calls == 1                    # never retried
        health = client.get("/health").json()
        assert health["circuit_breakers"]["lightweight_ranker"]["window_failures"] == 0
        assert health["contract_rejections"] == {"lightweight_ranker": 1}
        rejection = [r.getMessage() for r in caplog.records if "contract rejection" in r.getMessage()]
        assert len(rejection) == 1 and body["trace_id"] in rejection[0] and "stage=lightweight_ranker" in rejection[0]


def test_malformed_downstream_response_is_rejected(orchestrator, sample_users):
    with orchestrator(faults={"heavy_ranker": {"mode": "bad_body"}}) as (client, _):
        body = recommend(client, sample_users["warm"][0])
        assert fallbacks(body) == {("heavy_ranker", "contract_violation")}
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (2, "Skip heavy ranker")


def test_heavy_ranker_reply_without_task_scores_is_a_contract_violation(orchestrator, sample_users):
    """Without the five heads the value function cannot run, so the reply is rejected, not used as a score."""
    with orchestrator(faults={"heavy_ranker": {"mode": "no_task_scores"}}) as (client, _):
        body = recommend(client, sample_users["warm"][0])
        assert fallbacks(body) == {("heavy_ranker", "contract_violation")}
        assert (body["degradation"]["rung"], body["score_source"]) == (2, "lightweight_ranker")
        assert client.get("/health").json()["circuit_breakers"]["heavy_ranker"]["window_failures"] == 1


class _Broken:
    """Wraps a feature store and makes one method raise."""

    def __init__(self, store, method):
        self._store, self._method = store, method

    def __getattr__(self, name):
        if name == self._method:
            def fail(*args, **kwargs):
                raise ConnectionError("feature store unavailable")
            return fail
        return getattr(self._store, name)


def test_user_feature_failure_is_a_flag_not_a_rung(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    store = _Broken(open_feature_store(bundle_dir / "feature_store"), "get_user")
    with orchestrator(store=store) as (client, _):
        body = recommend(client, sample_users["warm"][0])
        assert (body["degradation"]["rung"], body["degradation"]["fallbacks"]) == (1, [])
        assert body["degradation"]["feature_defaults"]["lookup_failed"] == ["user_features"]


def test_item_feature_failure_skips_both_rankers(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    store = _Broken(open_feature_store(bundle_dir / "feature_store"), "get_user_items")
    with orchestrator(store=store) as (client, t):
        body = recommend(client, sample_users["warm"][0])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (4, "Skip lightweight ranker")
        assert fallbacks(body) == {("heavy_ranker", "features_unavailable"),
                                   ("lightweight_ranker", "features_unavailable")}
        assert body["degradation"]["feature_defaults"]["lookup_failed"] == ["item_features"]
        assert t["lightweight_ranker"].calls == 0 and t["heavy_ranker"].calls == 0
        assert_slate(body)


class _Networked:
    """Wraps a feature store as if it answered over a network; the named methods first sleep `delay_s`."""

    in_process = False

    def __init__(self, store, *slow):
        self._store, self._slow, self.delay_s = store, set(slow), 0.0

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        if name not in self._slow:
            return attr

        def slow(*args, **kwargs):
            time.sleep(self.delay_s)
            return attr(*args, **kwargs)
        return slow


def test_slow_networked_user_lookup_times_out_to_defaults(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    store = _Networked(open_feature_store(bundle_dir / "feature_store"), "get_user")
    with orchestrator(store=store, settings=relaxed_settings(feature_lookup_cap_ms=200.0)) as (client, _):
        user = sample_users["warm"][0]
        body = recommend(client, user)
        assert (body["degradation"]["rung"], body["degradation"]["feature_defaults"]["lookup_failed"]) == (1, [])
        store.delay_s = 2.0
        body = recommend(client, user)
        assert (body["degradation"]["rung"], body["degradation"]["fallbacks"]) == (1, [])
        assert body["degradation"]["feature_defaults"]["lookup_failed"] == ["user_features"]
        assert body["timings_ms"]["user_features"] < 1500


def test_slow_networked_item_lookup_times_out_and_skips_both_rankers(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    store = _Networked(open_feature_store(bundle_dir / "feature_store"), "get_items")
    with orchestrator(store=store, settings=relaxed_settings(feature_lookup_cap_ms=200.0)) as (client, t):
        store.delay_s = 2.0
        body = recommend(client, sample_users["warm"][0])
        assert (body["degradation"]["rung"], body["degradation"]["label"]) == (4, "Skip lightweight ranker")
        assert fallbacks(body) == {("heavy_ranker", "features_unavailable"),
                                   ("lightweight_ranker", "features_unavailable")}
        assert "timeout" in body["degradation"]["fallbacks"][0]["detail"]
        assert body["degradation"]["feature_defaults"]["lookup_failed"] == ["item_features"]
        assert t["lightweight_ranker"].calls == 0 and t["heavy_ranker"].calls == 0
        assert body["timings_ms"]["item_features"] < 1500
        assert_slate(body)


def test_stale_features_are_replaced_counted_and_flagged(orchestrator, bundle_dir, sample_users):
    from common.feature_store import open_feature_store

    store = open_feature_store(bundle_dir / "feature_store")
    user_age = store.max_age[store.user_feature_names[0]]
    with orchestrator(store=store) as (client, _):
        user = sample_users["warm"][0]
        fresh = recommend(client, user)["degradation"]["feature_defaults"]
        assert fresh["stale"] == {}
        late = store.reference_timestamp + user_age + 1           # user aggregates expired, items not yet
        body = recommend(client, user, body={"context": {"timestamp": late}})
        assert body["degradation"]["rung"] == 1
        stale = body["degradation"]["feature_defaults"]["stale"]
        assert {name: stale[name] for name in store.user_feature_names} == {n: 1 for n in store.user_feature_names}
        assert not set(stale) & set(store.item_feature_names)
        counted = client.get("/health").json()["feature_store"]["substitutions"]["stale"]
        assert counted[store.user_feature_names[0]] >= 1


def test_client_policy_version_pins_the_reranker_config(orchestrator, sample_users):
    with orchestrator() as (client, _):
        user = sample_users["warm"][0]
        assert recommend(client, user)["stage_versions"]["reranker_config"] == "vf-default-v1"
        pinned = recommend(client, user, body={"policy_version": "vf-growth-v1"})["stage_versions"]
        assert (pinned["reranker_config"], pinned["policy_pin"]) == ("vf-growth-v1", "honoured")
        missing = recommend(client, user, body={"policy_version": "vf-missing-v9"})["stage_versions"]
        assert (missing["reranker_config"], missing["policy_pin"]) == ("vf-default-v1", "unavailable")


def readme_curl_examples():
    """Every `curl ... /v1/recommend ... -d '<json>'` block in the README, with its headers."""
    text = (CHAPTER_DIR / "README.md").read_text(encoding="utf-8")
    pattern = re.compile(r"curl[^\n]*?/v1/recommend(?P<rest>(?:[^\n]*\\\n)*[^\n]*)")
    examples = []
    for match in pattern.finditer(text):
        command = match.group(0)
        payload = re.search(r"-d '(\{.*?\})'", command, re.S)
        headers = dict(re.findall(r'-H "([^:]+): ([^"]+)"', command))
        if payload:
            examples.append((json.loads(payload.group(1)), headers))
    return examples


def test_readme_curl_example_passes_the_client_contract(orchestrator):
    examples = readme_curl_examples()
    assert examples, "README has no curl example for /v1/recommend"
    with orchestrator() as (client, _):
        for payload, headers in examples:
            response = client.post("/v1/recommend", json=payload, headers=headers)
            assert response.status_code == 200, (payload, response.text)
            ClientResponse.model_validate(response.json())


@pytest.mark.parametrize("payload", [{"user_id": -5}, {"user_id": 1, "num_results": 0}, {"num_results": 5}])
def test_invalid_client_request_is_422(orchestrator, payload):
    with orchestrator() as (client, _):
        response = client.post("/v1/recommend", json=payload)
        body = response.json()
        assert response.status_code == 422
        assert body["error"] == "contract_violation" and body["trace_id"] and body["model_version"]


def test_bad_deadline_header_is_422(orchestrator):
    with orchestrator() as (client, _):
        response = client.post("/v1/recommend", json={"user_id": 1}, headers={"X-Request-Timeout-Ms": "soon"})
        assert response.status_code == 422
