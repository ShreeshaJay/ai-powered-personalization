"""Deadlines, budgets, circuit breaker and retry rules with a fake clock."""

import random

import pytest

from common.resilience import BudgetPlanner, CircuitBreaker, Deadline, RetryBudget, RetryPolicy
from services.orchestrator.settings import STAGES, OrchestratorSettings
from tests.helpers import FakeClock


@pytest.fixture
def planner():
    settings = OrchestratorSettings()
    return BudgetPlanner([settings.budgets[s] for s in STAGES], settings.response_reserve_ms)


def test_deadline_counts_down_and_expires():
    clock = FakeClock()
    deadline = Deadline(300, clock)
    assert deadline.remaining_ms() == pytest.approx(300)
    clock.advance(0.120)
    assert deadline.remaining_ms() == pytest.approx(180)
    assert deadline.elapsed_ms() == pytest.approx(120)
    clock.advance(0.5)
    assert deadline.remaining_ms() == 0 and deadline.expired()


def test_reserve_counts_later_stage_minimums(planner):
    # min_ms: retrieval 20, lightweight 15, heavy 25, reranker 10; response reserve 10
    assert planner.reserve_after("retrieval") == pytest.approx(15 + 25 + 10 + 10)
    assert planner.reserve_after("heavy_ranker") == pytest.approx(10 + 10)
    assert planner.reserve_after("reranker") == pytest.approx(10)


@pytest.mark.parametrize("stage,remaining,expected", [
    ("retrieval", 1000, 250),        # capped
    ("retrieval", 150, 90),          # 150 - 60 left for the later stages
    ("retrieval", 75, None),         # 15 < min 20: skip, fall back now
    ("heavy_ranker", 200, 180),      # 200 - 20
    ("heavy_ranker", 40, None),      # 20 < min 25
    ("reranker", 25, 15),            # 25 - 10
])
def test_stage_timeout_derives_from_remaining_time(planner, stage, remaining, expected):
    budget = planner.budget_for(stage, remaining)
    assert budget == (pytest.approx(expected) if expected is not None else None)


def _calls(breaker, clock, outcomes, spacing_s=0.1):
    for failed in outcomes:
        assert breaker.allow_request()
        breaker.record_failure() if failed else breaker.record_success()
        clock.advance(spacing_s)


def test_breaker_defaults_match_section_13_8_5():
    breaker = CircuitBreaker("x")
    settings = OrchestratorSettings()
    # "more than half of at least twenty calls in the last ten seconds"
    assert (breaker.failure_rate, breaker.min_calls, breaker.window_s) == (0.5, 20, 10.0)
    assert (settings.breaker_failure_rate, settings.breaker_min_calls, settings.breaker_window_s) == (0.5, 20, 10.0)


def test_a_streak_of_failures_does_not_open_the_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker("heavy_ranker", clock=clock)
    _calls(breaker, clock, [False] * 30 + [True] * 8 + [False] * 10)   # 8 in a row, 8 of 48 overall
    assert breaker.state == "closed" and breaker.times_opened == 0
    fresh = CircuitBreaker("cold", clock=clock)
    _calls(fresh, clock, [True] * 19)                                   # 19 in a row, below 20 calls
    assert fresh.state == "closed"


def test_breaker_opens_on_the_windowed_failure_rate():
    clock = FakeClock()
    breaker = CircuitBreaker("heavy_ranker", clock=clock)
    # Alternating outcomes, never more than two failures in a row.
    _calls(breaker, clock, [True, False] * 9 + [True])
    assert breaker.state == "closed"                                    # 10 of 19: below the minimum
    breaker.record_failure()                                            # 11 of 20 = 0.55
    assert breaker.snapshot()["window_calls"] == 20 and breaker.snapshot()["window_failures"] == 11
    assert breaker.state == "open" and not breaker.allow_request()


def test_exactly_half_does_not_open_and_old_calls_leave_the_window():
    clock = FakeClock()
    breaker = CircuitBreaker("x", clock=clock)
    _calls(breaker, clock, [True, False] * 10)                          # 10 of 20 is not "more than half"
    assert breaker.state == "closed"
    clock.advance(10)                                                    # all twenty calls age out
    assert breaker.snapshot()["window_calls"] == 0
    _calls(breaker, clock, [True] * 5)
    assert breaker.state == "closed"


def test_breaker_half_opens_after_cooldown_and_closes_on_a_successful_trial():
    clock = FakeClock()
    breaker = CircuitBreaker("heavy_ranker", min_calls=4, reset_timeout_s=10, clock=clock)
    _calls(breaker, clock, [True] * 4, spacing_s=0.0)
    assert breaker.state == "open" and not breaker.allow_request()

    clock.advance(9.9)
    assert not breaker.allow_request()
    clock.advance(0.2)
    assert breaker.state == "half_open"
    assert breaker.allow_request()            # the single trial call
    assert not breaker.allow_request()        # others wait for it
    breaker.record_failure()                  # trial failed: open again, new cool-down
    assert breaker.state == "open" and breaker.times_opened == 2

    clock.advance(10)
    assert breaker.allow_request()
    breaker.record_success()
    snapshot = breaker.snapshot()
    assert snapshot["state"] == "closed" and snapshot["window_calls"] == 0


def test_a_late_success_does_not_close_an_open_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker("x", min_calls=2, clock=clock)
    _calls(breaker, clock, [True, True], spacing_s=0.0)
    breaker.record_success()                  # a call admitted before the breaker opened
    assert breaker.state == "open"


def test_abandoned_probe_frees_the_slot():
    clock = FakeClock()
    breaker = CircuitBreaker("x", min_calls=1, reset_timeout_s=1, clock=clock)
    breaker.record_failure()
    clock.advance(1)
    assert breaker.allow_request()
    breaker.abandon()
    assert breaker.allow_request()


def test_retry_budget_drops_retries_once_the_bucket_is_empty():
    clock = FakeClock()
    budget = RetryBudget(per_second=1.0, burst=3, clock=clock)
    assert [budget.try_acquire() for _ in range(5)] == [True, True, True, False, False]
    assert budget.snapshot()["retries_spent"] == 3 and budget.snapshot()["retries_dropped"] == 2
    clock.advance(1.0)                         # one token back
    assert budget.try_acquire() and not budget.try_acquire()
    clock.advance(60)                          # refills only up to the burst
    assert budget.snapshot()["tokens"] == 3


def test_retry_only_for_calls_that_did_not_run_and_only_with_time_left():
    policy = RetryPolicy(max_retries=1, backoff_ms=10, jitter=0.0)
    backoff = policy.backoff_s(0, random.Random(0))
    assert backoff == pytest.approx(0.010)
    assert policy.should_retry(0, "connect_error", remaining_ms=100, min_ms=20, next_backoff_s=backoff)
    assert policy.should_retry(0, "http_503", remaining_ms=100, min_ms=20, next_backoff_s=backoff)
    assert not policy.should_retry(0, "timeout", remaining_ms=100, min_ms=20, next_backoff_s=backoff)
    assert not policy.should_retry(0, "http_500", remaining_ms=100, min_ms=20, next_backoff_s=backoff)
    assert not policy.should_retry(1, "connect_error", remaining_ms=100, min_ms=20, next_backoff_s=backoff)
    assert not policy.should_retry(0, "connect_error", remaining_ms=25, min_ms=20, next_backoff_s=backoff)
