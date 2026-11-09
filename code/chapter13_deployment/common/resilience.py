"""
Deadlines, stage budgets, circuit breakers, bounded retries and the
process-wide retry budget (Section 13.8).

These are small, dependency-free classes so the orchestrator logic can be
tested with a fake clock. Budgets and thresholds are configuration, not
measurements: Chapter 14 derives real values from load tests.
"""

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

Clock = Callable[[], float]
logger = logging.getLogger("ch13.resilience")


class Deadline:
    """An absolute point in time, created from a relative budget in milliseconds."""

    def __init__(self, budget_ms: float, clock: Clock = time.monotonic):
        self._clock = clock
        self.budget_ms = float(budget_ms)
        self._expires_at = clock() + self.budget_ms / 1000.0

    def remaining_ms(self) -> float:
        return max(0.0, (self._expires_at - self._clock()) * 1000.0)

    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def elapsed_ms(self) -> float:
        return self.budget_ms - self.remaining_ms()


@dataclass(frozen=True)
class StageBudget:
    """cap_ms bounds one stage; min_ms is the least time worth starting it with."""
    name: str
    cap_ms: float
    min_ms: float


class BudgetPlanner:
    """
    Derive each stage's timeout from the time left on the request deadline.

        timeout(stage) = min(cap_ms, remaining - reserve_after(stage))
        reserve_after(stage) = sum of min_ms of the stages after it + response reserve

    A stage whose timeout would fall below its own min_ms is skipped, so the
    request falls back early instead of starting work that cannot finish.
    """

    def __init__(self, stages: List[StageBudget], response_reserve_ms: float):
        self.stages = list(stages)
        self.response_reserve_ms = float(response_reserve_ms)
        self._index: Dict[str, int] = {stage.name: i for i, stage in enumerate(self.stages)}

    def reserve_after(self, name: str) -> float:
        later = self.stages[self._index[name] + 1:]
        return sum(stage.min_ms for stage in later) + self.response_reserve_ms

    def budget_for(self, name: str, remaining_ms: float) -> Optional[float]:
        stage = self.stages[self._index[name]]
        budget = min(stage.cap_ms, remaining_ms - self.reserve_after(name))
        return budget if budget >= stage.min_ms else None

    def min_ms(self, name: str) -> float:
        return self.stages[self._index[name]].min_ms


class CircuitBreaker:
    """
    Closed -> Open when the failure rate over the last `window_s` seconds
    exceeds `failure_rate` and the window holds at least `min_calls` calls
    (default: more than half of at least 20 calls in 10 s, Section 13.8.5).
    A run of consecutive failures alone never opens it: normal tail latency
    produces short streaks by chance.
    Open -> Half-open once `reset_timeout_s` has elapsed (cool-down).
    Half-open lets one trial call through: success closes the circuit and
    clears the window, failure reopens it for another cool-down.
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, name: str, failure_rate: float = 0.5, min_calls: int = 20, window_s: float = 10.0,
                 reset_timeout_s: float = 10.0, clock: Clock = time.monotonic):
        if not 0.0 < failure_rate < 1.0:
            raise ValueError("failure_rate must be between 0 and 1")
        if min_calls < 1:
            raise ValueError("min_calls must be >= 1")
        self.name = name
        self.failure_rate = failure_rate
        self.min_calls = min_calls
        self.window_s = window_s
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._outcomes: Deque[Tuple[float, bool]] = deque()   # (time, failed) within the window
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self.times_opened = 0

    def _prune(self) -> None:
        horizon = self._clock() - self.window_s
        while self._outcomes and self._outcomes[0][0] <= horizon:
            _, failed = self._outcomes.popleft()
            self._failures -= failed

    def _refresh(self) -> None:
        if self._state == self.OPEN and self._clock() - self._opened_at >= self.reset_timeout_s:
            self._state = self.HALF_OPEN
            self._probe_in_flight = False
            logger.info("circuit breaker %s: open -> half_open (cool-down elapsed)", self.name)

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh()
            return self._state

    def allow_request(self) -> bool:
        with self._lock:
            self._refresh()
            if self._state == self.CLOSED:
                return True
            if self._state == self.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._refresh()
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                self._outcomes.clear()
                self._failures = 0
                self._probe_in_flight = False
                logger.info("circuit breaker %s: half_open -> closed (trial call succeeded)", self.name)
            elif self._state == self.CLOSED:
                self._record(failed=False)

    def record_failure(self) -> None:
        with self._lock:
            self._refresh()
            if self._state == self.HALF_OPEN:
                self._open()
            elif self._state == self.CLOSED:
                self._record(failed=True)
                calls = len(self._outcomes)
                if calls >= self.min_calls and self._failures / calls > self.failure_rate:
                    self._open()

    def _record(self, failed: bool) -> None:
        self._prune()
        self._outcomes.append((self._clock(), failed))
        self._failures += failed

    def abandon(self) -> None:
        """A permitted call was cancelled before it finished; free the half-open probe slot."""
        with self._lock:
            self._probe_in_flight = False

    def _open(self) -> None:
        previous = self._state
        self._state = self.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False
        self.times_opened += 1
        logger.warning("circuit breaker %s: %s -> open (%d of %d calls failed in the window)", self.name,
                       previous, self._failures, len(self._outcomes))

    def snapshot(self) -> dict:
        with self._lock:
            self._refresh()
            self._prune()
            calls = len(self._outcomes)
            return {
                "state": self._state,
                "window_calls": calls,
                "window_failures": self._failures,
                "failure_rate": round(self._failures / calls, 4) if calls else 0.0,
                "times_opened": self.times_opened,
            }


@dataclass
class RetryPolicy:
    """
    At most `max_retries` extra attempts, only for errors where the request
    almost certainly did not run (connection refused, 502, 503) and only when
    the deadline still leaves the stage its minimum budget after the backoff.
    Timeouts are never retried: the budget is already spent. Every stage call
    is a read-only scoring request, so a retry cannot apply a change twice.
    """
    max_retries: int = 1
    backoff_ms: float = 10.0
    jitter: float = 0.5
    retryable: tuple = ("connect_error", "http_502", "http_503")

    def backoff_s(self, attempt: int, rng: random.Random = random) -> float:
        base = self.backoff_ms * (2 ** attempt)
        return base * (1.0 + self.jitter * rng.random()) / 1000.0

    def should_retry(self, attempt: int, kind: str, remaining_ms: float, min_ms: float,
                     next_backoff_s: float) -> bool:
        return (
            attempt < self.max_retries
            and kind in self.retryable
            and remaining_ms - next_backoff_s * 1000.0 >= min_ms
        )


class RetryBudget:
    """
    A token bucket shared by every stage call in the process (Section 13.8.4).

    Each retry spends one token; tokens refill at `per_second` up to `burst`.
    The default of one per second is the SRE book's example of 60 retries per
    minute in a process (Beyer et al., 2016). When the bucket is empty the
    retry is skipped and the stage falls back at once, so a failure that hits
    every request drops retries instead of multiplying the load.
    """

    def __init__(self, per_second: float = 1.0, burst: float = 10.0, clock: Clock = time.monotonic):
        if per_second < 0 or burst < 1:
            raise ValueError("per_second must be >= 0 and burst >= 1")
        self.per_second = float(per_second)
        self.burst = float(burst)
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = self.burst
        self._updated = clock()
        self.spent = 0
        self.dropped = 0

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.per_second)
        self._updated = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self.spent += 1
                return True
            self.dropped += 1
            return False

    def snapshot(self) -> dict:
        with self._lock:
            self._refill()
            return {"tokens": round(self._tokens, 3), "per_second": self.per_second, "burst": self.burst,
                    "retries_spent": self.spent, "retries_dropped": self.dropped}
