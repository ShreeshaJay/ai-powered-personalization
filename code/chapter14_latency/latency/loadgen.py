"""
Open-loop and closed-loop load generation against an in-process handler.

Closed loop: each client sends a request, waits for the response, optionally
thinks, and sends the next one. When the server stalls, the clients stop
sending, so the stall shows up as a handful of slow samples. This is
coordinated omission (Tene, 2012).

Open loop: requests are sent on a schedule fixed in advance, whatever the
server is doing. Latency is measured from each request's *intended* send
time, so time spent queued behind a stall is counted.

The "server" here is a thread pool of ``n_workers`` threads, which plays the
role of a service's worker pool. Requests waiting for a free worker are the
queueing stage (S1) of the attribution framework.

An open-loop generator can itself fall behind its schedule, for example when
the process it shares with the server is busy. Because latency is measured
from the intended send time, the lateness is still counted, but it is the
generator's delay and not the server's, and the server receives the late
requests in a burst instead of on the intended schedule. ``dispatch_lag_ms``
records the lateness per request and ``dispatch_lag_report`` flags a run
whose generator fell behind, so such a run is excluded instead of trusted.

Both generators pause Python's automatic garbage collection for the measured
window, after a full collection (``gc_paused``). The harness keeps every
record and trace until the run ends, so without the pause a full collection
of that growing heap lands inside the window and stalls the server and the
generator together.
"""

from __future__ import annotations

import gc
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from latency.timing import now_ns

# handler(request_index, intended_ns, submitted_ns, start_ns) -> any result
Handler = Callable[[int, int, int, int], Any]


@dataclass
class RequestRecord:
    index: int
    intended_ns: int
    submitted_ns: int
    start_ns: int
    end_ns: int
    result: Any = None
    error: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        """Response time measured from the intended send time."""
        return (self.end_ns - self.intended_ns) / 1e6

    @property
    def service_ms(self) -> float:
        """Time from a worker picking the request up to completion."""
        return (self.end_ns - self.start_ns) / 1e6

    @property
    def dispatch_lag_ms(self) -> float:
        """How late the load generator itself was in sending the request."""
        return (self.submitted_ns - self.intended_ns) / 1e6


def open_loop_schedule(rate_per_s: float, duration_s: float, seed: int = 0,
                       process: str = "poisson") -> np.ndarray:
    """Intended send offsets in seconds for an open-loop run.

    ``poisson`` draws exponential gaps (memoryless arrivals); ``uniform``
    sends exactly every ``1 / rate_per_s`` seconds.
    """
    if rate_per_s <= 0 or duration_s <= 0:
        raise ValueError("rate_per_s and duration_s must be positive")
    if process == "uniform":
        return np.arange(0.0, duration_s, 1.0 / rate_per_s)
    if process != "poisson":
        raise ValueError("process must be 'poisson' or 'uniform'")
    rng = np.random.default_rng(seed)
    expected = int(rate_per_s * duration_s * 1.2) + 16
    offsets = np.cumsum(rng.exponential(1.0 / rate_per_s, expected))
    while offsets[-1] < duration_s:
        more = np.cumsum(rng.exponential(1.0 / rate_per_s, expected)) + offsets[-1]
        offsets = np.concatenate([offsets, more])
    return offsets[offsets < duration_s]


@contextmanager
def gc_paused(enabled: bool = True) -> Iterator[None]:
    """Collect garbage, freeze the survivors, and pause automatic collection.

    Objects that later die are still freed by reference counting; only
    reference cycles wait for the collection after the window.
    """
    if not enabled:
        yield
        return
    was_enabled = gc.isenabled()
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.unfreeze()
        if was_enabled:
            gc.enable()


def _sleep_until(target_ns: int, spin_ns: int = 1_000_000) -> None:
    """Sleep coarsely, then yield-spin for the last ``spin_ns`` to hit the target."""
    while True:
        remaining = target_ns - now_ns()
        if remaining <= 0:
            return
        if remaining > spin_ns:
            time.sleep((remaining - spin_ns) / 1e9)
        else:
            time.sleep(0)


def _run_one(handler: Handler, index: int, intended_ns: int, submitted_ns: int) -> RequestRecord:
    start = now_ns()
    try:
        result = handler(index, intended_ns, submitted_ns, start)
        error = None
    except Exception as exc:  # recorded, never swallowed silently
        result, error = None, repr(exc)
    return RequestRecord(index, intended_ns, submitted_ns, start, now_ns(), result, error)


def run_open_loop(handler: Handler, offsets_s: np.ndarray, n_workers: int,
                  lead_s: float = 0.05, pause_gc: bool = True) -> List[RequestRecord]:
    """Send one request per offset to a pool of ``n_workers`` and wait for all of them."""
    with gc_paused(pause_gc):
        t0 = now_ns() + int(lead_s * 1e9)
        futures = []
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="worker") as pool:
            for i, off in enumerate(offsets_s):
                intended = t0 + int(off * 1e9)
                _sleep_until(intended)
                futures.append(pool.submit(_run_one, handler, i, intended, now_ns()))
            records = [f.result() for f in futures]
    return records


def run_closed_loop(handler: Handler, n_clients: int, duration_s: float,
                    think_s: float = 0.0, pause_gc: bool = True) -> List[RequestRecord]:
    """Each of ``n_clients`` threads sends, waits, thinks, and repeats until time is up."""
    with gc_paused(pause_gc):
        return _closed_loop(handler, n_clients, duration_s, think_s)


def _closed_loop(handler: Handler, n_clients: int, duration_s: float,
                 think_s: float) -> List[RequestRecord]:
    deadline = now_ns() + int(duration_s * 1e9)
    records: List[RequestRecord] = []
    lock = threading.Lock()
    counter = iter(range(10**12))

    def client() -> None:
        while now_ns() < deadline:
            with lock:
                i = next(counter)
            sent = now_ns()
            rec = _run_one(handler, i, sent, sent)
            with lock:
                records.append(rec)
            if think_s > 0:
                time.sleep(think_s)

    threads = [threading.Thread(target=client, name=f"client-{c}") for c in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records.sort(key=lambda r: r.intended_ns)
    return records


def throughput_per_s(records: List[RequestRecord]) -> float:
    """Completed requests per second over the span of the run."""
    if len(records) < 2:
        return float("nan")
    first = min(r.intended_ns for r in records)
    last = max(r.end_ns for r in records)
    return len(records) / ((last - first) / 1e9)


def dispatch_lag_report(lags_ms: Sequence[float], max_lag_ms: float) -> Dict[str, object]:
    """Summarize how late an open-loop generator sent its requests.

    ``generator_behind`` is true when the 99th percentile of the lag exceeds
    ``max_lag_ms``. Latencies from such a run include the generator's own
    delay and were offered in bursts, so they overstate what the server did
    under the intended arrival process and should not be compared.
    """
    lag = np.asarray(lags_ms, dtype=np.float64)
    if lag.size == 0:
        return {"dispatch_lag_p50_ms": math.nan, "dispatch_lag_p99_ms": math.nan,
                "dispatch_lag_max_ms": math.nan, "generator_behind": False}
    p99 = float(np.percentile(lag, 99))
    return {"dispatch_lag_p50_ms": float(np.percentile(lag, 50)), "dispatch_lag_p99_ms": p99,
            "dispatch_lag_max_ms": float(lag.max()), "generator_behind": bool(p99 > max_lag_ms)}
