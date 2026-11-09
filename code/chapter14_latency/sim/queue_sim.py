"""
Event-driven queueing simulator in virtual time.

Virtual time means no real sleeping: the simulator advances a clock from
event to event, so results are exact for the model and identical on every
operating system. Four experiments build on it:

1. ``stall_experiment``: a closed-loop and an open-loop generator measure
   the same server through a one-off stall (coordinated omission).
2. ``utilization_sweep``: mean and p99 response time as utilization rises,
   for several arrival and service-time distributions, against the
   single-server exponential (M/M/1) formulas.
3. ``fanout_experiment``: requests that wait for the slowest of N leaves.
4. ``hedging_experiment``: replicated servers where a backup request is sent
   after a delay and the loser is cancelled, measuring tail change and the
   extra work the backups cost.

Times are in arbitrary units; the experiments use milliseconds.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from latency.histogram import LatencyHistogram

Sampler = Callable[[int], np.ndarray]


# ------------------------------------------------------------------ samplers
def service_sampler(dist: str, mean: float, rng: np.random.Generator, sigma: float = 1.0,
                    p_slow: float = 0.01, slow_factor: float = 10.0) -> Sampler:
    """Return ``f(n)`` drawing ``n`` service times with the given mean.

    ``exponential``: memoryless, coefficient of variation 1.
    ``lognormal``: right-skewed; ``sigma`` is the log-scale standard deviation.
    ``bimodal``: exponential fast mode, plus a slow mode ``slow_factor`` times
    longer with probability ``p_slow`` (a cache miss or a slow replica).
    ``deterministic``: every job takes exactly ``mean``.
    """
    if dist == "exponential":
        return lambda n: rng.exponential(mean, n)
    if dist == "lognormal":
        mu = math.log(mean) - sigma ** 2 / 2
        return lambda n: rng.lognormal(mu, sigma, n)
    if dist == "bimodal":
        fast = mean / (1 - p_slow + p_slow * slow_factor)

        def draw(n: int) -> np.ndarray:
            slow = rng.random(n) < p_slow
            return rng.exponential(fast, n) * np.where(slow, slow_factor, 1.0)
        return draw
    if dist == "deterministic":
        return lambda n: np.full(n, float(mean))
    raise ValueError(f"unknown service distribution: {dist}")


def poisson_arrivals(rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.exponential(1.0 / rate, n))


def bursty_arrivals(rate: float, n: int, rng: np.random.Generator, burst_factor: float = 4.0,
                    burst_fraction: float = 0.2, arrivals_per_cycle: float = 50.0) -> np.ndarray:
    """Two-state Markov-modulated Poisson arrivals with long-run mean ``rate``.

    The source spends ``burst_fraction`` of the time sending at
    ``burst_factor * rate`` and the rest at a lower rate chosen so the
    average stays ``rate``.
    """
    if burst_factor * burst_fraction >= 1:
        raise ValueError("burst_factor * burst_fraction must be below 1")
    high = burst_factor * rate
    low = rate * (1 - burst_fraction * burst_factor) / (1 - burst_fraction)
    cycle = arrivals_per_cycle / rate
    times = np.empty(n)
    t, k, in_burst = 0.0, 0, False
    while k < n:
        length = rng.exponential(cycle * (burst_fraction if in_burst else 1 - burst_fraction))
        r = high if in_burst else low
        count = rng.poisson(r * length)
        if count:
            pts = np.sort(t + rng.random(count) * length)
            take = min(count, n - k)
            times[k:k + take] = pts[:take]
            k += take
        t += length
        in_burst = not in_burst
    return times


def arrivals(kind: str, rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "poisson":
        return poisson_arrivals(rate, n, rng)
    if kind == "bursty":
        return bursty_arrivals(rate, n, rng)
    raise ValueError(f"unknown arrival process: {kind}")


# ------------------------------------------------------- single-server queue
def fcfs_queue(arrival_times: np.ndarray, service_times: np.ndarray,
               n_servers: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """First-come-first-served queue. Returns (response times, waiting times)."""
    n = len(arrival_times)
    wait = np.empty(n)
    if n_servers == 1:
        free = 0.0
        for i in range(n):
            a = arrival_times[i]
            start = a if a > free else free
            wait[i] = start - a
            free = start + service_times[i]
    else:
        free_at = [0.0] * n_servers
        heapq.heapify(free_at)
        for i in range(n):
            a = arrival_times[i]
            f = heapq.heappop(free_at)
            start = a if a > f else f
            wait[i] = start - a
            heapq.heappush(free_at, start + service_times[i])
    return wait + service_times, wait


def mm1_theory(rho: float, mean_service: float = 1.0) -> Dict[str, float]:
    """Closed-form M/M/1 response time: exponential with rate (mu - lambda)."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    scale = mean_service / (1 - rho)
    return {"mean": scale, "p50": scale * math.log(2), "p99": scale * math.log(100),
            "p999": scale * math.log(1000)}


def time_average_in_system(arrival_times: np.ndarray, response: np.ndarray, n_probe: int,
                           rng: np.random.Generator) -> float:
    """Average number of jobs in the system, sampled at random instants (for Little's law)."""
    departures = np.sort(arrival_times + response)
    lo, hi = arrival_times[0], arrival_times[-1]
    probes = rng.uniform(lo, hi, n_probe)
    return float(np.mean(np.searchsorted(arrival_times, probes, side="right")
                         - np.searchsorted(departures, probes, side="right")))


def utilization_sweep(rhos: Sequence[float], n_jobs: int, seed: int,
                      arrival_kinds: Sequence[str] = ("poisson", "bursty"),
                      service_kinds: Sequence[str] = ("exponential", "lognormal", "bimodal"),
                      mean_service: float = 1.0, warmup_fraction: float = 0.1) -> List[Dict[str, object]]:
    rows = []
    for ai, a_kind in enumerate(arrival_kinds):
        for si, s_kind in enumerate(service_kinds):
            for j, rho in enumerate(rhos):
                rng = np.random.default_rng([seed, ai, si, j])
                svc = service_sampler(s_kind, mean_service, rng, sigma=1.0, p_slow=0.01, slow_factor=20)
                arr = arrivals(a_kind, rho / mean_service, n_jobs, rng)
                resp, wait = fcfs_queue(arr, svc(n_jobs))
                keep = slice(int(n_jobs * warmup_fraction), None)
                r = resp[keep]
                lam = (len(r) - 1) / (arr[keep][-1] - arr[keep][0])
                row = {
                    "arrivals": a_kind, "service": s_kind, "utilization": rho,
                    "mean_response": float(r.mean()), "p50_response": float(np.percentile(r, 50)),
                    "p99_response": float(np.percentile(r, 99)),
                    "mean_wait": float(wait[keep].mean()),
                    "littles_law_L_from_lambda_W": float(lam * r.mean()),
                    "littles_law_L_sampled": time_average_in_system(arr[keep], r, 20_000, rng),
                }
                if a_kind == "poisson" and s_kind == "exponential":
                    th = mm1_theory(rho, mean_service)
                    row["mm1_mean_theory"] = th["mean"]
                    row["mm1_p99_theory"] = th["p99"]
                rows.append(row)
    return rows


# --------------------------------------------------- coordinated omission
def _finish_with_stall(start: float, service: float, stall: Tuple[float, float]) -> float:
    """Finish time when the server freezes completely during ``stall = (begin, end)``."""
    s0, s1 = stall
    if start >= s1 or start + service <= s0:
        return start + service
    if start >= s0:
        return s1 + service
    return start + service + (s1 - s0)


def stall_experiment(rate: float = 100.0, mean_service: float = 2.0, duration: float = 60_000.0,
                     stall_start: float = 30_000.0, stall_length: float = 1_000.0,
                     n_clients: int = 2, seed: int = 0) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """Closed-loop versus open-loop measurement of one server through a stall.

    ``rate`` is requests per second and times are milliseconds. The closed-loop
    generator has ``n_clients`` clients, each thinking long enough between
    requests that the combined rate matches ``rate`` when the server is healthy.
    """
    rng = np.random.default_rng(seed)
    interval = 1000.0 / rate
    stall = (stall_start, stall_start + stall_length)
    svc = service_sampler("exponential", mean_service, rng)

    # Open loop: a fixed schedule, latency from the intended send time.
    intended = np.arange(0.0, duration, interval)
    s_open = svc(len(intended))
    free, open_lat = 0.0, np.empty(len(intended))
    for i, t in enumerate(intended):
        start = max(t, free)
        free = _finish_with_stall(start, s_open[i], stall)
        open_lat[i] = free - t

    # Closed loop: clients wait for each response, then think.
    per_client_interval = interval * n_clients
    think = max(per_client_interval - mean_service, 0.0)
    heap = [(c * interval, c) for c in range(n_clients)]
    heapq.heapify(heap)
    free, closed_lat = 0.0, []
    s_closed = svc(int(duration / interval) * 2 + 100)
    k = 0
    while heap:
        t, c = heapq.heappop(heap)
        if t >= duration:
            continue
        start = max(t, free)
        free = _finish_with_stall(start, s_closed[k], stall)
        k += 1
        closed_lat.append(free - t)
        heapq.heappush(heap, (free + think, c))
    closed_lat = np.array(closed_lat)

    corrected = LatencyHistogram()
    for v in closed_lat:
        corrected.record_corrected(float(v), per_client_interval)

    rows = []
    for name, hist in [("closed_loop_recorded", _hist(closed_lat)),
                       ("closed_loop_corrected", corrected),
                       ("open_loop", _hist(open_lat))]:
        rows.append({"generator": name, "samples": hist.total, "p50_ms": hist.percentile(50),
                     "p99_ms": hist.percentile(99), "p99.9_ms": hist.percentile(99.9),
                     "max_ms": hist.max_ms,
                     "samples_over_100ms": int(np.sum(hist.counts[hist._index(np.array([100.0]))[0]:]))})
    return rows, {"open": open_lat, "closed": closed_lat}


def _hist(values: np.ndarray) -> LatencyHistogram:
    h = LatencyHistogram()
    h.record_many(values)
    return h


# ------------------------------------------------------------------ fan-out
def fanout_experiment(widths: Sequence[int], slow_probs: Sequence[float], n_requests: int,
                      seed: int, slow_factor: float = 10.0, sigma: float = 0.25,
                      chunk: int = 2_000_000) -> List[Dict[str, object]]:
    """Requests that wait for the slowest of ``width`` independent leaves.

    Each leaf takes a lognormal time with median 1 and, with probability ``q``,
    is ``slow_factor`` times slower. A request is "delayed" when at least one
    of its leaves is slow; the closed form is ``1 - (1 - q) ** width``.
    """
    rows = []
    for qi, q in enumerate(slow_probs):
        for wi, width in enumerate(widths):
            rng = np.random.default_rng(seed * 7919 + qi * 101 + wi)
            per = max(1, chunk // width)
            delayed, lat = 0, []
            done = 0
            while done < n_requests:
                m = min(per, n_requests - done)
                leaf = rng.lognormal(0.0, sigma, (m, width))
                slow = rng.random((m, width)) < q
                leaf = np.where(slow, leaf * slow_factor, leaf)
                delayed += int(slow.any(axis=1).sum())
                lat.append(leaf.max(axis=1))
                done += m
            lat = np.concatenate(lat)
            rows.append({
                "leaf_slow_prob": q, "fanout": width,
                "share_delayed_sim": delayed / n_requests,
                "share_delayed_theory": 1 - (1 - q) ** width,
                "p50_request_latency": float(np.percentile(lat, 50)),
                "p99_request_latency": float(np.percentile(lat, 99)),
            })
    return rows


# ------------------------------------------------------------------ hedging
@dataclass
class _Job:
    request: int
    replica: int
    service: float
    state: int = 0  # 0 queued, 1 running, 2 done, 3 cancelled
    start: float = 0.0
    backup: bool = False


def simulate_replicas(arrival_times: np.ndarray, sampler: Sampler, n_replicas: int,
                      rng: np.random.Generator, hedge_delay: Optional[float] = None
                      ) -> Dict[str, object]:
    """FCFS replicas with optional hedged requests and cancellation.

    Each request goes to a random replica. If ``hedge_delay`` is set and the
    request has not finished by then, a backup goes to a different random
    replica. Whichever copy finishes first answers; the other is cancelled,
    removed from its queue if it has not started, or stopped if it has (the
    work it did until then still counts as load).

    Because a job's slowness belongs to the job here, cancelling a slow
    primary also saves the rest of its work, so total work can fall. The
    result reports backup work and net work separately for that reason.
    """
    n = len(arrival_times)
    primary_service = sampler(n)
    primary_replica = rng.integers(0, n_replicas, n)
    backup_offset = rng.integers(1, max(n_replicas, 2), n)
    backup_service = sampler(n)

    queues = [deque() for _ in range(n_replicas)]
    running: List[Optional[_Job]] = [None] * n_replicas
    jobs_of: List[List[_Job]] = [[] for _ in range(n)]
    done_at = np.full(n, np.nan)
    busy = 0.0
    backup_work = 0.0
    hedged = 0
    events: List[Tuple[float, int, int, object]] = []
    seq = 0

    def push(t: float, kind: int, payload: object) -> None:
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, payload))
        seq += 1

    def start_next(r: int, t: float) -> None:
        while queues[r]:
            job = queues[r].popleft()
            if job.state == 0:
                job.state, job.start = 1, t
                running[r] = job
                push(t + job.service, 2, job)
                return
        running[r] = None

    def submit(job: _Job, t: float) -> None:
        jobs_of[job.request].append(job)
        queues[job.replica].append(job)
        if running[job.replica] is None:
            start_next(job.replica, t)

    for i, a in enumerate(arrival_times):
        push(a, 0, i)

    ARRIVE, HEDGE, DONE = 0, 1, 2
    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == ARRIVE:
            i = payload
            submit(_Job(i, int(primary_replica[i]), float(primary_service[i])), t)
            if hedge_delay is not None and n_replicas > 1:
                push(t + hedge_delay, HEDGE, i)
        elif kind == HEDGE:
            i = payload
            if np.isnan(done_at[i]):
                hedged += 1
                r = int((primary_replica[i] + backup_offset[i]) % n_replicas)
                submit(_Job(i, r, float(backup_service[i]), backup=True), t)
        else:
            job = payload
            if job.state != 1:
                continue  # stale completion of a cancelled job
            job.state = 2
            busy += job.service
            if job.backup:
                backup_work += job.service
            i = job.request
            if np.isnan(done_at[i]):
                done_at[i] = t
            start_next(job.replica, t)
            for other in jobs_of[i]:
                if other is job:
                    continue
                if other.state == 0:
                    other.state = 3
                elif other.state == 1:
                    other.state = 3
                    busy += t - other.start
                    if other.backup:
                        backup_work += t - other.start
                    start_next(other.replica, t)

    return {"response": done_at - arrival_times, "busy_time": busy, "backup_work": backup_work,
            "hedged": hedged}


def hedging_experiment(n_requests: int = 200_000, n_replicas: int = 10,
                       utilizations: Sequence[float] = (0.3,), mean_service: float = 1.0,
                       p_slow: float = 0.01, slow_factor: float = 20.0,
                       hedge_percentiles: Sequence[float] = (95.0, 50.0), seed: int = 0
                       ) -> List[Dict[str, object]]:
    """Compare no hedging with hedging after percentile delays of the unhedged response.

    Repeated at each utilization (of the unhedged system), because hedging
    spends spare capacity: the same policy that removes the tail at low load
    can lengthen queues when replicas are busy.
    """
    rows = []
    for ui, utilization in enumerate(utilizations):
        def run(delay: Optional[float]) -> Dict[str, object]:
            rng = np.random.default_rng([seed, ui])
            sampler = service_sampler("bimodal", mean_service, rng, p_slow=p_slow, slow_factor=slow_factor)
            arr = poisson_arrivals(utilization * n_replicas / mean_service, n_requests, rng)
            return simulate_replicas(arr, sampler, n_replicas, rng, delay)

        base = run(None)
        base_resp = base["response"]
        rows.append(_hedge_row(utilization, "none", math.nan, base, base["busy_time"]))
        for q in hedge_percentiles:
            delay = float(np.percentile(base_resp, q))
            rows.append(_hedge_row(utilization, f"hedge_after_p{q:g}", delay, run(delay), base["busy_time"]))
    return rows


def _hedge_row(utilization: float, policy: str, delay: float, res: Dict[str, object],
               base_busy: float) -> Dict[str, object]:
    r = res["response"]
    return {"utilization": utilization, "policy": policy, "hedge_delay": delay,
            "p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95)),
            "p99": float(np.percentile(r, 99)), "p99.9": float(np.percentile(r, 99.9)),
            "backup_requests_per_request": res["hedged"] / len(r),
            "backup_work_share": res["backup_work"] / base_busy,
            "net_work_change": res["busy_time"] / base_busy - 1.0}
