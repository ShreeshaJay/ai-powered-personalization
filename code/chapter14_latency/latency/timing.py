"""
Spans, traces, critical-path extraction, and confidence intervals.

A trace is a tree of spans. Each span has a name, a stage label from the
chapter's attribution framework (S1 to S7, or R for the residual), a kind
("service" when the component is working, "wait" when it is queued or blocked
on a remote call), a start and end time from ``time.perf_counter_ns``, and a
parent.

The critical path of a span is the chain of child spans that determined when
it finished. Parallel branches that finished earlier than the slowest one are
off the critical path: making them faster cannot make the request faster.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

now_ns = time.perf_counter_ns

SERVICE = "service"
WAIT = "wait"
# Time the load generator itself sent a request late. It is measured so it is
# not hidden, and labelled separately because it is not the server's queueing.
GENERATOR_LAG = "generator_lag"
RESIDUAL_STAGE = "R"


@dataclass
class Span:
    name: str
    stage: str
    kind: str
    start_ns: int
    end_ns: int = 0
    span_id: int = 0
    parent_id: Optional[int] = None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass
class Trace:
    """All spans of one request. Thread-safe so parallel branches can add spans."""

    request_id: int
    spans: List[Span] = field(default_factory=list)
    _ids: Iterator[int] = field(default_factory=lambda: itertools.count(1), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, stage: str, kind: str, start_ns: int, end_ns: int,
            parent: Optional[Span] = None) -> Span:
        with self._lock:
            span = Span(name, stage, kind, start_ns, end_ns, next(self._ids),
                        parent.span_id if parent is not None else None)
            self.spans.append(span)
        return span

    def open(self, name: str, stage: str, kind: str = SERVICE, parent: Optional[Span] = None,
             start_ns: Optional[int] = None) -> Span:
        return self.add(name, stage, kind, now_ns() if start_ns is None else start_ns, 0, parent)

    @contextmanager
    def span(self, name: str, stage: str, kind: str = SERVICE,
             parent: Optional[Span] = None) -> Iterator[Span]:
        sp = self.open(name, stage, kind, parent)
        try:
            yield sp
        finally:
            sp.end_ns = now_ns()

    @property
    def root(self) -> Span:
        roots = [s for s in self.spans if s.parent_id is None]
        if len(roots) != 1:
            raise ValueError(f"trace {self.request_id} has {len(roots)} root spans")
        return roots[0]


def critical_path(spans: Sequence[Span], root: Optional[Span] = None) -> List[Tuple[Span, int]]:
    """Return ``(span, nanoseconds)`` segments that make up the critical path.

    The walk starts at the end of the root and repeatedly steps to the child
    that finished last before the current cursor, then continues from that
    child's start. Time in a span not covered by any critical child is
    attributed to the span itself (its "self time"). For the root span, that
    self time is the unattributed residual.
    """
    children: Dict[Optional[int], List[Span]] = {}
    for s in spans:
        children.setdefault(s.parent_id, []).append(s)
    if root is None:
        roots = children.get(None, [])
        if len(roots) != 1:
            raise ValueError("expected exactly one root span")
        root = roots[0]

    segments: List[Tuple[Span, int]] = []

    def walk(span: Span) -> None:
        kids = sorted(children.get(span.span_id, []), key=lambda c: c.end_ns, reverse=True)
        cursor = span.end_ns
        i = 0
        while i < len(kids):
            child = kids[i]
            if child.end_ns > cursor:
                i += 1
                continue
            if child.end_ns <= span.start_ns:
                break
            if cursor > child.end_ns:
                segments.append((span, cursor - child.end_ns))
            walk(child)
            cursor = max(child.start_ns, span.start_ns)
            i += 1
        if cursor > span.start_ns:
            segments.append((span, cursor - span.start_ns))

    walk(root)
    return segments


def attribute(spans: Sequence[Span], root: Optional[Span] = None) -> Dict[Tuple[str, str], int]:
    """Critical-path nanoseconds per ``(stage, kind)``; the root's self time is the residual."""
    root = root or next(s for s in spans if s.parent_id is None)
    totals: Dict[Tuple[str, str], int] = {}
    for span, ns in critical_path(spans, root):
        if span is root:
            key = (RESIDUAL_STAGE, "residual")
        else:
            key = (span.stage, span.kind)
        totals[key] = totals.get(key, 0) + ns
    return totals


def critical_span_names(spans: Sequence[Span], root: Optional[Span] = None) -> set:
    """Names of spans that contribute any time to the critical path."""
    return {span.name for span, ns in critical_path(spans, root) if ns > 0}


# ---------------------------------------------------------------- intervals
def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0, max_n: int = 50_000) -> Tuple[float, float, float]:
    """Percentile-bootstrap interval for ``stat(values)``.

    Very large samples are subsampled to ``max_n`` per resample, which keeps
    the cost bounded and makes the interval slightly conservative.
    """
    values = np.asarray(values, dtype=np.float64)
    point = float(stat(values))
    if values.size < 2:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    m = min(values.size, max_n)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat(values[rng.integers(0, values.size, m)])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def repeat_ci(per_run: Sequence[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and Student-t interval across independent repeated runs (Georges et al., 2007).

    Suitable for quantities that can be negative. For latencies use
    ``repeat_ci_log``, whose interval cannot fall below zero.
    """
    from scipy import stats as st

    x = np.asarray(per_run, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(x.mean())
    if x.size < 2:
        return mean, math.nan, math.nan
    half = float(st.t.ppf(1 - alpha / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return mean, mean - half, mean + half


def repeat_ci_log(per_run: Sequence[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Geometric mean and Student-t interval on the log scale across repeated runs.

    Latencies are positive and run-to-run noise on a shared machine is
    multiplicative (a busy neighbour slows everything by a factor), so the
    interval is computed for log latency and transformed back. It is
    asymmetric and never extends below zero. With two repeats it is still
    very wide, which honestly shows how little two runs say.
    """
    x = np.asarray(per_run, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    centre, lo, hi = repeat_ci(np.log(x), alpha)
    return math.exp(centre), (math.exp(lo) if math.isfinite(lo) else math.nan), \
        (math.exp(hi) if math.isfinite(hi) else math.nan)


def paired_ratio_ci(treatment: Sequence[float], reference: Sequence[float],
                    alpha: float = 0.05) -> Tuple[float, float, float]:
    """Ratio ``treatment / reference`` with a paired Student-t interval on the log scale.

    Repeat ``i`` of both configurations must share its inputs (the same
    arrival schedule or the same batches), so the pairing removes the
    variation the two runs have in common. A ratio interval that excludes 1
    is evidence of a real change.
    """
    t = np.asarray(treatment, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError("paired samples must have the same length")
    ok = np.isfinite(t) & np.isfinite(r) & (t > 0) & (r > 0)
    return repeat_ci_log(t[ok] / r[ok], alpha)


def percentile_ms(values_ns: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values_ns) / 1e6, q))
