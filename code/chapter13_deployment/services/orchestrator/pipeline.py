"""
One recommendation request under a deadline (Figures 13.2 and 13.6).

    user features (feature store)  ||  retrieval            in parallel
    item + user-item features      one batched lookup after retrieval
    lightweight ranker             300 -> 100
    heavy ranker                   100 scored with five task heads
    value function and re-ranker   100 -> num_results
    metadata hydration             from the item lookup above

Every remote call gets a timeout derived from the time left on the request
deadline, passes that timeout downstream in X-Request-Timeout-Ms, is
guarded by a circuit breaker for that service, and is retried at most once,
for errors where the call almost certainly did not run, while the
process-wide retry budget has a token.

When a stage fails, times out, has an open circuit, returns an invalid
response, or would not have enough time left, the request takes that
stage's rung of the degradation ladder in Figure 13.6:

    1 Full pipeline            all five services answer
    2 Skip heavy ranker        re-rank the lightweight scores
    3 Skip re-ranker           sort by best score, minimal hard rules
    4 Skip lightweight ranker  top candidates by similarity
    5 Popularity candidates    in-memory list, ranked if possible
    6 Cached list              precomputed, returned without ranking

Failures combine: the response reports the lowest rung reached and lists
every stage skipped, in the figure's order. Rung 6 serves the cached list
when every stage has failed or the deadline runs out. Feature values replaced
by defaults (lookup failed, missing or stale) are a flag on the response,
not a rung; a failed item lookup also skips both rankers, which have no
features to score.

Feature lookups against an in-process store (SQLite, Parquet) run inline on
the event loop. A worker thread would have to win the GIL back from the busy
event loop after every row SQLite returns, which under load turned a 5 ms
lookup into hundreds of milliseconds. Lookups against a networked store
(Redis) run in a worker thread under a timeout taken from the deadline; a
lookup that times out counts as failed.
"""

import asyncio
import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from common import published
from common.feature_store import FeatureSubstitutions, UserRecord
from common.resilience import BudgetPlanner, CircuitBreaker, Deadline, RetryBudget, RetryPolicy
from common.service import DEADLINE_HEADER, MODEL_VERSION_HEADER, TRACE_HEADER
from contracts import (ClientRequest, ClientResponse, Degradation, Fallback, FeatureDefaults, RankingCandidate,
                       RankingRequest, RankingResponse, RecommendedItem, RerankCandidate, RerankRequest,
                       RerankResponse, RetrievalRequest, RetrievalResponse)
from services.orchestrator.settings import STAGES, OrchestratorSettings

logger = logging.getLogger("ch13.orchestrator")
_rules = published.chapter8("business_rules")

# Figure 13.6, top to bottom: rung, label, what the rung serves, and the stage
# whose failure leads to it.
LADDER = (
    (1, "Full pipeline", "all five services answer", None),
    (2, "Skip heavy ranker", "re-rank the lightweight scores", "heavy_ranker"),
    (3, "Skip re-ranker", "sort by best score, minimal hard rules", "reranker"),
    (4, "Skip lightweight ranker", "top candidates by similarity", "lightweight_ranker"),
    (5, "Popularity candidates", "in-memory list, ranked if possible", "retrieval"),
    (6, "Cached list", "precomputed, returned without ranking", "pipeline"),
)
RUNG_LABELS = {rung: label for rung, label, _, _ in LADDER}
RUNG_DESCRIPTIONS = {rung: description for rung, _, description, _ in LADDER}
RUNG_OF_STAGE = {stage: rung for rung, _, _, stage in LADDER if stage is not None}
REMOTE_STAGES = frozenset(STAGES)


class StageError(Exception):
    def __init__(self, reason: str, detail: Optional[str] = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Run:
    trace_id: str
    deadline: Deadline
    fallbacks: List[Fallback] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)
    retries: Dict[str, int] = field(default_factory=dict)
    lookup_failed: List[str] = field(default_factory=list)
    substitutions: FeatureSubstitutions = field(default_factory=FeatureSubstitutions)

    def fallback(self, stage: str, reason: str, action: str, detail: Optional[str] = None) -> None:
        rung = RUNG_OF_STAGE[stage]
        self.fallbacks.append(Fallback(stage=stage, rung=rung, label=RUNG_LABELS[rung], reason=reason,
                                       action=action, detail=detail[:300] if detail else None))

    def failed_stages(self) -> set:
        return {f.stage for f in self.fallbacks}

    def degradation(self) -> Degradation:
        ordered = sorted(self.fallbacks, key=lambda f: f.rung)
        rung = max([f.rung for f in ordered], default=1)
        features = FeatureDefaults(lookup_failed=self.lookup_failed, missing=dict(self.substitutions.missing),
                                   stale=dict(self.substitutions.stale))
        return Degradation(rung=rung, label=RUNG_LABELS[rung], fallbacks=ordered, feature_defaults=features)


class RecommendationPipeline:
    def __init__(self, settings: OrchestratorSettings, store, popularity: List[int],
                 clients: Dict[str, httpx.AsyncClient], bundle_version: str, clock=time.monotonic,
                 rng: Optional[random.Random] = None):
        self.settings = settings
        self.store = store
        self.popularity = [int(i) for i in popularity]
        self.clients = clients
        self.bundle_version = bundle_version
        self.clock = clock
        self.rng = rng or random.Random()
        self.planner = BudgetPlanner([settings.budgets[s] for s in STAGES], settings.response_reserve_ms)
        self.breakers = {s: CircuitBreaker(s, failure_rate=settings.breaker_failure_rate,
                                           min_calls=settings.breaker_min_calls, window_s=settings.breaker_window_s,
                                           reset_timeout_s=settings.breaker_reset_s, clock=clock)
                         for s in STAGES}
        self.retry = RetryPolicy(max_retries=settings.retry_max, backoff_ms=settings.retry_backoff_ms)
        self.retry_budget = RetryBudget(settings.retry_budget_per_s, settings.retry_budget_burst, clock)
        self.rejections: Counter = Counter()
        self.pacing = _rules.PacingConfig()
        self.cached_items = self._cached_slate()

    # -- remote calls -------------------------------------------------------
    async def call(self, run: Run, stage: str, path: str, payload, response_model):
        budget = self.planner.budget_for(stage, run.deadline.remaining_ms())
        if budget is None:
            raise StageError("deadline_budget", f"{run.deadline.remaining_ms():.0f} ms left on the request")
        body = payload.model_dump(mode="json", exclude_none=True)
        client = self.clients[stage]
        breaker = self.breakers[stage]
        if not breaker.allow_request():
            raise StageError("circuit_open")
        attempt, started = 0, time.perf_counter()
        try:
            while True:
                headers = {TRACE_HEADER: run.trace_id, DEADLINE_HEADER: f"{budget:.0f}"}
                try:
                    response = await asyncio.wait_for(
                        client.post(path, json=body, headers=headers, timeout=budget / 1000.0), budget / 1000.0)
                except (asyncio.TimeoutError, httpx.TimeoutException):
                    breaker.record_failure()
                    raise StageError("timeout", f"no response within {budget:.0f} ms")
                except httpx.TransportError as exc:
                    kind, detail = "connect_error", type(exc).__name__
                else:
                    status = response.status_code
                    if status == 200:
                        try:
                            parsed = response_model.model_validate(response.json())
                        except (ValidationError, ValueError) as exc:
                            breaker.record_failure()
                            raise StageError("contract_violation", f"invalid {response_model.__name__}: {exc}")
                        breaker.record_success()
                        return parsed
                    if status == 422:
                        # The callee is healthy; the request broke its contract.
                        breaker.record_success()
                        self.rejections[stage] += 1
                        logger.warning("contract rejection trace_id=%s stage=%s orchestrator=%s callee=%s detail=%s",
                                       run.trace_id, stage, self.bundle_version,
                                       response.headers.get(MODEL_VERSION_HEADER, "unknown"), response.text[:1000])
                        raise StageError("contract_violation", response.text)
                    if status not in (502, 503):
                        breaker.record_failure()
                        raise StageError("timeout" if status == 504 else "error", f"HTTP {status}")
                    kind, detail = f"http_{status}", f"HTTP {status}"

                backoff = self.retry.backoff_s(attempt, self.rng)
                stage_time_left = run.deadline.remaining_ms() - self.planner.reserve_after(stage)
                if not self.retry.should_retry(attempt, kind, stage_time_left, self.planner.min_ms(stage), backoff):
                    breaker.record_failure()
                    raise StageError("error", detail)
                if not self.retry_budget.try_acquire():
                    breaker.record_failure()
                    raise StageError("error", f"{detail}; retry skipped: process retry budget exhausted")
                attempt += 1
                run.retries[stage] = attempt
                await asyncio.sleep(backoff)
                budget = self.planner.budget_for(stage, run.deadline.remaining_ms())
                if budget is None:
                    breaker.record_failure()
                    raise StageError("deadline_budget", "no time left for a retry")
        except asyncio.CancelledError:
            breaker.abandon()
            raise
        except StageError:
            raise
        except Exception as exc:
            # Every allowed call must settle with the breaker: a half-open trial
            # call that never reports back would keep the circuit open for good.
            breaker.record_failure()
            raise StageError("error", type(exc).__name__) from exc
        finally:
            run.timings[stage] = round((time.perf_counter() - started) * 1000.0, 2)

    # -- feature lookups -----------------------------------------------------
    async def _lookup(self, run: Run, lookup, *args):
        """Run a feature lookup inline (in-process store) or in a worker thread under a deadline timeout."""
        if self.store.in_process:
            return lookup(*args)
        timeout_ms = min(self.settings.feature_lookup_cap_ms,
                         run.deadline.remaining_ms() - self.planner.reserve_after("retrieval"))
        if timeout_ms <= 0:
            raise StageError("deadline_budget", f"{run.deadline.remaining_ms():.0f} ms left on the request")
        try:
            return await asyncio.wait_for(run_in_threadpool(lookup, *args), timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            raise StageError("timeout", f"no answer within {timeout_ms:.0f} ms") from None

    @staticmethod
    def _lookup_error(exc: Exception) -> str:
        return str(exc) if isinstance(exc, StageError) else type(exc).__name__

    # -- stages --------------------------------------------------------------
    async def _user_features(self, user_id: int, as_of: int, run: Run) -> UserRecord:
        started = time.perf_counter()
        seen = FeatureSubstitutions()           # a timed-out lookup's thread must not write into the run
        try:
            user = await self._lookup(run, self.store.get_user, user_id, as_of, seen)
            run.substitutions.add(seen)
            return user
        except Exception as exc:
            run.lookup_failed.append("user_features")
            logger.warning("trace_id=%s user feature lookup failed (%s); using contract defaults", run.trace_id,
                           self._lookup_error(exc))
            return UserRecord(user_id, False, dict(self.store.defaults["user"]), [])
        finally:
            run.timings["user_features"] = round((time.perf_counter() - started) * 1000.0, 2)

    async def _retrieve(self, user_id: int, run: Run) -> Optional[List[Tuple[int, float]]]:
        try:
            response = await self.call(run, "retrieval", "/v1/retrieve",
                                       RetrievalRequest(user_id=user_id,
                                                        num_candidates=self.settings.retrieval_candidates),
                                       RetrievalResponse)
        except StageError as exc:
            run.fallback("retrieval", exc.reason, "used the popularity list", exc.detail)
            return None
        run.versions["retrieval"] = response.model_version
        run.versions["index"] = response.index_version
        run.versions["retrieval_source"] = response.source
        return [(c.item_id, c.score) for c in response.candidates]

    def _item_lookup(self, user: UserRecord, item_ids: List[int], as_of: int, report: FeatureSubstitutions):
        return (self.store.get_items(item_ids, as_of, report),
                self.store.get_user_items(user.user_id, item_ids, as_of, user.updated_at, report))

    def _popular(self, exclude, n: int) -> List[Tuple[int, float]]:
        items = [i for i in self.popularity if i not in exclude][:n]
        return [(i, float(len(items) - rank) / len(items)) for rank, i in enumerate(items)]

    def _cached_slate(self) -> List[RecommendedItem]:
        ids = self.popularity[:100]
        try:
            records = self.store.get_items(ids)
        except Exception:
            records = {}
        slate = []
        for rank, item_id in enumerate(ids):
            record = records.get(item_id)
            slate.append(RecommendedItem(
                rank=rank + 1, item_id=item_id, score=float(len(ids) - rank) / len(ids),
                artist_id=record.artist_id if record else None, album_id=record.album_id if record else None,
                track_length_seconds=record.track_length_seconds if record else None))
        return slate

    # -- request -------------------------------------------------------------
    async def recommend(self, request: ClientRequest, trace_id: str, deadline: Deadline) -> ClientResponse:
        run = Run(trace_id, deadline)
        started = time.perf_counter()
        guard_s = max(0.0, deadline.remaining_ms() - self.settings.response_reserve_ms) / 1000.0
        try:
            response = await asyncio.wait_for(self._run(request, run), timeout=guard_s)
        except asyncio.TimeoutError:
            run.fallback("pipeline", "timeout", "deadline reached; served the cached list")
            response = self._cached_response(request, run)
        except Exception as exc:  # never fail the client: degrade to the last rung
            logger.exception("Pipeline error for trace %s", trace_id)
            run.fallback("pipeline", "error", "served the cached list", type(exc).__name__)
            response = self._cached_response(request, run)
        response.timings_ms["total"] = round((time.perf_counter() - started) * 1000.0, 2)
        self._log(request, response)
        return response

    @staticmethod
    def _log(request: ClientRequest, response: ClientResponse) -> None:
        """One structured record per request (Section 13.8.1); WARNING when any stage was skipped."""
        degradation = response.degradation
        record = {
            "trace_id": response.trace_id, "user_id": request.user_id,
            "rung": degradation.rung, "label": degradation.label,
            "skipped": [f"{f.rung} {f.label}: {f.stage} {f.reason}" for f in degradation.fallbacks],
            "feature_defaults": degradation.feature_defaults.model_dump(exclude_defaults=True),
            "score_source": response.score_source, "versions": response.stage_versions,
            "timings_ms": response.timings_ms,
        }
        level = logging.WARNING if degradation.fallbacks else logging.INFO
        logger.log(level, "request %s", json.dumps(record, separators=(",", ":")))

    def _hard_rules(self, scored: List[Tuple[int, float]], candidates: List[RerankCandidate], exclude: set,
                    n: int) -> List[Tuple[int, float]]:
        """Rung 3: the minimal copy of the hard rules, the exclusions and Chapter 8's artist and album pacing."""
        score_of = dict(scored)
        ordered = [i for i, _ in sorted(scored, key=lambda pair: -pair[1]) if i not in exclude]
        pacer = _rules.ArtistPacer({c.item_id: c.artist_id for c in candidates if c.artist_id is not None},
                                   {c.item_id: c.album_id for c in candidates if c.album_id is not None},
                                   self.pacing)
        return [(i, score_of[i]) for i in pacer.apply_pacing(ordered)[:n]]

    def _cached_response(self, request: ClientRequest, run: Run) -> ClientResponse:
        return ClientResponse(
            trace_id=run.trace_id, model_version=self.bundle_version, user_id=request.user_id,
            items=self.cached_items[:request.num_results], score_source="popularity",
            degradation=run.degradation(), stage_versions=dict(run.versions), timings_ms=dict(run.timings),
            deadline_ms=run.deadline.budget_ms)

    async def _run(self, request: ClientRequest, run: Run) -> ClientResponse:
        user_id, n = request.user_id, request.num_results
        context = request.context
        if context.timestamp is None:
            context = context.model_copy(update={"timestamp": self.store.reference_timestamp})
        run.versions["feature_set"] = self.store.feature_set_version
        as_of = context.timestamp

        retrieved, user = await asyncio.gather(self._retrieve(user_id, run),
                                               self._user_features(user_id, as_of, run))
        if retrieved is None:
            candidates, score_source = self._popular(set(user.recent_item_ids), self.settings.retrieval_candidates), \
                "popularity"
        else:
            candidates, score_source = retrieved, "retrieval"
        if not candidates:
            run.fallback("pipeline", "error", "served the cached list", "no candidates")
            return self._cached_response(request, run)

        started = time.perf_counter()
        seen = FeatureSubstitutions()
        try:
            items, pairs = await self._lookup(run, self._item_lookup, user, [i for i, _ in candidates], as_of, seen)
            run.substitutions.add(seen)
            # Both rankers need the track length, and an item never played has
            # none and no honest default, so it cannot be ranked.
            candidates = [(i, s) for i, s in candidates
                          if items[i].found and items[i].track_length_seconds is not None]
        except Exception as exc:
            run.lookup_failed.append("item_features")
            items, pairs, lookup_error = None, None, self._lookup_error(exc)
        run.timings["item_features"] = round((time.perf_counter() - started) * 1000.0, 2)
        if not candidates:
            run.fallback("pipeline", "error", "served the cached list", "no candidate has a track length")
            return self._cached_response(request, run)

        keep = self.settings.lightweight_keep
        scored = sorted(candidates, key=lambda pair: -pair[1])[:keep]
        task_scores = {}
        if items is None:
            run.fallback("lightweight_ranker", "features_unavailable",
                         f"kept the top {keep} candidates by {score_source} score", f"item lookup: {lookup_error}")
            run.fallback("heavy_ranker", "features_unavailable", f"used {score_source} scores",
                         f"item lookup: {lookup_error}")
        elif candidates:
            by_id = {i: RankingCandidate(item_id=i, features={**items[i].features, **pairs[i]}, retrieval_score=s)
                     for i, s in candidates}
            common_fields = dict(user_id=user_id, context=context, user_features=user.features,
                                 feature_set_version=self.store.feature_set_version)
            try:
                light = await self.call(run, "lightweight_ranker", "/v1/rank",
                                        RankingRequest(candidates=list(by_id.values()), num_results=keep,
                                                       **common_fields), RankingResponse)
                scored = [(r.item_id, r.score) for r in light.ranked_items]
                score_source = "lightweight_ranker"
                run.versions["lightweight_ranker"] = light.model_version
            except StageError as exc:
                run.fallback("lightweight_ranker", exc.reason,
                             f"kept the top {keep} candidates by {score_source} score", exc.detail)
            try:
                heavy = await self.call(run, "heavy_ranker", "/v1/rank",
                                        RankingRequest(candidates=[by_id[i] for i, _ in scored], **common_fields),
                                        RankingResponse)
                scored = [(r.item_id, r.score) for r in heavy.ranked_items]
                task_scores = {r.item_id: r.task_scores for r in heavy.ranked_items}
                score_source = "heavy_ranker"
                run.versions["heavy_ranker"] = heavy.model_version
            except StageError as exc:
                run.fallback("heavy_ranker", exc.reason, f"used {score_source} scores", exc.detail)

        def attribute(item_id: int, name: str):
            return getattr(items[item_id], name) if items is not None else None

        rerank_candidates = [RerankCandidate(item_id=i, score=s, task_scores=task_scores.get(i),
                                             artist_id=attribute(i, "artist_id"), album_id=attribute(i, "album_id"))
                             for i, s in scored]
        try:
            reranked = await self.call(run, "reranker", "/v1/rerank",
                                       RerankRequest(user_id=user_id, candidates=rerank_candidates, num_results=n,
                                                     policy_version=request.policy_version),
                                       RerankResponse)
            final = [(r.item_id, r.score) for r in reranked.items]
            if reranked.value_function_applied:
                score_source = "value_function"
            run.versions["reranker_config"] = reranked.config_version
            if reranked.policy_pin != "none":
                run.versions["policy_pin"] = reranked.policy_pin
        except StageError as exc:
            if REMOTE_STAGES <= run.failed_stages() | {"reranker"}:
                run.fallback("reranker", exc.reason, "every stage failed", exc.detail)
                run.fallback("pipeline", "error", "served the cached list", "every stage failed")
                return self._cached_response(request, run)
            run.fallback("reranker", exc.reason,
                         "sorted by model score, removed the user's recent items and applied artist pacing",
                         exc.detail)
            final = self._hard_rules(scored, rerank_candidates, set(user.recent_item_ids), n)

        started = time.perf_counter()
        results = [RecommendedItem(rank=rank + 1, item_id=i, score=s, artist_id=attribute(i, "artist_id"),
                                   album_id=attribute(i, "album_id"),
                                   track_length_seconds=attribute(i, "track_length_seconds"))
                   for rank, (i, s) in enumerate(final)]
        run.timings["hydrate"] = round((time.perf_counter() - started) * 1000.0, 2)
        return ClientResponse(
            trace_id=run.trace_id, model_version=self.bundle_version, user_id=user_id, items=results,
            score_source=score_source, degradation=run.degradation(), stage_versions=dict(run.versions),
            timings_ms=dict(run.timings), deadline_ms=run.deadline.budget_ms)

    def status(self) -> dict:
        return {stage: breaker.snapshot() for stage, breaker in self.breakers.items()}
