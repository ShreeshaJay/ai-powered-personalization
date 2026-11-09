"""
Request and response contracts for every hop of the pipeline (Figure 13.3).

Client -> Orchestrator      POST /v1/recommend   ClientRequest  -> ClientResponse
Orchestrator -> Retrieval   POST /v1/retrieve    RetrievalRequest -> RetrievalResponse
Orchestrator -> Rankers     POST /v1/rank        RankingRequest -> RankingResponse
Orchestrator -> Re-ranker   POST /v1/rerank      RerankRequest  -> RerankResponse

Every response carries `trace_id` and `model_version`. A request that fails
validation is rejected with HTTP 422 before any model code runs.

Evolution rules (checked by contracts/compat.py and tests/test_contracts.py):
adding an optional field is compatible; removing a field, changing its type,
or adding a required field needs a new path version (/v2/).

Readers ignore fields they do not know (`extra="ignore"`), so a newer sender
can add optional fields without breaking an older receiver. Non-finite floats
(NaN, inf) are rejected everywhere.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "v1"

# Multi-task heads of the Chapter 7 MMoE, in model order.
TASK_NAMES = ("engagement", "completion", "play_ratio", "like", "dislike")

# The Chapter 5 user tower reads at most this many recent items.
MAX_HISTORY_ITEMS = 20


class Contract(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False, protected_namespaces=())


class ServiceResponse(Contract):
    trace_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)


def _unique_item_ids(items: list) -> list:
    seen = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"duplicate item_id {item.item_id}")
        seen.add(item.item_id)
    return items


# ---------------------------------------------------------------------------
# Client <-> Orchestrator
# ---------------------------------------------------------------------------

class RequestContext(Contract):
    timestamp: Optional[int] = Field(
        None, ge=0,
        description="Event time in seconds on Yambda's clock. Defaults to the end of the training window.",
    )
    is_organic: int = Field(
        0, ge=0, le=1,
        description="Yambda's is_organic flag; 0 means the track was surfaced by a recommender.",
    )


POLICY_VERSION_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class ClientRequest(Contract):
    user_id: int = Field(ge=0)
    context: RequestContext = Field(default_factory=RequestContext)
    num_results: int = Field(20, ge=1, le=100)
    policy_version: Optional[str] = Field(
        None, pattern=POLICY_VERSION_PATTERN,
        description="Pin a value-function configuration, for example an experiment arm; passed to the re-ranker.",
    )


class RecommendedItem(Contract):
    rank: int = Field(ge=1)
    item_id: int = Field(ge=0)
    score: float
    artist_id: Optional[int] = None
    album_id: Optional[int] = None
    track_length_seconds: Optional[int] = None


class Fallback(Contract):
    """One stage skipped or substituted, with the rung of Figure 13.6 that its failure leads to."""
    stage: Literal["heavy_ranker", "reranker", "lightweight_ranker", "retrieval", "pipeline"]
    rung: int = Field(ge=2, le=6)
    label: str = Field(description="The rung's name in Figure 13.6, for example 'Skip heavy ranker'.")
    reason: Literal["timeout", "error", "circuit_open", "deadline_budget", "contract_violation",
                    "features_unavailable"]
    action: str
    detail: Optional[str] = None


class FeatureDefaults(Contract):
    """
    Feature values replaced by the feature contract's defaults (Section 13.7.2).
    Recorded as a flag on the response, not as a rung of the ladder.
    """
    lookup_failed: List[Literal["user_features", "item_features"]] = Field(default_factory=list)
    missing: Dict[str, int] = Field(default_factory=dict, description="Feature -> values absent from the store.")
    stale: Dict[str, int] = Field(default_factory=dict,
                                  description="Feature -> values older than the contract's maximum age.")


class Degradation(Contract):
    """The lowest rung reached (Figure 13.6) and every fallback taken, in the figure's order."""
    rung: int = Field(ge=1, le=6)
    label: str
    fallbacks: List[Fallback] = Field(default_factory=list)
    feature_defaults: FeatureDefaults = Field(default_factory=FeatureDefaults)


class ClientResponse(ServiceResponse):
    user_id: int
    items: List[RecommendedItem]
    score_source: Literal["value_function", "heavy_ranker", "lightweight_ranker", "retrieval", "popularity"]
    degradation: Degradation
    stage_versions: Dict[str, str] = Field(default_factory=dict)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    deadline_ms: float


# ---------------------------------------------------------------------------
# Orchestrator <-> Retrieval
# ---------------------------------------------------------------------------

class RetrievalRequest(Contract):
    user_id: int = Field(ge=0)
    recent_item_ids: Optional[List[int]] = Field(
        None, max_length=MAX_HISTORY_ITEMS,
        description=(
            "Chronological, most recent last, as Chapter 5's user tower expects. "
            "When omitted, the service reads the user's history from the feature store."
        ),
    )
    exclude_item_ids: List[int] = Field(default_factory=list, max_length=1000)
    num_candidates: int = Field(300, ge=1, le=1000)


class Candidate(Contract):
    item_id: int = Field(ge=0)
    score: float


class RetrievalResponse(ServiceResponse):
    user_id: int
    candidates: List[Candidate]
    source: Literal["two_tower", "popularity_cold_start"]
    index_version: str
    history_length: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Orchestrator <-> Lightweight ranker and heavy ranker
# ---------------------------------------------------------------------------

class RankingCandidate(Contract):
    item_id: int = Field(ge=0)
    features: Dict[str, float] = Field(description="Item and user-item features from the feature store.")
    retrieval_score: Optional[float] = None


class RankingRequest(Contract):
    user_id: int = Field(ge=0)
    context: RequestContext = Field(default_factory=RequestContext)
    user_features: Dict[str, float]
    candidates: List[RankingCandidate] = Field(min_length=1, max_length=1000)
    num_results: Optional[int] = Field(None, ge=1, le=1000, description="Return the top N; all when omitted.")
    feature_set_version: Optional[str] = None

    @field_validator("candidates")
    @classmethod
    def _candidates_unique(cls, candidates):
        return _unique_item_ids(candidates)


class TaskScores(Contract):
    engagement: float = Field(ge=0, le=1)
    completion: float = Field(ge=0, le=1)
    play_ratio: float = Field(ge=0, le=1, description="Expected fraction of the track played, a regression output.")
    like: float = Field(ge=0, le=1)
    dislike: float = Field(ge=0, le=1)


class RankedItem(Contract):
    item_id: int = Field(ge=0)
    score: float
    task_scores: Optional[TaskScores] = None


class RankingResponse(ServiceResponse):
    model_name: Literal["xgboost", "mmoe"]
    score_name: str = Field(description="What `score` estimates, for example p_completion.")
    ranked_items: List[RankedItem]

    @model_validator(mode="after")
    def _mmoe_items_carry_task_scores(self):
        # Without them the re-ranker would silently fall back to the raw score.
        if self.model_name == "mmoe":
            missing = [item.item_id for item in self.ranked_items if item.task_scores is None]
            if missing:
                raise ValueError(f"mmoe response without task_scores for {len(missing)} item(s), "
                                 f"for example item {missing[0]}")
        return self


# ---------------------------------------------------------------------------
# Orchestrator <-> Value function and re-ranker
# ---------------------------------------------------------------------------

class RerankCandidate(Contract):
    item_id: int = Field(ge=0)
    score: float = Field(description="Model score from the last ranking stage that ran.")
    task_scores: Optional[TaskScores] = None
    artist_id: Optional[int] = None
    album_id: Optional[int] = None


class RerankRequest(Contract):
    user_id: int = Field(ge=0)
    candidates: List[RerankCandidate] = Field(min_length=1, max_length=1000)
    num_results: int = Field(20, ge=1, le=100)
    policy_version: Optional[str] = Field(
        None, pattern=POLICY_VERSION_PATTERN,
        description="Serve this value-function configuration instead of the active one (Section 13.2.1).",
    )

    @field_validator("candidates")
    @classmethod
    def _candidates_unique(cls, candidates):
        return _unique_item_ids(candidates)


class RerankedItem(Contract):
    item_id: int = Field(ge=0)
    score: float = Field(description="Value-function score, or the model score when task scores were absent.")


class RerankResponse(ServiceResponse):
    items: List[RerankedItem]
    config_version: str = Field(description="The configuration that produced this response.")
    policy_pin: Literal["none", "honoured", "unavailable"] = Field(
        "none",
        description="none: no policy_version requested; honoured: config_version is the pinned version; "
                    "unavailable: the pinned version is missing or invalid, so the active one served.",
    )
    value_function_applied: bool
    diversity_applied: bool
    pacing_applied: bool


# ---------------------------------------------------------------------------
# Shared error body (HTTP 422 and 504)
# ---------------------------------------------------------------------------

class ErrorResponse(Contract):
    error: Literal["contract_violation", "deadline_exceeded", "not_ready", "internal_error"]
    detail: Any
    trace_id: str
    model_version: str


ALL_CONTRACTS = {
    cls.__name__: cls for cls in (
        ClientRequest, ClientResponse, RetrievalRequest, RetrievalResponse,
        RankingRequest, RankingResponse, RerankRequest, RerankResponse, ErrorResponse,
    )
}
