"""Contract rules: validation, version fields in every response, and the compatibility check."""

import math

import pytest
from pydantic import ValidationError

from contracts import (ALL_CONTRACTS, ClientRequest, RankingRequest, RankingResponse, RerankRequest, RetrievalRequest,
                       ServiceResponse, TaskScores)
from contracts import compat


def test_every_response_contract_carries_trace_id_and_model_version():
    responses = [model for name, model in ALL_CONTRACTS.items() if name.endswith("Response")]
    assert len(responses) >= 5
    for model in responses:
        fields = model.model_fields
        assert "trace_id" in fields and fields["trace_id"].is_required(), model.__name__
        assert "model_version" in fields and fields["model_version"].is_required(), model.__name__


def test_service_responses_share_the_base():
    for name in ("ClientResponse", "RetrievalResponse", "RankingResponse", "RerankResponse"):
        assert issubclass(ALL_CONTRACTS[name], ServiceResponse)


@pytest.mark.parametrize("payload", [
    {"user_id": -1},
    {"user_id": 1, "num_results": 0},
    {"user_id": 1, "num_results": 101},
    {"user_id": 1, "context": {"is_organic": 2}},
    {"user_id": "abc"},
    {},
])
def test_client_request_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        ClientRequest.model_validate(payload)


def test_client_request_ignores_unknown_fields_and_applies_defaults():
    request = ClientRequest.model_validate({"user_id": 7, "device": "phone"})
    assert request.num_results == 20
    assert request.context.is_organic == 0 and request.context.timestamp is None


def _ranking(candidates, **extra):
    return {"user_id": 1, "context": {"timestamp": 10}, "user_features": {"a": 1.0},
            "candidates": candidates, **extra}


def test_ranking_request_rejects_duplicates_and_non_finite_features():
    with pytest.raises(ValidationError):
        RankingRequest.model_validate(_ranking([{"item_id": 1, "features": {}}, {"item_id": 1, "features": {}}]))
    with pytest.raises(ValidationError):
        RankingRequest.model_validate(_ranking([{"item_id": 1, "features": {"x": math.nan}}]))
    with pytest.raises(ValidationError):
        RankingRequest.model_validate(_ranking([]))


def test_rerank_and_retrieval_bounds():
    with pytest.raises(ValidationError):
        RerankRequest.model_validate({"user_id": 1, "candidates": [{"item_id": 1, "score": 1.0},
                                                                  {"item_id": 1, "score": 0.5}]})
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate({"user_id": 1, "num_candidates": 1001})
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate({"user_id": 1, "recent_item_ids": list(range(21))})


def test_task_scores_are_probabilities():
    with pytest.raises(ValidationError):
        TaskScores(engagement=1.2, completion=0.5, play_ratio=0.5, like=0.1, dislike=0.1)


def test_mmoe_ranking_response_requires_task_scores_on_every_item():
    scores = {"engagement": 0.5, "completion": 0.5, "play_ratio": 0.5, "like": 0.1, "dislike": 0.1}

    def response(model_name, items):
        return {"trace_id": "t", "model_version": "m", "model_name": model_name, "score_name": "p_completion",
                "ranked_items": items}

    RankingResponse.model_validate(response("mmoe", [{"item_id": 1, "score": 0.5, "task_scores": scores}]))
    RankingResponse.model_validate(response("xgboost", [{"item_id": 1, "score": 0.5}]))
    with pytest.raises(ValidationError, match="task_scores"):
        RankingResponse.model_validate(response("mmoe", [{"item_id": 1, "score": 0.5, "task_scores": scores},
                                                         {"item_id": 2, "score": 0.4}]))


def test_current_contracts_have_no_breaking_change_against_the_snapshot():
    breaking, _ = compat.compare(compat.load_snapshot(), compat.current_schemas())
    assert breaking == []


def test_compat_classifies_changes():
    old = {"Req": {"properties": {"a": {"type": "integer"}}, "required": ["a"]},
           "Resp": {"properties": {"x": {"type": "string"}}, "required": ["x"]}}

    def with_changes(req=None, resp=None):
        new = {"Req": {**old["Req"], **(req or {})}, "Resp": {**old["Resp"], **(resp or {})}}
        return compat.compare(old, new)

    breaking, compatible = with_changes(req={"properties": {"a": {"type": "integer"}, "b": {"type": "string"}}})
    assert breaking == [] and compatible
    breaking, _ = with_changes(req={"properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
                                    "required": ["a", "b"]})
    assert breaking
    breaking, _ = with_changes(req={"properties": {"a": {"type": "string"}}})
    assert breaking
    breaking, _ = with_changes(resp={"properties": {}, "required": []})
    assert breaking
