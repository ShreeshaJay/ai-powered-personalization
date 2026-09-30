import json

import pytest

from adjudicate_pilots import (
    build_prompt,
    parse_and_validate,
    resolve_candidate_choice,
    response_schema,
)


def compatibility_row() -> dict:
    return {
        "item_id": "compat_test",
        "adjudication_input": {
            "item_id": "compat_test",
            "query_context": "phone case",
            "anchor_product": {"product_id": "a", "title": "Phone"},
            "candidate_product": {"product_id": "b", "title": "Case"},
        },
    }


def test_compatibility_response_parses() -> None:
    row = compatibility_row()
    raw = json.dumps(
        {
            "labels": [
                {
                    "item_id": "compat_test",
                    "label": "explicit_fit",
                    "confidence": "high",
                    "evidence": ["The model is named."],
                    "rationale": "The case explicitly fits the phone.",
                }
            ]
        }
    )
    parsed = parse_and_validate("compatibility", raw, [row])
    assert parsed[0]["label"] == "explicit_fit"


def test_missing_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing item IDs"):
        parse_and_validate("compatibility", '{"labels":[]}', [compatibility_row()])


def test_brand_candidate_outside_options_is_rejected() -> None:
    row = {
        "item_id": "brand_test",
        "adjudication_input": {
            "item_id": "brand_test",
            "query": "example",
            "brand_candidates": [
                {"id": "brand_0", "value": "Example"},
                {"id": "none_or_other", "value": "none"},
            ],
            "category_candidates": [
                {"id": "category_0", "value": "Example category"},
                {"id": "other_or_ambiguous", "value": "other"},
            ],
        },
    }
    raw = json.dumps(
        {
            "labels": [
                {
                    "item_id": "brand_test",
                    "brand_intent": "explicit_brand",
                    "brand_choice": "brand_99",
                    "category_choice": "category_0",
                    "multi_target": False,
                    "confidence": "high",
                    "rationale": "Example.",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="Invalid brand candidate"):
        parse_and_validate("brand_category", raw, [row])


def test_prompt_excludes_sampling_context() -> None:
    row = compatibility_row()
    row["sampling_context"] = {"hidden": "must not appear"}
    prompt = build_prompt("compatibility", [row])
    assert "must not appear" not in prompt


def test_candidate_display_value_resolves_to_id() -> None:
    candidates = [
        {"id": "brand_0", "value": "Example Brand"},
        {"id": "none_or_other", "value": "none or other brand"},
    ]
    assert (
        resolve_candidate_choice("Example Brand", candidates, "none_or_other")
        == "brand_0"
    )
    assert (
        resolve_candidate_choice("none", candidates, "none_or_other")
        == "none_or_other"
    )


def test_schema_is_strict_at_object_boundaries() -> None:
    schema = response_schema("query_segmentation", 20)
    assert schema["additionalProperties"] is False
    assert (
        schema["properties"]["labels"]["items"]["additionalProperties"] is False
    )

