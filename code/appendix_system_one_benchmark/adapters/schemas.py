"""Zero-shot typed-decision schemas and compact state builders."""

from __future__ import annotations

from typing import Any

from adjudicate_pilots import ENUMS, TASK_INSTRUCTIONS


PROMPT_VERSION = "zero_shot_typed_v1"
PRODUCT_TITLE_LIMIT = 240
PRODUCT_BULLET_LIMIT = 180
PRODUCT_ATTR_LIMIT = 80

TASK_FIELDS = {
    "compatibility": ["label"],
    "query_segmentation": ["goal", "object", "specificity", "commerce_scope"],
    "brand_category": [
        "brand_intent",
        "brand_choice",
        "category_choice",
        "multi_target",
    ],
    "esci": ["esci_label"],
}

BRAND_CHOICE_IDS = [f"brand_{index}" for index in range(8)] + ["none_or_other"]
CATEGORY_CHOICE_IDS = [f"category_{index}" for index in range(8)] + [
    "other_or_ambiguous"
]

CRITERIA = {
    "label": {
        "incompatible": (
            "Explicit mismatch or no meaningful use of the candidate with the anchor"
        ),
        "insufficient_evidence": (
            "A relationship is plausible, but the supplied metadata cannot establish fit"
        ),
        "generic_compatible": (
            "Usable together without model-specific fit, or general interoperability is established"
        ),
        "explicit_fit": (
            "Metadata explicitly establishes model, family, or specification fit"
        ),
    },
    "goal": {
        "known_item_navigation": (
            "Reach a named model, title, person, distinct item, brand store, retailer, or website"
        ),
        "product_discovery": (
            "Find products by generic type, attributes, recipient, occasion, or need"
        ),
        "comparison_decision": "Compare alternatives, seek recommendations, or decide what to buy",
        "informational_support": (
            "Learn, troubleshoot, obtain instructions, or answer a product-related question"
        ),
        "service_account": (
            "Account, login, order, payment, delivery, subscription, repair, or customer service"
        ),
        "other_unclear": "Malformed, non-commerce, or insufficiently clear",
    },
    "object": {
        "product": (
            "A distinct named item, model, title, or identifier, including an accessory constrained by a named model"
        ),
        "category": (
            "A generic product type or merchandise family, with or without ordinary constraints"
        ),
        "brand_store": "A brand, retailer, or storefront destination with no product type requested",
        "service_content": (
            "Account, workflow, website, instructions, support, restaurant, or other non-product target"
        ),
        "unclear": "No defensible target object",
    },
    "specificity": {
        "exact_model_or_identifier": (
            "Exact model, generation, part identifier, or named-model accessory constraint"
        ),
        "named_entity_or_title": (
            "Brand or store only, person, media title, proprietary line, or named item"
        ),
        "product_type_with_constraints": (
            "Product type plus brand, size, color, audience, quantity, feature, or other constraint"
        ),
        "product_type_only": "Generic product type with no explicit constraint",
        "broad_need_or_occasion": (
            "Broad need, activity, recipient, theme, or occasion without one definite product type"
        ),
        "none_or_unclear": "No defensible specificity",
    },
    "commerce_scope": {
        "in_scope": "The query is in shopping or merchandise scope",
        "out_of_scope": "The query is not a shopping or merchandise request",
        "ambiguous": "Scope cannot be determined from the query alone",
    },
    "brand_intent": {
        "explicit_brand": "The query explicitly names a brand or unmistakable brand alias",
        "inferred_product_line": (
            "The query names a proprietary product line whose brand can be resolved from common usage"
        ),
        "no_brand": "No brand preference is expressed",
        "ambiguous_brand": (
            "A token might be a brand or an ordinary term, and the query does not resolve it"
        ),
    },
    "esci_label": {
        "E": "Exact: directly satisfies the query and its explicit constraints",
        "S": "Substitute: plausible alternative for the same main purpose",
        "C": "Complement: useful with the requested item but not a replacement",
        "I": "Irrelevant: neither satisfies nor meaningfully complements the query",
    },
}

ESCI_INSTRUCTIONS = (
    "Classify the query-product relationship using Amazon ESCI relevance."
)
MAJORITY_PRIORS = {
    "compatibility": {"label": "incompatible"},
    "query_segmentation": {
        "goal": "product_discovery",
        "object": "category",
        "specificity": "product_type_with_constraints",
        "commerce_scope": "in_scope",
    },
    "brand_category": {
        "brand_intent": "no_brand",
        "brand_choice": "none_or_other",
        "category_choice": "other_or_ambiguous",
        "multi_target": False,
    },
    "esci": {"esci_label": "I"},
}


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def compact_product(product: dict[str, Any] | None) -> dict[str, str]:
    product = product or {}
    title = product.get("title") or product.get("product_title") or ""
    brand = product.get("brand") or product.get("product_brand") or ""
    color = product.get("color") or product.get("product_color") or ""
    bullet = (
        product.get("bullet_point")
        or product.get("product_bullet_point")
        or ""
    )
    return {
        "title": _clip(title, PRODUCT_TITLE_LIMIT),
        "brand": _clip(brand, PRODUCT_ATTR_LIMIT),
        "color": _clip(color, PRODUCT_ATTR_LIMIT),
        "bullet_point": _clip(bullet, PRODUCT_BULLET_LIMIT),
    }


def _candidate_lines(candidates: list[dict[str, Any]]) -> list[str]:
    lines = []
    for candidate in candidates:
        lines.append(f"{candidate['id']}: {candidate.get('value', '')}")
    return lines


def compact_state(task: str, adjudication_input: dict[str, Any]) -> dict[str, Any]:
    """Build the model-visible state. Sampling metadata is never included."""
    payload = adjudication_input
    if task == "compatibility":
        return {
            "query_context": payload.get("query_context", ""),
            "anchor_product": compact_product(payload.get("anchor_product")),
            "candidate_product": compact_product(payload.get("candidate_product")),
        }
    if task == "query_segmentation":
        return {"query": payload.get("query", "")}
    if task == "brand_category":
        return {
            "query": payload.get("query", ""),
            "brand_candidates": _candidate_lines(payload.get("brand_candidates", [])),
            "category_candidates": _candidate_lines(
                payload.get("category_candidates", [])
            ),
        }
    if task == "esci":
        return {
            "query": payload.get("query", ""),
            "product": compact_product(payload.get("product")),
        }
    raise ValueError(f"Unknown task: {task}")


def _id_criteria(ids: list[str], kind: str) -> dict[str, str]:
    criteria = {
        option_id: f"The {kind} listed beside {option_id} in the state"
        for option_id in ids
        if option_id not in {"none_or_other", "other_or_ambiguous"}
    }
    if "none_or_other" in ids:
        criteria["none_or_other"] = (
            "No offered brand is expressed, or the expressed brand is absent from the list"
        )
    if "other_or_ambiguous" in ids:
        criteria["other_or_ambiguous"] = (
            "No offered category is defensible, or the query targets multiple unrelated types"
        )
    return criteria


def laya_questions(task: str) -> dict[str, dict[str, Any]]:
    if task == "compatibility":
        return {
            "label": {
                "type": "choice",
                "instructions": TASK_INSTRUCTIONS["compatibility"],
                "criteria": CRITERIA["label"],
            }
        }
    if task == "query_segmentation":
        return {
            field: {
                "type": "choice",
                "instructions": TASK_INSTRUCTIONS["query_segmentation"],
                "criteria": CRITERIA[field],
            }
            for field in TASK_FIELDS[task]
        }
    if task == "brand_category":
        return {
            "brand_intent": {
                "type": "choice",
                "instructions": TASK_INSTRUCTIONS["brand_category"],
                "criteria": CRITERIA["brand_intent"],
            },
            "brand_choice": {
                "type": "choice",
                "instructions": TASK_INSTRUCTIONS["brand_category"],
                "criteria": _id_criteria(BRAND_CHOICE_IDS, "brand"),
            },
            "category_choice": {
                "type": "choice",
                "instructions": TASK_INSTRUCTIONS["brand_category"],
                "criteria": _id_criteria(CATEGORY_CHOICE_IDS, "category"),
            },
            "multi_target": {
                "type": "noul",
                "instructions": (
                    "The query requests multiple distinct product or brand targets."
                ),
                "criteria": {
                    "false": "The query has one requested target",
                    "true": "The query has multiple distinct requested targets",
                },
            },
        }
    if task == "esci":
        return {
            "esci_label": {
                "type": "choice",
                "instructions": ESCI_INSTRUCTIONS,
                "criteria": CRITERIA["esci_label"],
            }
        }
    raise ValueError(f"Unknown task: {task}")


def field_label_space(task: str, field: str) -> list[Any]:
    if field == "label":
        return list(ENUMS["compatibility_label"])
    if field == "esci_label":
        return ["E", "S", "C", "I"]
    if field == "brand_choice":
        return list(BRAND_CHOICE_IDS)
    if field == "category_choice":
        return list(CATEGORY_CHOICE_IDS)
    if field == "multi_target":
        return [False, True]
    return list(ENUMS[field])
