import json
from pathlib import Path

import pytest

from adapters.jev import (
    JevAdapter,
    build_systemone_payload,
    estimate_input_tokens,
    token_cost_usd,
)
from adapters.jevlite import JevLiteAdapter
from adapters.kev import KevAdapter, build_kev_payload
from adapters.laya import decode_laya_result
from adapters.majority import MajorityAdapter
from adapters.schemas import compact_state, field_label_space, laya_questions
from eval.datasets import load_examples
from eval.metrics import field_metrics, top_label_ece
from eval.runner import evaluate_task, prediction_from_dict
from run_benchmark import output_slug

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMPAT = ROOT / "references" / "compatibility_reference_2000.jsonl"
REFERENCE_CONSENSUS = ROOT / "references" / "consensus" / "compatibility_consensus.jsonl"


def test_compact_state_omits_sampling_and_item_id() -> None:
    state = compact_state(
        "compatibility",
        {
            "item_id": "compat_test",
            "query_context": "phone case",
            "anchor_product": {"title": "Phone", "brand": "Acme"},
            "candidate_product": {"title": "Case", "bullet_point": "x" * 500},
        },
    )
    rendered = str(state)
    assert "compat_test" not in rendered
    assert "must not appear" not in rendered
    assert len(state["candidate_product"]["bullet_point"]) == 180
    assert state["query_context"] == "phone case"


def test_brand_state_lists_candidate_ids() -> None:
    state = compact_state(
        "brand_category",
        {
            "query": "6ku bike",
            "brand_candidates": [
                {"id": "brand_5", "value": "6KU"},
                {"id": "none_or_other", "value": "none or other brand"},
            ],
            "category_candidates": [
                {"id": "category_2", "value": "Sports & Outdoors > Cycling"},
                {"id": "other_or_ambiguous", "value": "other"},
            ],
        },
    )
    assert "brand_5: 6KU" in state["brand_candidates"]
    assert "category_2: Sports & Outdoors > Cycling" in state["category_candidates"]


def test_laya_questions_cover_task_fields() -> None:
    questions = laya_questions("query_segmentation")
    assert set(questions) == {"goal", "object", "specificity", "commerce_scope"}
    assert questions["goal"]["type"] == "choice"
    brand_questions = laya_questions("brand_category")
    assert brand_questions["multi_target"]["type"] == "noul"
    assert "none_or_other" in brand_questions["brand_choice"]["criteria"]


def test_decode_laya_choice_and_noul() -> None:
    prediction = decode_laya_result(
        "brand_category_ref_00001",
        "brand_category",
        {
            "answers": {
                "brand_intent": {
                    "type": "choice",
                    "choice": "no_brand",
                    "probabilities": {"no_brand": 0.7, "explicit_brand": 0.3},
                    "confidence": 0.4,
                },
                "brand_choice": {
                    "type": "choice",
                    "choice": "none_or_other",
                    "probabilities": {"none_or_other": 0.9, "brand_0": 0.1},
                    "confidence": 0.8,
                },
                "category_choice": {
                    "type": "choice",
                    "choice": "category_1",
                    "probabilities": {"category_1": 0.6, "other_or_ambiguous": 0.4},
                    "confidence": 0.2,
                },
                "multi_target": {
                    "type": "noul",
                    "noul": 0.12,
                    "confidence": 0.88,
                },
            },
            "usage": {"input_tokens": 40},
        },
    )
    assert prediction.fields["brand_intent"].predicted == "no_brand"
    assert prediction.fields["multi_target"].predicted is False
    assert prediction.fields["multi_target"].probabilities["false"] == 0.88
    assert prediction.input_tokens == 40


def test_perfect_predictions_have_zero_calibration_error() -> None:
    report = field_metrics(
        ["E", "S"],
        ["E", "S"],
        [{"E": 1.0, "S": 0.0, "C": 0.0, "I": 0.0}, {"E": 0.0, "S": 1.0, "C": 0.0, "I": 0.0}],
        [1.0, 1.0],
    )
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["ece"] == 0.0
    assert report["brier"] == 0.0
    assert report["nll"] == 0.0


def test_confident_wrong_predictions_have_unit_ece() -> None:
    ece = top_label_ece([1.0, 1.0], [False, False], bins=10)
    assert ece == 1.0


def test_majority_adapter_uses_prior_not_item_text() -> None:
    adapter = MajorityAdapter()
    predictions = adapter.predict_batch(
        "compatibility",
        [
            {
                "item_id": "compat_test",
                "adjudication_input": {"query_context": "unused"},
            }
        ],
    )
    assert predictions[0].fields["label"].predicted == "incompatible"
    assert set(predictions[0].fields["label"].probabilities) == set(
        field_label_space("compatibility", "label")
    )


@pytest.mark.skipif(
    not (REFERENCE_COMPAT.is_file() and REFERENCE_CONSENSUS.is_file()),
    reason="reference manifests and consensus labels are not shipped in this public drop",
)
def test_reference_examples_withhold_disagreements() -> None:
    examples = load_examples("compatibility", dataset="reference")
    assert len(examples) == 2000
    labeled = [row for row in examples if row["labels"]["label"] is not None]
    withheld = [row for row in examples if row["labels"]["label"] is None]
    assert len(labeled) == 1633
    assert len(withheld) == 367
    assert "sampling_context" not in examples[0]["adjudication_input"]


def test_evaluate_task_exact_match_requires_complete_labels() -> None:
    examples = [
        {
            "item_id": "a",
            "labels": {"label": "incompatible"},
            "complete_labels": {"label": "incompatible"},
            "complete_agreement": True,
            "sampling_context": {"source_stratum": "x"},
        },
        {
            "item_id": "b",
            "labels": {"label": None},
            "complete_labels": None,
            "complete_agreement": False,
            "sampling_context": {"source_stratum": "y"},
        },
    ]
    raw = {
        "item_id": "a",
        "task": "compatibility",
        "fields": {
            "label": {
                "field": "label",
                "predicted": "incompatible",
                "probabilities": {"incompatible": 1.0},
                "confidence": 1.0,
            }
        },
        "input_tokens": 0,
        "elapsed_ms": 10.0,
    }
    other = dict(raw)
    other["item_id"] = "b"
    other["fields"] = {
        "label": {
            "field": "label",
            "predicted": "explicit_fit",
            "probabilities": {"explicit_fit": 1.0},
            "confidence": 1.0,
        }
    }
    metrics = evaluate_task(
        examples,
        [prediction_from_dict(raw), prediction_from_dict(other)],
        "compatibility",
    )
    assert metrics["fields"]["label"]["items"] == 1
    assert metrics["fields"]["label"]["accuracy"] == 1.0
    assert metrics["complete_exact_match"] == {"items": 1, "exact_match": 1.0}


def test_jev_payload_uses_compact_state_and_shared_questions() -> None:
    payload = build_systemone_payload(
        "compatibility",
        {
            "item_id": "compat_test",
            "query_context": "phone case",
            "anchor_product": {"title": "Phone"},
            "candidate_product": {"title": "Case"},
            "sampling_context": {"hidden": "must not appear"},
        },
    )
    assert payload["model"] == "jev-1.13.0"
    assert payload["questions"]["label"]["type"] == "choice"
    rendered = json.dumps(payload)
    assert "must not appear" not in rendered
    assert "compat_test" not in rendered
    assert estimate_input_tokens(payload) >= 1
    assert token_cost_usd(1_000_000) == pytest.approx(0.042)


def test_decode_jev_noul_without_confidence() -> None:
    prediction = decode_laya_result(
        "brand_category_ref_00001",
        "brand_category",
        {
            "answers": {
                "brand_intent": {
                    "type": "choice",
                    "choice": "no_brand",
                    "probabilities": {"no_brand": 1.0},
                    "confidence": 1.0,
                },
                "brand_choice": {
                    "type": "choice",
                    "choice": "none_or_other",
                    "probabilities": {"none_or_other": 1.0},
                    "confidence": 1.0,
                },
                "category_choice": {
                    "type": "choice",
                    "choice": "other_or_ambiguous",
                    "probabilities": {"other_or_ambiguous": 1.0},
                    "confidence": 1.0,
                },
                "multi_target": {"type": "noul", "noul": 0.2},
            },
            "usage": {"input_tokens": 88, "output_tokens": 12},
        },
    )
    assert prediction.fields["multi_target"].predicted is False
    assert prediction.fields["multi_target"].confidence == pytest.approx(0.8)


def test_jev_budget_blocks_before_request() -> None:
    adapter = JevAdapter(budget_usd=0.0000001)
    adapter._api_key = "test-key"
    with pytest.raises(RuntimeError, match="budget"):
        adapter._reserve_tokens(100_000)


def test_kev_payload_reuses_shared_questions() -> None:
    payload = build_kev_payload(
        "compatibility",
        {
            "item_id": "compat_test",
            "query_context": "phone case",
            "anchor_product": {"title": "Phone"},
            "candidate_product": {"title": "Case"},
            "sampling_context": {"hidden": "must not appear"},
        },
    )
    assert payload["model"] == "kev-latest"
    assert payload["questions"]["label"]["type"] == "choice"
    assert "must not appear" not in json.dumps(payload)


def test_kev_adapter_cache_key_is_the_run() -> None:
    adapter = KevAdapter(run="jaredpalmer/kev-0.8b", autostart=False)
    assert adapter.name == "kev"
    assert adapter.cache_model_id == "jaredpalmer/kev-0.8b"
    with pytest.raises(RuntimeError, match="not reachable"):
        adapter.load()


def test_pack_colab_bundle_lists_required_files() -> None:
    from pack_colab_bundle import collect_files

    names = {path.name for path in collect_files()}
    assert "esci_eval_slice.jsonl" in names
    assert "run_open_models.py" in names
    assert "run_benchmark.py" in names


def test_output_slug_distinguishes_kev_sizes() -> None:
    assert output_slug("kev", "jaredpalmer/kev-0.8b") == "kev_0_8b"
    assert output_slug("kev", "jaredpalmer/kev-4b") == "kev_4b"
    assert output_slug("jevlite", "vagmi/jev-lite") == "jevlite"


def test_jevlite_adapter_defaults_to_local_serve() -> None:
    adapter = JevLiteAdapter(autostart=False)
    assert adapter.name == "jevlite"
    assert adapter.cache_model_id == "vagmi/jev-lite"
    assert adapter.base_url.endswith(":8000")
    assert adapter.concurrency == 1
