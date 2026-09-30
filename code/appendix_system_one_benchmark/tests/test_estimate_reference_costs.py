from copy import deepcopy

import pytest

from estimate_reference_costs import DEFAULT_CONFIG, estimate, load_config, render_markdown


def test_batch_nominal_costs() -> None:
    result = estimate(load_config(DEFAULT_CONFIG), use_batch=True)
    nominal = result["scenarios"]["nominal"]

    assert result["base_items"] == 6_000
    assert nominal["input_tokens_per_model"] == 1_827_000
    assert nominal["output_tokens_per_model"] == 325_500
    assert nominal["model_costs"]["claude-opus-5"]["total_cost_usd"] == pytest.approx(
        8.6363
    )
    assert nominal["model_costs"]["gpt-5.6-sol"]["total_cost_usd"] == pytest.approx(
        6.909
    )
    assert nominal["combined_cost_usd"] == pytest.approx(15.5452)
    assert nominal["within_budget"] is True


def test_standard_cost_is_double_batch_cost() -> None:
    config = load_config(DEFAULT_CONFIG)
    batch = estimate(config, use_batch=True)["scenarios"]["reasoning_reserve"]
    standard = estimate(config, use_batch=False)["scenarios"]["reasoning_reserve"]

    assert standard["combined_cost_usd"] == pytest.approx(
        2 * batch["combined_cost_usd"], abs=0.0002
    )


def test_large_esci_run_exposes_budget_risk() -> None:
    config = deepcopy(load_config(DEFAULT_CONFIG))
    config["tasks"]["esci_new_enriched"]["items"] = 100_000
    result = estimate(config, use_batch=True)

    assert result["scenarios"]["nominal"]["within_budget"] is True
    assert result["scenarios"]["reasoning_reserve"]["within_budget"] is False


def test_markdown_contains_both_models() -> None:
    rendered = render_markdown(estimate(load_config(DEFAULT_CONFIG), use_batch=True))

    assert "claude-opus-5" in rendered
    assert "gpt-5.6-sol" in rendered
    assert "planning estimates" in rendered

