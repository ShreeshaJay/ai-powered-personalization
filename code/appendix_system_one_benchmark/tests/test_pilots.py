import random
from pathlib import Path

import pytest

from build_pilots import (
    ROOT,
    bounded_candidates,
    explicit_brand_matches,
    normalized,
)
from validate_pilots import validate_all
from validate_reference_sets import validate_all as validate_references


def test_normalized_text() -> None:
    assert normalized("  iPhone-13 Pro! ") == "iphone 13 pro"


def test_explicit_brand_matching_uses_token_boundaries() -> None:
    assert explicit_brand_matches(
        "vortex diamondback 10x42", ["Vortex", "Ort", "Other Brand"]
    ) == ["Vortex"]


def test_bounded_candidates_are_unique_and_deterministic() -> None:
    first = bounded_candidates(
        ["Target"], ["Target", "A", "B", "C"], 3, random.Random(42), "brand"
    )
    second = bounded_candidates(
        ["Target"], ["Target", "A", "B", "C"], 3, random.Random(42), "brand"
    )
    assert first == second
    assert len({candidate["value"] for candidate in first}) == 3


def test_generated_pilots_validate() -> None:
    result = validate_all(ROOT / "pilots")
    assert result["valid"] is True
    assert all(dataset["rows"] == 100 for dataset in result["datasets"].values())


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "references" / "compatibility_reference_2000.jsonl").is_file(),
    reason="reference manifests are not shipped in this public drop",
)
def test_generated_reference_sets_validate() -> None:
    result = validate_references(ROOT / "references")
    assert result["valid"] is True
    assert all(dataset["rows"] == 2000 for dataset in result["datasets"].values())

