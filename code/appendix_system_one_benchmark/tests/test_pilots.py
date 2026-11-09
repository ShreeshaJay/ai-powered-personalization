import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from build_pilots import (
    NOT_DISTRIBUTED,
    ROOT,
    SOURCE_FILES,
    bounded_candidates,
    explicit_brand_matches,
    normalized,
)
from eval.replay_pack import sha256_lf
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


def test_pilot_summary_names_inputs_without_paths_and_matches_the_files() -> None:
    summary = json.loads((ROOT / "pilots" / "pilot_summary.json").read_text(encoding="utf-8"))
    assert summary["source_files"] == {name: f"{file_name} {NOT_DISTRIBUTED}" for name, file_name in SOURCE_FILES.items()}
    for path in (meta["path"] for meta in summary["datasets"].values()):
        assert "\\" not in path and ":" not in path and not PurePosixPath(path).is_absolute()
    for name, meta in summary["datasets"].items():
        assert meta["path"] == f"pilots/{name}_pilot_100.jsonl"
        file = ROOT / meta["path"]
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert sha256_lf(file) == meta["sha256"]
        assert len(rows) == meta["rows"] == 100
        assert Counter(row["sampling_context"]["source_stratum"] for row in rows) == meta["source_strata"]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "references" / "compatibility_reference_2000.jsonl").is_file(),
    reason="reference manifests are not shipped in this public drop",
)
def test_generated_reference_sets_validate() -> None:
    result = validate_references(ROOT / "references")
    assert result["valid"] is True
    assert all(dataset["rows"] == 2000 for dataset in result["datasets"].values())

