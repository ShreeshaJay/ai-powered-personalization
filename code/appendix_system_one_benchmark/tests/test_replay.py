import json
import random
from pathlib import Path

import numpy as np
import pytest

from adapters.schemas import TASK_FIELDS, field_label_space
from eval.calibration import (
    ece_equal_mass,
    ece_equal_width,
    equal_width_bin_index,
    macro_f1_fixed,
    macro_f1_union,
    rescaled_margin,
    risk_coverage,
)
from eval.intervals import ModelArrays, bootstrap_field, encode, paired_differences, resample_indices
from eval.metrics import _key, f1_report
from eval.replay_pack import (
    MANIFEST_NAME,
    ORCAS_STRATUM,
    PROVENANCE_NAME,
    check_manifest,
    collapse_stratum,
    examples_from_label_rows,
    jsonl_gz_bytes,
    load_pack,
    read_jsonl_gz,
    sha256_bytes,
    sha256_lf,
)
from eval.runner import evaluate_task, prediction_from_dict
from replay_scores import (
    EXIT_MANIFEST,
    EXIT_MISMATCH,
    EXIT_OK,
    PILOT_OVERLAP_CHECKS,
    build_report,
    collapse_orcas_strata,
    judge_agreement,
    label_space,
    render_report,
    run_verify,
)

ROOT = Path(__file__).resolve().parents[1]
SLUGS = ("majority", "model_a")
TASKS = ("compatibility", "query_segmentation")
RAW_QUERY_STRATA = ("esci_single_product", "esci_bundle_or_set", "orcas_alpha", "orcas_beta")


def _labels(task: str, count: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Pack label rows (collapsed strata) and the same rows with the raw strata."""
    packed, raw = [], []
    for index in range(count):
        fields = {}
        for field in TASK_FIELDS[task]:
            space = field_label_space(task, field)
            judge_a = space[rng.randrange(len(space))]
            judge_b = judge_a if rng.random() < 0.8 else space[rng.randrange(len(space))]
            fields[field] = {"reference": judge_a if judge_a == judge_b else None, "judge_a": judge_a, "judge_b": judge_b}
        stratum = RAW_QUERY_STRATA[index % len(RAW_QUERY_STRATA)] if task == "query_segmentation" else ("s1", "s2")[index % 2]
        row = {"item_id": f"{task}_ref_{index:05d}", "task": task, "stratum": stratum, "label_source": "dual_judge_consensus", "fields": fields}
        raw.append(row)
        packed.append({**row, "stratum": collapse_stratum(stratum)})
    return packed, raw


def _predictions(slug: str, task: str, rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for row in rows:
        fields = {}
        for field in TASK_FIELDS[task]:
            space = field_label_space(task, field)
            if slug == "majority":
                weights = [1.0] + [0.0] * (len(space) - 1)
            else:
                weights = [rng.random() for _ in space]
            total = sum(weights)
            probabilities = {_key(label): weight / total for label, weight in zip(space, weights)}
            predicted = space[int(np.argmax(weights))]
            fields[field] = {"predicted": predicted, "probabilities": probabilities, "confidence": max(probabilities.values())}
        out.append({"item_id": row["item_id"], "task": task, "elapsed_ms": 1.0 + rng.random(), "input_tokens": 0, "fields": fields})
    return out


def _sign(pack: Path, root: Path) -> None:
    """(Re)write MANIFEST.json for the pack and summaries under root."""
    files = {}
    for path in sorted(pack.rglob("*.jsonl.gz")):
        data = path.read_bytes()
        files[path.relative_to(pack).as_posix()] = {"rows": len(read_jsonl_gz(path)), "bytes": len(data), "sha256": sha256_bytes(data)}
    manifest = {
        "build_date": "2000-01-01",
        "builder": {"commit": "test"},
        "prompt_version": "test",
        "model_order": list(SLUGS),
        "models": {
            slug: {
                "model_id": f"{slug}_id",
                "role_label": slug,
                "summary": f"outputs/benchmark/{slug}/summary.json",
                "summary_sha256_lf": sha256_lf(root / "outputs" / "benchmark" / slug / "summary.json"),
            }
            for slug in SLUGS
        },
        "rubrics": {},
        "files": files,
        "label_provenance": {"path": PROVENANCE_NAME, "sha256_lf": sha256_lf(pack / PROVENANCE_NAME)},
    }
    (pack / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@pytest.fixture()
def synthetic(tmp_path: Path) -> tuple[Path, Path]:
    rng = random.Random(7)
    root = tmp_path / "root"
    pack = tmp_path / "pack"
    (pack / "labels").mkdir(parents=True)
    labels, raw = {}, {}
    for task, count in zip(TASKS, (40, 30)):
        labels[task], raw[task] = _labels(task, count, rng)
        (pack / "labels" / f"{task}.jsonl.gz").write_bytes(jsonl_gz_bytes(labels[task]))
    for slug in SLUGS:
        blocks = {}
        for task in TASKS:
            rows = _predictions(slug, task, labels[task], rng)
            path = pack / "predictions" / slug / f"{task}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(jsonl_gz_bytes(rows))
            metrics = evaluate_task(examples_from_label_rows(raw[task]), [prediction_from_dict(row) for row in rows], task)
            blocks[task] = {**metrics, "adapter": slug, "model_id": f"{slug}_id", "wall_seconds": 1.0, "memory": None}
        summary = root / "outputs" / "benchmark" / slug / "summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"model": slug, "model_id": f"{slug}_id", "tasks": blocks}, indent=2), encoding="utf-8")
    (pack / PROVENANCE_NAME).write_text(json.dumps({"agreement_by_stratum": {}}), encoding="utf-8")
    _sign(pack, root)
    return pack, root


def _edit_summary(root: Path, slug: str, edit) -> None:
    path = root / "outputs" / "benchmark" / slug / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    edit(summary)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def test_verify_passes_and_checks_orcas_strata_in_aggregate(synthetic) -> None:
    pack, root = synthetic
    code, report = run_verify(pack, root, results_file=None)
    assert code == EXIT_OK, report.get("mismatch_details")
    assert report["mismatches"] == 0
    fields = len(TASK_FIELDS["query_segmentation"])
    assert report["values_verified_in_aggregate"] == len(SLUGS) * fields * 2 * 2
    assert report["values_compared_exactly"] + report["values_verified_in_aggregate"] == report["values_stored"]
    assert report["models_without_stored_token_counts"] == list(SLUGS)


def test_verify_exits_1_on_a_perturbed_metric(synthetic) -> None:
    pack, root = synthetic

    def perturb(summary: dict) -> None:
        summary["tasks"]["compatibility"]["fields"]["label"]["accuracy"] += 0.0001

    _edit_summary(root, "model_a", perturb)
    _sign(pack, root)
    code, report = run_verify(pack, root, results_file=None)
    assert code == EXIT_MISMATCH
    assert [item["path"] for item in report["mismatch_details"]] == ["model_a.compatibility.fields.label.accuracy"]


def test_verify_exits_1_on_a_perturbed_orcas_entry(synthetic) -> None:
    pack, root = synthetic

    def perturb(summary: dict) -> None:
        strata = summary["tasks"]["query_segmentation"]["fields"]["goal"]["accuracy_by_stratum"]
        strata["orcas_alpha"]["items"] += 1

    _edit_summary(root, "model_a", perturb)
    _sign(pack, root)
    code, _report = run_verify(pack, root, results_file=None)
    assert code == EXIT_MISMATCH


def test_verify_exits_2_when_inputs_differ_from_the_manifest(synthetic) -> None:
    pack, root = synthetic
    _edit_summary(root, "majority", lambda summary: summary.update(model="changed"))
    code, report = run_verify(pack, root, results_file=None)
    assert code == EXIT_MANIFEST
    assert any("summary changed" in problem for problem in report["manifest_problems"])


def test_verify_exits_2_on_a_changed_or_unlisted_pack_file(synthetic) -> None:
    pack, root = synthetic
    target = pack / "labels" / "compatibility.jsonl.gz"
    original = target.read_bytes()
    changed = bytearray(original)
    changed[-5] ^= 0xFF
    target.write_bytes(bytes(changed))
    assert run_verify(pack, root, results_file=None)[0] == EXIT_MANIFEST
    target.write_bytes(original)
    assert run_verify(pack, root, results_file=None)[0] == EXIT_OK
    (pack / "notes.txt").write_text("x", encoding="utf-8")
    code, report = run_verify(pack, root, results_file=None)
    assert code == EXIT_MANIFEST
    assert report["manifest_problems"] == ["unlisted file in pack: notes.txt"]


def test_report_is_deterministic(synthetic) -> None:
    pack, _root = synthetic
    first = build_report(load_pack(pack), resamples=50, seed=1, bins=15)
    second = build_report(load_pack(pack), resamples=50, seed=1, bins=15)
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)
    text = render_report(first)
    assert "## Paired differences" in text
    assert "orcas_alpha" not in text and "orcas_beta" not in text


def test_class_support_counts_the_labelled_items_of_each_field(synthetic) -> None:
    pack, _root = synthetic
    data = build_report(load_pack(pack), resamples=20, seed=1, bins=15)
    assert set(data["class_support"]) == set(data["fields"])
    for key, support in data["class_support"].items():
        entry = data["fields"][key]
        assert support["labelled_items"] == entry["models"]["majority"]["items"]
        assert sum(support["counts"].values()) == support["labelled_items"]
        assert list(support["counts"])[: len(label_space(entry["task"], entry["field"]))] == label_space(entry["task"], entry["field"])
    assert "## Class support" in render_report(data)


def test_esci_strata_are_pooled_for_judge_agreement(synthetic) -> None:
    pack, _root = synthetic
    loaded = load_pack(pack)
    data = build_report(loaded, resamples=20, seed=1, bins=15)
    pooled = data["agreement_aggregates"]["query_segmentation"]["esci_*"]
    strata = data["agreement_by_stratum"]["query_segmentation"]
    esci = [name for name in strata if name.startswith("esci_")]
    assert len(esci) == 2
    assert pooled["items"] == sum(strata[name]["items"] for name in esci)
    weighted = sum(strata[name]["complete_agreement"] * strata[name]["items"] for name in esci) / pooled["items"]
    assert pooled["complete_agreement"] == pytest.approx(weighted)
    rows = [row for row in loaded.labels["query_segmentation"] if row["stratum"].startswith("esci_")]
    assert pooled == judge_agreement("query_segmentation", rows)
    assert "compatibility" not in data["agreement_aggregates"]
    assert "Pooled over all `esci_*` strata" in render_report(data)


def test_probability_diagnostic_totals_sum_every_field(synthetic) -> None:
    pack, _root = synthetic
    data = build_report(load_pack(pack), resamples=20, seed=1, bins=15)
    totals = {row["model"]: row for row in data["probability_diagnostics_totals"]}
    assert list(totals) == list(SLUGS)
    for slug, total in totals.items():
        rows = [row for row in data["probability_diagnostics"] if row["model"] == slug]
        assert total["fields"] == sum(len(TASK_FIELDS[task]) for task in TASKS) == len(rows)
        for key in ("items", "ties_at_max", "argmax_mismatches"):
            assert total[key] == sum(row[key] for row in rows)
    assert "Totals per model" in render_report(data)


def test_pilot_overlaps_count_shared_exclusion_keys() -> None:
    from build_replay_pack import PILOT_OVERLAP_KEYS, pilot_overlaps

    def compat(query_id: int, anchor: str, candidate: str) -> dict:
        return {
            "sampling_context": {"query_id": query_id},
            "adjudication_input": {"anchor_product": {"product_id": anchor}, "candidate_product": {"product_id": candidate}},
        }

    def query(text: str, query_id: int) -> dict:
        return {"sampling_context": {"query_id": query_id}, "adjudication_input": {"query": text}}

    references = {
        "compatibility": [compat(1, "A", "X"), compat(1, "A", "Y"), compat(1, "B", "X"), compat(2, "A", "X")],
        "query_segmentation": [query("Red  Shoes", 5), query("blue hat", 6)],
        "brand_category": [query("q", 7), query("r", 8), query("s", 9)],
    }
    pilots = {
        "compatibility": [compat(1, "A", "Z")],
        "query_segmentation": [query("red shoes", 50)],
        "brand_category": [query("other text", 8), query("more", 9)],
    }
    counts = pilot_overlaps(references, pilots)
    assert counts == {
        "compatibility_pilot_query_anchor_overlap": 2,
        "query_segmentation_pilot_query_overlap": 1,
        "brand_category_pilot_query_overlap": 2,
    }
    assert {name for name, _key_fn in PILOT_OVERLAP_KEYS.values()} == {name for name, _ in PILOT_OVERLAP_CHECKS.values()}


def test_pilot_overlap_is_reported_only_when_recorded(synthetic) -> None:
    pack, _root = synthetic
    loaded = load_pack(pack)
    data = build_report(loaded, resamples=20, seed=1, bins=15)
    assert data["pilot_overlap"] == {}
    assert "Overlap with the 100-item pilot sets" not in render_report(data)
    loaded.manifest["build_checks"] = {"compatibility_pilot_query_anchor_overlap": 3}
    loaded.manifest["tasks"] = {"compatibility": {"items": 40}}
    data = build_report(loaded, resamples=20, seed=1, bins=15)
    assert data["pilot_overlap"] == {
        "compatibility": {
            "check": "compatibility_pilot_query_anchor_overlap",
            "key": "query and anchor product",
            "overlapping_items": 3,
            "reference_items": 40,
        }
    }
    text = render_report(data)
    assert "`compatibility`, 3 of 40 reference items share the query and anchor product of a pilot item" in text
    assert "cannot be recomputed" in text


def test_readme_maps_every_role_label_to_its_model_key() -> None:
    from build_replay_pack import ROLE_LABELS

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for slug, label in ROLE_LABELS.items():
        assert f"| {label} | `{slug}` |" in readme


def test_collapse_orcas_strata_merges_counts() -> None:
    block = {
        "fields": {
            "goal": {
                "accuracy_by_stratum": {
                    "esci_x": {"items": 3, "accuracy": 0.6667},
                    "orcas_alpha": {"items": 3, "accuracy": 0.3333},
                    "orcas_beta": {"items": 7, "accuracy": 0.5714},
                }
            }
        }
    }
    collapsed, merged = collapse_orcas_strata(block)
    assert collapsed["fields"]["goal"]["accuracy_by_stratum"] == {
        "esci_x": {"items": 3, "accuracy": 0.6667},
        ORCAS_STRATUM: {"items": 10, "accuracy": 0.5},
    }
    assert merged == [{"field": "goal", "stored_entries": 2, "stored_values": 4, "items": 10, "correct": 5, "counts_recovered_exactly": True}]
    assert "orcas_alpha" in block["fields"]["goal"]["accuracy_by_stratum"]


def test_equal_width_bins_follow_the_committed_convention() -> None:
    assert equal_width_bin_index([0.0, 1 / 15, 1 / 15 + 1e-9, 0.5, 1.0, 1.5], 15).tolist() == [0, 0, 1, 7, 14, -1]
    assert ece_equal_width([0.9, 0.9, 0.2, 0.2], [True, False, False, False], 15) == pytest.approx(0.3)


def test_top_label_and_margin_confidence_give_different_ece() -> None:
    probabilities = [0.5, 0.3, 0.1, 0.1]
    top = max(probabilities)
    margin = rescaled_margin(top, len(probabilities))
    correct = [True, False]
    assert margin == pytest.approx(1 / 3)
    assert ece_equal_width([top, top], correct) == pytest.approx(0.0)
    assert ece_equal_width([margin, margin], correct) == pytest.approx(1 / 6)


def test_equal_mass_ece_and_risk_coverage_known_cases() -> None:
    assert ece_equal_mass([0.1, 0.2, 0.8, 0.9], [False, False, True, True], bins=2) == pytest.approx(0.15)
    result = risk_coverage([0.9, 0.8, 0.7], [True, False, True])
    assert result["aurc"] == pytest.approx((0 + 0.5 + 1 / 3) / 3)
    assert result["coverage_accuracy"]["acc_at_1.00"] == pytest.approx(0.6667)


def test_macro_f1_fixed_scores_absent_classes_as_zero() -> None:
    y_true, y_pred = ["a", "a", "b"], ["a", "b", "b"]
    assert macro_f1_fixed(y_true, y_pred, ["a", "b", "c"]) == pytest.approx(4 / 9)
    assert macro_f1_union(y_true, y_pred) == pytest.approx(2 / 3)
    assert round(macro_f1_union(y_true, y_pred), 4) == f1_report(y_true, y_pred)["macro_f1"]


def _arrays(seed: int, n: int = 60) -> dict[str, ModelArrays]:
    rng = np.random.default_rng(seed)
    space = ["a", "b", "c"]
    reference = rng.choice(space, n).tolist()
    models = {}
    for name in ("base", "x", "y"):
        predicted = [ref if rng.random() < 0.6 else rng.choice(space) for ref in reference]
        models[name] = ModelArrays(encode(reference, space), encode(predicted, space), rng.random(n))
    return models


def test_bootstrap_is_seeded_and_matches_scalar_metrics() -> None:
    models = _arrays(3)
    first = bootstrap_field(models, 3, resamples=40, seed=11, stream=2)
    again = bootstrap_field(models, 3, resamples=40, seed=11, stream=2)
    other = bootstrap_field(models, 3, resamples=40, seed=11, stream=3)
    for name in models:
        for metric, samples in first[name].items():
            assert np.array_equal(samples, again[name][metric])
    assert not np.array_equal(first["x"]["accuracy"], other["x"]["accuracy"])
    index = next(resample_indices(60, 40, 11, 2))[0]
    arrays = models["x"]
    reference, predicted = arrays.reference[index], arrays.predicted[index]
    assert first["x"]["accuracy"][0] == pytest.approx(np.mean(reference == predicted))
    assert first["x"]["macro_f1_fixed"][0] == pytest.approx(macro_f1_fixed(reference.tolist(), predicted.tolist(), [0, 1, 2]))
    assert first["x"]["ece_top_label_width"][0] == pytest.approx(ece_equal_width(arrays.confidence[index], reference == predicted))


def test_paired_differences_are_zero_for_identical_models() -> None:
    models = _arrays(5)
    models["copy"] = models["x"]
    distributions = bootstrap_field(models, 3, resamples=30, seed=1)
    estimates = {name: {metric: 0.0 for metric in distributions[name]} for name in models}
    rows = [row for row in paired_differences(distributions, estimates, "base") if {row["a"], row["b"]} == {"x", "copy"}]
    assert rows and all(row["low"] == row["high"] == 0.0 and not row["excludes_zero"] for row in rows)


@pytest.mark.skipif(not (ROOT / "replay" / MANIFEST_NAME).is_file(), reason="replay pack not present")
def test_figures_are_written_at_print_size(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import struct

    import make_figures

    assert make_figures.main(["--output-dir", str(tmp_path)]) == 0
    for name in ("confidence_meanings.png", "calibration_two_numbers.png", "judge_agreement_by_stratum.png"):
        data = (tmp_path / name).read_bytes()
        width = struct.unpack(">I", data[16:20])[0]
        assert width == make_figures.WIDTH_IN * make_figures.DPI
        assert b"tEXt" not in data and b"iTXt" not in data


@pytest.mark.skipif(not (ROOT / "replay" / MANIFEST_NAME).is_file(), reason="replay pack not present")
def test_committed_pack_matches_manifest_and_schema() -> None:
    assert check_manifest(ROOT / "replay", ROOT) == []
    checks = json.loads((ROOT / "replay" / MANIFEST_NAME).read_text(encoding="utf-8"))["build_checks"]
    assert all(isinstance(checks[name], int) for name, _ in PILOT_OVERLAP_CHECKS.values())
    for task in TASK_FIELDS:
        rows = read_jsonl_gz(ROOT / "replay" / "labels" / f"{task}.jsonl.gz")
        for row in rows:
            assert set(row) == {"item_id", "task", "stratum", "label_source", "fields"}
            assert set(row["fields"]) == set(TASK_FIELDS[task])
            assert row["stratum"] == ORCAS_STRATUM or not row["stratum"].startswith("orcas")
            judged = {"reference", "judge_a", "judge_b"} if task != "esci" else {"reference"}
            assert all(set(values) == judged for values in row["fields"].values())
        predictions = read_jsonl_gz(ROOT / "replay" / "predictions" / "jev_1_13_0" / f"{task}.jsonl.gz")
        assert [row["item_id"] for row in predictions] == [row["item_id"] for row in rows]
        assert all(set(row) == {"item_id", "task", "elapsed_ms", "input_tokens", "fields"} for row in predictions)
