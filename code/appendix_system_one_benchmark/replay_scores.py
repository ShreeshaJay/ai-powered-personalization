"""Recompute the committed benchmark numbers from the text-free replay pack, on CPU.

    python replay_scores.py verify    # exit 0 pass, 1 metric or results-file mismatch, 2 manifest mismatch
    python replay_scores.py report    # outputs/replay/REPLAY_RESULTS.md and replay_metrics.json
    python replay_scores.py figures   # outputs/replay/figures/*.png (needs matplotlib)

`verify` recomputes every metric in every committed `summary.json` from `replay/`
with the committed `evaluate_task` and compares at stored precision. It then
regenerates `outputs/benchmark/ZERO_SHOT_RESULTS.md` in memory and compares it
with the committed file. No network, no GPU, no API keys; seeds are fixed, so
repeated runs write identical files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from adapters.schemas import TASK_FIELDS, field_label_space
from eval.calibration import (
    DEFAULT_BINS,
    binned_brier_decomposition,
    confidence_semantics,
    ece_equal_mass,
    ece_equal_width,
    macro_f1_fixed,
    macro_f1_union,
    reliability_bins,
    risk_coverage,
    zero_probability_true_labels,
)
from eval.intervals import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    METRICS,
    ModelArrays,
    bootstrap_field,
    encode,
    paired_differences,
    percentile_interval,
)
from eval.metrics import _key, multiclass_brier, negative_log_likelihood
from eval.replay_pack import (
    DEFAULT_PACK_DIR,
    ORCAS_PREFIX,
    ORCAS_STRATUM,
    PROVENANCE_NAME,
    ReplayPack,
    check_manifest,
    field_view,
    is_degenerate,
    load_pack,
    majority_share,
    sha256_lf,
)
from eval.runner import evaluate_task

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "replay"
RESULTS_FILE = ROOT / "outputs" / "benchmark" / "ZERO_SHOT_RESULTS.md"
EXIT_OK, EXIT_MISMATCH, EXIT_MANIFEST = 0, 1, 2
RUN_ONLY_KEYS = frozenset(
    {
        "adapter",
        "model_id",
        "prompt_version",
        "dataset",
        "task",
        "cached_items",
        "new_items",
        "wall_seconds",
        "memory",
    }
)
STRATUM_KEY = "accuracy_by_stratum"
BASELINE = "majority"
SNAPSHOT_FIELDS = (
    ("compatibility", "label"),
    ("query_segmentation", "goal"),
    ("brand_category", "brand_choice"),
    ("esci", "esci_label"),
)
STRATUM_FIELDS = (
    ("compatibility", "label"),
    ("query_segmentation", "goal"),
    ("brand_category", "brand_choice"),
)
POSITION_FIELDS = (("brand_category", "brand_choice"), ("brand_category", "category_choice"))
CONFIDENCE_LEVEL = 0.95


# ---------------------------------------------------------------- verify


def same(stored: Any, recomputed: Any) -> bool:
    if isinstance(stored, float) and isinstance(recomputed, float) and math.isnan(stored) and math.isnan(recomputed):
        return True
    if isinstance(stored, bool) or isinstance(recomputed, bool):
        return stored is recomputed
    return stored == recomputed


def diff(stored: Any, recomputed: Any, path: str, out: list[dict[str, Any]]) -> None:
    if isinstance(stored, dict) and isinstance(recomputed, dict):
        for key in sorted(set(stored) | set(recomputed)):
            if key not in stored:
                out.append({"path": f"{path}.{key}", "stored": "<absent>", "recomputed": recomputed[key]})
            elif key not in recomputed:
                out.append({"path": f"{path}.{key}", "stored": stored[key], "recomputed": "<absent>"})
            else:
                diff(stored[key], recomputed[key], f"{path}.{key}", out)
        return
    if not same(stored, recomputed):
        out.append({"path": path, "stored": stored, "recomputed": recomputed})


def count_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_leaves(item) for item in value.values())
    return 1


def collapse_orcas_strata(block: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge a stored block's per-category ORCAS-I stratum entries into one `orcas_i` entry.

    The pack does not carry the ORCAS-I category, so those entries can only be
    checked in aggregate: correct counts are recovered as round(accuracy * items),
    which is exact because accuracy is stored to 4 decimals and every stratum has
    fewer than 10,000 items. The returned records name fields, not categories.
    """
    block = copy.deepcopy(block)
    merged = []
    for field_name, report in (block.get("fields") or {}).items():
        strata = report.get(STRATUM_KEY) or {}
        orcas = {key: value for key, value in strata.items() if key.startswith(ORCAS_PREFIX)}
        if not orcas:
            continue
        items = correct = 0
        exact = True
        for entry in orcas.values():
            count = int(entry["items"])
            hits = int(round(entry["accuracy"] * count))
            exact = exact and count < 10_000 and round(hits / count, 4) == entry["accuracy"]
            items += count
            correct += hits
        for key in orcas:
            del strata[key]
        strata[ORCAS_STRATUM] = {"items": items, "accuracy": round(correct / items, 4)}
        report[STRATUM_KEY] = dict(sorted(strata.items()))
        merged.append(
            {
                "field": field_name,
                "stored_entries": len(orcas),
                "stored_values": 2 * len(orcas),
                "items": items,
                "correct": correct,
                "counts_recovered_exactly": exact,
            }
        )
    return block, merged


def recompute_task(pack: ReplayPack, slug: str, task: str, stored: dict[str, Any]) -> dict[str, Any]:
    predictions = pack.predictions[(slug, task)]
    metrics = evaluate_task(pack.examples[task], predictions, task)
    if "input_tokens" in stored:
        metrics["input_tokens"] = sum(prediction.input_tokens for prediction in predictions)
    if "cost_usd" in stored:
        from adapters.jev import INPUT_USD_PER_MILLION

        metrics["cost_usd"] = round(metrics["input_tokens"] * INPUT_USD_PER_MILLION / 1_000_000, 6)
    return metrics


def compare_task(stored_block: dict[str, Any], recomputed: dict[str, Any], path: str) -> dict[str, Any]:
    stored = {key: value for key, value in stored_block.items() if key not in RUN_ONLY_KEYS}
    total = count_leaves(stored)
    collapsed, merged = collapse_orcas_strata(stored)
    mismatches: list[dict[str, Any]] = []
    diff(collapsed, recomputed, path, mismatches)
    for record in merged:
        if not record["counts_recovered_exactly"]:
            mismatches.append({"path": f"{path}.fields.{record['field']}.{STRATUM_KEY}", "stored": "ORCAS-I entries", "recomputed": "counts not recoverable"})
    aggregate = sum(record["stored_values"] for record in merged)
    return {
        "values": total,
        "compared_exactly": total - aggregate,
        "verified_in_aggregate": aggregate,
        "token_and_cost_values": sum(key in stored for key in ("input_tokens", "cost_usd")),
        "orcas_i_merges": merged,
        "mismatches": mismatches,
    }


def summaries_for(pack: ReplayPack, root: Path) -> dict[str, dict[str, Any]]:
    return {
        slug: json.loads((root / pack.manifest["models"][slug]["summary"]).read_text(encoding="utf-8"))
        for slug in pack.slugs
    }


def run_verify(
    pack_dir: Path = DEFAULT_PACK_DIR,
    root: Path = ROOT,
    results_file: Path | None = RESULTS_FILE,
) -> tuple[int, dict[str, Any]]:
    problems = check_manifest(pack_dir, root)
    report: dict[str, Any] = {"manifest_problems": problems}
    if problems:
        report["status"] = "manifest_mismatch"
        return EXIT_MANIFEST, report

    pack = load_pack(pack_dir)
    summaries = summaries_for(pack, root)
    metadata = []
    per_task = []
    for slug, summary in summaries.items():
        if summary.get("model_id") != pack.model_id(slug):
            metadata.append(f"{slug}: summary model_id {summary.get('model_id')!r} is not {pack.model_id(slug)!r}")
        stored_tasks = summary.get("tasks") or {}
        if sorted(stored_tasks) != sorted(pack.tasks):
            metadata.append(f"{slug}: summary tasks {sorted(stored_tasks)} are not the pack tasks {sorted(pack.tasks)}")
        for task in pack.tasks:
            if task not in stored_tasks:
                continue
            ids = [prediction.item_id for prediction in pack.predictions[(slug, task)]]
            if ids != [example["item_id"] for example in pack.examples[task]]:
                metadata.append(f"{slug}.{task}: prediction rows do not match label rows")
                continue
            result = compare_task(stored_tasks[task], recompute_task(pack, slug, task, stored_tasks[task]), f"{slug}.{task}")
            per_task.append({"model": slug, "task": task, **result})

    mismatches = [item for result in per_task for item in result["mismatches"]]
    report.update(
        {
            "summaries": len(summaries),
            "task_blocks": len(per_task),
            "values_stored": sum(result["values"] for result in per_task),
            "values_compared_exactly": sum(result["compared_exactly"] for result in per_task),
            "values_verified_in_aggregate": sum(result["verified_in_aggregate"] for result in per_task),
            "token_and_cost_values_compared": sum(result["token_and_cost_values"] for result in per_task),
            "models_without_stored_token_counts": sorted(
                {result["model"] for result in per_task if not result["token_and_cost_values"]}
            ),
            "run_only_keys_not_compared": sorted(RUN_ONLY_KEYS),
            "metadata_problems": metadata,
            "mismatches": len(mismatches),
            "mismatch_details": mismatches[:50],
            "per_task": [
                {key: value for key, value in result.items() if key != "mismatches"} | {"mismatches": len(result["mismatches"])}
                for result in per_task
            ],
        }
    )
    if results_file is not None:
        from write_comparison_report import render

        expected = render(summaries, pack)
        committed = results_file.read_text(encoding="utf-8").replace("\r\n", "\n") if results_file.is_file() else None
        report["results_file"] = {
            "path": results_file.relative_to(root).as_posix() if results_file.is_relative_to(root) else results_file.name,
            "matches_regenerated": committed == expected,
        }
    failed = bool(mismatches or metadata) or (results_file is not None and not report["results_file"]["matches_regenerated"])
    report["status"] = "mismatch" if failed else "pass"
    return (EXIT_MISMATCH if failed else EXIT_OK), report


# ---------------------------------------------------------------- report


def rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, digits)
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def label_space(task: str, field: str) -> list[str]:
    return [_key(value) for value in field_label_space(task, field)]


def model_metrics(task: str, field: str, view: Any, bins: int) -> dict[str, Any]:
    space = field_label_space(task, field)
    n = len(view)
    top = view.top_label_probability
    reported = view.reported_confidence
    return {
        "items": n,
        "accuracy": sum(view.correct) / n,
        "macro_f1_union": macro_f1_union(view.reference, view.predicted),
        "macro_f1_fixed": macro_f1_fixed(view.reference, view.predicted, space),
        "ece_top_label_width": ece_equal_width(top, view.correct, bins),
        "ece_top_label_mass": ece_equal_mass(top, view.correct, bins),
        "ece_confidence_width": ece_equal_width(reported, view.correct, bins),
        "ece_confidence_width_10_bins": ece_equal_width(reported, view.correct, 10),
        "mean_top_label_probability": sum(top) / n,
        "mean_reported_confidence": sum(reported) / n,
        "log_loss": negative_log_likelihood(view.reference, view.probabilities),
        "brier_multiclass": multiclass_brier(view.reference, view.probabilities),
        "zero_probability_true_labels": zero_probability_true_labels(view.reference, view.probabilities),
        "risk_coverage_top_label": risk_coverage(top, view.correct),
        "top_label_brier_decomposition": binned_brier_decomposition(top, view.correct, bins),
        "reliability_equal_width": reliability_bins(top, view.correct, bins, "width"),
        "reliability_equal_mass": reliability_bins(top, view.correct, bins, "mass"),
    }


def bootstrap_for_field(
    pack: ReplayPack,
    task: str,
    field: str,
    views: dict[str, Any],
    estimates: dict[str, dict[str, float]],
    *,
    stream: int,
    resamples: int,
    seed: int,
    bins: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    space = label_space(task, field)
    arrays = {}
    for slug, view in views.items():
        reference = [_key(value) for value in view.reference]
        unknown = sorted(set(reference) - set(space))
        if unknown:
            raise ValueError(f"{task}.{field}: reference labels outside the label space: {unknown}")
        arrays[slug] = ModelArrays(
            reference=encode(reference, space),
            predicted=encode([_key(value) for value in view.predicted], space),
            confidence=np.asarray(view.top_label_probability, dtype=float),
        )
    distributions = bootstrap_field(arrays, len(space), resamples=resamples, seed=seed, stream=stream, bins=bins)
    intervals = {}
    for slug, metrics in distributions.items():
        intervals[slug] = {}
        for metric, samples in metrics.items():
            low, high = percentile_interval(samples, CONFIDENCE_LEVEL)
            intervals[slug][metric] = {"estimate": estimates[slug][metric], "low": low, "high": high}
    baseline = BASELINE if BASELINE in views else None
    return intervals, paired_differences(distributions, estimates, baseline, CONFIDENCE_LEVEL)


def label_source_sensitivity(pack: ReplayPack, task: str, field: str) -> dict[str, Any]:
    rows = pack.labels[task]
    sources = {"consensus": "reference", "judge_a": "judge_a", "judge_b": "judge_b"}
    accuracy: dict[str, dict[str, float]] = {}
    items = {}
    for slug in pack.slugs:
        by_id = {prediction.item_id: prediction for prediction in pack.predictions[(slug, task)]}
        accuracy[slug] = {}
        for name, key in sources.items():
            pairs = [
                (row["fields"][field][key], by_id[row["item_id"]].fields[field].predicted)
                for row in rows
                if row["fields"][field].get(key) is not None
            ]
            items[name] = len(pairs)
            accuracy[slug][name] = sum(_key(ref) == _key(pred) for ref, pred in pairs) / len(pairs) if pairs else math.nan
    ranked = [slug for slug in pack.slugs if slug != BASELINE]
    flips = []
    for a, b in combinations(ranked, 2):
        signs = {name: np.sign(accuracy[a][name] - accuracy[b][name]) for name in sources}
        if len({value for value in signs.values() if value != 0}) > 1:
            flips.append({"a": a, "b": b, **{name: float(accuracy[a][name] - accuracy[b][name]) for name in sources}})
    return {"items": items, "accuracy": accuracy, "rank_flips": flips, "pairs_compared": len(list(combinations(ranked, 2)))}


def accuracy_by_stratum(pack: ReplayPack, task: str, field: str, views: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slug, view in views.items():
        grouped: dict[str, list[bool]] = defaultdict(list)
        for stratum, correct in zip(view.strata, view.correct, strict=True):
            grouped[stratum or "unknown"].append(correct)
        for stratum, hits in sorted(grouped.items()):
            out.setdefault(stratum, {"items": len(hits)})[slug] = sum(hits) / len(hits)
    return out


def agreement_by_stratum(pack: ReplayPack, task: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pack.labels[task]:
        grouped[row["stratum"] or "unknown"].append(row)
    out = {}
    for stratum, rows in sorted(grouped.items()):
        agree = [
            {field: _key(row["fields"][field]["judge_a"]) == _key(row["fields"][field]["judge_b"]) for field in TASK_FIELDS[task]}
            for row in rows
        ]
        out[stratum] = {
            "items": len(rows),
            "complete_agreement": sum(all(item.values()) for item in agree) / len(rows),
            **{field: sum(item[field] for item in agree) / len(rows) for field in TASK_FIELDS[task]},
        }
    return out


def gold_position(label: Any) -> str:
    text = _key(label)
    prefix, _, suffix = text.rpartition("_")
    return suffix if prefix in {"brand", "category"} and suffix.isdigit() else text


def accuracy_by_gold_position(pack: ReplayPack, views: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slug, view in views.items():
        grouped: dict[str, list[bool]] = defaultdict(list)
        for reference, correct in zip(view.reference, view.correct, strict=True):
            grouped[gold_position(reference)].append(correct)
        for position, hits in grouped.items():
            out.setdefault(position, {"items": len(hits)})[slug] = sum(hits) / len(hits)
    return dict(sorted(out.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else 0, item[0])))


def probability_diagnostics(pack: ReplayPack) -> list[dict[str, Any]]:
    rows = []
    for slug in pack.slugs:
        for task in pack.tasks:
            for field in TASK_FIELDS[task]:
                ties = mismatches = 0
                predictions = pack.predictions[(slug, task)]
                for prediction in predictions:
                    values = prediction.fields[field].probabilities
                    top = max(values.values())
                    ties += sum(value == top for value in values.values()) > 1
                    mismatches += float(values.get(_key(prediction.fields[field].predicted), 0.0)) < top
                rows.append({"model": slug, "task": task, "field": field, "items": len(predictions), "ties_at_max": ties, "argmax_mismatches": mismatches})
    return rows


def build_report(pack: ReplayPack, *, resamples: int, seed: int, bins: int) -> dict[str, Any]:
    from write_comparison_report import FIELD_ROWS

    fields = {}
    for stream, (task, field, title) in enumerate(FIELD_ROWS):
        if task not in pack.labels:
            continue
        views = {slug: field_view(pack, slug, task, field) for slug in pack.slugs}
        metrics = {slug: model_metrics(task, field, view, bins) for slug, view in views.items()}
        degenerate = is_degenerate(pack.examples[task], field)
        share, top, labelled = majority_share(pack.examples[task], field)
        entry: dict[str, Any] = {
            "task": task,
            "field": field,
            "title": title,
            "degenerate": degenerate,
            "majority_class_share": share,
            "majority_class_items": top,
            "labelled_items": labelled,
            "bootstrap_stream": stream,
            "models": metrics,
        }
        if not degenerate:
            estimates = {slug: {metric: metrics[slug][metric] for metric in METRICS} for slug in views}
            intervals, differences = bootstrap_for_field(
                pack, task, field, views, estimates, stream=stream, resamples=resamples, seed=seed, bins=bins
            )
            entry["intervals"] = intervals
            entry["paired_differences"] = differences
        if task != "esci":
            entry["label_source_sensitivity"] = label_source_sensitivity(pack, task, field)
        if (task, field) in STRATUM_FIELDS:
            entry["accuracy_by_stratum"] = accuracy_by_stratum(pack, task, field, views)
        if (task, field) in POSITION_FIELDS:
            entry["accuracy_by_gold_position"] = accuracy_by_gold_position(pack, views)
        if (task, field) in SNAPSHOT_FIELDS:
            entry["confidence_semantics"] = {
                slug: confidence_semantics(view.reported_confidence, view.probabilities) for slug, view in views.items()
            }
        fields[f"{task}.{field}"] = entry
    manifest = pack.manifest
    return {
        "pack": {
            "build_date": manifest["build_date"],
            "builder_commit": manifest["builder"]["commit"],
            "prompt_version": manifest["prompt_version"],
        },
        "settings": {
            "bins": bins,
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "interval": "percentile",
            "level": CONFIDENCE_LEVEL,
            "baseline": BASELINE,
        },
        "models": {slug: {"model_id": pack.model_id(slug), "role_label": manifest["models"][slug]["role_label"]} for slug in pack.slugs},
        "fields": fields,
        "agreement_by_stratum": {task: agreement_by_stratum(pack, task) for task in pack.tasks if task != "esci"},
        "probability_diagnostics": probability_diagnostics(pack),
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def with_interval(cell: dict[str, float], kind: str) -> str:
    if kind == "pct":
        return f"{pct(cell['estimate'])} [{pct(cell['low'])}, {pct(cell['high'])}]"
    return f"{num(cell['estimate'])} [{num(cell['low'])}, {num(cell['high'])}]"


def signed(value: float, kind: str) -> str:
    return f"{100 * value:+.1f} pp" if kind == "pct" else f"{value:+.3f}"


def difference_cell(row: dict[str, Any], kind: str) -> str:
    return f"{signed(row['estimate'], kind)} [{signed(row['low'], kind)}, {signed(row['high'], kind)}]"


def render_report(data: dict[str, Any]) -> str:
    from write_comparison_report import LAYA_NOTE

    settings = data["settings"]
    bins = settings["bins"]
    models = data["models"]
    lines = [
        "# Replay Results",
        "",
        "Generated by `python replay_scores.py report` from the replay pack in `replay/`. Do not edit by hand.",
        "",
        f"- Pack built {data['pack']['build_date']} when the repository HEAD was `{data['pack']['builder_commit']}`; "
        f"prompt version `{data['pack']['prompt_version']}`.",
        f"- Intervals: {int(100 * settings['level'])}% percentile intervals from a paired bootstrap over items, "
        f"{settings['bootstrap_resamples']:,} resamples, seed {settings['bootstrap_seed']}. Every model on a field "
        "is resampled with the same item indices.",
        f"- Calibration: top-label ECE with {bins} bins, from the probability each model assigned to the label it returned.",
        "",
        "## Models",
        "",
        "| Model | Model ID | Role label |",
        "|---|---|---|",
        *[f"| `{slug}` | `{meta['model_id']}` | {meta['role_label']} |" for slug, meta in models.items()],
        "",
        LAYA_NOTE,
        "",
        "## Definitions",
        "",
        "- **Accuracy:** share of labelled items whose returned label equals the reference label.",
        "- **Macro-F1 (fixed):** per-class F1 averaged over every class in the field's label space; a class that never "
        "occurs scores 0. **Macro-F1 (union)** is the `summary.json` definition, averaged over classes present in the "
        "reference or predicted labels.",
        "- Accuracy and macro-F1 are computed at full precision and rounded once for display; `summary.json` stores "
        "the same values rounded to 4 decimals.",
        f"- **Top-label ECE:** expected calibration error of the top-label probability; equal-width uses {bins} bins on "
        f"[0, 1] (first bin closed below), equal-mass uses {bins} groups of near-equal size after a stable sort, "
        f"using `numpy.array_split`, so the first n mod {bins} groups hold one extra item.",
        "- **ECE on reported `confidence`:** the same equal-width estimator applied to the adapter's `confidence` field. "
        "Adapters define that field differently (see confidence semantics below); it is a comparison column only.",
        "- **Log loss:** mean negative log probability of the reference label, floored at 1e-12. "
        "**Brier:** multiclass Brier score over the label space.",
        "- **AURC:** area under the risk-coverage curve, items ranked by top-label probability (highest first, ties in pack order).",
        "",
    ]

    def snapshot_rows(keys: list[str]) -> list[str]:
        out = [
            "| Field | Model | n | Accuracy [95% interval] | Macro-F1, fixed [95% interval] "
            f"| Top-label ECE, {bins} equal-width [95% interval] | Log loss |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for key in keys:
            entry = data["fields"][key]
            for slug, metrics in entry["models"].items():
                cells = entry["intervals"][slug]
                out.append(
                    f"| {entry['title']} | {models[slug]['role_label']} | {metrics['items']:,} "
                    f"| {with_interval(cells['accuracy'], 'pct')} | {with_interval(cells['macro_f1_fixed'], '3f')} "
                    f"| {with_interval(cells['ece_top_label_width'], '3f')} | {num(metrics['log_loss'])} |"
                )
        return out

    def present(pairs: tuple[tuple[str, str], ...]) -> list[str]:
        return [f"{task}.{field}" for task, field in pairs if f"{task}.{field}" in data["fields"]]

    snapshot = present(SNAPSHOT_FIELDS)
    headline = [key for key, entry in data["fields"].items() if not entry["degenerate"]]
    degenerate = [key for key, entry in data["fields"].items() if entry["degenerate"]]
    lines += ["## Snapshot", "", *snapshot_rows(snapshot), ""]

    lines += [
        "## Classification",
        "",
        "| Field | Model | n | Accuracy | Macro-F1 (union) | Macro-F1 (fixed) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in headline + degenerate:
        entry = data["fields"][key]
        for slug, metrics in entry["models"].items():
            lines.append(
                f"| {entry['title']} | `{slug}` | {metrics['items']:,} | {pct(metrics['accuracy'])} "
                f"| {num(metrics['macro_f1_union'])} | {num(metrics['macro_f1_fixed'])} |"
            )
    lines += [
        "",
        "## Probability quality",
        "",
        f"| Field | Model | Top-label ECE ({bins} equal-width) | Top-label ECE ({bins} equal-mass) "
        f"| ECE on reported `confidence` ({bins} equal-width) | Mean top-label probability "
        "| Mean reported `confidence` | Log loss | Brier | AURC | Reference label at probability 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in headline + degenerate:
        entry = data["fields"][key]
        for slug, metrics in entry["models"].items():
            lines.append(
                f"| {entry['title']} | `{slug}` | {num(metrics['ece_top_label_width'])} | {num(metrics['ece_top_label_mass'])} "
                f"| {num(metrics['ece_confidence_width'])} | {num(metrics['mean_top_label_probability'])} "
                f"| {num(metrics['mean_reported_confidence'])} | {num(metrics['log_loss'])} | {num(metrics['brier_multiclass'])} "
                f"| {num(metrics['risk_coverage_top_label']['aurc'])} | {metrics['zero_probability_true_labels']:,} |"
            )

    lines += [
        "",
        "## Paired differences",
        "",
        "Each row is model A minus model B on identical bootstrap resamples. For ECE, a negative difference means "
        "A's top-label ECE is lower.",
        "",
    ]
    counts: Counter[tuple[str, bool]] = Counter()
    for key in headline:
        entry = data["fields"][key]
        grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in entry["paired_differences"]:
            grouped[(row["a"], row["b"])][row["metric"]] = row
            counts[(row["metric"], row["excludes_zero"])] += 1
        lines += [
            f"### {entry['title']}",
            "",
            "| A | B | Accuracy | Macro-F1 (fixed) | Top-label ECE |",
            "|---|---|---:|---:|---:|",
        ]
        for (a, b), metrics in grouped.items():
            lines.append(
                f"| `{a}` | `{b}` | {difference_cell(metrics['accuracy'], 'pct')} "
                f"| {difference_cell(metrics['macro_f1_fixed'], '3f')} | {difference_cell(metrics['ece_top_label_width'], '3f')} |"
            )
        lines.append("")
    lines += ["| Metric | Comparisons | Interval excludes 0 |", "|---|---:|---:|"]
    for metric in METRICS:
        total = counts[(metric, True)] + counts[(metric, False)]
        lines.append(f"| `{metric}` | {total} | {counts[(metric, True)]} |")

    lines += [
        "",
        "## Label-source sensitivity",
        "",
        "Accuracy against the field consensus (items where the two judges agreed) and against each judge's labels "
        "on all items. A rank flip is a pair of non-baseline models whose accuracy order differs between label sources.",
        "",
        "| Field | Model | Consensus | Judge A | Judge B |",
        "|---|---|---:|---:|---:|",
    ]
    flip_lines = []
    for key in headline + degenerate:
        entry = data["fields"][key]
        sensitivity = entry.get("label_source_sensitivity")
        if not sensitivity:
            continue
        for slug, values in sensitivity["accuracy"].items():
            lines.append(
                f"| {entry['title']} | `{slug}` | {pct(values['consensus'])} | {pct(values['judge_a'])} | {pct(values['judge_b'])} |"
            )
        flip_lines.append(
            f"- **{entry['title']}:** {len(sensitivity['rank_flips'])} of {sensitivity['pairs_compared']} pairs"
            + (": " + "; ".join(f"`{flip['a']}` and `{flip['b']}`" for flip in sensitivity["rank_flips"]) if sensitivity["rank_flips"] else "")
            + "."
        )
    lines += ["", "Rank flips:", "", *flip_lines, ""]

    lines += ["## Accuracy by sampling stratum", ""]
    for key in present(STRATUM_FIELDS):
        entry = data["fields"][key]
        slugs = list(entry["models"])
        lines += [
            f"### {entry['title']}",
            "",
            "| Stratum | n | " + " | ".join(f"`{slug}`" for slug in slugs) + " |",
            "|---|---:|" + "---:|" * len(slugs),
        ]
        for stratum, values in entry["accuracy_by_stratum"].items():
            lines.append(f"| `{stratum}` | {values['items']:,} | " + " | ".join(pct(values.get(slug)) for slug in slugs) + " |")
        lines.append("")

    lines += ["## Judge agreement by sampling stratum", ""]
    for task, strata in data["agreement_by_stratum"].items():
        fields = TASK_FIELDS[task]
        lines += [
            f"### {task}",
            "",
            "| Stratum | n | All fields | " + " | ".join(f"`{field}`" for field in fields) + " |",
            "|---|---:|---:|" + "---:|" * len(fields),
        ]
        for stratum, values in strata.items():
            lines.append(
                f"| `{stratum}` | {values['items']:,} | {pct(values['complete_agreement'])} | "
                + " | ".join(pct(values[field]) for field in fields)
                + " |"
            )
        lines.append("")

    lines += ["## Accuracy by position of the reference option", ""]
    for key in present(POSITION_FIELDS):
        entry = data["fields"][key]
        slugs = list(entry["models"])
        lines += [
            f"### {entry['title']}",
            "",
            "| Reference option | n | " + " | ".join(f"`{slug}`" for slug in slugs) + " |",
            "|---|---:|" + "---:|" * len(slugs),
        ]
        for position, values in entry["accuracy_by_gold_position"].items():
            label = f"position {position}" if position.isdigit() else f"`{position}`"
            lines.append(f"| {label} | {values['items']:,} | " + " | ".join(pct(values.get(slug)) for slug in slugs) + " |")
        lines.append("")

    lines += [
        "## Probability-vector diagnostics",
        "",
        "Over all items, labelled or not. A tie is a probability vector whose maximum is shared by two or more labels; "
        "an argmax mismatch is a returned label whose probability is below the vector's maximum.",
        "",
        "| Model | Field | Items | Ties at the maximum | Argmax mismatches |",
        "|---|---|---:|---:|---:|",
    ]
    nonzero = [row for row in data["probability_diagnostics"] if row["ties_at_max"] or row["argmax_mismatches"]]
    for row in nonzero:
        lines.append(f"| `{row['model']}` | {row['task']}.{row['field']} | {row['items']:,} | {row['ties_at_max']:,} | {row['argmax_mismatches']:,} |")
    lines += ["", f"All other model and field pairs ({len(data['probability_diagnostics']) - len(nonzero)}): no ties and no mismatches.", ""]

    lines += [
        "## Confidence semantics",
        "",
        "Share of labelled items whose reported `confidence` is within 0.01 of each formula, where p is the top-label "
        "probability and k the number of options: p itself; the rescaled margin (p - 1/k) / (1 - 1/k); "
        "and 1 minus the normalised entropy of the probability vector.",
        "",
        "| Field | Model | p | Rescaled margin | 1 - normalised entropy |",
        "|---|---|---:|---:|---:|",
    ]
    for key in snapshot:
        entry = data["fields"][key]
        for slug, shares in entry["confidence_semantics"].items():
            lines.append(
                f"| {entry['title']} | `{slug}` | {pct(shares['top_label_probability'])} | {pct(shares['rescaled_margin'])} "
                f"| {pct(shares['normalised_entropy'])} |"
            )
    lines.append("")

    for key in degenerate:
        entry = data["fields"][key]
        lines += [
            f"## Degenerate field: {entry['title']}",
            "",
            f"One class holds {entry['majority_class_items']:,} of {entry['labelled_items']:,} labelled items "
            f"({100 * entry['majority_class_share']:.2f}%). The field appears in the classification, probability-quality and "
            "label-source tables above with no intervals or paired differences, and it is not in the snapshot.",
            "",
        ]

    lines += [
        "## What the pack can and cannot verify",
        "",
        "- `replay_scores.py verify` recomputes every value stored in each `summary.json` task block except run metadata "
        "(" + ", ".join(f"`{key}`" for key in sorted(RUN_ONLY_KEYS)) + "), and compares it at stored precision.",
        "- Query-segmentation items sampled from ORCAS-I carry the stratum `orcas_i`. The summaries break their accuracy "
        "out by ORCAS-I category; the pack does not carry that category, so those entries are verified in aggregate "
        "(item and correct counts summed over categories), not one by one.",
        "- Token counts and cost are compared only where a summary stores them.",
        "- Latency values are recomputed from each prediction's recorded `elapsed_ms`. They describe each run on its own "
        "hardware and batch size, not a controlled latency comparison.",
        "- The pack holds no query or product text, so prompts, model inputs and judge rationales cannot be inspected from it.",
        "",
    ]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("verify", "report", "figures"))
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    if args.command == "verify":
        code, report = run_verify(args.pack_dir)
        write_text(args.output_dir / "verify_report.json", json.dumps(report, indent=2) + "\n")
        print(
            f"verify: {report['status']}; values stored {report.get('values_stored', 0):,}; "
            f"compared exactly {report.get('values_compared_exactly', 0):,}; "
            f"verified in aggregate {report.get('values_verified_in_aggregate', 0):,}; "
            f"mismatches {report.get('mismatches', 0)}; manifest problems {len(report['manifest_problems'])}"
        )
        for problem in report["manifest_problems"] + report.get("metadata_problems", []):
            print(f"  {problem}")
        for item in report.get("mismatch_details", [])[:10]:
            print(f"  {item['path']}: stored {item['stored']!r}, recomputed {item['recomputed']!r}")
        if "results_file" in report and not report["results_file"]["matches_regenerated"]:
            print(f"  {report['results_file']['path']} differs from write_comparison_report.py output")
    elif args.command == "report":
        problems = check_manifest(args.pack_dir, ROOT)
        if problems:
            print("report: pack does not match its manifest; run verify", *problems, sep="\n  ")
            return EXIT_MANIFEST
        data = build_report(load_pack(args.pack_dir), resamples=args.resamples, seed=args.seed, bins=args.bins)
        write_text(args.output_dir / "replay_metrics.json", json.dumps(rounded(data), indent=2) + "\n")
        write_text(args.output_dir / "REPLAY_RESULTS.md", render_report(rounded(data)))
        print(f"report: wrote {(args.output_dir / 'REPLAY_RESULTS.md').name} and replay_metrics.json")
        code = EXIT_OK
    else:
        import make_figures

        code = make_figures.main(["--pack-dir", str(args.pack_dir), "--output-dir", str(args.output_dir / "figures")])
    print(f"{args.command}: {time.perf_counter() - started:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
