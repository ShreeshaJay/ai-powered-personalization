"""Build the text-free replay pack in `replay/` from the private benchmark workspace.

    python build_replay_pack.py --private-root PATH      # or set APPE_PRIVATE_ROOT

The private root is a benchmark workspace laid out like this folder, holding
`references/` (reference manifests, `consensus/`, `esci_eval_slice.jsonl`),
`outputs/benchmark/<model>/predictions/<task>.jsonl` and
`outputs/adjudication_cache.sqlite`. Those inputs contain query and product text
and are not published. The pack keeps item IDs, reference and judge labels,
predictions, probability vectors, reported confidence, input token counts and
elapsed time; nothing else.

The pack is written only if all of these hold:
  1. the private predictions reproduce every committed `summary.json` value
     (the `replay_scores.py verify` check, run on the new pack);
  2. every item ID matches a fixed pattern and every other string in the pack
     belongs to a closed vocabulary of task, field, label, stratum and
     label-source names;
  3. no ORCAS-I intent category name appears anywhere in the pack.
Run it twice to confirm the output is byte-identical.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import adjudicate_pilots
import eval.datasets as datasets
from adapters.schemas import PROMPT_VERSION, TASK_FIELDS, field_label_space
from eval.metrics import _key
from eval.replay_pack import (
    DEFAULT_PACK_DIR,
    MANIFEST_NAME,
    ORCAS_PREFIX,
    ORCAS_STRATUM,
    PROVENANCE_NAME,
    check_manifest,
    collapse_stratum,
    jsonl_gz_bytes,
    sha256_bytes,
    sha256_lf,
)
from write_comparison_report import MODEL_SLUGS, summary_path

ROOT = Path(__file__).resolve().parent
PRIVATE_ROOT_ENV = "APPE_PRIVATE_ROOT"
CACHE_NAME = Path("outputs") / "adjudication_cache.sqlite"
ITEM_ID = re.compile(r"^[a-z][a-z_]*_[0-9]+$")
JUDGES = {"judge_a": "anthropic", "judge_b": "openai"}
PREDICTION_KEYS = ("predicted", "probabilities", "confidence")
KNOWN_STRATA = {
    "generated_cross_sell",
    "generated_direct_accessory",
    "human_irrelevant_negative",
    "human_substitute_hard_negative",
    "out_of_category_negative",
    "esci_single_product",
    "esci_accessory_for_product",
    "esci_bundle_or_set",
    "esci_broad_gift_intent",
    "esci_ambiguous",
    "esci_unknown",
    ORCAS_STRATUM,
    "explicit_brand_with_category",
    "generic_query_with_category",
    "ambiguous_or_unmapped",
    "human",
}
ROLE_LABELS = {
    "majority": "Majority prior",
    "laya_typed_decisions": "Encoder decision model",
    "jev_1_13_0": "Hosted typed-decision API",
    "kev_0_8b": "Open adapter, small",
    "kev_4b": "Open adapter, mid",
    "jevlite": "Option-letter adapter",
}
LABEL_SOURCES = {
    "dual_judge_consensus": (
        "Field consensus of two LLM judges (judge_a and judge_b) under the listed rubric. "
        "A field is labelled only where both judges returned the same value; otherwise its reference is null "
        "and the field is not scored for that item."
    ),
    "human": "The ESCI human relevance labels (esci_label: E, S, C or I) from the Amazon Shopping Queries Dataset.",
}
LICENCES = [
    {
        "source": "Amazon Shopping Queries Dataset (ESCI)",
        "url": "https://github.com/amazon-science/esci-data",
        "licence": "Apache-2.0",
        "use": (
            "Queries and products behind the compatibility, brand/category and ESCI items, 1,000 of the 2,000 "
            "query-segmentation queries, and the ESCI human labels for the ESCI slice. No query or product text is "
            "in this pack."
        ),
    },
    {
        "source": "ESCI-S product metadata",
        "url": "https://github.com/shuttie/esci-s",
        "licence": "Apache-2.0",
        "licence_checked": (
            "2026-11-09: the repository LICENSE file is the Apache License 2.0 and the README 'License' section states "
            "Apache-2.0. The README 'Methodology' section states that the product metadata was scraped from the "
            "product pages."
        ),
        "use": (
            "US category paths. The brand/category task's category options, shown to models and judges, are ESCI-S "
            "category paths; this pack stores only the option IDs (category_0 to category_7, other_or_ambiguous)."
        ),
    },
    {
        "source": "ORCAS-I gold (Alexander, Kusa and de Vries, SIGIR 2022)",
        "licence": "CC BY-NC 4.0",
        "use": (
            "1,000 of the 2,000 query-segmentation queries were sampled from ORCAS-I. The queries and their ORCAS-I "
            "intent labels are not redistributed: those items carry the single stratum value orcas_i, and the "
            "ORCAS-I intent label is not in this pack. The judge labels and model predictions on those items are "
            "this project's outputs. The committed summary.json files report query-segmentation accuracy "
            "aggregated by ORCAS-I intent category; those aggregates derive from ORCAS-I and are credited under "
            "CC BY-NC 4.0."
        ),
    },
    {
        "source": "GPT-5.1 enrichment of ESCI queries",
        "licence": "OpenAI model output; not redistributed except as the stratum names below.",
        "use": (
            "Query taxonomy, complement candidates and category matches used to form sampling strata and candidate "
            "pools. The only enrichment-derived values in this pack are stratum names: each esci_* stratum is the "
            "query's GPT-5.1 product-type intent, and several compatibility and brand/category strata name pools "
            "built from the enrichment. No enrichment text is in this pack, and none was shown to models or judges "
            "except the ESCI-S category paths it matched."
        ),
    },
    {
        "source": (
            "Judge labels: judge_a (Anthropic claude-opus-5), judge_b (OpenAI gpt-5.6-sol), and their field consensus"
        ),
        "licence": (
            "AI-generated outputs of the two named models, released with this repository as evaluation labels; "
            "Anthropic Commercial Terms and OpenAI Services Agreement and Sharing and Publication Policy checked "
            "2026-11-09."
        ),
        "use": "Reference labels for compatibility, query segmentation and brand/category.",
    },
]


def resolve_private_root(value: Path | None) -> Path:
    raw = value or (Path(os.environ[PRIVATE_ROOT_ENV]) if os.environ.get(PRIVATE_ROOT_ENV) else None)
    if raw is None:
        raise SystemExit(f"Pass --private-root or set {PRIVATE_ROOT_ENV}.")
    root = raw.expanduser().resolve()
    required = [root / "references" / "consensus", root / "references" / "esci_eval_slice.jsonl", root / CACHE_NAME]
    required += [root / "outputs" / "benchmark" / slug / "predictions" for slug in MODEL_SLUGS]
    missing = [path.relative_to(root).as_posix() for path in required if not path.exists()]
    if missing:
        raise SystemExit("Private root is missing: " + ", ".join(missing))
    return root


def use_private_inputs(private_root: Path) -> None:
    """Point the committed loaders at the private reference files."""
    references = private_root / "references"
    reference_files = adjudicate_pilots.DATASET_TASK_FILES["reference"]
    for task, path in list(reference_files.items()):
        reference_files[task] = references / Path(path).name
    datasets.CONSENSUS_DIRS["reference"] = references / "consensus"
    datasets.ESCI_SLICE_PATH = references / "esci_eval_slice.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def head_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def label_rows(task: str, examples: list[dict[str, Any]], private_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Text-free label rows in reference-manifest order, and the count of consensus-rule violations."""
    if task == "esci":
        rows = [
            {
                "item_id": example["item_id"],
                "task": task,
                "stratum": example["sampling_context"].get("source_stratum"),
                "label_source": example["label_source"],
                "fields": {"esci_label": {"reference": example["labels"]["esci_label"]}},
            }
            for example in examples
        ]
        return rows, 0
    judged = {row["item_id"]: row for row in read_jsonl(private_root / "references" / "consensus" / f"{task}_consensus.jsonl")}
    rows = []
    violations = 0
    for example in examples:
        judges = judged[example["item_id"]]
        fields = {}
        for field in TASK_FIELDS[task]:
            reference = example["labels"].get(field)
            judge_a = (judges.get(JUDGES["judge_a"]) or {}).get(field)
            judge_b = (judges.get(JUDGES["judge_b"]) or {}).get(field)
            expected = judge_a if judge_a is not None and _key(judge_a) == _key(judge_b) else None
            if (reference is None) != (expected is None) or (reference is not None and _key(reference) != _key(expected)):
                violations += 1
            fields[field] = {"reference": reference, "judge_a": judge_a, "judge_b": judge_b}
        rows.append(
            {
                "item_id": example["item_id"],
                "task": task,
                "stratum": collapse_stratum(example["sampling_context"].get("source_stratum")),
                "label_source": example["label_source"],
                "fields": fields,
            }
        )
    return rows, violations


def prediction_rows(slug: str, task: str, order: list[str], private_root: Path) -> list[dict[str, Any]]:
    path = private_root / "outputs" / "benchmark" / slug / "predictions" / f"{task}.jsonl"
    source = read_jsonl(path)
    by_id = {row["item_id"]: row for row in source}
    if len(by_id) != len(source) or set(by_id) != set(order):
        raise SystemExit(f"{slug}/{task}: prediction item IDs do not match the reference manifest")
    rows = []
    for item_id in order:
        row = by_id[item_id]
        if set(row["fields"]) != set(TASK_FIELDS[task]):
            raise SystemExit(f"{slug}/{task}/{item_id}: prediction fields differ from TASK_FIELDS")
        rows.append(
            {
                "item_id": item_id,
                "task": row["task"],
                "elapsed_ms": row["elapsed_ms"],
                "input_tokens": row["input_tokens"],
                "fields": {field: {key: row["fields"][field][key] for key in PREDICTION_KEYS} for field in TASK_FIELDS[task]},
            }
        )
    return rows


def vocabulary() -> set[str]:
    allowed = set(MODEL_SLUGS) | set(TASK_FIELDS) | set(LABEL_SOURCES)
    allowed |= {"item_id", "task", "stratum", "label_source", "fields", "reference", *JUDGES, "elapsed_ms", "input_tokens", *PREDICTION_KEYS}
    for task, fields in TASK_FIELDS.items():
        allowed |= set(fields)
        for field in fields:
            allowed |= {_key(value) for value in field_label_space(task, field)}
    allowed |= KNOWN_STRATA
    return allowed


def scan_strings(files: dict[str, list[dict[str, Any]]], allowed: set[str]) -> Counter[str]:
    """Count strings outside the closed vocabulary; item IDs must match ITEM_ID."""
    unexpected: Counter[str] = Counter()

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                if name not in allowed:
                    unexpected["key"] += 1
                walk(item, name)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif isinstance(value, str):
            ok = bool(ITEM_ID.match(value)) if key == "item_id" else value in allowed
            if not ok:
                unexpected["item_id" if key == "item_id" else "value"] += 1

    for rows in files.values():
        for row in rows:
            walk(row)
    return unexpected


def orcas_terms_found(pack_dir: Path, terms: set[str]) -> int:
    """Occurrences of ORCAS-I category names in any pack file (decompressed, case-insensitive)."""
    found = 0
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        text = (gzip.decompress(data) if path.name.endswith(".gz") else data).decode("utf-8").lower()
        found += sum(text.count(term) for term in terms)
    return found


def label_provenance(cache: Path, labels: dict[str, list[dict[str, Any]]], violations: dict[str, int]) -> dict[str, Any]:
    """Aggregate label-generation counts from a temporary copy of the adjudication cache.

    Only grouped counts, token totals, costs and timestamps are read; label JSON,
    raw responses and error text are never selected.
    """
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / cache.name
        shutil.copyfile(cache, copy)
        wal = cache.with_name(cache.name + "-wal")
        if wal.is_file():
            shutil.copyfile(wal, copy.with_name(copy.name + "-wal"))
        connection = sqlite3.connect(f"file:{copy.as_posix()}?mode=ro", uri=True)
        try:
            generated = [
                dict(zip(("task", "provider", "model", "dataset", "labels", "input_tokens", "output_tokens", "cost_usd", "first_created_at", "last_created_at"), row))
                for row in connection.execute(
                    "SELECT task, provider, model, "
                    "CASE WHEN item_id LIKE '%\\_ref\\_%' ESCAPE '\\' THEN 'reference' ELSE 'pilot' END, "
                    "COUNT(*), SUM(input_tokens), SUM(output_tokens), ROUND(SUM(cost_usd), 4), MIN(created_at), MAX(created_at) "
                    "FROM labels GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 4"
                )
            ]
            calls = [
                dict(zip(("provider", "status", "calls", "cost_usd"), row))
                for row in connection.execute(
                    "SELECT provider, status, COUNT(*), ROUND(SUM(cost_usd), 4) FROM calls GROUP BY 1, 2 ORDER BY 1, 2"
                )
            ]
        finally:
            connection.close()
    agreement = {}
    for task, rows in labels.items():
        if task == "esci":
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["stratum"] or "unknown"].append(row)
        grouped["all"] = rows
        agreement[task] = {}
        for stratum in sorted(grouped, key=lambda name: (name == "all", name)):
            subset = grouped[stratum]
            agree = [
                {field: _key(row["fields"][field]["judge_a"]) == _key(row["fields"][field]["judge_b"]) for field in TASK_FIELDS[task]}
                for row in subset
            ]
            agreement[task][stratum] = {
                "items": len(subset),
                "complete_agreement": round(sum(all(item.values()) for item in agree) / len(subset), 4),
                "fields": {field: round(sum(item[field] for item in agree) / len(subset), 4) for field in TASK_FIELDS[task]},
            }
    return {
        "judges": {name: {"provider": provider, "model": adjudicate_pilots.PROVIDERS[provider]["model"]} for name, provider in JUDGES.items()},
        "agreement_by_stratum": agreement,
        "consensus_rule_violations": violations,
        "label_generation": generated,
        "provider_calls": calls,
    }


def text_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def build(private_root: Path, staging: Path, build_date: str, builder_commit: str) -> dict[str, Any]:
    use_private_inputs(private_root)
    examples = {task: datasets.load_examples(task, dataset="reference") for task in TASK_FIELDS}
    raw_strata = {
        example["sampling_context"].get("source_stratum") or ""
        for rows in examples.values()
        for example in rows
    }
    orcas_terms = {name[len(ORCAS_PREFIX):].lower() for name in raw_strata if name.startswith(ORCAS_PREFIX)}

    labels: dict[str, list[dict[str, Any]]] = {}
    violations: dict[str, int] = {}
    for task, rows in examples.items():
        labels[task], violations[task] = label_rows(task, rows, private_root)
    files: dict[str, list[dict[str, Any]]] = {f"labels/{task}.jsonl.gz": rows for task, rows in labels.items()}
    for slug in MODEL_SLUGS:
        for task, rows in examples.items():
            files[f"predictions/{slug}/{task}.jsonl.gz"] = prediction_rows(slug, task, [row["item_id"] for row in rows], private_root)

    unexpected = scan_strings(files, vocabulary())
    if sum(unexpected.values()):
        raise SystemExit(f"Pack contains strings outside the closed vocabulary: {dict(unexpected)}")
    if any(violations.values()):
        raise SystemExit(f"Reference labels break the consensus rule: {violations}")

    manifest_files = {}
    for rel, rows in sorted(files.items()):
        data = jsonl_gz_bytes(rows)
        path = staging / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest_files[rel] = {"rows": len(rows), "bytes": len(data), "sha256": sha256_bytes(data)}

    provenance = label_provenance(private_root / CACHE_NAME, labels, violations)
    (staging / PROVENANCE_NAME).write_bytes(text_bytes(provenance))

    rubrics = {
        task: Path(path).resolve().relative_to(ROOT).as_posix()
        for task, path in adjudicate_pilots.TASK_RUBRIC_PATHS.items()
    }
    summaries = {slug: json.loads(summary_path(slug).read_text(encoding="utf-8")) for slug in MODEL_SLUGS}
    private_inputs = sorted(
        [private_root / "references" / Path(path).name for path in adjudicate_pilots.DATASET_TASK_FILES["reference"].values()]
        + [private_root / "references" / "consensus" / f"{task}_consensus.jsonl" for task in TASK_FIELDS if task != "esci"]
        + [private_root / "references" / "esci_eval_slice.jsonl"]
        + [private_root / "outputs" / "benchmark" / slug / "predictions" / f"{task}.jsonl" for slug in MODEL_SLUGS for task in TASK_FIELDS]
    )
    manifest = {
        "schema_version": 1,
        "build_date": build_date,
        "builder": {
            "script": "build_replay_pack.py",
            "script_sha256_lf": sha256_lf(Path(__file__)),
            "commit": builder_commit,
            "commit_note": "Repository HEAD when the pack was built.",
        },
        "prompt_version": PROMPT_VERSION,
        "model_order": list(MODEL_SLUGS),
        "models": {
            slug: {
                "model_id": summaries[slug]["model_id"],
                "role_label": ROLE_LABELS[slug],
                "summary": summary_path(slug).relative_to(ROOT).as_posix(),
                "summary_sha256_lf": sha256_lf(summary_path(slug)),
            }
            for slug in MODEL_SLUGS
        },
        "tasks": {
            task: {
                "items": len(labels[task]),
                "fields": list(TASK_FIELDS[task]),
                "label_source": sorted({row["label_source"] for row in labels[task]}),
                "strata": dict(sorted(Counter(row["stratum"] for row in labels[task]).items())),
            }
            for task in TASK_FIELDS
        },
        "label_sources": LABEL_SOURCES,
        "judges": provenance["judges"],
        "strata_note": (
            f"stratum is the sampling stratum each item was drawn from. Query-segmentation items sampled from ORCAS-I "
            f"all carry {ORCAS_STRATUM}; their ORCAS-I intent category is not included. ESCI items carry human."
        ),
        "rubrics": {task: {"path": rel, "sha256_lf": sha256_lf(ROOT / rel)} for task, rel in rubrics.items() if task in labels},
        "licences": LICENCES,
        "row_schema": {
            "labels": (
                "item_id, task, stratum, label_source, fields.<field>.reference (null when withheld); "
                "dual-judge tasks add fields.<field>.judge_a and judge_b"
            ),
            "predictions": (
                "item_id, task, elapsed_ms, input_tokens, fields.<field>.predicted, probabilities "
                "(label -> probability), confidence (the adapter's reported confidence field)"
            ),
        },
        "private_inputs_sha256": {
            path.relative_to(private_root).as_posix(): sha256_bytes(path.read_bytes()) for path in private_inputs
        },
        "files": manifest_files,
        "label_provenance": {"path": PROVENANCE_NAME, "sha256_lf": sha256_lf(staging / PROVENANCE_NAME)},
    }
    (staging / MANIFEST_NAME).write_bytes(text_bytes(manifest))

    problems = check_manifest(staging, ROOT)
    if problems:
        raise SystemExit("New pack does not match its own manifest: " + "; ".join(problems))
    from replay_scores import run_verify

    code, report = run_verify(staging, ROOT, results_file=None)
    if code:
        raise SystemExit(f"Committed summaries do not recompute from the new pack (exit {code}): {report.get('mismatch_details', [])[:5]}")
    found = orcas_terms_found(staging, orcas_terms)
    if found:
        raise SystemExit(f"ORCAS-I category names found in the pack: {found}")
    manifest["build_checks"] = {
        "summary_values_stored": report["values_stored"],
        "summary_values_compared_exactly": report["values_compared_exactly"],
        "summary_values_verified_in_aggregate": report["values_verified_in_aggregate"],
        "summary_mismatches": report["mismatches"],
        "strings_outside_vocabulary": 0,
        "orcas_i_category_names_found": 0,
        "consensus_rule_violations": sum(violations.values()),
    }
    (staging / MANIFEST_NAME).write_bytes(text_bytes(manifest))
    return manifest


def replace_pack(staging: Path, output_dir: Path) -> None:
    if output_dir.exists():
        extra = [
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file() and not (path.name in {MANIFEST_NAME, PROVENANCE_NAME} or path.name.endswith(".jsonl.gz"))
        ]
        if extra:
            raise SystemExit(f"Refusing to replace {output_dir}: it holds non-pack files {extra[:5]}")
        shutil.rmtree(output_dir)
    shutil.copytree(staging, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--build-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--builder-commit", default=None, help="defaults to the repository HEAD (git rev-parse --short HEAD)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    private_root = resolve_private_root(args.private_root)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "replay"
        staging.mkdir()
        manifest = build(private_root, staging, args.build_date, args.builder_commit or head_commit())
        replace_pack(staging, args.output_dir.resolve())
    total = sum(meta["bytes"] for meta in manifest["files"].values())
    total += sum((args.output_dir / name).stat().st_size for name in (MANIFEST_NAME, PROVENANCE_NAME))
    checks = manifest["build_checks"]
    print(
        f"replay pack: {len(manifest['files'])} data files, {total:,} bytes; "
        f"summary values {checks['summary_values_stored']:,} "
        f"({checks['summary_values_compared_exactly']:,} exact, {checks['summary_values_verified_in_aggregate']:,} in aggregate), "
        f"mismatches {checks['summary_mismatches']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
