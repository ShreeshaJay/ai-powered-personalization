"""Read the text-free replay pack in `replay/` and check it against its manifest.

The pack holds, per item, the reference label each model is scored against (the
dual-judge field consensus with both judges' labels, or the ESCI human label) and
every model's prediction, probabilities and reported confidence. It holds no query
or product text. `build_replay_pack.py` writes it; everything here only reads it.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapters.base import ItemPrediction
from adapters.schemas import TASK_FIELDS
from eval.metrics import _key
from eval.runner import prediction_from_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK_DIR = ROOT / "replay"
MANIFEST_NAME = "MANIFEST.json"
PROVENANCE_NAME = "label_provenance.json"
ORCAS_PREFIX = "orcas_"
ORCAS_STRATUM = "orcas_i"
DEGENERATE_MAJORITY_SHARE = 0.99


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_lf(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF (git checkouts differ by platform)."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def jsonl_gz_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Compact sorted-key JSON lines, gzip level 9, mtime 0, no stored file name."""
    raw = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def collapse_stratum(stratum: str | None) -> str | None:
    """Map every ORCAS-I category stratum to the single value `orcas_i`."""
    if stratum and stratum.startswith(ORCAS_PREFIX):
        return ORCAS_STRATUM
    return stratum


def examples_from_label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild `evaluate_task` examples from pack label rows.

    A field is labelled when its `reference` is not null; withheld fields are skipped
    by `labeled_pairs`, as in `eval/datasets.py`. An item counts towards
    `complete_exact_match` only when every field has a reference label.
    """
    examples = []
    for row in rows:
        reference = {name: values["reference"] for name, values in row["fields"].items()}
        complete = all(value is not None for value in reference.values())
        examples.append(
            {
                "item_id": row["item_id"],
                "task": row["task"],
                "labels": {name: value for name, value in reference.items() if value is not None},
                "complete_labels": reference if complete else None,
                "complete_agreement": complete,
                "sampling_context": {"source_stratum": row["stratum"]} if row["stratum"] else {},
                "label_source": row["label_source"],
            }
        )
    return examples


def load_manifest(pack_dir: Path = DEFAULT_PACK_DIR) -> dict[str, Any]:
    return json.loads((pack_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def check_manifest(pack_dir: Path = DEFAULT_PACK_DIR, root: Path = ROOT) -> list[str]:
    """Return every way the pack, summaries or rubrics differ from the manifest."""
    manifest_path = pack_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing {MANIFEST_NAME}"]
    manifest = load_manifest(pack_dir)
    problems = []
    listed = set(manifest["files"]) | {MANIFEST_NAME, manifest["label_provenance"]["path"]}
    for path in sorted(pack_dir.rglob("*")):
        if path.is_file() and path.relative_to(pack_dir).as_posix() not in listed:
            problems.append(f"unlisted file in pack: {path.relative_to(pack_dir).as_posix()}")
    for rel, meta in sorted(manifest["files"].items()):
        path = pack_dir / rel
        if not path.is_file():
            problems.append(f"missing pack file: {rel}")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != meta["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
            continue
        rows = sum(1 for line in gzip.decompress(data).splitlines() if line.strip())
        if rows != meta["rows"]:
            problems.append(f"row count mismatch: {rel} ({rows} against {meta['rows']})")
    provenance = manifest["label_provenance"]
    provenance_path = pack_dir / provenance["path"]
    if not provenance_path.is_file():
        problems.append(f"missing pack file: {provenance['path']}")
    elif sha256_lf(provenance_path) != provenance["sha256_lf"]:
        problems.append(f"sha256 mismatch: {provenance['path']}")
    for slug, meta in sorted(manifest["models"].items()):
        path = root / meta["summary"]
        if not path.is_file():
            problems.append(f"missing summary for {slug}: {meta['summary']}")
        elif sha256_lf(path) != meta["summary_sha256_lf"]:
            problems.append(f"summary changed since the pack was built: {meta['summary']}")
    for task, meta in sorted(manifest["rubrics"].items()):
        path = root / meta["path"]
        if not path.is_file() or sha256_lf(path) != meta["sha256_lf"]:
            problems.append(f"rubric differs from the one the labels were made with ({task}): {meta['path']}")
    return problems


@dataclass
class ReplayPack:
    manifest: dict[str, Any]
    labels: dict[str, list[dict[str, Any]]]
    examples: dict[str, list[dict[str, Any]]]
    prediction_rows: dict[tuple[str, str], list[dict[str, Any]]]
    predictions: dict[tuple[str, str], list[ItemPrediction]] = field(default_factory=dict)

    @property
    def slugs(self) -> list[str]:
        return list(self.manifest["model_order"])

    @property
    def tasks(self) -> list[str]:
        return [task for task in TASK_FIELDS if task in self.labels]

    def model_id(self, slug: str) -> str:
        return self.manifest["models"][slug]["model_id"]


def load_pack(pack_dir: Path = DEFAULT_PACK_DIR) -> ReplayPack:
    manifest = load_manifest(pack_dir)
    labels = {}
    for task in TASK_FIELDS:
        path = pack_dir / "labels" / f"{task}.jsonl.gz"
        if path.is_file():
            labels[task] = read_jsonl_gz(path)
    examples = {task: examples_from_label_rows(rows) for task, rows in labels.items()}
    prediction_rows = {}
    predictions = {}
    for slug in manifest["model_order"]:
        for task in labels:
            rows = read_jsonl_gz(pack_dir / "predictions" / slug / f"{task}.jsonl.gz")
            prediction_rows[(slug, task)] = rows
            predictions[(slug, task)] = [prediction_from_dict(row) for row in rows]
    return ReplayPack(manifest, labels, examples, prediction_rows, predictions)


@dataclass(frozen=True)
class FieldView:
    """One model's labelled predictions on one field, in pack (manifest) order."""

    item_ids: list[str]
    strata: list[str | None]
    reference: list[Any]
    predicted: list[Any]
    probabilities: list[dict[str, float]]
    reported_confidence: list[float]
    top_label_probability: list[float]
    correct: list[bool]

    def __len__(self) -> int:
        return len(self.item_ids)


def field_view(pack: ReplayPack, slug: str, task: str, field_name: str) -> FieldView:
    """Pair labelled items with predictions the way `eval.datasets.labeled_pairs` does."""
    by_id = {prediction.item_id: prediction for prediction in pack.predictions[(slug, task)]}
    columns: dict[str, list[Any]] = {name: [] for name in FieldView.__dataclass_fields__}
    for example in pack.examples[task]:
        if field_name not in example["labels"]:
            continue
        prediction = by_id[example["item_id"]].fields[field_name]
        reference = example["labels"][field_name]
        columns["item_ids"].append(example["item_id"])
        columns["strata"].append(example["sampling_context"].get("source_stratum"))
        columns["reference"].append(reference)
        columns["predicted"].append(prediction.predicted)
        columns["probabilities"].append(prediction.probabilities)
        columns["reported_confidence"].append(float(prediction.confidence))
        columns["top_label_probability"].append(
            float(prediction.probabilities.get(_key(prediction.predicted), 0.0))
        )
        columns["correct"].append(_key(reference) == _key(prediction.predicted))
    return FieldView(**columns)


def majority_share(examples: list[dict[str, Any]], field_name: str) -> tuple[float, int, int]:
    """Share of labelled items in the most frequent class: (share, count, labelled)."""
    counts = Counter(_key(example["labels"][field_name]) for example in examples if field_name in example["labels"])
    labelled = sum(counts.values())
    if not labelled:
        return 0.0, 0, 0
    top = counts.most_common(1)[0][1]
    return top / labelled, top, labelled


def is_degenerate(examples: list[dict[str, Any]], field_name: str) -> bool:
    """A field is degenerate when one class holds at least 99% of its labelled items."""
    share, _top, labelled = majority_share(examples, field_name)
    return labelled > 0 and share >= DEGENERATE_MAJORITY_SHARE
