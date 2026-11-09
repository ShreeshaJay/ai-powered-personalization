"""
Locate, verify and switch artifact bundles.

A bundle is an immutable, versioned directory written by
`python -m pipeline_build.build_artifacts`:

    artifacts/<bundle_version>/
        bundle_manifest.json      versions, source data, metrics, SHA-256 of every file
        feature_store/            online features (SQLite and Parquet) and their manifest
        retrieval/                user tower, item vectors in a FAISS index, index manifest
        lightweight_ranker/       Chapter 6 XGBoost model and feature pipeline
        heavy_ranker/             Chapter 7 MMoE checkpoint and feature processor
        reranker/                 content embeddings for MMR
        orchestrator/             popularity fallback list

`artifacts/ACTIVE` holds the name of the bundle services load by default.
Promoting or rolling back a bundle is a one-line atomic rewrite of that file
followed by a restart (or a new revision in a managed platform).

Overrides: CH13_BUNDLE_DIR points at one bundle directory; CH13_ARTIFACT_ROOT
points at a directory that holds ACTIVE and bundles.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

CHAPTER_DIR = Path(__file__).resolve().parents[1]
BUILD_HINT = (
    "Build artifacts first, for example:\n"
    "    python -m pipeline_build.build_artifacts --profile synthetic\n"
    "or point CH13_BUNDLE_DIR at an existing bundle directory."
)


class ArtifactError(RuntimeError):
    """A required artifact is missing, corrupt, or does not match its manifest."""


def artifact_root() -> Path:
    return Path(os.environ.get("CH13_ARTIFACT_ROOT", CHAPTER_DIR / "artifacts"))


def resolve_bundle_dir(explicit: Optional[Path] = None) -> Path:
    if explicit is None and os.environ.get("CH13_BUNDLE_DIR"):
        explicit = Path(os.environ["CH13_BUNDLE_DIR"])
    if explicit is not None:
        bundle = Path(explicit)
    else:
        pointer = artifact_root() / "ACTIVE"
        if not pointer.is_file():
            raise ArtifactError(f"No active artifact bundle: {pointer} does not exist.\n{BUILD_HINT}")
        name = pointer.read_text(encoding="utf-8").strip()
        bundle = artifact_root() / name
    if not (bundle / "bundle_manifest.json").is_file():
        raise ArtifactError(f"Not an artifact bundle (no bundle_manifest.json): {bundle}\n{BUILD_HINT}")
    return bundle


def require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Missing {what}: {path}\n{BUILD_HINT}")
    return path


def require_dir(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise ArtifactError(f"Missing {what}: {path}\n{BUILD_HINT}")
    return path


def read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(require_file(path, what).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Corrupt {what} at {path}: {exc}") from exc


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(component_dir: Path, expected: dict, what: str) -> None:
    """Check each file named in `expected` ({relative path: sha256}) exists and matches."""
    for relative, digest in expected.items():
        path = require_file(Path(component_dir) / relative, f"{what} file {relative}")
        actual = sha256_file(path)
        if actual != digest:
            raise ArtifactError(
                f"{what} file {relative} does not match its manifest "
                f"(expected sha256 {digest[:12]}, found {actual[:12]}). "
                "The bundle was modified or mixed with files from another build; rebuild it."
            )


def check_published_code(bundle_dir: Path, loaded: Dict[str, str]) -> None:
    """
    Refuse a bundle that was built with different published code from the code
    this process runs. A changed feature function or model class would
    otherwise compute serving features that differ from the training features
    without raising any error.
    """
    manifest = read_json(Path(bundle_dir) / "bundle_manifest.json", "bundle manifest")
    built_with = manifest.get("published_code") or {}
    if not built_with.get("files"):
        raise ArtifactError(f"Bundle {Path(bundle_dir).name} does not record the published code it was built "
                            f"with; rebuild it.\n{BUILD_HINT}")
    changed = sorted(path for path, digest in loaded.items() if built_with["files"].get(path) != digest)
    if changed:
        raise ArtifactError(
            f"vendor/ differs from the published code bundle {manifest.get('bundle_version')} was built with "
            f"(commit {str(built_with.get('commit'))[:12]}): {', '.join(changed)}. Rebuild the bundle with the "
            f"current vendor/, or restore vendor/ to that commit.\n{BUILD_HINT}")


def set_active_bundle(bundle_version: str, root: Optional[Path] = None) -> Path:
    """Point ACTIVE at a bundle with an atomic replace, so readers never see a partial file."""
    root = Path(root) if root is not None else artifact_root()
    if not (root / bundle_version / "bundle_manifest.json").is_file():
        raise ArtifactError(f"Cannot activate {bundle_version}: no bundle at {root / bundle_version}")
    pointer = root / "ACTIVE"
    tmp = root / "ACTIVE.tmp"
    tmp.write_text(bundle_version + "\n", encoding="utf-8")
    os.replace(tmp, pointer)
    return pointer
