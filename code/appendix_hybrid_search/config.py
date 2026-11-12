"""Settings for Appendix B, Hybrid Search with Vector Databases.

Importing this module does not create directories and does not read a data
path outside the process environment. Call ``ensure_dirs`` before writing.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PLAN_PATH = PACKAGE_ROOT / "analysis_plan.yaml"
PLAN_HASH_PATH = PACKAGE_ROOT / "analysis_plan.sha256"

OFFICIAL_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
LINEAR_GAINS = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}
LABELS = ("E", "S", "C", "I")

TEXT_TEMPLATE = "{title}. {bullet_point}. Brand: {brand}"
TEXT_RULE = "skip empty parts, join with '. ', collapse whitespace"

EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"

TRACKED_PACKAGES = (
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "bm25s",
    "faiss-cpu",
    "pyarrow",
    "qdrant-client",
    "datasets",
    "accelerate",
    "ir-measures",
    "PyYAML",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_plan(path: Path | None = None) -> dict[str, Any]:
    plan_path = path or PLAN_PATH
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if not isinstance(plan, dict):
        raise ValueError(f"Analysis plan at {plan_path} is not a mapping.")
    if plan.get("contains_results"):
        raise ValueError("The analysis plan must not contain results.")
    return plan


def require_plan_hash(plan_path: Path | None = None, hash_path: Path | None = None) -> str:
    """Return the analysis_plan.yaml digest, or raise if the sidecar does not match."""
    plan_path = plan_path or PLAN_PATH
    hash_path = hash_path or PLAN_HASH_PATH
    digest = sha256_file(plan_path)
    if not hash_path.exists():
        raise FileNotFoundError(
            "analysis_plan.sha256 is missing. Test scoring is refused until the "
            "fixed protocol file has a recorded digest."
        )
    recorded = hash_path.read_text(encoding="utf-8").strip()
    if recorded != digest:
        raise ValueError(
            "analysis_plan.yaml does not match analysis_plan.sha256. "
            "Test scoring is refused."
        )
    return digest


def data_dir_from_env() -> Path | None:
    raw = os.environ.get("ESCI_DATA_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def resolve_data_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    env_dir = data_dir_from_env()
    if env_dir is not None:
        return env_dir
    raise FileNotFoundError(
        "No ESCI directory was given. Pass --data-dir or set ESCI_DATA_DIR. "
        "There is no default path."
    )


def resolve_device(requested: str | None) -> str:
    """Use CUDA when it is present and the caller asked for auto."""
    choice = (requested or "auto").strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device {requested!r}. Use auto, cpu, or cuda.")
    if choice == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested and torch.cuda.is_available() is false.")
    return choice


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class BM25Settings:
    method: str = "lucene"
    k1: float = 1.5
    b: float = 0.75
    stopwords: str = "en"


@dataclass
class DenseSettings:
    repo: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    trust_remote_code: bool = False
    embedding_dim: int = 384
    max_length: int = 256
    normalize: bool = True
    index_type: str = "IndexFlatIP"
    batch_size: int = 64
    query_batch_size: int = 128
    faiss_index_file: str = "dense_faiss.index"
    embeddings_file: str = "dense_embeddings.npz"


@dataclass
class Paths:
    data_dir: Path | None = None
    output_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs")
    cache_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "cache")

    def runs_dir(self, split: str) -> Path:
        return self.output_dir / "runs" / split

    def manifest_dir(self) -> Path:
        return self.output_dir / "manifests"
