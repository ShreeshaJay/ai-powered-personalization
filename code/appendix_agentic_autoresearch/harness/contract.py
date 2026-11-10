"""Paths and the pre-registered contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from harness.yaml_util import load_yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_NAME = "contract.yaml"
SEARCH_SPACE_NAME = "search_space.yaml"
ARTIFACT_DIR = "artifacts"


def contract_path() -> Path:
    return ROOT / CONTRACT_NAME


def search_space_path() -> Path:
    return ROOT / SEARCH_SPACE_NAME


def load_contract(path: Path | None = None) -> dict[str, Any]:
    loaded = load_yaml(path or contract_path())
    if not isinstance(loaded, dict):
        raise ValueError("contract.yaml must be a mapping")
    _require(loaded)
    return loaded


def load_search_space(path: Path | None = None) -> dict[str, Any]:
    loaded = load_yaml(path or search_space_path())
    if not isinstance(loaded, dict):
        raise ValueError("search_space.yaml must be a mapping")
    objective = loaded.get("objective") or {}
    if objective.get("split") != "validation":
        raise ValueError("search_space objective split must be validation")
    if objective.get("metric") != "auc_roc":
        raise ValueError("search_space objective metric must be auc_roc")
    return loaded


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifacts_dir(run_dir: Path) -> Path:
    return run_dir / ARTIFACT_DIR


def trial_budget(contract: dict[str, Any], synthetic: bool) -> int:
    if synthetic:
        return int(contract["synthetic"]["trial_budget_per_arm"])
    return int(contract["trial_budget_per_arm"])


def bootstrap_replicates(contract: dict[str, Any], synthetic: bool) -> int:
    if synthetic:
        return int(contract["synthetic"]["bootstrap_replicates"])
    return int(contract["noise_band"]["bootstrap_replicates"])


def apply_runtime_clip(config: dict[str, Any], contract: dict[str, Any], synthetic: bool) -> dict[str, Any]:
    """Cap tree size for the synthetic minute-scale run.

    The pre-registered baseline and search space stay as written. The clip is
    applied to the config that actually trains.
    """

    clipped = dict(config)
    if not synthetic:
        return clipped
    limits = contract["synthetic"]
    clipped["n_estimators"] = min(int(clipped["n_estimators"]), int(limits["clip_n_estimators"]))
    clipped["max_depth"] = min(int(clipped["max_depth"]), int(limits["clip_max_depth"]))
    clipped["early_stopping_rounds"] = min(
        int(clipped["early_stopping_rounds"]),
        int(limits["clip_early_stopping_rounds"]),
    )
    return clipped


def _require(contract: dict[str, Any]) -> None:
    if contract.get("primary_split") != "validation":
        raise ValueError("contract primary_split must be validation")
    if int(contract.get("test_budget", 0)) != 3:
        raise ValueError("contract test_budget must be 3")
    noise = contract.get("noise_band") or {}
    if noise.get("method") != "baseline_seeds_plus_user_bootstrap":
        raise ValueError("unexpected noise-band method")
    if int(noise.get("baseline_seeds", 0)) != 5:
        raise ValueError("noise band requires 5 baseline seeds")
    if float(contract["split"]["early_stopping_fraction"]) != 0.10:
        raise ValueError("early stopping fraction must be the last 10 percent")
    rule = (contract.get("promotion_threshold") or {}).get("rule")
    if rule != "winner_validation_auc_minus_baseline_mean_gt_noise_band":
        raise ValueError("unexpected promotion threshold")
