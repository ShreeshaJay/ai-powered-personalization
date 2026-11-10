"""Hash seal and the three-evaluation test budget.

Only the auditor (``build_harness.py``) and the promotion script open the
seal directory. The research loop does not import this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.contract import file_sha256
from harness.ledger import append_jsonl, read_jsonl, utc_now

SEAL_DIR = "sealed"
SEALED_FILES = ("test_rows.parquet", "test_labels.parquet")
RECORD_NAME = "seal_record.json"
BUDGET_NAME = "budget_state.json"
TEST_LEDGER_NAME = "test_ledger.jsonl"


class SealError(RuntimeError):
    """Hashes, the contract, or the seal record do not agree."""


class BudgetError(RuntimeError):
    """The pre-registered test budget is spent."""


def seal_dir(run_dir: Path) -> Path:
    return run_dir / SEAL_DIR


def record_path(run_dir: Path) -> Path:
    return run_dir / "artifacts" / RECORD_NAME


def budget_path(run_dir: Path) -> Path:
    return seal_dir(run_dir) / BUDGET_NAME


def test_ledger_path(run_dir: Path) -> Path:
    return run_dir / TEST_LEDGER_NAME


def write_seal(run_dir: Path, *, contract_hash: str, test_budget: int) -> dict[str, Any]:
    hashes = {
        name: file_sha256(seal_dir(run_dir) / name)
        for name in SEALED_FILES
    }
    record = {
        "algorithm": "sha256",
        "contract_sha256": contract_hash,
        "files": hashes,
        "test_budget": int(test_budget),
    }
    _write_json(record_path(run_dir), record)
    _write_json(seal_dir(run_dir) / "seal.json", record)
    _write_json(
        budget_path(run_dir),
        {"budget": int(test_budget), "used": 0, "evaluations": []},
    )
    return record


def verify_hashes(run_dir: Path, contract_hash: str) -> dict[str, Any]:
    record = _read_json(record_path(run_dir))
    sealed_copy = _read_json(seal_dir(run_dir) / "seal.json")
    if record != sealed_copy:
        raise SealError("Seal record and sealed copy disagree")
    if record.get("contract_sha256") != contract_hash:
        raise SealError("Contract hash does not match the harness build")
    if int(record.get("test_budget", 0)) != 3:
        raise SealError("Seal record test budget is not 3")
    for name, expected in record["files"].items():
        actual = file_sha256(seal_dir(run_dir) / name)
        if actual != expected:
            raise SealError(f"Hash mismatch for {name}")
    return record


def spent_count(run_dir: Path) -> int:
    ledger = [
        row for row in read_jsonl(test_ledger_path(run_dir))
        if row.get("spent") is True
    ]
    state = _read_json(budget_path(run_dir))
    used = int(state.get("used", 0))
    if used != len(ledger):
        raise BudgetError("Test ledger and seal budget disagree")
    if used != len(state.get("evaluations", [])):
        raise BudgetError("Seal budget evaluations do not match the counter")
    return used


def enforce_budget(run_dir: Path, budget: int = 3) -> int:
    used = spent_count(run_dir)
    if used >= int(budget):
        raise BudgetError(f"Test budget of {budget} is exhausted ({used} already spent)")
    return used


def record_spend(run_dir: Path, evaluation: dict[str, Any]) -> None:
    """Append one spent evaluation and advance the seal counter together."""

    state = _read_json(budget_path(run_dir))
    budget = int(state["budget"])
    enforce_budget(run_dir, budget)
    evaluation = dict(evaluation)
    evaluation["spent"] = True
    evaluation.setdefault("created_at_utc", utc_now())
    append_jsonl(test_ledger_path(run_dir), evaluation)
    state["used"] = int(state["used"]) + 1
    state["evaluations"] = list(state.get("evaluations", [])) + [evaluation["trial_id"]]
    _write_json(budget_path(run_dir), state)
    if spent_count(run_dir) > budget:
        raise BudgetError("Spend exceeded the test budget")


def record_rejection(run_dir: Path, rejection: dict[str, Any]) -> None:
    """A gate rejection is logged and does not spend the test budget."""

    rejection = dict(rejection)
    rejection["spent"] = False
    rejection.setdefault("created_at_utc", utc_now())
    append_jsonl(test_ledger_path(run_dir), rejection)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
