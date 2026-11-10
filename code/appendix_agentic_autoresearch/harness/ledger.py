"""Append-only JSONL ledgers. The research ledger stores validation only."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def research_ledger_path(run_dir: Path) -> Path:
    return run_dir / "ledger.jsonl"


def append_research_record(run_dir: Path, record: dict[str, Any]) -> None:
    if "test_metrics" in record:
        raise ValueError("The research ledger cannot store test metrics")
    metrics = record.get("validation_metrics") or {}
    if metrics.get("split") not in (None, "validation"):
        raise ValueError("Research metrics must be validation metrics")
    record = dict(record)
    record.setdefault("created_at_utc", utc_now())
    append_jsonl(research_ledger_path(run_dir), record)


def public_error(exc: BaseException, *roots: Path) -> str:
    """Exception text with machine paths removed."""

    text = f"{type(exc).__name__}: {exc}"
    for root in roots:
        text = text.replace(str(root), ".")
    cleaned = []
    for token in text.split():
        if len(token) > 2 and token[1] == ":" and token[2] in ("\\", "/"):
            cleaned.append(".")
        elif token.startswith("/Users/") or token.startswith("/home/"):
            cleaned.append(".")
        else:
            cleaned.append(token)
    return " ".join(cleaned)[:500]
