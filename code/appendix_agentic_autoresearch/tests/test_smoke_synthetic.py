"""Full synthetic pipeline, including the seal, without Yambda."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from build_harness import build
from harness.ledger import read_jsonl
from loop import run_search
from promote_and_test import promote
from report import write_report


class SealAccessError(RuntimeError):
    pass


def _install_guard(monkeypatch, sealed: Path) -> None:
    sealed = sealed.resolve()

    def check(path) -> None:
        if not isinstance(path, (str, Path)):
            return
        try:
            resolved = Path(path).resolve()
        except OSError:
            return
        if resolved == sealed or sealed in getattr(resolved, "parents", []):
            raise SealAccessError(str(resolved))

    real_open = Path.open

    def guarded_open(self, *args, **kwargs):
        check(self)
        return real_open(self, *args, **kwargs)

    real_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        check(self)
        return real_read_text(self, *args, **kwargs)

    real_read_bytes = Path.read_bytes

    def guarded_read_bytes(self, *args, **kwargs):
        check(self)
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    import builtins

    real_builtin = builtins.open

    def guarded_builtin(file, *args, **kwargs):
        check(file)
        return real_builtin(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_builtin)
    real_read = pl.read_parquet

    def guarded_read(source, *args, **kwargs):
        check(source)
        return real_read(source, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", guarded_read)
    real_scan = pl.scan_parquet

    def guarded_scan(source, *args, **kwargs):
        check(source)
        return real_scan(source, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", guarded_scan)


def test_loop_cannot_read_the_seal(tmp_path, monkeypatch):
    """The loop scores validation only. Training is stubbed so this stays fast."""

    run_dir = tmp_path / "run"
    build(run_dir, synthetic=True, data_dir=None, force=True)
    labels = pl.read_parquet(run_dir / "artifacts" / "validation_labels.parquet")

    def fake_train(**kwargs):
        trial_dir = kwargs["trial_dir"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "row_id": labels["row_id"],
                "prediction": [0.45] * labels.height,
            }
        ).write_parquet(trial_dir / "validation_predictions.parquet")
        manifest = {
            "feature_columns": ["track_length_seconds"],
            "metadata": {
                "early_stopping_split": "train_tail",
                "early_stopping_fraction": 0.1,
                "model_selection_split": "validation",
                "optimization_split": "validation",
                "preprocessor_fit_split": "train",
                "primary_metric": "auc_roc",
                "primary_split": "validation",
            },
        }
        (trial_dir / "submission_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return {"status": "completed"}

    monkeypatch.setattr("loop.train_trial", fake_train)
    _install_guard(monkeypatch, run_dir / "sealed")
    summary = run_search(run_dir, synthetic=True, in_process=True)
    assert summary["failed"] == 0
    assert summary["completed"] == summary["trials"]
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    assert ledger
    for row in ledger:
        assert "test_metrics" not in row
        if row["status"] == "completed":
            assert row["validation_metrics"]["split"] == "validation"


def test_synthetic_end_to_end(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    build(run_dir, synthetic=True, data_dir=None, force=True)
    _install_guard(monkeypatch, run_dir / "sealed")
    summary = run_search(run_dir, synthetic=True, in_process=True)
    monkeypatch.undo()
    assert summary["completed"] >= 7
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    assert any(row["arm"] == "replay" for row in ledger)
    for row in ledger:
        assert "test_metrics" not in row
        if row["status"] == "completed":
            assert row["validation_metrics"]["split"] == "validation"
    manifest_path = next((run_dir / "trials").glob("*/submission_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["early_stopping_split"] == "train_tail"
    assert manifest["metadata"]["model_selection_split"] == "validation"
    assert manifest["metadata"]["optimization_split"] == "validation"
    assert manifest["metadata"]["primary_split"] == "validation"
    predictions = pl.read_parquet(
        manifest_path.parent / "validation_predictions.parquet"
    )
    assert "label" not in predictions.columns
    for candidate in ("baseline", "random", "greedy"):
        promote(run_dir, candidate)
    write_report(run_dir)
    decisions = read_jsonl(run_dir / "test_ledger.jsonl")
    spent = [row for row in decisions if row.get("spent") is True]
    assert len(spent) <= 3
    assert (run_dir / "report.md").is_file()
    for row in spent:
        assert row["test_metrics"]["split"] == "test"
        assert "optimism_gap" in row
