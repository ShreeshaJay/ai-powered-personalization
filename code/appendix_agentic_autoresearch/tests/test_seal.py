"""Seal, budget, objective path, and proposer tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from build_harness import build
from harness.contract import ROOT, contract_path, file_sha256, load_contract, load_search_space
from harness.seal import BudgetError, SealError, enforce_budget, record_spend, verify_hashes
from harness.train import fit_xgboost, train_trial
from proposers.base import build_proposer
from proposers.greedy import GreedyProposer
from proposers.replay import ReplayProposer


def test_contract_and_search_space_are_validation_only():
    contract = load_contract()
    space = load_search_space()
    assert contract["primary_split"] == "validation"
    assert contract["primary_metric"] == "auc_roc"
    assert contract["test_budget"] == 3
    assert contract["noise_band"]["method"] == "baseline_seeds_plus_user_bootstrap"
    assert contract["noise_band"]["baseline_seeds"] == 5
    assert contract["split"]["early_stopping_fraction"] == pytest.approx(0.10)
    assert space["objective"]["split"] == "validation"
    assert space["objective"]["metric"] == "auc_roc"
    for path in (contract_path(), ROOT / "search_space.yaml", ROOT / "proposers" / "proposals_replay.jsonl"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users" not in text
        assert "/Users/" not in text
        assert "api_key" not in text.lower()


def test_trainer_early_stops_on_the_train_tail_only():
    source = inspect.getsource(fit_xgboost)
    assert "early_stopping_rounds" in source
    assert "x_tail" in source
    assert "sealed" not in source.lower()
    assert "validation_rows" not in source
    parameters = inspect.signature(train_trial).parameters
    assert "test" not in parameters
    for name in ("loop.py", "run_trial.py"):
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "sealed" not in text
        assert "test_labels" not in text
        assert "test_rows" not in text


def test_hash_mismatch_aborts(tmp_path):
    run_dir = tmp_path / "run"
    build(run_dir, synthetic=True, data_dir=None, force=True)
    labels = run_dir / "sealed" / "test_labels.parquet"
    frame = pl.read_parquet(labels)
    flipped = frame.with_columns((1 - pl.col("label")).cast(pl.Int8).alias("label"))
    flipped.write_parquet(labels)
    with pytest.raises(SealError, match="Hash mismatch"):
        verify_hashes(run_dir, file_sha256(contract_path()))


def test_test_budget_is_three(tmp_path):
    run_dir = tmp_path / "run"
    build(run_dir, synthetic=True, data_dir=None, force=True)
    for index in range(3):
        record_spend(
            run_dir,
            {
                "trial_id": f"fixture-{index}",
                "candidate": "baseline",
                "critic_verdict": "approve",
                "validation_auc": 0.5,
                "test_metrics": {"auc_roc": 0.5, "split": "test"},
                "optimism_gap": 0.0,
                "reason": "fixture",
            },
        )
    with pytest.raises(BudgetError):
        enforce_budget(run_dir, 3)
    with pytest.raises(BudgetError):
        record_spend(
            run_dir,
            {
                "trial_id": "fixture-3",
                "candidate": "random",
                "critic_verdict": "approve",
                "validation_auc": 0.5,
                "test_metrics": {"auc_roc": 0.5, "split": "test"},
                "optimism_gap": 0.0,
                "reason": "fixture",
            },
        )


def test_replay_proposer_is_file_only_and_deterministic():
    source = (ROOT / "proposers" / "replay.py").read_text(encoding="utf-8").lower()
    for banned in ("openai", "anthropic", "requests", "urllib", "httpx", "api_key"):
        assert banned not in source
    proposer = ReplayProposer(ROOT / "proposers" / "proposals_replay.jsonl")
    first = proposer.propose([])
    again = ReplayProposer(ROOT / "proposers" / "proposals_replay.jsonl").propose([])
    assert first == again
    assert "learning_rate" in first
    contract = load_contract()
    space = load_search_space()
    left = GreedyProposer(space, contract["baseline"])
    right = GreedyProposer(space, contract["baseline"])
    assert left.propose([]) == right.propose([])
    random_a = build_proposer("random", contract=contract, seed=20261109)
    random_b = build_proposer("random", contract=contract, seed=20261109)
    assert random_a.propose([]) == random_b.propose([])


def test_subprocess_timeout_stops_a_sleep(tmp_path):
    """The Yambda loop kills an over-time trial with subprocess timeout."""

    script = tmp_path / "sleep_once.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    import subprocess
    import sys
    import time

    started = time.perf_counter()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, str(script)], timeout=1, check=False)
    assert time.perf_counter() - started < 10
