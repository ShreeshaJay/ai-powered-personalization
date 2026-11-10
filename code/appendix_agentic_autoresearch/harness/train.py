"""One trial: Chapter 6 features, train-tail early stopping, validation scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from harness.chapter6 import polars_pipeline
from harness.contract import artifacts_dir, load_contract

EARLY_STOP_SPLIT = "train_tail"


def train_trial(
    *,
    run_dir: Path,
    trial_dir: Path,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fit one ranker and write validation predictions plus a manifest.

    The scoring frame is the validation day. Early stopping watches the last
    10 percent of the training window. This function has no parameter for a
    later window.
    """

    contract = load_contract()
    fraction = float(contract["split"]["early_stopping_fraction"])
    n_jobs = int(contract["n_jobs"])
    artifacts = artifacts_dir(run_dir)
    train_df = pl.read_parquet(artifacts / "train_rows.parquet")
    score_df = pl.read_parquet(artifacts / "validation_rows.parquet")
    prepared = prepare_frames(
        train_df,
        score_df,
        artifacts,
        fraction,
    )
    x_train, y_train = _matrix(prepared["train"], prepared["columns"])
    x_tail, y_tail = _matrix(prepared["tail"], prepared["columns"])
    x_score, _ = _matrix(prepared["score"], prepared["columns"])
    _require_both_classes(y_train, "fit rows")
    _require_both_classes(y_tail, "train tail")
    model = fit_xgboost(
        x_train,
        y_train,
        x_tail,
        y_tail,
        config=config,
        seed=seed,
        n_jobs=n_jobs,
    )
    probabilities = model.predict_proba(x_score)[:, 1]
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    predictions = pl.DataFrame(
        {
            "row_id": prepared["score"]["row_id"],
            "prediction": probabilities.astype(np.float64),
        }
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = trial_dir / "validation_predictions.parquet"
    predictions.write_parquet(predictions_path)
    model.save_model(str(trial_dir / "model.json"))
    prepared["pipeline"].save(str(trial_dir / "pipeline"))
    best_iteration = getattr(model, "best_iteration", None)
    manifest = {
        "prediction_column": "prediction",
        "predictions_file": predictions_path.name,
        "feature_columns": list(prepared["columns"]),
        "config": _plain_config(config),
        "seed": int(seed),
        "metadata": {
            "early_stopping_split": EARLY_STOP_SPLIT,
            "early_stopping_fraction": fraction,
            "model_selection_split": "validation",
            "optimization_split": "validation",
            "preprocessor_fit_split": "train",
            "primary_metric": "auc_roc",
            "primary_split": "validation",
            "historical_feature_policy": "pit aggregates from an earlier window",
            "eval_set_description": "fit rows, then the train tail last",
            "metrics_computed_by": "harness",
            "train_fit_rows": prepared["fit_rows"],
            "train_tail_rows": prepared["tail_rows"],
            "validation_rows": prepared["score_rows"],
            "train_tail_start_timestamp": prepared["tail_start"],
            "best_iteration": None if best_iteration is None else int(best_iteration),
            "eval_metric_watched_on_train_tail": str(config["eval_metric"]),
        },
    }
    manifest_path = trial_dir / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "completed",
        "validation_rows": prepared["score_rows"],
        "train_tail_rows": prepared["tail_rows"],
    }


def prepare_frames(
    train_df: pl.DataFrame,
    score_df: pl.DataFrame,
    listens_dir: Path,
    early_stop_fraction: float,
) -> dict[str, Any]:
    """Fit the Chapter 6 pipeline on the early training rows and transform the rest."""

    if int(score_df["timestamp"].min()) <= int(train_df["timestamp"].max()):
        raise ValueError("Scoring rows overlap the training window")
    chapter6 = polars_pipeline()
    fit_df, tail_df = chapter6.split_validation_window(train_df, float(early_stop_fraction))
    if fit_df.height == 0 or tail_df.height == 0:
        raise ValueError("The train-tail split is empty")
    if int(fit_df["timestamp"].max()) >= int(tail_df["timestamp"].min()):
        raise ValueError("The train tail is not strictly after the fit rows")
    pipeline = chapter6.PolarsPipeline()
    pipeline.fit(fit_df)
    fit_rows, history = chapter6.split_training_history(str(listens_dir), fit_df, 0)
    if history.height == 0:
        raise ValueError("The history window used for training-row aggregates is empty")
    train_history = chapter6.PolarsHistoricalFeatures().fit(history)
    train_out, columns = pipeline.transform(fit_rows, history=train_history)
    tail_out, tail_columns = pipeline.transform(tail_df)
    score_out, score_columns = pipeline.transform(score_df)
    if list(tail_columns) != list(columns) or list(score_columns) != list(columns):
        raise ValueError("Feature columns differ across the fit, tail, and scoring frames")
    return {
        "pipeline": pipeline,
        "columns": list(columns),
        "train": train_out,
        "tail": tail_out,
        "score": score_out,
        "fit_rows": int(train_out.height),
        "tail_rows": int(tail_out.height),
        "score_rows": int(score_out.height),
        "tail_start": int(tail_df["timestamp"].min()),
    }


def fit_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_tail: np.ndarray,
    y_tail: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> xgb.XGBClassifier:
    """Early-stop on the train tail.

    ``early_stopping_rounds`` is applied only because this function passes an
    eval set, and that eval set is the train tail. The scoring day is not an
    argument.
    """

    model = xgb.XGBClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        scale_pos_weight=float(config["scale_pos_weight"]),
        objective="binary:logistic",
        eval_metric=str(config["eval_metric"]),
        tree_method="hist",
        device="cpu",
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbosity=0,
    )
    model.set_params(early_stopping_rounds=int(config["early_stopping_rounds"]))
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_tail, y_tail)],
        verbose=False,
    )
    return model


def predict_with_saved_pipeline(
    trial_dir: Path,
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Apply a trial's saved pipeline and model to rows from the same schema."""

    manifest = json.loads((trial_dir / "submission_manifest.json").read_text(encoding="utf-8"))
    columns = list(manifest["feature_columns"])
    chapter6 = polars_pipeline()
    pipeline = chapter6.PolarsPipeline().load(str(trial_dir / "pipeline"))
    transformed, available = pipeline.transform(frame)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"Saved pipeline did not produce columns: {missing}")
    matrix = transformed.select(columns).to_pandas().to_numpy(dtype=np.float32)
    model = xgb.XGBClassifier()
    model.load_model(str(trial_dir / "model.json"))
    probabilities = np.clip(model.predict_proba(matrix)[:, 1], 1e-6, 1.0 - 1e-6)
    return pl.DataFrame(
        {
            "row_id": transformed["row_id"],
            "prediction": probabilities.astype(np.float64),
        }
    )


def _matrix(frame: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame.select(columns).to_pandas().to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy().astype(int)
    return values, labels


def _require_both_classes(labels: np.ndarray, name: str) -> None:
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{name} contain a single class, so AUC early stopping cannot run")


def _plain_config(config: dict[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (np.floating, float)):
            plain[key] = float(value)
        elif isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            plain[key] = int(value)
        else:
            plain[key] = value
    return plain
