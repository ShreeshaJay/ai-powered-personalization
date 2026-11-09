"""
Train the lightweight ranker (Chapter 6 XGBoost) and the heavy ranker
(Chapter 7 MMoE) on one feature snapshot.

Time is split into a feature window (first part of the data) and a label
window (the rest). Chapter 6's HistoricalFeatureBuilder is fitted on the
feature window only, and both rankers train on listens from the label window,
so no training row contributes to its own aggregate features. The feature
store serves exactly this snapshot, which is what makes the serving features
equal the training features (tests/test_services.py checks it).

Both models use the published classes and training functions:
  Chapter 6: XGBoostRanker.fit / evaluate / save (label: played_ratio_pct >= 50).
  Chapter 7: MMoEFeatureProcessor, MMoE, MultiTaskLoss, MMoEDataset, collate_fn,
             train_epoch, evaluate, and _add_time_windowed_labels for like and
             dislike labels (LIKE_WINDOW_TS, 24 hours, cut at the end of each split).
"""

import copy
import dataclasses
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader

from common import published
from common.artifacts import sha256_file, write_json

logger = logging.getLogger("ch13.build")

COMPLETION_THRESHOLD = 50


def add_labels(rows: pd.DataFrame, likes: pd.DataFrame, dislikes: pd.DataFrame,
               label_end_ts: Optional[int] = None) -> pd.DataFrame:
    """
    Chapter 7's five task labels, computed with its own windowed-join function.
    Likes and dislikes at or after `label_end_ts` (the end of the rows' split)
    are ignored, so training labels never read events from the validation period.
    """
    train_mmoe = published.chapter7_train_mmoe()
    lf = pl.from_pandas(rows).lazy().with_columns([
        (pl.col("played_ratio_pct") > 0).cast(pl.Int8).alias("label_engagement"),
        (pl.col("played_ratio_pct") >= COMPLETION_THRESHOLD).cast(pl.Int8).alias("label"),
        (pl.col("played_ratio_pct") / 100.0).clip(0, 1).alias("label_play_ratio"),
    ])
    lf = train_mmoe._add_time_windowed_labels(lf, pl.from_pandas(likes).lazy(), "label_liked",
                                              train_mmoe.LIKE_WINDOW_TS, label_end_ts)
    lf = train_mmoe._add_time_windowed_labels(lf, pl.from_pandas(dislikes).lazy(), "label_disliked",
                                              train_mmoe.LIKE_WINDOW_TS, label_end_ts)
    labelled = lf.collect().to_pandas()
    labelled = labelled.astype({c: rows[c].dtype for c in rows.columns})
    # Polars 2.x joins and group_by do not keep row order, so ties are broken on every column.
    keys = ["timestamp", "uid", "item_id"]
    keys += sorted(c for c in labelled.columns if c not in keys)
    return labelled.sort_values(keys, kind="stable").reset_index(drop=True)


def train_lightweight(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, *, model_version: str,
                      feature_set_version: str, n_estimators: int, n_jobs: int, seed: int) -> dict:
    ch6 = published.chapter6("xgboost_ranker")
    started = time.perf_counter()
    y_train = (train_df["played_ratio_pct"] >= COMPLETION_THRESHOLD).astype(int)
    y_val = (val_df["played_ratio_pct"] >= COMPLETION_THRESHOLD).astype(int)
    model = ch6.XGBoostRanker(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    model.fit(train_df, y_train, eval_set=[(val_df, y_val)], verbose=False)
    metrics = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in model.evaluate(val_df, y_val).items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))

    pipeline = model.feature_pipeline
    required = {
        "context": ["timestamp", "is_organic"],
        "categorical": list(pipeline.categorical_columns),
        "numerical": list(pipeline.numerical_columns),
        "historical": list(pipeline.historical_columns),
    }
    files = {name: sha256_file(out_dir / name) for name in
             ("model.json", "feature_encoder.pkl", "feature_pipeline.pkl", "metadata.json")}
    write_json(out_dir / "serving_manifest.json", {
        "model_name": "xgboost",
        "model_version": model_version,
        "feature_set_version": feature_set_version,
        "score_name": "p_completion",
        "label": f"played_ratio_pct >= {COMPLETION_THRESHOLD}",
        "required_columns": required,
        "files": files,
    })
    metrics["seconds"] = round(time.perf_counter() - started, 1)
    logger.info("Lightweight ranker: %s", metrics)
    return metrics


def _processor_state(processor) -> dict:
    # Same keys as Chapter 7's export_inference_artifacts writes to inference/feature_processor.pkl.
    return {
        "sparse_features": processor.sparse_features,
        "dense_features": processor.dense_features,
        "task_names": processor.task_names,
        "vocab_sizes": processor.vocab_sizes,
        "sparse_encoders": processor.sparse_encoders,
        "dense_scaler": processor.dense_scaler,
    }


def train_heavy(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, *, model_version: str,
                feature_set_version: str, epochs: int, batch_size: int, seed: int) -> dict:
    tm = published.chapter7_train_mmoe()
    mm = published.chapter7_mmoe()
    add_derived = published.chapter6("feature_encoder").add_derived_features
    started = time.perf_counter()
    torch.manual_seed(seed)

    train_df = add_derived(train_df)
    val_df = add_derived(val_df)
    missing = [c for c in tm.SPARSE_FEATURES + tm.DENSE_FEATURES if c not in train_df.columns]
    if missing:
        raise RuntimeError(f"MMoE training frame is missing features: {missing}")

    processor = tm.MMoEFeatureProcessor(tm.SPARSE_FEATURES, tm.DENSE_FEATURES, tm.TASK_NAMES)
    train_sparse, train_dense, train_labels = processor.fit_transform(train_df)
    val_sparse, val_dense, val_labels = processor.transform(val_df)
    train_loader = DataLoader(mm.MMoEDataset(train_sparse, train_dense, train_labels), batch_size=batch_size,
                              shuffle=True, collate_fn=mm.collate_fn, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(mm.MMoEDataset(val_sparse, val_dense, val_labels), batch_size=batch_size,
                            shuffle=False, collate_fn=mm.collate_fn)

    config = mm.MMoEConfig(
        sparse_features=processor.vocab_sizes,
        dense_features=list(tm.DENSE_FEATURES),
        num_tasks=len(tm.TASK_NAMES),
        task_names=list(tm.TASK_NAMES),
    )
    model = mm.MMoE(config)
    # Loss settings are train_mmoe.py's command-line defaults.
    criterion = mm.MultiTaskLoss(
        task_names=tm.TASK_NAMES,
        task_weights={"engagement": 1.0, "completion": 1.0, "play_ratio": 1.0, "like": 2.0, "dislike": 2.0},
        regression_tasks=tm.REGRESSION_TASKS,
        use_focal_loss={name: False for name in tm.TASK_NAMES},
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    device = torch.device("cpu")

    best_auc, best_state, best_metrics, history = -1.0, None, None, []
    for epoch in range(epochs):
        loss, _ = tm.train_epoch(model, train_loader, optimizer, criterion, device)
        val = tm.evaluate(model, val_loader, criterion, device)
        auc = float(val["completion"]["auc_roc"])
        history.append({"epoch": epoch + 1, "train_loss": round(float(loss), 4), "val_completion_auc": round(auc, 4)})
        logger.info("MMoE epoch %d/%d: train loss %.4f, val completion AUC %.4f", epoch + 1, epochs, loss, auc)
        if auc > best_auc:
            best_auc, best_state, best_metrics = auc, copy.deepcopy(model.state_dict()), val
    model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(config),
        "model_version": model_version,
        "feature_set_version": feature_set_version,
    }, out_dir / "mmoe.pt")
    with open(out_dir / "feature_processor.pkl", "wb") as handle:
        pickle.dump(_processor_state(processor), handle)

    task_metrics = {}
    for task in tm.TASK_NAMES:
        values = best_metrics[task]
        key = "mse" if task in tm.REGRESSION_TASKS else "auc_roc"
        task_metrics[task] = {key: round(float(values[key]), 4)}
        if "positive_rate" in values:
            task_metrics[task]["positive_rate"] = round(float(values["positive_rate"]), 4)
    files = {name: sha256_file(out_dir / name) for name in ("mmoe.pt", "feature_processor.pkl")}
    write_json(out_dir / "serving_manifest.json", {
        "model_name": "mmoe",
        "model_version": model_version,
        "feature_set_version": feature_set_version,
        "score_name": "p_completion",
        "task_names": list(tm.TASK_NAMES),
        "sparse_features": list(tm.SPARSE_FEATURES),
        "dense_features": list(tm.DENSE_FEATURES),
        "files": files,
    })
    metrics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "best_val_completion_auc": round(best_auc, 4),
        "tasks": task_metrics,
        "history": history,
        "seconds": round(time.perf_counter() - started, 1),
    }
    logger.info("Heavy ranker: best val completion AUC %.4f (%.1fs)", best_auc, metrics["seconds"])
    return metrics
