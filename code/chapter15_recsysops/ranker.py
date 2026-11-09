"""
Chapter 15: The Chapter 6 Ranker, Retrained for Monitoring
==========================================================
Retrains the Chapter 6 pointwise XGBoost ranker (listen completion,
played_ratio_pct >= 50) on a fixed, CPU-sized Yambda window, using the
Chapter 6 Polars feature pipeline.

What this module takes from Chapter 6
-------------------------------------
- `PolarsPipeline`: categorical encoding, the label, the calendar features,
  and the user and item window-aggregate features with their cold-start
  defaults
- `PolarsHistoricalFeatures`: the window-aggregate builder, fit per window
- `split_training_history`: the window before the fit window, which
  supplies the training rows' aggregates
- `POLARS_DTYPES`: the column dtypes the pipeline expects

All of them come from `ch6_polars_pipeline_vendored.py`, a verbatim copy of
`chapter6_ranking/utils/polars_pipeline.py` whose header records the source
commit and SHA-256. `--ch6_dir` points the exercise at another copy instead.

What this module does itself
----------------------------
1. The global temporal split, in days: [train window][30-minute gap]
   [test day], ending at the last event. Yambda timestamps are seconds (at
   5-second granularity), so a day is 86,400 units.
2. Window carving. The train window is split, in time order, into a fit
   window, a validation window (the last `val_ratio` of the rest, used only
   for early stopping), and a reference day (the last day before the gap,
   never used for training or model selection). The test day is never
   passed to `fit`.
3. Aggregates that never include a row's own label. Training rows take
   their user and item aggregates from the fit-window-length window before
   the fit window, validation rows from the fit window, and the reference
   and test days from the fit-window-length history that ends where each
   day starts, the way a daily feature-store job would serve them. The
   encoders stay as fitted on the fit window.
4. Point-in-time user-item counts. `previous_listen_count` and
   `has_listened_before` are recomputed across all loaded windows in time
   order, counting only the pair's plays before each event and within the
   same fit-window-length lookback, so every window sees equally long
   histories.
5. A total row order. Each transformed frame is sorted on all its columns,
   so training does not depend on the row order a Polars engine returns.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from monitoring_signals import ColumnContract

logger = logging.getLogger(__name__)

DATA_DIR_ENV = "YAMBDA_DATA_DIR"
VENDORED_CH6 = Path(__file__).resolve().parent / "ch6_polars_pipeline_vendored.py"

# Published SHA-256 of flat/50m/listens.parquet on huggingface.co/datasets/yandex/yambda
YAMBDA_50M_LISTENS_SHA256 = "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1"

# Yambda timestamps count seconds, rounded to multiples of 5. Listen counts in
# flat/50m repeat with a period of 86,400 units, so one unit is one second.
SECONDS_PER_DAY = 86_400
TS_UNITS_PER_DAY = SECONDS_PER_DAY
GAP_SECONDS = 1800
WINDOW_ORDER = ("history", "train", "val", "reference", "test")
SERVED_WINDOWS = ("reference", "test")
IDENTIFIER_PREFIXES = ("uid_", "item_id_")


# =============================================================================
# Loading the Chapter 6 code
# =============================================================================

def locate_ch6_pipeline(ch6_dir: Optional[str] = None) -> Path:
    """The vendored copy, or utils/polars_pipeline.py under an explicit chapter6_ranking folder."""
    if ch6_dir is None:
        return VENDORED_CH6
    path = Path(ch6_dir).expanduser().resolve() / "utils" / "polars_pipeline.py"
    if not path.is_file():
        raise FileNotFoundError(f"--ch6_dir has no utils/polars_pipeline.py: {path}")
    return path


def import_ch6_pipeline(path: Path) -> ModuleType:
    """Import the Chapter 6 pipeline module from a file path without touching sys.path."""
    spec = importlib.util.spec_from_file_location("ch6_polars_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_vendored(path: Path = VENDORED_CH6) -> Tuple[Dict[str, str], bytes]:
    """The source fields recorded in the vendored header, and the body after it (LF endings)."""
    lines = Path(path).read_bytes().replace(b"\r\n", b"\n").splitlines(keepends=True)
    rules = [i for i, line in enumerate(lines) if line.startswith(b"# =====")]
    header = b"".join(lines[: rules[1] + 1]).decode()
    fields = {
        "source_path": header.split("Source:")[1].split("\n")[1].strip("# \n"),
        "source_commit": header.split("at commit")[1].split()[0],
        "source_sha256": header.split("SHA-256 of the source file:")[1].split()[0],
    }
    return fields, b"".join(lines[rules[1] + 1:])


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def ch6_provenance(path: Path) -> Dict[str, object]:
    """Which Chapter 6 pipeline this run used.

    For the vendored copy: the source path, commit and SHA-256 from its header,
    and whether the body still matches that hash. For an --ch6_dir copy: its
    file hash and git state.
    """
    path = Path(path)
    if path == VENDORED_CH6:
        fields, body = split_vendored(path)
        return {"vendored_file": path.name, **fields,
                "polars_pipeline_sha256": hashlib.sha256(body).hexdigest(),
                "body_matches_source": hashlib.sha256(body).hexdigest() == fields["source_sha256"]}
    repo = path.parent
    status = _git(["status", "--porcelain", "--", path.name], repo)
    return {
        "polars_pipeline_path": str(path),
        "polars_pipeline_sha256": file_sha256(path),
        "repo_head_commit": _git(["rev-parse", "HEAD"], repo),
        "polars_pipeline_last_commit": _git(["log", "-1", "--format=%H", "--", path.name], repo),
        "polars_pipeline_uncommitted_changes": None if status is None else bool(status),
    }


def resolve_data_dir(explicit: Optional[str] = None) -> Path:
    """The Yambda flat directory holding listens.parquet: --data_dir, else YAMBDA_DATA_DIR."""
    value = explicit or os.environ.get(DATA_DIR_ENV)
    if not value:
        raise ValueError(
            f"No data directory. Pass --data_dir or set {DATA_DIR_ENV} to the Yambda flat/50m folder "
            "(download: https://huggingface.co/datasets/yandex/yambda), or use --synthetic."
        )
    path = Path(value).expanduser().resolve()
    if not (path / "listens.parquet").is_file():
        raise FileNotFoundError(f"listens.parquet not found in {path}")
    return path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RankerConfig:
    """Chapter 6 defaults for the model; windows sized for a CPU run."""
    train_days: int = 30
    test_days: int = 1
    reference_days: int = 1
    val_ratio: float = 0.1
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10
    seed: int = 42
    n_jobs: int = -1


# =============================================================================
# Windows and point-in-time features
# =============================================================================

def load_global_split(data_dir: Path, train_days: int, test_days: int,
                      dtypes: Dict[str, pl.DataType],
                      gap_seconds: int = GAP_SECONDS) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """[train_days][gap][test_days], ending at the last event, as in the Chapter 6 split.

    The test window includes the last timestamp; the train window excludes its end.
    """
    ts = pl.col("timestamp")
    lf = pl.scan_parquet(Path(data_dir) / "listens.parquet")
    lo, hi = lf.select(ts.min().alias("lo"), ts.max().alias("hi")).collect().row(0)
    test_start = hi - test_days * TS_UNITS_PER_DAY
    train_end = test_start - gap_seconds
    train_start = max(train_end - train_days * TS_UNITS_PER_DAY, lo)
    names = lf.collect_schema().names()
    cast = [pl.col(c).cast(t) for c, t in dtypes.items() if c in names]
    train = lf.filter((ts >= train_start) & (ts < train_end)).with_columns(cast).collect()
    test = lf.filter((ts >= test_start) & (ts <= hi)).with_columns(cast).collect()
    logger.info("Global split: train [%d, %d), gap %d s, test [%d, %d]", train_start, train_end,
                gap_seconds, test_start, hi)
    return train, test


def total_order(df: pl.DataFrame) -> pl.DataFrame:
    """Sort on every column, time first, so the row order does not depend on the Polars engine."""
    lead = [c for c in ("timestamp", "uid", "item_id") if c in df.columns]
    return df.sort(lead + [c for c in df.columns if c not in lead])


def refresh_lookups(pipeline, ch6: ModuleType, history: pl.DataFrame):
    """A copy of the fitted pipeline whose user and item lookup tables are rebuilt from history.

    The categorical encoders are shared with the original, so every encoded
    input keeps the meaning the model was trained on.
    """
    served = copy.copy(pipeline)
    served.hist_builder = ch6.PolarsHistoricalFeatures().fit(history)
    return served


def carve_windows(train_df: pl.DataFrame, reference_days: int,
                  val_ratio: float) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, int]]:
    """Split the Chapter 6 train window into fit, validation, and reference windows, in time order.

    reference: the last `reference_days` of the train window
    val:       the last `val_ratio` of what remains, by timestamp quantile
    fit:       everything earlier
    """
    if reference_days < 1:
        raise ValueError("reference_days must be at least 1")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    ts = pl.col("timestamp")
    ts_max = int(train_df["timestamp"].max())
    reference_start = ts_max + 1 - reference_days * TS_UNITS_PER_DAY
    reference = train_df.filter(ts >= reference_start)
    rest = train_df.filter(ts < reference_start)
    if rest.height == 0:
        raise ValueError("The reference window covers the whole train window; use a longer train_days")
    val_start = int(rest["timestamp"].quantile(1 - val_ratio, interpolation="nearest"))
    fit = rest.filter(ts < val_start)
    val = rest.filter(ts >= val_start)
    if min(fit.height, val.height, reference.height) == 0:
        raise ValueError("A window is empty; use a longer train_days")
    bounds = {"fit_start": int(fit["timestamp"].min()), "val_start": val_start,
              "reference_start": reference_start, "train_end": ts_max + 1}
    return fit, val, reference, bounds


def add_pit_user_item_features(frames: Dict[str, pl.DataFrame],
                               lookback_units: Optional[int] = None) -> Dict[str, pl.DataFrame]:
    """Recompute previous_listen_count and has_listened_before point in time.

    All windows are stacked in time order and, for each (uid, item_id) pair,
    each event counts the pair's earlier events in the loaded history, or only
    those in [t - lookback_units, t) when a lookback is given. Ties on
    timestamp keep window order, then row order.
    """
    names = [n for n in WINDOW_ORDER if n in frames]
    pair = ["uid", "item_id"]
    parts = [
        frames[n].select("uid", "item_id", "timestamp").with_columns(
            pl.lit(i, dtype=pl.Int8).alias("_w"), pl.int_range(frames[n].height).alias("_r"))
        for i, n in enumerate(names)
    ]
    history = (pl.concat(parts).sort(["timestamp", "_w", "_r"])
               .with_columns((pl.col("uid").cum_count().over(pair) - 1).alias("_k")))
    count = pl.col("_k")
    if lookback_units is not None:
        # Plays of the pair at or before t - lookback - 1, taken from the last such play's rank.
        last_rank = (history.group_by(pair + ["timestamp"]).agg(pl.col("_k").max().alias("_k_last"))
                     .rename({"timestamp": "_ts_last"}).sort("_ts_last"))
        history = (history.with_columns((pl.col("timestamp") - lookback_units - 1).alias("_cut"))
                   .sort("_cut")
                   .join_asof(last_rank, left_on="_cut", right_on="_ts_last", by=pair,
                              strategy="backward", check_sortedness=False))
        count = pl.col("_k") - (pl.col("_k_last") + 1).fill_null(0)
    history = (history.with_columns(count.cast(pl.Int32).alias("previous_listen_count"))
               .with_columns((pl.col("previous_listen_count") > 0).cast(pl.Int8).alias("has_listened_before")))
    out = {}
    for i, n in enumerate(names):
        pit = history.filter(pl.col("_w") == i).sort("_r")
        out[n] = frames[n].with_columns(pit["previous_listen_count"], pit["has_listened_before"])
    return out


@dataclass
class PreparedData:
    frames: Dict[str, pl.DataFrame]          # train (fit), val, reference, test; features + label
    feature_columns: List[str]
    pipeline: object                         # Chapter 6 PolarsPipeline fitted on the fit window
    served_pipelines: Dict[str, object]      # train, reference, test: the same encoders, own lookups
    bounds: Dict[str, int]
    window_days: Dict[str, float]

    @property
    def row_counts(self) -> Dict[str, int]:
        return {n: f.height for n, f in self.frames.items()}

    def lookup_keys(self, window: str) -> Dict[str, np.ndarray]:
        """The user and item keys of the lookup tables a window was scored with."""
        hb = self.served_pipelines.get(window, self.pipeline).hist_builder
        return {"uid": hb.user_features["uid"].to_numpy(), "item_id": hb.item_features["item_id"].to_numpy()}


def prepare_data(data_dir: Path, cfg: RankerConfig, ch6: ModuleType) -> PreparedData:
    """Split, carve windows, and build features with the Chapter 6 pipeline.

    The encoders and the validation rows' aggregates are fit on the fit
    window. Training rows take their aggregates from the fit-window-length
    window before it (`split_training_history`), so no row's own label is in
    its features; when the data do not reach back that far, Chapter 6 moves
    the earliest fit rows into that history. The reference and test days use
    aggregates rebuilt from the fit-window-length history that ends at the
    start of each day.
    """
    ts = pl.col("timestamp")
    train_raw, test_raw = load_global_split(data_dir, cfg.train_days, cfg.test_days, ch6.POLARS_DTYPES)
    if test_raw.height == 0:
        raise ValueError("The test window is empty")
    fit_raw, val_raw, reference_raw, bounds = carve_windows(train_raw, cfg.reference_days, cfg.val_ratio)
    bounds["test_start"] = int(test_raw["timestamp"].min())
    bounds["test_end"] = int(test_raw["timestamp"].max()) + 1
    bounds["lookback_units"] = bounds["val_start"] - bounds["fit_start"]

    pipeline = ch6.PolarsPipeline().fit(fit_raw)
    train_rows, train_history = ch6.split_training_history(str(Path(data_dir)), fit_raw)
    if train_rows.height == 0 or train_history.height == 0:
        raise ValueError("No history before the training rows; use a longer dataset or shorter train_days")
    bounds["train_history_start"] = int(train_history["timestamp"].min())
    bounds["train_rows_start"] = int(train_rows["timestamp"].min())
    served = {"train": refresh_lookups(pipeline, ch6, train_history)}
    train_df, feature_columns = pipeline.transform(train_rows, history=served["train"].hist_builder)
    frames = {"train": train_df, "val": pipeline.transform(val_raw)[0]}
    history = pl.concat([fit_raw, val_raw, reference_raw])
    for name, raw, start in (("reference", reference_raw, bounds["reference_start"]),
                             ("test", test_raw, bounds["test_start"])):
        window = history.filter((ts >= start - bounds["lookback_units"]) & (ts < start))
        served[name] = refresh_lookups(pipeline, ch6, window)
        frames[name] = served[name].transform(raw)[0]
    frames = {n: total_order(f) for n, f in frames.items()}
    frames = add_pit_user_item_features({"history": total_order(train_history), **frames},
                                        lookback_units=bounds["lookback_units"])
    del frames["history"]
    window_days = {
        "train": (bounds["val_start"] - bounds["train_rows_start"]) / TS_UNITS_PER_DAY,
        "val": (bounds["reference_start"] - bounds["val_start"]) / TS_UNITS_PER_DAY,
        "reference": float(cfg.reference_days),
        "test": float(cfg.test_days),
    }
    logger.info("Rows per window: %s", {n: f.height for n, f in frames.items()})
    return PreparedData(frames, list(feature_columns), pipeline, served, bounds, window_days)


# =============================================================================
# Training and scoring
# =============================================================================

@dataclass
class TrainedRanker:
    model: xgb.XGBClassifier
    feature_columns: List[str]
    best_iteration: int
    best_val_auc: float
    importance: pl.DataFrame


def to_matrix(df: pl.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.select(columns).to_pandas()


def gain_importance(model: xgb.XGBClassifier, columns: List[str]) -> pl.DataFrame:
    """Average gain, total gain, and split count per feature; unused features get zeros."""
    booster = model.get_booster()
    kinds = {k: booster.get_score(importance_type=k) for k in ("gain", "total_gain", "weight")}
    return pl.DataFrame({
        "feature": columns,
        "gain": [float(kinds["gain"].get(c, 0.0)) for c in columns],
        "total_gain": [float(kinds["total_gain"].get(c, 0.0)) for c in columns],
        "splits": [int(kinds["weight"].get(c, 0)) for c in columns],
    }).sort("gain", descending=True).with_row_index("gain_rank", offset=1)


def train_ranker(prepared: PreparedData, cfg: RankerConfig) -> TrainedRanker:
    """Fit on the fit window; early-stop on the validation window. The test window is not used."""
    cols = prepared.feature_columns
    fit, val = prepared.frames["train"], prepared.frames["val"]
    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
        min_child_weight=cfg.min_child_weight, subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree, objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=cfg.early_stopping_rounds, tree_method="hist",
        random_state=cfg.seed, n_jobs=cfg.n_jobs,
    )
    model.fit(to_matrix(fit, cols), fit["label"].to_numpy(),
              eval_set=[(to_matrix(val, cols), val["label"].to_numpy())], verbose=False)
    trained = TrainedRanker(model, cols, int(model.best_iteration), float(model.best_score),
                            gain_importance(model, cols))
    logger.info("Best iteration %d, validation AUC %.4f", trained.best_iteration, trained.best_val_auc)
    return trained


def predict_scores(trained: TrainedRanker, df: pl.DataFrame) -> np.ndarray:
    """Completion probability; uses the early-stopped best iteration."""
    return trained.model.predict_proba(to_matrix(df, trained.feature_columns))[:, 1]


# =============================================================================
# Target features and the data contract
# =============================================================================

TARGET_SCOPES = {"any": None, "user": "user_", "item": "item_"}


def select_target_features(importance: pl.DataFrame, train_df: pl.DataFrame,
                           n: int = 2, scope: str = "any") -> Tuple[List[str], pl.DataFrame]:
    """Top-n features by gain among non-identifier numeric features with more than two values.

    Excluded: identifier encodings (uid_*, item_id_*), label-encoded
    categoricals (*_encoded), and binary or constant features, for which a
    mean-fill would round to an existing category and repeat the zero-fill.
    scope 'user' or 'item' further restricts the choice to the user_* or
    item_* lookup features, which are constant across one user's (one
    item's) rows.
    """
    if scope not in TARGET_SCOPES:
        raise ValueError(f"scope must be one of {sorted(TARGET_SCOPES)}, got {scope!r}")
    prefix = TARGET_SCOPES[scope]

    def reason(feature: str) -> str:
        if feature.startswith(IDENTIFIER_PREFIXES):
            return "identifier encoding"
        if feature.endswith("_encoded"):
            return "categorical code"
        if train_df[feature].n_unique() <= 2:
            return "binary or constant"
        if prefix is not None and not feature.startswith(prefix):
            return f"outside target scope '{scope}'"
        return "eligible"

    table = importance.with_columns(
        pl.col("feature").map_elements(reason, return_dtype=pl.String).alias("eligibility"))
    eligible = table.filter((pl.col("eligibility") == "eligible") & (pl.col("gain") > 0))
    targets = eligible["feature"].head(n).to_list()
    if len(targets) < n:
        raise ValueError(f"Only {len(targets)} eligible target features")
    return targets, table.with_columns(pl.col("feature").is_in(targets).alias("is_target"))


_UNIT_INTERVAL = ("user_organic_ratio", "item_organic_ratio", "item_like_ratio", "has_listened_before")


def feature_contract(df: pl.DataFrame, feature_columns: List[str]) -> List[ColumnContract]:
    """The type-and-range contract a schema validator would enforce on the scoring frame.

    dtypes come from the training frame; bounds are the natural domain of each
    feature: frequencies and ratios in [0, 1], label codes >= -1 (unseen),
    calendar fields in range, and everything else non-negative.
    """
    contract = []
    for c in feature_columns:
        lower, upper = 0.0, None
        if c.endswith("_freq") or c in _UNIT_INTERVAL:
            upper = 1.0
        elif c.endswith("_encoded"):
            lower = -1.0
        elif c == "hour_of_day":
            upper = 23.0
        elif c == "day_of_week":
            upper = 6.0
        contract.append(ColumnContract(c, str(df.schema[c]), lower, upper))
    return contract


# =============================================================================
# Persistence
# =============================================================================

def save_ranker(out_dir: Path, trained: TrainedRanker, prepared: PreparedData,
                cfg: RankerConfig) -> Path:
    """Write model.json, the Chapter 6 pipeline lookup tables, and the window metadata."""
    model_dir = Path(out_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    trained.model.save_model(str(model_dir / "model.json"))
    prepared.pipeline.save(str(model_dir / "pipeline"))
    meta = {
        "feature_columns": trained.feature_columns,
        "best_iteration": trained.best_iteration,
        "best_val_auc": trained.best_val_auc,
        "bounds": prepared.bounds,
        "ranker_config": asdict(cfg),
        "note": "After the Chapter 6 transform, previous_listen_count and has_listened_before are "
                "recomputed point in time (ranker.add_pit_user_item_features). pipeline/ holds the "
                "encoders and the fit-window lookups used for validation rows; training rows used "
                "lookups from the window before the fit window, and scored days use lookups rebuilt "
                "by ranker.refresh_lookups.",
    }
    (model_dir / "ch15_model_metadata.json").write_text(json.dumps(meta, indent=2))
    return model_dir
