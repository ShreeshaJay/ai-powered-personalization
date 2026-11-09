"""
Chapter 15: Silent Degradation Exercise
=======================================
Entry script. Retrains the Chapter 6 XGBoost ranker on a 30-day Yambda
window, corrupts its scoring path in three schema-valid ways, and records
which monitors fire and which stay green, for the aggregate and for a hashed
30 percent user segment.

Scenarios (the test day is corrupted; training is untouched)
------------------------------------------------------------
- clean              control: the test day as it arrived
- zero_fill_all      the top two eligible features arrive as 0 for everyone
- zero_fill_segment  the same, for the hashed user segment only
- mean_fill_all      the same features arrive as their training-set mean

Every monitor compares the test day with the reference day (the last day of
the train window, never used for fitting or early stopping). The clean
control shows how far each signal moves without any corruption.

Usage
-----
# Smoke run on generated data (no dataset needed)
python run_exercise.py --synthetic --run_name smoke

# Reader run on Yambda 50M (CPU only)
#   Linux/macOS:  export YAMBDA_DATA_DIR=/path/to/yambda/flat/50m
#   PowerShell:   $env:YAMBDA_DATA_DIR = "C:\\path\\to\\yambda\\flat\\50m"
python run_exercise.py

Outputs are written to outputs/<run_name>/ (see README.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

import corruption as cx
import monitoring_signals as ms
import ranker as rk
from synthetic_yambda import write_synthetic_yambda

logger = logging.getLogger("ch15")

HERE = Path(__file__).resolve().parent
SLICES = ("aggregate", "segment", "rest")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExerciseConfig:
    ranker: rk.RankerConfig = field(default_factory=rk.RankerConfig)
    n_target_features: int = 2
    target_scope: str = "any"
    segment_fraction: float = 0.3
    segment_salt: int = 15
    ndcg_k: int = 10
    psi_bins: int = 10
    thresholds: ms.Thresholds = field(default_factory=ms.Thresholds)
    data_dir: Optional[str] = None
    ch6_dir: Optional[str] = None
    synthetic: bool = False
    synthetic_n_users: int = 400
    output_dir: str = str(HERE / "outputs")
    run_name: str = "yambda_30d"
    verify_data_hash: bool = True
    save_model: bool = True


@dataclass
class Batch:
    """One slice of one scored window: everything a monitor looks at."""
    frame: pl.DataFrame
    scores: np.ndarray
    tiebreak: np.ndarray
    days: float
    lookup_keys: Dict[str, np.ndarray]


@dataclass
class ExerciseResult:
    config: ExerciseConfig
    prepared: rk.PreparedData
    trained: rk.TrainedRanker
    targets: List[str]
    eligibility: pl.DataFrame
    fill_values: Dict[str, Dict[str, float]]
    segment: Dict[str, np.ndarray]
    reference_scores: np.ndarray
    scored: Dict[str, Dict[str, object]]
    signals: pl.DataFrame
    summary: pl.DataFrame
    quality: pl.DataFrame
    reliability: pl.DataFrame
    manifest: Dict[str, object]
    out_dir: Path


# =============================================================================
# Measurements
# =============================================================================

def _slice(frame: pl.DataFrame, scores: np.ndarray, tiebreak: np.ndarray,
           mask: np.ndarray, days: float, lookup_keys: Dict[str, np.ndarray]) -> Batch:
    return Batch(frame.filter(pl.Series(mask)), scores[mask], tiebreak[mask], days, lookup_keys)


def _pipeline_measures(b: Batch, contract) -> Dict[str, float]:
    schema = ms.schema_pass_rate(b.frame, contract)
    return {
        "schema_pass_rate": schema["pass_rate"],
        "schema_dtype_ok": schema["dtype_ok"],
        "rows_per_day": b.frame.height / b.days,
        "join_rate_user": ms.join_match_rate(b.frame["uid"].to_numpy(), b.lookup_keys["uid"]),
        "join_rate_item": ms.join_match_rate(b.frame["item_id"].to_numpy(), b.lookup_keys["item_id"]),
    }


def _quality(b: Batch, cfg: ExerciseConfig) -> Dict[str, float]:
    return ms.quality_metrics(b.frame["uid"].to_numpy(), b.frame["label"].to_numpy(), b.scores,
                              k=cfg.ndcg_k, tiebreak=b.tiebreak)


def compare(reference: Batch, current: Batch, feature_cols: List[str], contract,
            cfg: ExerciseConfig) -> List[Dict[str, object]]:
    """Every monitor, current batch against reference batch, in a fixed row order."""
    th = cfg.thresholds
    rows: List[Dict[str, object]] = []
    for r in ms.pipeline_signals(_pipeline_measures(reference, contract),
                                 _pipeline_measures(current, contract), th):
        rows.append({**r, "layer": "pipeline", "feature": "(all features)"})
    for f in feature_cols:
        for r in ms.distribution_signals(reference.frame[f], current.frame[f], th, cfg.psi_bins):
            rows.append({**r, "layer": "feature", "feature": f})
    for r in ms.score_signals(reference.scores, current.scores, th, cfg.psi_bins):
        rows.append({**r, "layer": "score", "feature": "(model score)"})
    for r in ms.quality_signals(_quality(reference, cfg), _quality(current, cfg), th, cfg.ndcg_k):
        rows.append({**r, "layer": "quality", "feature": "(model score)"})
    return rows


def monitor_exemptions(feature_cols: List[str], binary_features=()) -> Dict[str, str]:
    """Monitors computed but not alerted on, with the reason.

    Keys are a feature name (every distribution monitor on it) or
    'feature:signal' (that monitor only).
    """
    out = {}
    for f in feature_cols:
        if f.endswith("_encoded") and f.startswith(rk.IDENTIFIER_PREFIXES):
            out[f] = ("label-encoded identifier: its distribution tracks who is active that day, "
                      "not data health; the join-match-rate monitor covers it")
        elif f == "day_of_week":
            out[f] = ("calendar feature: a one-day window holds at most two of its seven values, so "
                      "two different days differ by construction")
        elif f in binary_features:
            out[f"{f}:quantiles"] = ("two-valued feature: its quantiles jump between the two values "
                                     "whenever its share crosses a quantile level")
    return out


def _exemption(exemptions: Dict[str, str], feature: str, signal: str) -> Optional[str]:
    return exemptions.get(feature) or exemptions.get(f"{feature}:{signal}")


def _fires(stat: float, threshold: float) -> Optional[bool]:
    return None if stat is None or math.isnan(stat) else bool(stat > threshold)


STATUS_LEGEND = {
    "FIRES": "over threshold on the scenario and not on the clean control",
    "FIRES, clean also over": ("over threshold on both, and the scenario statistic exceeds the clean "
                               "statistic by more than the threshold"),
    "clean also over": ("over threshold on both, with no material difference from clean: "
                        "the monitor cannot attribute the change"),
    "green": "under threshold",
    "false alarm": "over threshold on the clean control (no corruption)",
    "exempt": "computed but not alerted on (see exempt_reason)",
    "n/a": "statistic undefined for this slice",
}


def _status(scenario: str, stat: float, clean_stat: float, threshold: float,
            exempt: bool = False) -> str:
    fires, fires_clean = _fires(stat, threshold), _fires(clean_stat, threshold)
    if exempt:
        return "exempt"
    if fires is None:
        return "n/a"
    if scenario == "clean":
        return "false alarm" if fires else "green"
    if not fires:
        return "green"
    if not fires_clean:
        return "FIRES"
    return "FIRES, clean also over" if stat - clean_stat > threshold else "clean also over"


# =============================================================================
# Summaries
# =============================================================================

FAMILIES = [
    ("Schema validation pass rate", "pipeline", ("schema_pass",), False),
    ("Rows per day", "pipeline", ("rows_per_day",), False),
    ("Join match rate (user, item)", "pipeline", ("join_rate_user", "join_rate_item"), False),
    ("Null rate", "feature", ("null_rate",), True),
    ("Zero rate", "feature", ("zero_rate",), True),
    ("Mean", "feature", ("mean",), True),
    ("Quantiles", "feature", ("quantiles",), True),
    ("KS statistic", "feature", ("ks",), True),
    ("PSI", "feature", ("psi",), True),
    ("Score mean", "score", ("score_mean",), None),
    ("Score quantiles, KS, PSI, entropy", "score",
     ("score_quantiles", "score_ks", "score_psi", "score_entropy"), None),
    ("AUC", "quality", ("auc",), None),
    ("GAUC", "quality", ("gauc",), None),
    ("NDCG@10", "quality", ("ndcg@10",), None),
    ("Log loss", "quality", ("log_loss",), None),
    ("ECE", "quality", ("ece",), None),
    ("Calibration gap", "quality", ("calibration_gap",), None),
]


_PRIORITY = ("false alarm", "FIRES", "FIRES, clean also over", "clean also over", "green", "n/a")


def summarize(signals: pl.DataFrame, ndcg_k: int) -> pl.DataFrame:
    """Collapse signal rows into one status per (scenario, slice, signal family).

    A family takes the most alarming status among its rows, in the order of
    _PRIORITY. Exempt rows are left out.
    """
    out = []
    for (scenario, slc), grp in signals.group_by(["scenario", "slice"], maintain_order=True):
        for label, layer, names, targets_only in FAMILIES:
            names = tuple(n.replace("ndcg@10", f"ndcg@{ndcg_k}") for n in names)
            label = label.replace("NDCG@10", f"NDCG@{ndcg_k}")
            rows = grp.filter((pl.col("layer") == layer) & pl.col("signal").is_in(names)
                              & (pl.col("status") != "exempt"))
            if targets_only:
                rows = rows.filter(pl.col("is_target"))
            statuses = rows["status"].to_list()
            status = next((s for s in _PRIORITY if s in statuses), "n/a")
            out.append({"scenario": scenario, "slice": slc, "family": label, "status": status,
                        "rows": len(statuses),
                        "rows_over_threshold": int(rows["fires"].fill_null(False).sum()),
                        "rows_over_threshold_on_clean": int(rows["fires_on_clean"].fill_null(False).sum()),
                        "max_statistic": float(rows["statistic"].max()) if rows.height else float("nan")})
    return pl.DataFrame(out)


def clean_false_alarms(signals: pl.DataFrame) -> pl.DataFrame:
    """Monitors over threshold on the uncorrupted test day: the noise floor of each threshold."""
    return (signals.filter((pl.col("scenario") == "clean") & (pl.col("status") == "false alarm"))
            .select("slice", "layer", "signal", "feature", "threshold", "statistic"))


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.4f}" if abs(x) < 1000 else f"{x:,.0f}"


def summary_markdown(result_parts: Dict[str, object]) -> str:
    """The headline tables, written to signal_summary.md."""
    summary: pl.DataFrame = result_parts["summary"]
    quality: pl.DataFrame = result_parts["quality"]
    signals: pl.DataFrame = result_parts["signals"]
    targets = result_parts["targets"]
    fills = result_parts["fill_values"]
    columns = [("clean", "aggregate"), ("zero_fill_all", "aggregate"),
               ("zero_fill_segment", "aggregate"), ("zero_fill_segment", "segment"),
               ("zero_fill_segment", "rest"), ("mean_fill_all", "aggregate")]
    lines = ["# Chapter 15 exercise: which monitors fire", "",
             f"Run: `{result_parts['run_name']}`. Target features: "
             + ", ".join(f"`{t}` (zero-fill {_fmt(fills['zero'][t])}, mean-fill {_fmt(fills['mean'][t])})"
                         for t in targets) + ".", "",
             "Every monitor compares the scored test day with the reference day. Status legend:", ""]
    lines += [f"- **{k}**: {v}" for k, v in STATUS_LEGEND.items()]
    lines += ["", "## Status by signal family", "",
             "| Signal | " + " | ".join(f"{s} / {c}" for s, c in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for family in summary["family"].unique(maintain_order=True):
        cells = []
        for s, c in columns:
            row = summary.filter((pl.col("scenario") == s) & (pl.col("slice") == c)
                                 & (pl.col("family") == family))
            cells.append(row["status"][0] if row.height else "n/a")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")

    lines += ["", "## Target-feature statistics (aggregate)", "",
              "| Scenario | Feature | Signal | Statistic | Clean statistic | Threshold | Status |",
              "|---|---|---|---|---|---|---|"]
    tf = signals.filter(pl.col("is_target") & (pl.col("slice") == "aggregate")
                        & (pl.col("scenario") != "clean"))
    for r in tf.iter_rows(named=True):
        lines.append(f"| {r['scenario']} | {r['feature']} | {r['signal']} | {_fmt(r['statistic'])} | "
                     f"{_fmt(r['clean_statistic'])} | {_fmt(r['threshold'])} | {r['status']} |")

    alarms = clean_false_alarms(signals)
    lines += ["", "## False alarms on the clean test day", "",
              "Monitors already over threshold with no corruption. On untouched features these "
              "statistics are identical in every scenario.", ""]
    if alarms.height:
        lines += ["| Slice | Signal | Feature | Statistic | Threshold |", "|---|---|---|---|---|"]
        for r in alarms.iter_rows(named=True):
            lines.append(f"| {r['slice']} | {r['signal']} | {r['feature']} | {_fmt(r['statistic'])} | "
                         f"{_fmt(r['threshold'])} |")
    else:
        lines.append("None.")

    exempt = (signals.filter(pl.col("status") == "exempt")
              .group_by(["feature", "exempt_reason"], maintain_order=True)
              .agg(pl.col("signal").unique(maintain_order=True)))
    if exempt.height:
        lines += ["", "## Exempt monitors", ""]
        lines += [f"- `{r['feature']}` ({', '.join(r['signal'])}): {r['exempt_reason']}"
                  for r in exempt.iter_rows(named=True)]

    for slc in SLICES:
        lines += ["", f"## Quality metrics, slice: {slc}", "",
                  "| Metric | Reference day | " + " | ".join(s.name for s in cx.SCENARIOS) + " |",
                  "|---|---|" + "---|" * len(cx.SCENARIOS)]
        q = quality.filter(pl.col("slice") == slc)
        for metric in q["metric"].unique(maintain_order=True):
            m = q.filter(pl.col("metric") == metric)
            ref = m["reference"][0]
            vals = [m.filter(pl.col("scenario") == s.name)["value"][0] for s in cx.SCENARIOS]
            lines.append(f"| {metric} | {_fmt(ref)} | " + " | ".join(_fmt(v) for v in vals) + " |")
    return "\n".join(lines) + "\n"


# =============================================================================
# Provenance
# =============================================================================

def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "polars", "pyarrow", "scipy", "scikit-learn", "xgboost"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def peak_rss_mb() -> Optional[float]:
    """Peak resident memory of this process so far, where the platform reports it (not Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024


def data_provenance(data_dir: Path, verify: bool) -> Dict[str, object]:
    listens = data_dir / "listens.parquet"
    info: Dict[str, object] = {"listens_bytes": listens.stat().st_size}
    if verify:
        digest = rk.file_sha256(listens)
        info["listens_sha256"] = digest
        info["matches_published_yambda_flat_50m"] = digest == rk.YAMBDA_50M_LISTENS_SHA256
    return info


# =============================================================================
# The experiment
# =============================================================================

def run_experiment(cfg: ExerciseConfig) -> ExerciseResult:
    """Train, corrupt, monitor, and write every result file. Returns everything for the notebook."""
    t0 = time.time()
    timings: Dict[str, float] = {}
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.ranker.seed)

    ch6_path = rk.locate_ch6_pipeline(cfg.ch6_dir)
    ch6 = rk.import_ch6_pipeline(ch6_path)
    provenance = {"chapter6": rk.ch6_provenance(ch6_path)}
    logger.info("Chapter 6 pipeline: %s (sha256 %s)", ch6_path,
                provenance["chapter6"]["polars_pipeline_sha256"][:12])
    if provenance["chapter6"].get("body_matches_source") is False:
        logger.warning("The vendored Chapter 6 pipeline no longer matches its recorded source hash")

    tmp_dir = None
    if cfg.synthetic:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ch15_synthetic_"))
        data_dir = write_synthetic_yambda(tmp_dir, n_users=cfg.synthetic_n_users, seed=cfg.ranker.seed)
        provenance["data"] = {"synthetic": True}
    else:
        data_dir = rk.resolve_data_dir(cfg.data_dir)
        provenance["data"] = data_provenance(data_dir, cfg.verify_data_hash)
    try:
        logger.info("Preparing data from %s", "synthetic data" if cfg.synthetic else data_dir)
        prepared = rk.prepare_data(data_dir, cfg.ranker, ch6)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    timings["prepare_data_s"] = time.time() - t0

    t1 = time.time()
    trained = rk.train_ranker(prepared, cfg.ranker)
    timings["train_s"] = time.time() - t1
    if cfg.save_model:
        rk.save_ranker(out_dir, trained, prepared, cfg.ranker)

    t2 = time.time()
    frames = prepared.frames
    feature_cols = prepared.feature_columns
    targets, eligibility = rk.select_target_features(trained.importance, frames["train"],
                                                     cfg.n_target_features, cfg.target_scope)
    fill_values = {fill: cx.compute_fill_values(frames["train"], targets, fill)
                   for fill in ("zero", "mean")}
    contract = rk.feature_contract(frames["train"], feature_cols)
    binary = [f for f in feature_cols if frames["train"][f].n_unique() <= 2]
    exemptions = monitor_exemptions(feature_cols, binary)
    lookup_keys = {w: prepared.lookup_keys(w) for w in ("reference", "test")}

    segment = {w: cx.segment_mask(frames[w]["uid"].to_numpy(), cfg.segment_fraction, cfg.segment_salt)
               for w in ("reference", "test")}
    tiebreak = {w: cx.splitmix64(np.arange(frames[w].height, dtype=np.uint64) + np.uint64(cfg.ranker.seed))
                for w in ("reference", "test")}
    masks = {w: {"aggregate": np.ones(frames[w].height, dtype=bool), "segment": segment[w],
                 "rest": ~segment[w]} for w in ("reference", "test")}

    reference_scores = rk.predict_scores(trained, frames["reference"])
    scored: Dict[str, Dict[str, object]] = {}
    for scenario in cx.SCENARIOS:
        frame = cx.corrupt(frames["test"], scenario, fill_values, segment["test"])
        scored[scenario.name] = {"frame": frame, "scores": rk.predict_scores(trained, frame)}

    def batch(window: str, frame: pl.DataFrame, scores: np.ndarray, slc: str) -> Batch:
        return _slice(frame, scores, tiebreak[window], masks[window][slc],
                      prepared.window_days[window], lookup_keys[window])

    signal_rows, quality_rows, reliability_rows = [], [], []
    for slc in SLICES:
        ref_b = batch("reference", frames["reference"], reference_scores, slc)
        clean_b = batch("test", scored["clean"]["frame"], scored["clean"]["scores"], slc)
        clean_rows = compare(ref_b, clean_b, feature_cols, contract, cfg)
        q_ref = _quality(ref_b, cfg)
        for scenario in cx.SCENARIOS:
            cur_b = batch("test", scored[scenario.name]["frame"], scored[scenario.name]["scores"], slc)
            rows = clean_rows if scenario.name == "clean" else compare(
                ref_b, cur_b, feature_cols, contract, cfg)
            for r, c in zip(rows, clean_rows):
                reason = _exemption(exemptions, r["feature"], r["signal"]) if r["layer"] == "feature" else None
                signal_rows.append({
                    "scenario": scenario.name, "slice": slc, "layer": r["layer"], "signal": r["signal"],
                    "feature": r["feature"], "is_target": r["feature"] in targets,
                    "threshold": r["threshold"], "reference_value": r["reference_value"],
                    "clean_value": c["current_value"], "scenario_value": r["current_value"],
                    "clean_statistic": c["statistic"], "statistic": r["statistic"],
                    "ks_p_value": r.get("p_value", float("nan")),
                    "fires_on_clean": _fires(c["statistic"], c["threshold"]),
                    "fires": _fires(r["statistic"], r["threshold"]),
                    "status": _status(scenario.name, r["statistic"], c["statistic"], r["threshold"],
                                      exempt=reason is not None),
                    "exempt_reason": reason,
                })
            q_cur = _quality(cur_b, cfg)
            for metric, value in q_cur.items():
                quality_rows.append({"scenario": scenario.name, "slice": slc, "metric": metric,
                                     "reference": q_ref[metric], "value": value,
                                     "delta_vs_reference": value - q_ref[metric]})
            for row in ms.reliability_table(cur_b.frame["label"].to_numpy(), cur_b.scores):
                reliability_rows.append({"scenario": scenario.name, "slice": slc, **row})
        for row in ms.reliability_table(ref_b.frame["label"].to_numpy(), ref_b.scores):
            reliability_rows.append({"scenario": "reference_day", "slice": slc, **row})

    signals = pl.DataFrame(signal_rows, infer_schema_length=None)
    quality = pl.DataFrame(quality_rows)
    clean_q = quality.filter(pl.col("scenario") == "clean").select(
        "slice", "metric", pl.col("value").alias("clean"))
    quality = quality.join(clean_q, on=["slice", "metric"], how="left", maintain_order="left").with_columns(
        (pl.col("value") - pl.col("clean")).alias("delta_vs_clean"))
    reliability = pl.DataFrame(reliability_rows)
    summary = summarize(signals, cfg.ndcg_k)
    timings["monitor_s"] = time.time() - t2
    timings["total_s"] = time.time() - t0

    test = frames["test"]
    manifest = {
        "run_name": cfg.run_name,
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("ranker", "thresholds")},
        "ranker_config": asdict(cfg.ranker),
        "thresholds": cfg.thresholds.to_dict(),
        "provenance": provenance,
        "versions": _versions(),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count(), "peak_rss_mb": peak_rss_mb()},
        "windows": {"rows": prepared.row_counts, "days": prepared.window_days, "bounds": prepared.bounds,
                    "positive_rate": {n: float(f["label"].mean()) for n, f in frames.items()},
                    "users": {n: int(f["uid"].n_unique()) for n, f in frames.items()}},
        "model": {"feature_columns": feature_cols, "best_iteration": trained.best_iteration,
                  "best_val_auc": trained.best_val_auc},
        "targets": targets,
        "monitor_exemptions": exemptions,
        "binary_features": binary,
        "fill_values": fill_values,
        "segment": {"fraction_requested": cfg.segment_fraction, "salt": cfg.segment_salt,
                    "test_user_share": float(test.filter(pl.Series(segment["test"]))["uid"].n_unique()
                                             / max(test["uid"].n_unique(), 1)),
                    "test_row_share": float(segment["test"].mean())},
        "timings_s": timings,
    }

    parts = {"summary": summary, "quality": quality, "signals": signals, "targets": targets,
             "fill_values": fill_values, "run_name": cfg.run_name}
    write_outputs(out_dir, manifest, trained, eligibility, signals, summary, quality, reliability, parts)
    logger.info("Done in %.1f s; results in %s", timings["total_s"], out_dir)
    return ExerciseResult(cfg, prepared, trained, targets, eligibility, fill_values, segment,
                          reference_scores, scored, signals, summary, quality, reliability, manifest, out_dir)


def write_outputs(out_dir: Path, manifest, trained, eligibility, signals, summary, quality,
                  reliability, parts) -> None:
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    eligibility.write_csv(out_dir / "feature_importance.csv")
    signals.write_csv(out_dir / "signals_long.csv")
    summary.write_csv(out_dir / "signal_summary.csv")
    clean_false_alarms(signals).write_csv(out_dir / "clean_false_alarms.csv")
    quality.write_csv(out_dir / "quality_metrics.csv")
    reliability.write_csv(out_dir / "reliability_bins.csv")
    (out_dir / "signal_summary.md").write_text(summary_markdown(parts), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None) -> ExerciseConfig:
    d, r = ExerciseConfig(), rk.RankerConfig()
    p = argparse.ArgumentParser(description="Chapter 15 silent-degradation exercise")
    p.add_argument("--data_dir", default=None, help=f"Yambda flat/50m folder (default: ${rk.DATA_DIR_ENV})")
    p.add_argument("--ch6_dir", default=None,
                   help="chapter6_ranking folder to use instead of the vendored pipeline copy")
    p.add_argument("--synthetic", action="store_true", help="Use generated Yambda-shaped data (smoke run)")
    p.add_argument("--run_name", default=None, help="Output subfolder (default: yambda_<train_days>d or smoke)")
    p.add_argument("--output_dir", default=d.output_dir)
    p.add_argument("--train_days", type=int, default=r.train_days)
    p.add_argument("--test_days", type=int, default=r.test_days)
    p.add_argument("--reference_days", type=int, default=r.reference_days)
    p.add_argument("--val_ratio", type=float, default=r.val_ratio)
    p.add_argument("--n_estimators", type=int, default=r.n_estimators)
    p.add_argument("--max_depth", type=int, default=r.max_depth)
    p.add_argument("--learning_rate", type=float, default=r.learning_rate)
    p.add_argument("--early_stopping_rounds", type=int, default=r.early_stopping_rounds)
    p.add_argument("--seed", type=int, default=r.seed)
    p.add_argument("--n_jobs", type=int, default=r.n_jobs)
    p.add_argument("--n_target_features", type=int, default=d.n_target_features)
    p.add_argument("--target_scope", choices=sorted(rk.TARGET_SCOPES), default=d.target_scope,
                   help="Restrict the corrupted features to user_* or item_* lookup features")
    p.add_argument("--segment_fraction", type=float, default=d.segment_fraction)
    p.add_argument("--segment_salt", type=int, default=d.segment_salt)
    p.add_argument("--ndcg_k", type=int, default=d.ndcg_k)
    p.add_argument("--no_verify_data_hash", action="store_true", help="Skip the SHA-256 check of listens.parquet")
    p.add_argument("--no_save_model", action="store_true")
    a = p.parse_args(argv)
    ranker_cfg = rk.RankerConfig(
        train_days=a.train_days, test_days=a.test_days, reference_days=a.reference_days,
        val_ratio=a.val_ratio, n_estimators=a.n_estimators, max_depth=a.max_depth,
        learning_rate=a.learning_rate, early_stopping_rounds=a.early_stopping_rounds,
        seed=a.seed, n_jobs=a.n_jobs)
    run_name = a.run_name or ("smoke" if a.synthetic else f"yambda_{a.train_days}d")
    return ExerciseConfig(
        ranker=ranker_cfg, n_target_features=a.n_target_features, target_scope=a.target_scope,
        segment_fraction=a.segment_fraction,
        segment_salt=a.segment_salt, ndcg_k=a.ndcg_k, data_dir=a.data_dir, ch6_dir=a.ch6_dir,
        synthetic=a.synthetic, output_dir=a.output_dir, run_name=run_name,
        verify_data_hash=not a.no_verify_data_hash, save_model=not a.no_save_model)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = parse_args(argv)
    result = run_experiment(cfg)
    print(summary_markdown({"summary": result.summary, "quality": result.quality,
                            "signals": result.signals, "targets": result.targets,
                            "fill_values": result.fill_values, "run_name": cfg.run_name}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
