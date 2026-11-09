"""
Build a versioned artifact bundle for the five services.

    python -m pipeline_build.build_artifacts --profile synthetic
    python -m pipeline_build.build_artifacts --profile sample --data-dir <Yambda folder>
    python -m pipeline_build.build_artifacts --profile full   --data-dir <Yambda folder>

Profiles:
  tiny       A smaller synthetic bundle, built by the test suite in a temp folder.
  synthetic  Generated Yambda-shaped data. No dataset needed.
  sample     A random sample of Yambda users and their most-played items.
  full       All Yambda users and every played item that has an audio embedding.
             Uses the chapters' own epoch counts. This is the path to full-size
             artifacts; it needs a machine with much more memory and time than
             the sample (it was not run while building this companion code;
             CLOUD_RUN.md describes a staged run).

The bundle is written to artifacts/<bundle_version>/ and, unless --no-activate
is given, artifacts/ACTIVE is switched to it atomically.
"""

import argparse
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import published  # noqa: E402
from common.artifacts import artifact_root, set_active_bundle, sha256_file, write_json  # noqa: E402
from pipeline_build import data as data_sources  # noqa: E402
from pipeline_build.feature_store_build import LIKE_COUNT_FEATURES, build_feature_store  # noqa: E402

logger = logging.getLogger("ch13.build")

PROFILES = {
    "tiny": dict(users=120, items=600, catalog=None, xgb_trees=20, mmoe_epochs=1, mmoe_batch=512,
                 tt_epochs=2, tt_warmup=1, tt_batch=256, tt_examples_per_user=60),
    "synthetic": dict(users=300, items=2000, catalog=None, xgb_trees=60, mmoe_epochs=2, mmoe_batch=1024,
                      tt_epochs=3, tt_warmup=1, tt_batch=256, tt_examples_per_user=200),
    "sample": dict(users=400, items=None, catalog=20000, xgb_trees=100, mmoe_epochs=3, mmoe_batch=4096,
                   tt_epochs=4, tt_warmup=1, tt_batch=512, tt_examples_per_user=300),
    "full": dict(users=None, items=None, catalog=None, xgb_trees=100, mmoe_epochs=10, mmoe_batch=4096,
                 tt_epochs=10, tt_warmup=2, tt_batch=512, tt_examples_per_user=10**9),
}
FEATURE_WINDOW = 0.80   # share of listens (by time) used to fit the historical features
VALIDATION_FROM = 0.95  # listens after this time quantile are held out for validation
HISTORY_LENGTH = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("YAMBDA_DATA_DIR"),
                        help="Yambda folder holding embeddings.parquet and flat/50m/ or flat/ (or set YAMBDA_DATA_DIR)")
    parser.add_argument("--users", type=int, default=None, help="Override the profile's number of users")
    parser.add_argument("--catalog-size", type=int, default=None, help="Override the profile's catalog size")
    parser.add_argument("--out", type=Path, default=None, help="Artifact root (default: artifacts/)")
    parser.add_argument("--bundle-version", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-activate", action="store_true", help="Do not switch artifacts/ACTIVE")
    return parser.parse_args(argv)


def most_played(listens, n: int = 1000) -> list:
    plays = listens.groupby("item_id").size().sort_values(ascending=False, kind="stable")
    return [int(i) for i in plays.index[:n]]


def peak_memory_mb() -> float:
    """Peak resident memory of this process so far."""
    if sys.platform == "win32":
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 2**20, 1)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)


def build(args) -> Path:
    profile = dict(PROFILES[args.profile])
    if args.users is not None:
        profile["users"] = args.users
    if args.catalog_size is not None:
        profile["catalog"] = args.catalog_size
    root = Path(args.out) if args.out else artifact_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    bundle_version = args.bundle_version or f"{args.profile}-{stamp}"
    bundle = root / bundle_version
    if bundle.exists():
        raise SystemExit(f"Bundle {bundle} already exists; bundles are immutable. Choose another --bundle-version.")
    bundle.mkdir(parents=True)
    timings, started = {}, time.perf_counter()

    import torch
    # OpenMP loops in XGBoost and PyTorch slow down sharply when they ask for
    # every core on a machine that is busy with other work, so cap them.
    threads = int(os.environ.get("BUILD_THREADS", max(1, min(4, (os.cpu_count() or 2) // 2))))
    torch.set_num_threads(threads)

    # 1. Data -----------------------------------------------------------------
    t0 = time.perf_counter()
    if args.profile in ("tiny", "synthetic"):
        data = data_sources.make_synthetic(num_users=profile["users"], num_items=profile["items"], seed=args.seed)
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir (or YAMBDA_DATA_DIR) is required for the sample and full profiles.")
        data = data_sources.load_yambda(Path(args.data_dir), num_users=profile["users"],
                                        catalog_size=profile["catalog"], seed=args.seed)
    timings["data"] = round(time.perf_counter() - t0, 1)

    listens = data.listens
    t_feature = int(np.quantile(listens["timestamp"], FEATURE_WINDOW))
    t_validation = int(np.quantile(listens["timestamp"], VALIDATION_FROM))
    feature_window = listens[listens["timestamp"] < t_feature]
    label_window = listens[listens["timestamp"] >= t_feature]
    popularity = most_played(label_window)
    # The retrieval baseline is scored on the held-out plays, so its list must not count them.
    baseline_popularity = most_played(label_window[label_window["timestamp"] < t_validation])

    versions = {
        "bundle_version": bundle_version,
        "feature_set_version": f"fs-{bundle_version}",
        "retrieval_model_version": f"two-tower-{bundle_version}",
        "index_version": f"index-{bundle_version}",
        "lightweight_model_version": f"xgboost-{bundle_version}",
        "heavy_model_version": f"mmoe-{bundle_version}",
    }

    # 2. Historical features (Chapter 6) and the feature store ----------------
    t0 = time.perf_counter()
    likes_before = data.likes[data.likes["timestamp"] < t_feature]
    dislikes_before = data.dislikes[data.dislikes["timestamp"] < t_feature]
    builder = published.chapter6("historical_features").HistoricalFeatureBuilder().fit(
        feature_window, likes_before, dislikes_before)
    # Chapter 6 declares no default for the two like counts; an item without
    # plays in the feature window has none.
    builder.item_defaults.update({name: 0 for name in LIKE_COUNT_FEATURES})
    fs_manifest = build_feature_store(bundle / "feature_store", builder, data,
                                      feature_set_version=versions["feature_set_version"],
                                      history_length=HISTORY_LENGTH)
    timings["features"] = round(time.perf_counter() - t0, 1)

    # 3. Rankers (Chapters 6 and 7) -----------------------------------------------
    from pipeline_build.train_rankers import add_labels, train_heavy, train_lightweight

    t0 = time.perf_counter()
    in_training = label_window["timestamp"] < t_validation
    labelled = pd.concat([add_labels(label_window[in_training], data.likes, data.dislikes,
                                     label_end_ts=t_validation),
                          add_labels(label_window[~in_training], data.likes, data.dislikes)], ignore_index=True)
    enriched = builder.transform(labelled)
    train_rows = enriched[enriched["timestamp"] < t_validation].reset_index(drop=True)
    val_rows = enriched[enriched["timestamp"] >= t_validation].reset_index(drop=True)
    light_metrics = train_lightweight(train_rows, val_rows, bundle / "lightweight_ranker",
                                      model_version=versions["lightweight_model_version"],
                                      feature_set_version=versions["feature_set_version"],
                                      n_estimators=profile["xgb_trees"], n_jobs=threads, seed=args.seed)
    timings["lightweight_ranker"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    heavy_metrics = train_heavy(train_rows, val_rows, bundle / "heavy_ranker",
                                model_version=versions["heavy_model_version"],
                                feature_set_version=versions["feature_set_version"],
                                epochs=profile["mmoe_epochs"], batch_size=profile["mmoe_batch"], seed=args.seed)
    timings["heavy_ranker"] = round(time.perf_counter() - t0, 1)
    val_rows.head(200).to_parquet(bundle / "lightweight_ranker" / "consistency_sample.parquet", index=False)

    # 4. Retrieval (Chapter 5) --------------------------------------------------
    from pipeline_build.train_retrieval import train_retrieval

    t0 = time.perf_counter()
    retrieval_metrics = train_retrieval(
        data, bundle / "retrieval", split_timestamp=t_validation, popularity=popularity,
        baseline_popularity=baseline_popularity, model_version=versions["retrieval_model_version"], index_version=versions["index_version"],
        epochs=profile["tt_epochs"], warmup_epochs=profile["tt_warmup"], batch_size=profile["tt_batch"],
        max_examples_per_user=profile["tt_examples_per_user"], seed=args.seed)
    timings["retrieval"] = round(time.perf_counter() - t0, 1)

    # 5. Re-ranker and orchestrator inputs -------------------------------------
    reranker_dir = bundle / "reranker"
    reranker_dir.mkdir()
    np.save(reranker_dir / "content_embeddings.npy", data.content_embeddings.astype(np.float32))
    np.save(reranker_dir / "item_ids.npy", data.item_ids.astype(np.int64))
    write_json(reranker_dir / "reranker_manifest.json", {
        "embedding_version": f"audio-{bundle_version}",
        "similarity": "cosine on Yambda normalized_embed (Chapter 8 MMR)",
        "files": {name: sha256_file(reranker_dir / name) for name in ("content_embeddings.npy", "item_ids.npy")},
    })
    orchestrator_dir = bundle / "orchestrator"
    orchestrator_dir.mkdir()
    write_json(orchestrator_dir / "popularity.json", {
        "description": "Most-played catalog items in the label window, the retrieval fallback list.",
        "item_ids": popularity,
    })

    known_users = set(int(u) for u in listens["uid"].unique())
    warm = sorted(int(u) for u in label_window["uid"].unique())
    rng = np.random.default_rng(args.seed)
    warm = sorted(rng.choice(warm, size=min(200, len(warm)), replace=False).tolist())
    cold, candidate = [], max(known_users) + 1000
    while len(cold) < 20:
        if candidate not in known_users:
            cold.append(candidate)
        candidate += 7
    write_json(bundle / "sample_users.json", {"warm": warm, "cold": cold})

    # 6. Bundle manifest -------------------------------------------------------
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            files[path.relative_to(bundle).as_posix()] = sha256_file(path)
    vendor_manifest = (CHAPTER_DIR / "vendor" / "MANIFEST.json")
    import json
    vendor_commit = json.loads(vendor_manifest.read_text(encoding="utf-8"))["source_commit"]
    timings["total"] = round(time.perf_counter() - started, 1)
    manifest = {
        **versions,
        "profile": args.profile,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_data": data.source,
        "data_summary": data.summary(),
        "splits": {"feature_window_end_timestamp": t_feature, "validation_start_timestamp": t_validation,
                   "feature_window_share": FEATURE_WINDOW, "validation_share": round(1 - VALIDATION_FROM, 2),
                   "ranker_train_rows": int(len(train_rows)), "ranker_val_rows": int(len(val_rows))},
        "published_code": {"commit": vendor_commit, "files": published.vendored_files()},
        "metrics": {"retrieval": retrieval_metrics, "lightweight_ranker": light_metrics, "heavy_ranker": heavy_metrics},
        "feature_store": {"counts": fs_manifest["counts"], "checks": fs_manifest["checks"]},
        "build_seconds": timings,
        "build_host": {"python": platform.python_version(), "platform": platform.platform(terse=True),
                       "cpu_count": os.cpu_count(), "build_threads": threads,
                       "peak_memory_mb": peak_memory_mb()},
        "files": files,
    }
    write_json(bundle / "bundle_manifest.json", manifest)
    if not args.no_activate:
        set_active_bundle(bundle_version, root)
        logger.info("Activated %s", bundle_version)
    logger.info("Bundle %s written in %.1fs", bundle, timings["total"])
    return bundle


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("ch6_models", "ch7_models", "ch8_value_functions", "faiss.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
