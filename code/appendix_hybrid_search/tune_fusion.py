"""Tune fusion on development queries and freeze the chosen settings.

The selection metric is development P2 Recall@100 of judged Exact products.
Ties break on condensed nDCG@10, then on the system name.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import OFFICIAL_GAINS, load_plan
from data_loader import QueryJudgments
from evaluate import macro, score_p2
from fusion import Run, convex, dbsf, rrf, simplex_weights, truncate, write_run


def _metric_key(run: Run, queries: dict[str, QueryJudgments], gains: dict[str, float]) -> tuple[float, float]:
    per_query = score_p2(run, queries, gains)
    recall = macro(per_query, "recall_at_100_exact")["mean"]
    condensed = macro(per_query, "condensed_ndcg_at_10")["mean"]
    return (
        -1.0 if recall is None else float(recall),
        -1.0 if condensed is None else float(condensed),
    )


def _best(names: list[str], scores: dict[str, tuple[float, float]]) -> str:
    return sorted(names, key=lambda name: (-scores[name][0], -scores[name][1], name))[0]


def _per_query(runs: dict[str, Run], query_ids: list[str], fuse) -> Run:
    fused: Run = {}
    for query_id in query_ids:
        legs = [runs[name].get(query_id, []) for name in fuse["legs"]]
        fused[query_id] = fuse["function"](legs)[:1000]
    return fused


def tune(
    runs: dict[str, Run],
    queries: dict[str, QueryJudgments],
    output_dir: Path,
    split: str = "development",
) -> dict:
    plan = load_plan()
    gains = dict(OFFICIAL_GAINS)
    query_ids = sorted(queries)
    variants = [name for name in plan["fusion"]["selection"]["splade_pool"] if name in runs]
    if "bm25" not in runs or "dense" not in runs:
        raise RuntimeError("Tuning needs bm25 and dense runs.")
    if not variants:
        raise RuntimeError("Tuning needs at least one SPLADE run.")

    single_names = [name for name in plan["fusion"]["selection"]["single_pool"] if name in runs]
    single_scores = {name: _metric_key(runs[name], queries, gains) for name in single_names}
    best_variant = _best(variants, {name: single_scores[name] for name in variants})
    best_single = _best(single_names, single_scores)

    alpha_grid = [float(value) for value in plan["fusion"]["f4"]["alpha_grid"]]
    k_grid = [int(value) for value in plan["fusion"]["f7"]["k_grid"]]
    fused_runs: dict[str, Run] = {}
    tuned: dict[str, dict] = {}

    def add(name: str, legs: list[str], function) -> None:
        fused_runs[name] = _per_query(runs, query_ids, {"legs": legs, "function": function})

    add("f1_rrf_bm25_dense", ["bm25", "dense"], lambda legs: rrf(legs, k=60))
    for variant in variants:
        add(
            f"f2_rrf_bm25_{variant}_dense",
            ["bm25", variant, "dense"],
            lambda legs: rrf(legs, k=60),
        )
        add(f"f3_rrf_{variant}_dense", [variant, "dense"], lambda legs: rrf(legs, k=60))

    best_alpha = None
    best_alpha_key = (-1.0, -1.0)
    for alpha in alpha_grid:
        name = f"f4_alpha_{alpha:.2f}"
        add(
            name,
            ["bm25", "dense"],
            lambda legs, alpha=alpha: convex(legs, [alpha, 1.0 - alpha]),
        )
        key = _metric_key(fused_runs[name], queries, gains)
        if (key[0], key[1], -alpha) > (best_alpha_key[0], best_alpha_key[1], -(best_alpha or 0)):
            best_alpha = alpha
            best_alpha_key = key
    fused_runs["f4_convex_bm25_dense"] = fused_runs[f"f4_alpha_{best_alpha:.2f}"]
    tuned["f4_alpha"] = best_alpha

    weight_grid = simplex_weights(float(plan["fusion"]["f5"]["simplex_step"]))
    best_weights = None
    best_weight_key = (-1.0, -1.0)
    for weights in weight_grid:
        label = "_".join(f"{weight:.1f}" for weight in weights)
        name = f"f5_{label}"
        add(
            name,
            ["bm25", best_variant, "dense"],
            lambda legs, weights=weights: convex(legs, list(weights)),
        )
        key = _metric_key(fused_runs[name], queries, gains)
        if key > best_weight_key or (key == best_weight_key and (best_weights is None or weights < best_weights)):
            best_weights = weights
            best_weight_key = key
    fused_runs["f5_convex_three"] = fused_runs["f5_" + "_".join(f"{weight:.1f}" for weight in best_weights)]
    tuned["f5_weights"] = list(best_weights)

    add("f6_dbsf_three", ["bm25", best_variant, "dense"], lambda legs: dbsf(legs))

    best_k = None
    best_k_key = (-1.0, -1.0)
    for constant in k_grid:
        name = f"f7_k_{constant}"
        add(
            name,
            ["bm25", best_variant, "dense"],
            lambda legs, constant=constant: rrf(legs, k=constant),
        )
        key = _metric_key(fused_runs[name], queries, gains)
        if key > best_k_key or (key == best_k_key and (best_k is None or constant < best_k)):
            best_k = constant
            best_k_key = key
    fused_runs["f7_rrf_tuned"] = fused_runs[f"f7_k_{best_k}"]
    tuned["f7_k"] = best_k

    hybrid_names = [
        "f1_rrf_bm25_dense",
        *[f"f2_rrf_bm25_{variant}_dense" for variant in variants],
        *[f"f3_rrf_{variant}_dense" for variant in variants],
        "f4_convex_bm25_dense",
        "f5_convex_three",
        "f6_dbsf_three",
        "f7_rrf_tuned",
    ]
    hybrid_scores = {name: _metric_key(fused_runs[name], queries, gains) for name in hybrid_names}
    best_hybrid = _best(hybrid_names, hybrid_scores)

    if split == "development":
        for depth in plan["fusion"]["depth_sensitivity"]["depths"]:
            depth = int(depth)
            for variant in variants:
                name = f"depth_{depth}_f2_{variant}"
                fused: Run = {}
                for query_id in query_ids:
                    legs = [
                        truncate(runs["bm25"].get(query_id, []), depth),
                        truncate(runs[variant].get(query_id, []), depth),
                        truncate(runs["dense"].get(query_id, []), depth),
                    ]
                    fused[query_id] = rrf(legs, k=60)[:depth]
                fused_runs[name] = fused

    run_dir = output_dir / "runs" / split
    kept = {}
    for name, run in fused_runs.items():
        if name.startswith("f4_alpha_") or name.startswith("f5_") and name != "f5_convex_three":
            if name.startswith("f7_k_"):
                continue
        if name.startswith("f4_alpha_") or (name.startswith("f5_") and name != "f5_convex_three") or name.startswith("f7_k_"):
            continue
        write_run(run_dir / f"{name}.p2.trec", run, name, depth=1000)
        kept[name] = name
    selection = {
        "split": split,
        "best_splade_variant": best_variant,
        "best_single": best_single,
        "best_hybrid": best_hybrid,
        "tuned": tuned,
        "single_scores": {name: {"recall_at_100_exact": scores[0], "condensed_ndcg_at_10": scores[1]} for name, scores in single_scores.items()},
        "hybrid_scores": {name: {"recall_at_100_exact": scores[0], "condensed_ndcg_at_10": scores[1]} for name, scores in hybrid_scores.items()},
        "placeholders": {"{best_hybrid}": best_hybrid, "{best_single}": best_single},
    }
    selection_path = output_dir / "tuning" / f"selection_{split}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selection


def apply_frozen(runs: dict[str, Run], query_ids: list[str], selection: dict, depth: int = 1000) -> dict[str, Run]:
    """Apply development-chosen fusion settings. This does not retune."""
    variant = selection["best_splade_variant"]
    alpha = float(selection["tuned"]["f4_alpha"])
    weights = [float(value) for value in selection["tuned"]["f5_weights"]]
    constant = int(selection["tuned"]["f7_k"])
    present = [name for name in ("splade_naver", "splade_prithivi", "splade_esci") if name in runs]
    specs = {
        "f1_rrf_bm25_dense": (["bm25", "dense"], lambda legs: rrf(legs, k=60)),
        "f4_convex_bm25_dense": (["bm25", "dense"], lambda legs: convex(legs, [alpha, 1.0 - alpha])),
        "f5_convex_three": (["bm25", variant, "dense"], lambda legs: convex(legs, weights)),
        "f6_dbsf_three": (["bm25", variant, "dense"], lambda legs: dbsf(legs)),
        "f7_rrf_tuned": (["bm25", variant, "dense"], lambda legs: rrf(legs, k=constant)),
    }
    for name in present:
        specs[f"f2_rrf_bm25_{name}_dense"] = (["bm25", name, "dense"], lambda legs: rrf(legs, k=60))
        specs[f"f3_rrf_{name}_dense"] = ([name, "dense"], lambda legs: rrf(legs, k=60))
    fused: dict[str, Run] = {}
    for name, (legs, function) in specs.items():
        if any(leg not in runs for leg in legs):
            continue
        run: Run = {}
        for query_id in query_ids:
            run[query_id] = function([runs[leg].get(query_id, []) for leg in legs])[:depth]
        fused[name] = run
    return fused
