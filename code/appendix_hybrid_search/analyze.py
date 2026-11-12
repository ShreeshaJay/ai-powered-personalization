"""Summaries, primary comparisons, and query slices.

Per-query values stay in the output directory. Files written under a results
directory are checked against the 1 MB limit.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from config import OFFICIAL_GAINS, load_plan
from data_loader import QueryJudgments
from evaluate import (
    assert_result_size,
    guard_test_split,
    macro,
    random_order_floor,
    report_primary_family,
    score_p1,
    score_p2,
    write_json,
)
from fusion import Run, read_run

NEGATION = re.compile(r"\b(?:without|not|no)\b", re.IGNORECASE)


def slice_membership(text: str) -> dict[str, bool]:
    words = text.split()
    return {
        "digits": bool(re.search(r"\d", text)),
        "negation": bool(NEGATION.search(text)),
        "long": len(words) >= 6,
        "short": len(words) <= 2,
        "all": True,
    }


def load_queries(data_queries: dict[str, QueryJudgments], ids: list[str]) -> dict[str, QueryJudgments]:
    return {query_id: data_queries[query_id] for query_id in ids if query_id in data_queries}


def score_run_file(
    path: Path,
    queries: dict[str, QueryJudgments],
    protocol: str,
    gains: dict[str, float] | None = None,
) -> dict[str, dict]:
    gains = gains or dict(OFFICIAL_GAINS)
    run = read_run(path) if path.exists() else {}
    for query_id in queries:
        run.setdefault(query_id, [])
    if protocol == "p1":
        return score_p1(run, queries, gains)
    if protocol == "p2":
        return score_p2(run, queries, gains)
    raise ValueError(f"Unknown protocol {protocol}.")


def summarize(per_query: dict[str, dict], metrics: list[str]) -> dict[str, dict]:
    return {metric: macro(per_query, metric) for metric in metrics}


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if "results" in path.parts:
        assert_result_size(path)


def evaluate_split(
    output_dir: Path,
    split: str,
    queries: dict[str, QueryJudgments],
    systems: list[str],
    selection: dict | None = None,
) -> dict:
    plan = load_plan()
    gains = dict(OFFICIAL_GAINS)
    per_protocol: dict[str, dict[str, dict]] = {"p1": {}, "p2": {}}
    rows = []
    for system in systems:
        for protocol, metrics in (
            ("p1", ["ndcg", "ndcg_at_10"]),
            ("p2", ["recall_at_100_exact", "recall_at_1000_exact", "condensed_ndcg_at_10", "judged_at_10"]),
        ):
            path = output_dir / "runs" / split / f"{system}.{protocol}.trec"
            if not path.exists():
                continue
            key = f"{split}|{protocol}|{system}|official"
            guard_test_split(output_dir, key, split)
            per_query = score_run_file(path, queries, protocol, gains)
            per_protocol[protocol][system] = per_query
            write_json(output_dir / "per_query" / split / f"{system}.{protocol}.json", per_query)
            for metric, summary in summarize(per_query, metrics).items():
                rows.append({"split": split, "system": system, "protocol": protocol, **summary})
    family = []
    if selection is not None:
        family = report_primary_family(
            per_protocol["p1"],
            per_protocol["p2"],
            plan["primary_family"],
            int(plan["statistics"]["bootstrap_resamples"]),
            int(plan["statistics"]["bootstrap_seed"]),
            float(plan["statistics"]["holm_alpha"]),
            float(plan["statistics"]["min_relative_difference_to_interpret"]),
            selection.get("placeholders"),
        )
    floor = random_order_floor(
        queries,
        gains,
        int(plan["protocols"]["p1"]["random_floor_shuffles"]),
        int(plan["protocols"]["p1"]["random_floor_seed"]),
    )
    payload = {"split": split, "rows": rows, "primary_family": family, "random_floor": floor}
    summary_path = output_dir / "summaries" / f"{split}.json"
    write_json(summary_path, payload)
    slice_rows = []
    texts = {query_id: item.query_text for query_id, item in queries.items()}
    for system, per_query in per_protocol["p2"].items():
        for slice_id in ("digits", "negation", "long", "short", "all"):
            chosen = {
                query_id: row
                for query_id, row in per_query.items()
                if slice_membership(texts[query_id])[slice_id]
            }
            summary = macro(chosen, "recall_at_100_exact")
            slice_rows.append({"split": split, "system": system, "slice": slice_id, **summary})
    if slice_rows:
        write_summary_csv(output_dir / "summaries" / f"{split}_slices.csv", slice_rows)
    return payload


def run_dict_from_hits(query_ids: list[str], hit_lists: list) -> Run:
    return {query_id: hits for query_id, hits in zip(query_ids, hit_lists)}
