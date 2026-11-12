"""Convert a saved label file and reconstruct the query sample.

The historical labeler is not run from this folder. This module only reads a
label file the reader already has. Labels are not included here.

The sample is the published retrieval-diversity rule: sort query ids by
disagreement descending (stable), take the top fraction, then draw the rest
with numpy RandomState(seed).choice. Parameters are read from the label file
when it stores them. The published command-line default was 2000 queries,
while the sampling design named in the protocol is 10000. Reconstruction uses
the parameters stored with the file and does not guess.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import load_plan
from evaluate import hajek_mean, macro, score_p2, write_json
from fusion import read_run


def select_query_ids(
    query_ids: list,
    disagreement: dict[str, float],
    max_queries: int,
    high_disagree_fraction: float,
    seed: int,
) -> list[str]:
    """Match the published labeler's retrieval-diversity selection."""
    ordered = [str(query_id) for query_id in query_ids]
    if max_queries >= len(ordered):
        max_queries = len(ordered)
    sorted_ids = sorted(ordered, key=lambda query_id: disagreement.get(str(query_id), 0.0), reverse=True)
    n_high = int(max_queries * high_disagree_fraction)
    n_uniform = max_queries - n_high
    high = sorted_ids[:n_high]
    remaining = sorted_ids[n_high:]
    if n_uniform <= len(remaining) and n_uniform > 0:
        generator = np.random.RandomState(seed)
        uniform = generator.choice(remaining, size=n_uniform, replace=False).tolist()
    else:
        uniform = remaining
    return [str(query_id) for query_id in high + uniform]


def load_label_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "labels" not in payload:
        raise ValueError("Label file has no 'labels' object.")
    return payload


def stored_sampling(payload: dict) -> dict:
    plan = load_plan()["protocols"]["p3"]["sampling"]
    metadata = payload.get("metadata") or {}
    sampling = metadata.get("sampling") or {}
    return {
        "high_disagree_fraction": float(sampling.get("high_disagree_fraction", plan["high_disagree_fraction"])),
        "seed": int(sampling.get("seed", plan["seed"])),
        "max_queries": int(sampling["max_queries"]) if "max_queries" in sampling else None,
        "top_k": int(sampling.get("top_k", plan["top_k"])),
        "methods": list(sampling.get("methods", plan["methods"])),
        "parameters_were_stored": "max_queries" in sampling,
    }


def labels_for_queries(payload: dict) -> dict[str, dict[str, str]]:
    converted: dict[str, dict[str, str]] = {}
    for query_id, products in payload["labels"].items():
        converted[str(query_id)] = {}
        for product_id, row in products.items():
            if isinstance(row, dict):
                converted[str(query_id)][str(product_id)] = str(row.get("label", ""))
            else:
                converted[str(query_id)][str(product_id)] = str(row)
    return converted


def maybe_p3(output_dir: Path, split: str, labels: dict[str, dict[str, str]], sampling: dict) -> dict:
    """Report Hajek nDCG@10 only when the sample is reconstructed and judged@10 is high enough."""
    plan = load_plan()["protocols"]["p3"]
    report = {
        "reported": False,
        "reason": "The sample was not reconstructed from stored parameters.",
    }
    if not sampling["parameters_were_stored"]:
        write_json(output_dir / "summaries" / f"{split}_p3.json", report)
        return report
    run_path = output_dir / "runs" / split / "bm25.p2.trec"
    if not run_path.exists():
        report["reason"] = "No BM25 run is available to score judged@10."
        write_json(output_dir / "summaries" / f"{split}_p3.json", report)
        return report
    from data_loader import QueryJudgments

    queries = {}
    for query_id, pairs in labels.items():
        queries[query_id] = QueryJudgments(query_id, query_id, split, pairs, list(pairs))
    per_query = score_p2(read_run(run_path), queries, {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0, "Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0})
    judged = macro(per_query, "judged_at_10")["mean"]
    threshold = float(plan["report_when_judged_at_10_at_least"])
    report["judged_at_10"] = judged
    if judged is None or judged < threshold:
        report["reason"] = f"judged@10 is below {threshold}."
        write_json(output_dir / "summaries" / f"{split}_p3.json", report)
        return report
    probabilities = payload_probabilities(labels)
    if probabilities is None:
        report["reason"] = "Inclusion probabilities were not stored, so the Hajek mean is not reported."
        write_json(output_dir / "summaries" / f"{split}_p3.json", report)
        return report
    values = [float(per_query[query_id]["condensed_ndcg_at_10"] or 0.0) for query_id in probabilities]
    report["reported"] = True
    report["reason"] = "Sample reconstructed and judged@10 met the threshold."
    report["hajek_condensed_ndcg_at_10"] = hajek_mean(values, list(probabilities.values()))
    write_json(output_dir / "summaries" / f"{split}_p3.json", report)
    return report


def payload_probabilities(labels: dict) -> dict[str, float] | None:
    return None


def convert(path: Path, output_dir: Path, disagreement_path: Path | None, split: str) -> dict:
    payload = load_label_file(path)
    sampling = stored_sampling(payload)
    labels = labels_for_queries(payload)
    reconstructed = None
    if disagreement_path is not None and disagreement_path.exists() and sampling["max_queries"] is not None:
        raw = json.loads(disagreement_path.read_text(encoding="utf-8"))
        scores = raw.get("scores", raw)
        scores = {str(key): float(value) for key, value in scores.items()}
        reconstructed = select_query_ids(
            list(scores),
            scores,
            sampling["max_queries"],
            sampling["high_disagree_fraction"],
            sampling["seed"],
        )
    destination = output_dir / "labels" / "converted_labels.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "n_queries": len(labels),
        "sampling": sampling,
        "reconstructed_query_ids": reconstructed,
        "labels": labels,
    }
    destination.write_text(json.dumps(body) + "\n", encoding="utf-8")
    p3 = maybe_p3(output_dir, split, labels, sampling)
    return {"converted": str(destination.name), "p3_reported": p3["reported"], "reason": p3["reason"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a saved label file. Does not call a model API.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--disagreement", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    disagreement = Path(args.disagreement) if args.disagreement else None
    result = convert(Path(args.labels), Path(args.output_dir), disagreement, args.split)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
