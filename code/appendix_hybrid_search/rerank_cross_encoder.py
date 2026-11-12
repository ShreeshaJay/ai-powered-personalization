"""Optional cross-encoder rerank of a fused top-k list.

The model is cross-encoder/ms-marco-MiniLM-L-6-v2 at the revision in
analysis_plan.yaml. trust_remote_code is false. This step is not part of
the required retrieval path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import load_plan, resolve_data_dir, resolve_device
from data_loader import load_esci
from fusion import Run, host_array, ranking_from_scores, read_run, write_run
from run_retrieval import write_manifest


def rerank_run(
    run: Run,
    queries: dict,
    texts: dict[str, str],
    depth: int,
    device: str,
    batch_size: int,
) -> Run:
    spec = load_plan()["models"]["cross_encoder"]
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        spec["repo"],
        revision=spec["revision"],
        trust_remote_code=False,
        device=device,
    )
    reranked: Run = {}
    for query_id, hits in run.items():
        item = queries.get(query_id)
        if item is None:
            continue
        head = hits[:depth]
        if not head:
            reranked[query_id] = []
            continue
        pairs = [(item.query_text, texts.get(product_id, "")) for product_id, _score in head]
        scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = host_array(scores)
        reranked[query_id] = ranking_from_scores(
            [product_id for product_id, _score in head],
            scores,
            depth=None,
        )
    return reranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a fused run with a cross-encoder.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()
    plan = load_plan()
    spec = plan["models"]["cross_encoder"]
    depth = int(args.depth or spec["rerank_depth"])
    device = resolve_device(args.device)
    data = load_esci(resolve_data_dir(args.data_dir))
    output = Path(args.output_dir)
    source = output / "runs" / args.split / f"{args.run_name}.p2.trec"
    run = read_run(source)
    queries = {query_id: data.queries[query_id] for query_id in run if query_id in data.queries}
    batch = 8 if device == "cpu" else int(spec["l4_batch_size"])
    reranked = rerank_run(run, queries, data.catalog.texts, depth, device, batch)
    name = f"{args.run_name}_cross_encoder"
    destination = output / "runs" / args.split / f"{name}.p2.trec"
    write_run(destination, reranked, name, depth=depth)
    write_manifest(
        output / "manifests" / args.split / f"{name}.json",
        {
            "system": name,
            "source_run": args.run_name,
            "repo": spec["repo"],
            "revision": spec["revision"],
            "trust_remote_code": False,
            "depth": depth,
            "device": device,
            "optional": True,
        },
    )
    print(f"Wrote {destination.name}")


if __name__ == "__main__":
    main()
