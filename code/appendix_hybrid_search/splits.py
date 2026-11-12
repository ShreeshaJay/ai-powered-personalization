"""Frozen query lists for test, development, and the optional fine-tune.

Development is drawn only after two exclusions: query text that appears in
the Qdrant checkpoint's training prefix, and query ids reserved for the
Exercise 3 pairs. Reserving those pairs first removes the circular
dependency between the development sample and the fine-tune. The test
split drops a query when its text matches that prefix under the same
comparison, and later scores use the queries that remain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_plan, resolve_data_dir, sha256_text
from data_loader import ESCIData, load_esci, normalize_query, short_label

POSITIVE_LABELS = {"E", "S", "Exact", "Substitute", "exact", "substitute"}
PREFIX_REVISION = "8113b17a5d4099e20243282c926f1bc1a08a4d13"


@dataclass
class SplitBundle:
    test_ids: list[str]
    development_ids: list[str]
    exercise3_query_ids: list[str]
    exercise3_pairs: list[tuple[str, str, str]]
    prefix_texts: list[str]
    prefix_source: str
    digests: dict[str, str]
    excluded_test_ids: list[str]

    def as_dict(self) -> dict:
        return {
            "test_ids": self.test_ids,
            "excluded_test_ids": self.excluded_test_ids,
            "n_excluded_test_ids": len(self.excluded_test_ids),
            "development_ids": self.development_ids,
            "exercise3_query_ids": self.exercise3_query_ids,
            "n_exercise3_pairs": len(self.exercise3_pairs),
            "n_prefix_texts": len(self.prefix_texts),
            "prefix_source": self.prefix_source,
            "digests": self.digests,
        }


def _digest(ids: list[str]) -> str:
    return sha256_text("\n".join(ids) + ("\n" if ids else ""))


def _is_positive(label: object) -> bool:
    text = str(label).strip()
    if text in POSITIVE_LABELS:
        return True
    try:
        return short_label(text) in {"E", "S"}
    except ValueError:
        return False


def prefix_texts_from_frame(frame: pd.DataFrame, n_pairs: int = 100_000) -> list[str]:
    """Replay the tutorial filter: US Exact or Substitute, file order, first n."""
    required = {"query", "product_locale", "esci_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prefix frame is missing {sorted(missing)}.")
    rows = frame
    if "split" in rows.columns:
        rows = rows[rows["split"].astype(str) == "train"]
    locale = rows["product_locale"].astype(str).str.lower() == "us"
    labels = rows["esci_label"].map(_is_positive)
    kept = rows[locale & labels]
    head = kept.head(n_pairs)
    return [normalize_query(value) for value in head["query"].tolist()]


def load_prefix_texts(source: str, n_pairs: int = 100_000, revision: str = PREFIX_REVISION) -> tuple[list[str], str]:
    """Load the prefix from a local table or from ``tasksource/esci``."""
    path = Path(source)
    if path.exists():
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "queries" in payload:
                return [normalize_query(item) for item in payload["queries"]], str(path)
            frame = pd.DataFrame(payload)
        else:
            raise ValueError(f"Unsupported prefix file {path}.")
        return prefix_texts_from_frame(frame, n_pairs=n_pairs), str(path)

    if source in {"tasksource/esci", "hf:tasksource/esci"}:
        from datasets import load_dataset

        dataset = load_dataset(
            "tasksource/esci",
            split="train",
            revision=revision,
            streaming=True,
            trust_remote_code=False,
        )
        kept: list[str] = []
        for row in dataset:
            if str(row.get("product_locale", "")).lower() != "us":
                continue
            if not _is_positive(row.get("esci_label")):
                continue
            kept.append(normalize_query(row.get("query", "")))
            if len(kept) >= n_pairs:
                break
        return kept, f"tasksource/esci@{revision}"
    raise FileNotFoundError(
        "Prefix source was not a local file and was not tasksource/esci. "
        "Pass a parquet exported in file order, or 'tasksource/esci'."
    )


def _prefix_set(prefix_texts: list[str]) -> set[str]:
    return {normalize_query(text) for text in prefix_texts}


def _query_text_in_prefix(text: object, prefix_set: set[str]) -> bool:
    return normalize_query(text) in prefix_set


def _sample_ids(ids: list[str], size: int, seed: int) -> list[str]:
    if size > len(ids):
        raise ValueError(f"Cannot sample {size} ids from a pool of {len(ids)}.")
    if size == len(ids):
        return sorted(ids)
    generator = np.random.default_rng(seed)
    chosen = generator.choice(np.array(sorted(ids), dtype=object), size=size, replace=False)
    return sorted(str(item) for item in chosen.tolist())


def build_splits(
    data: ESCIData,
    prefix_texts: list[str],
    prefix_source: str,
    development_size: int = 1000,
    development_seed: int = 20261110,
    exercise3_pairs: int = 100_000,
) -> SplitBundle:
    prefix_set = _prefix_set(prefix_texts)
    official_test_ids = sorted(qid for qid, item in data.queries.items() if item.split == "test")
    excluded_test_ids = [
        qid
        for qid in official_test_ids
        if _query_text_in_prefix(data.queries[qid].query_text, prefix_set)
    ]
    excluded_lookup = set(excluded_test_ids)
    test_ids = [qid for qid in official_test_ids if qid not in excluded_lookup]
    train_items = [item for item in data.queries.values() if item.split == "train"]

    reserved_pairs: list[tuple[str, str, str]] = []
    reserved_queries: set[str] = set()
    examples = data.examples
    train_positive = examples[
        (examples["split"] == "train") & (examples["esci_label"].isin(["E", "S"]))
    ]
    for query_id, product_id, query_text, label in zip(
        train_positive["query_id"].tolist(),
        train_positive["product_id"].tolist(),
        train_positive["query"].tolist(),
        train_positive["esci_label"].tolist(),
    ):
        if _query_text_in_prefix(query_text, prefix_set):
            continue
        reserved_pairs.append((str(query_id), str(product_id), str(label)))
        reserved_queries.add(str(query_id))
        if len(reserved_pairs) >= exercise3_pairs:
            break

    eligible = [
        item.query_id
        for item in train_items
        if item.query_id not in reserved_queries and not _query_text_in_prefix(item.query_text, prefix_set)
    ]
    development_ids = _sample_ids(eligible, development_size, development_seed)

    leaked = set(development_ids) & set(test_ids)
    leaked |= set(development_ids) & reserved_queries
    leaked |= {qid for qid in test_ids if _query_text_in_prefix(data.queries[qid].query_text, prefix_set)}
    if leaked:
        raise RuntimeError(f"Split leakage: {sorted(leaked)[:10]}")

    digests = {
        "test_query_ids": _digest(test_ids),
        "excluded_test_query_ids": _digest(excluded_test_ids),
        "development_query_ids": _digest(development_ids),
        "exercise3_query_ids": _digest(sorted(reserved_queries)),
        "prefix_texts": _digest(sorted(prefix_set)),
        "catalog": data.catalog.catalog_sha256,
        "template": data.catalog.template_sha256,
        "product_text": data.catalog.text_sha256,
    }
    return SplitBundle(
        test_ids=test_ids,
        development_ids=development_ids,
        exercise3_query_ids=sorted(reserved_queries),
        exercise3_pairs=reserved_pairs,
        prefix_texts=sorted(prefix_set),
        prefix_source=prefix_source,
        digests=digests,
        excluded_test_ids=excluded_test_ids,
    )


def exercise_slice(
    data: ESCIData,
    n_test_queries: int,
    n_distractors: int,
    seed: int,
    prefix_texts: list[str] | None = None,
) -> dict[str, list[str]]:
    """Judged products of seeded test queries, plus random distractors.

    prefix_texts, when given, uses the same comparison as the test split.
    """
    prefix_set = _prefix_set(prefix_texts or [])
    test_ids = sorted(
        qid
        for qid, item in data.queries.items()
        if item.split == "test" and not _query_text_in_prefix(item.query_text, prefix_set)
    )
    chosen_queries = _sample_ids(test_ids, min(n_test_queries, len(test_ids)), seed)
    judged: set[str] = set()
    for query_id in chosen_queries:
        judged.update(data.queries[query_id].labels)
    remaining = [pid for pid in data.catalog.product_ids if pid not in judged]
    generator = np.random.default_rng(seed)
    take = min(n_distractors, len(remaining))
    if take:
        picked = generator.choice(np.array(sorted(remaining), dtype=object), size=take, replace=False)
        distractors = sorted(str(item) for item in picked.tolist())
    else:
        distractors = []
    product_ids = sorted(judged | set(distractors))
    return {
        "query_ids": chosen_queries,
        "product_ids": product_ids,
        "judged_product_ids": sorted(judged),
        "distractor_product_ids": distractors,
    }


def write_bundle(path: Path, bundle: SplitBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.as_dict()
    payload["exercise3_pairs"] = [
        {"query_id": qid, "product_id": pid, "label": label}
        for qid, pid, label in bundle.exercise3_pairs
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze ESCI query lists and an optional slice.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--prefix-source", default="tasksource/esci")
    parser.add_argument("--slice", action="store_true", dest="make_slice")
    parser.add_argument("--slice-queries", type=int, default=None)
    parser.add_argument("--slice-distractors", type=int, default=None)
    args = parser.parse_args()
    plan = load_plan()
    data = load_esci(resolve_data_dir(args.data_dir))
    prefix, source = load_prefix_texts(args.prefix_source, revision=plan["splits"]["qdrant_prefix"]["revision"])
    bundle = build_splits(
        data,
        prefix,
        source,
        development_size=int(plan["splits"]["development"]["size"]),
        development_seed=int(plan["splits"]["development"]["seed"]),
        exercise3_pairs=int(plan["splits"]["exercise3_reservation"]["pairs"]),
    )
    output = Path(args.output_dir)
    write_bundle(output / "splits" / "splits.json", bundle)
    if args.make_slice:
        spec = plan["splits"]["exercise1_slice"]
        slice_info = exercise_slice(
            data,
            n_test_queries=args.slice_queries or int(spec["test_queries"]),
            n_distractors=args.slice_distractors or int(spec["distractor_products"]),
            seed=int(spec["seed"]),
            prefix_texts=prefix,
        )
        slice_path = output / "splits" / "exercise_slice.json"
        slice_path.parent.mkdir(parents=True, exist_ok=True)
        slice_path.write_text(json.dumps(slice_info, indent=2) + "\n", encoding="utf-8")
        print(f"Slice products: {len(slice_info['product_ids'])}")
    print(f"Test queries: {len(bundle.test_ids)}")
    print(f"Excluded test queries: {len(bundle.excluded_test_ids)}")
    print(f"Excluded test ids: {bundle.excluded_test_ids}")
    print(f"Development queries: {len(bundle.development_ids)}")
    print(f"Exercise 3 pairs: {len(bundle.exercise3_pairs)}")


if __name__ == "__main__":
    main()
