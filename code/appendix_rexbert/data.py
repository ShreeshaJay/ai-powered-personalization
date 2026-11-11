"""Amazon-M2 (KDD Cup 2023) loader, session split, and pair builder.

The training files are the whole universe used here. The split is a hash of
the filtered session row index, so training pairs are built only after the
split and only from training sessions.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    LOCALE,
    PAIR_CAP_SEED,
    SAMPLE_SEED,
    SPLIT_NAMES,
    SPLIT_SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TEXT_TEMPLATE,
    bucket_to_split,
    split_bucket,
)

logger = logging.getLogger("appendix_rexbert.data")

PRODUCT_COLUMNS = ["id", "locale", "title", "brand", "desc"]
SESSION_COLUMNS = ["prev_items", "next_item", "locale"]


def resolve_data_dir(cli_value: str | None = None) -> Path:
    raw = cli_value or os.environ.get("AMAZON_KDD_DATA_DIR")
    if not raw:
        raise SystemExit(
            "Set --data_dir or AMAZON_KDD_DATA_DIR to the folder that contains "
            "products_train.csv and sessions_train.csv. See the README for the download page."
        )
    path = Path(raw)
    missing = [
        name
        for name in ("products_train.csv", "sessions_train.csv")
        if not (path / name).is_file()
    ]
    if missing:
        raise SystemExit(
            "The data directory is missing "
            + " and ".join(missing)
            + ". See the README for the download page."
        )
    return path


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_prev_items(value: object) -> list[str]:
    """Parse a session cell.

    The released training file stores previous items as a bracketed list of
    quoted ids separated by spaces, for example ``['B09W9FND7K' 'B09JSPLN1M']``.
    """
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    if text[0] == "[":
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    items: list[str] = []
    for token in text.split():
        token = token.strip("'\" ,\t")
        if token:
            items.append(token)
    return items


def render_text(title: str, brand: str, description: str, template: str = TEXT_TEMPLATE) -> str:
    text = template.format(title=title, brand=brand, description=description)
    return " ".join(text.split())


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q, method="linear"))


@dataclass
class PreparedData:
    catalog_ids: list[str]
    texts: dict[str, str]
    query_ids: list[str]
    target_ids: list[str]
    prev_len: np.ndarray
    split: np.ndarray
    degenerate: np.ndarray
    pair_query: list[str]
    pair_positive: list[str]
    pair_session: np.ndarray
    pair_keys_before_cap: set[tuple[str, str]]
    transitions: dict[str, dict[str, int]]
    popularity: dict[str, int]
    seen_train_items: set[str]
    chapter10_map: dict[str, set[str]]
    card: dict
    file_hashes: dict[str, str] = field(default_factory=dict)

    def session_frame_mask(self, split_id: int, drop_degenerate: bool = True) -> np.ndarray:
        mask = self.split == split_id
        if drop_degenerate:
            mask = mask & ~self.degenerate
        return mask


def load_catalog(products_path: Path, locale: str = LOCALE) -> tuple[list[str], dict[str, str], dict]:
    ids: list[str] = []
    texts: dict[str, str] = {}
    word_counts: list[int] = []
    n_locale = 0
    n_duplicate = 0
    for chunk in pd.read_csv(
        products_path,
        usecols=PRODUCT_COLUMNS,
        dtype=str,
        chunksize=50_000,
        keep_default_na=True,
    ):
        chunk = chunk.loc[chunk["locale"] == locale]
        if chunk.empty:
            continue
        n_locale += int(len(chunk))
        titles = chunk["title"].fillna("")
        brands = chunk["brand"].fillna("")
        descs = chunk["desc"].fillna("")
        rendered = [
            render_text(title, brand, desc)
            for title, brand, desc in zip(titles.tolist(), brands.tolist(), descs.tolist())
        ]
        for product_id, text, desc in zip(chunk["id"].tolist(), rendered, descs.tolist()):
            if product_id in texts:
                n_duplicate += 1
                continue
            texts[product_id] = text
            ids.append(product_id)
            word_counts.append(_word_count("" if desc is None else str(desc)))
    words = np.asarray(word_counts, dtype=np.int32)
    card = {
        "locale": locale,
        "n_product_rows_in_locale": n_locale,
        "n_products": len(ids),
        "n_duplicate_product_ids_skipped": n_duplicate,
        "description_words_median": _percentile(words, 50),
        "description_words_p90": _percentile(words, 90),
        "description_words_mean": float(words.mean()) if words.size else float("nan"),
        "n_products_with_empty_description": int(np.sum(words == 0)),
        "text_template": TEXT_TEMPLATE,
    }
    logger.info("Loaded %s %s products", card["n_products"], locale)
    return ids, texts, card


def _empty_string(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def load_sessions(
    sessions_path: Path,
    catalog: dict[str, str],
    *,
    locale: str = LOCALE,
    split_seed: int = SPLIT_SEED,
    pair_cap: int = 500_000,
    pair_cap_seed: int = PAIR_CAP_SEED,
    chapter10_min_prev: int = 2,
) -> dict:
    """Filter to one locale, split by row index, and build training pairs."""
    query_ids: list[str] = []
    target_ids: list[str] = []
    prev_lens: list[int] = []
    splits: list[int] = []
    degenerates: list[bool] = []
    transitions: dict[str, dict[str, int]] = {}
    popularity: dict[str, int] = {}
    seen_train: set[str] = set()
    raw_pairs: list[tuple[str, str, int]] = []
    chapter10: dict[str, set[str]] = defaultdict(set)

    n_locale = 0
    n_dropped_missing = 0
    n_dropped_empty_prev = 0
    kept = 0

    for chunk in pd.read_csv(
        sessions_path,
        usecols=SESSION_COLUMNS,
        dtype=str,
        chunksize=100_000,
        keep_default_na=True,
    ):
        chunk = chunk.loc[chunk["locale"] == locale]
        if chunk.empty:
            continue
        for prev_raw, nxt in zip(chunk["prev_items"].tolist(), chunk["next_item"].tolist()):
            n_locale += 1
            if _empty_string(nxt):
                n_dropped_missing += 1
                continue
            prev = parse_prev_items(prev_raw)
            if not prev:
                n_dropped_empty_prev += 1
                continue
            target = str(nxt).strip()
            query = prev[-1]
            if query not in catalog or target not in catalog:
                n_dropped_missing += 1
                continue
            split_id = bucket_to_split(split_bucket(kept, split_seed))
            degenerate = query == target
            query_ids.append(query)
            target_ids.append(target)
            prev_lens.append(len(prev))
            splits.append(split_id)
            degenerates.append(degenerate)

            if len(prev) >= chapter10_min_prev and not degenerate:
                chapter10[query].add(target)

            if split_id == SPLIT_TRAIN:
                seen_train.update(prev)
                seen_train.add(target)
                if not degenerate:
                    sequence = prev + [target]
                    for left, right in zip(sequence, sequence[1:]):
                        if left == right:
                            continue
                        row = transitions.get(left)
                        if row is None:
                            transitions[left] = {right: 1}
                        else:
                            row[right] = row.get(right, 0) + 1
                    popularity[target] = popularity.get(target, 0) + 1
                    raw_pairs.append((query, target, kept))
            kept += 1
            if kept % 400_000 == 0:
                logger.info("Kept %s sessions", kept)

    first_session: dict[tuple[str, str], int] = {}
    for query, positive, session_index in raw_pairs:
        first_session.setdefault((query, positive), session_index)
    n_before_dedup = len(raw_pairs)
    n_after_dedup = len(first_session)
    keys = list(first_session.keys())
    if pair_cap > 0 and len(keys) > pair_cap:
        rng = np.random.RandomState(pair_cap_seed)
        chosen = np.sort(rng.choice(len(keys), size=pair_cap, replace=False))
        keys = [keys[int(i)] for i in chosen]
    del raw_pairs

    prev_len = np.asarray(prev_lens, dtype=np.int32)
    split = np.asarray(splits, dtype=np.int8)
    degenerate_arr = np.asarray(degenerates, dtype=bool)
    card = {
        "n_session_rows_in_locale": n_locale,
        "n_sessions_kept": kept,
        "n_sessions_dropped_empty_prev": n_dropped_empty_prev,
        "n_sessions_dropped_missing_catalog_or_target": n_dropped_missing,
        "prev_items_length_median": _percentile(prev_len, 50),
        "prev_items_length_p90": _percentile(prev_len, 90),
        "prev_items_length_mean": float(prev_len.mean()) if prev_len.size else float("nan"),
        "n_degenerate_sessions": int(degenerate_arr.sum()),
        "n_degenerate_by_split": {
            SPLIT_NAMES[split_id]: int(np.sum(degenerate_arr & (split == split_id)))
            for split_id in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
        },
        "n_sessions_by_split": {
            SPLIT_NAMES[split_id]: int(np.sum(split == split_id))
            for split_id in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
        },
        "n_train_pairs_before_dedup": n_before_dedup,
        "n_train_pairs_after_dedup": n_after_dedup,
        "n_train_pairs_after_cap": len(keys),
        "pair_cap": pair_cap,
        "pair_cap_seed": pair_cap_seed,
        "split_seed": split_seed,
        "split_rule": "sha256(f'{seed}:{filtered_row_index}') mod 10; 0-7 train, 8 val, 9 test",
        "chapter10_min_prev_items": chapter10_min_prev,
        "n_chapter10_query_items": len(chapter10),
    }
    logger.info(
        "Sessions kept=%s train_pairs=%s degenerate=%s",
        kept,
        len(keys),
        card["n_degenerate_sessions"],
    )
    return {
        "query_ids": query_ids,
        "target_ids": target_ids,
        "prev_len": prev_len,
        "split": split,
        "degenerate": degenerate_arr,
        "pair_query": [query for query, _ in keys],
        "pair_positive": [positive for _, positive in keys],
        "pair_session": np.asarray([first_session[key] for key in keys], dtype=np.int64),
        "pair_keys_before_cap": set(first_session.keys()),
        "transitions": transitions,
        "popularity": popularity,
        "seen_train_items": seen_train,
        "chapter10_map": dict(chapter10),
        "card": card,
    }


def prepare_dataset(
    data_dir: Path,
    *,
    pair_cap: int,
    locale: str = LOCALE,
    split_seed: int = SPLIT_SEED,
) -> PreparedData:
    products_path = data_dir / "products_train.csv"
    sessions_path = data_dir / "sessions_train.csv"
    hashes = {
        "products_train.csv": file_sha256(products_path),
        "sessions_train.csv": file_sha256(sessions_path),
    }
    catalog_ids, texts, product_card = load_catalog(products_path, locale=locale)
    session_payload = load_sessions(
        sessions_path,
        texts,
        locale=locale,
        split_seed=split_seed,
        pair_cap=pair_cap,
    )
    card = {}
    card.update(product_card)
    card.update(session_payload["card"])
    return PreparedData(
        catalog_ids=catalog_ids,
        texts=texts,
        query_ids=session_payload["query_ids"],
        target_ids=session_payload["target_ids"],
        prev_len=session_payload["prev_len"],
        split=session_payload["split"],
        degenerate=session_payload["degenerate"],
        pair_query=session_payload["pair_query"],
        pair_positive=session_payload["pair_positive"],
        pair_session=session_payload["pair_session"],
        pair_keys_before_cap=session_payload["pair_keys_before_cap"],
        transitions=session_payload["transitions"],
        popularity=session_payload["popularity"],
        seen_train_items=session_payload["seen_train_items"],
        chapter10_map=session_payload["chapter10_map"],
        card=card,
        file_hashes=hashes,
    )


def sample_indices(indices: np.ndarray, n: int | None, seed: int = SAMPLE_SEED) -> np.ndarray:
    """Sample session indices with an independent RandomState(seed).

    The result is sorted so the set, not the draw order, determines later loops.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if n is None or len(indices) <= n:
        return np.sort(indices)
    rng = np.random.RandomState(seed)
    picked = rng.choice(indices, size=n, replace=False)
    return np.sort(picked.astype(np.int64))


def sample_like_chapter10(ids: list[str], n: int, seed: int) -> list[str]:
    """Match evaluate_amazon.py: RandomState(seed).choice without replacement, unsorted."""
    if len(ids) <= n:
        return list(ids)
    rng = np.random.RandomState(seed)
    picked = rng.choice(ids, size=n, replace=False)
    return [str(item) for item in picked]


def chapter10_query_ids(
    item_ids_in_catalog_order: list[str],
    next_item_map: dict[str, set[str]],
    n: int,
    seed: int,
) -> list[str]:
    candidates = [
        product_id
        for product_id in item_ids_in_catalog_order
        if product_id in next_item_map and len(next_item_map[product_id]) > 0
    ]
    return sample_like_chapter10(candidates, n, seed)


def sample_catalog(
    catalog_ids: list[str],
    must_include: list[str],
    n: int | None,
    seed: int = SAMPLE_SEED,
) -> tuple[list[str], dict]:
    """Return catalog ids in file order, forcing must_include into a sample of size n."""
    catalog_set = set(catalog_ids)
    must: list[str] = []
    seen: set[str] = set()
    n_missing = 0
    for product_id in must_include:
        if product_id not in catalog_set:
            n_missing += 1
            continue
        if product_id in seen:
            continue
        seen.add(product_id)
        must.append(product_id)
    if n is None or n >= len(catalog_ids):
        return list(catalog_ids), {
            "n_catalog": len(catalog_ids),
            "n_must_include": len(must),
            "n_must_missing": n_missing,
            "expanded_past_cap": False,
            "full_catalog": True,
        }
    expanded = len(must) > n
    chosen = set(must)
    if not expanded:
        remaining = [product_id for product_id in catalog_ids if product_id not in chosen]
        need = n - len(chosen)
        if need > 0 and remaining:
            rng = np.random.RandomState(seed)
            pick = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            for position in pick:
                chosen.add(remaining[int(position)])
    ordered = [product_id for product_id in catalog_ids if product_id in chosen]
    return ordered, {
        "n_catalog": len(ordered),
        "n_must_include": len(must),
        "n_must_missing": n_missing,
        "expanded_past_cap": expanded,
        "full_catalog": False,
        "requested_n": n,
    }


def texts_for(ids: list[str], texts: dict[str, str]) -> list[str]:
    return [texts[product_id] for product_id in ids]


def print_data_card(card: dict) -> None:
    print("Amazon-M2 data card")
    for key in (
        "locale",
        "n_products",
        "n_sessions_kept",
        "description_words_median",
        "description_words_p90",
        "prev_items_length_median",
        "prev_items_length_p90",
        "n_sessions_by_split",
        "n_degenerate_sessions",
        "n_degenerate_by_split",
        "n_train_pairs_before_dedup",
        "n_train_pairs_after_dedup",
        "n_train_pairs_after_cap",
        "text_template",
    ):
        print(f"  {key}: {card.get(key)}")
