"""Popularity and co-visitation baselines.

Popularity ranks by how often an item is the next item in a training session.
Co-visitation ranks by the count of consecutive transitions in training
sessions, then by popularity, then by catalog order. The query item is excluded.
Both rankings are inside the evaluation catalog.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def popularity_array(catalog_ids: Sequence[str], popularity: dict[str, int]) -> np.ndarray:
    return np.asarray([int(popularity.get(product_id, 0)) for product_id in catalog_ids], dtype=np.int64)


def popularity_positions(pop_array: np.ndarray) -> np.ndarray:
    """0-based position in the popularity ranking. Ties break toward the earlier catalog row."""
    n = len(pop_array)
    order = np.lexsort((np.arange(n, dtype=np.int64), -pop_array))
    position = np.empty(n, dtype=np.int32)
    position[order] = np.arange(n, dtype=np.int32)
    return position


def popularity_rank(query_row: int, target_row: int, position: np.ndarray) -> int:
    """1-based rank of the target after removing the query row."""
    if int(query_row) == int(target_row):
        raise ValueError("Query and target rows must differ")
    target_pos = int(position[target_row])
    query_pos = int(position[query_row])
    if query_pos < target_pos:
        return target_pos
    return target_pos + 1


def _pop_greater_cut(pop_array: np.ndarray) -> np.ndarray:
    """Descending popularity order, used to count items with popularity above a threshold."""
    return np.lexsort((np.arange(len(pop_array), dtype=np.int64), -pop_array))


def rank_covisitation(
    query: str,
    target: str,
    trans_row: dict[str, int],
    popularity: dict[str, int],
    id_to_row: dict[str, int],
    pop_array: np.ndarray,
    indices_with_pop: dict[int, list[int]],
    pop_desc_order: np.ndarray,
) -> int:
    """1-based lexicographic rank: transition count, then popularity, then catalog index."""
    if query == target:
        raise ValueError("Query and target must differ")
    target_row = id_to_row[target]
    target_count = int(trans_row.get(target, 0))
    target_pop = int(popularity.get(target, 0))
    better = 0
    if target_count > 0:
        for item, count in trans_row.items():
            if item == query or item == target or item not in id_to_row:
                continue
            count = int(count)
            item_pop = int(popularity.get(item, 0))
            item_row = id_to_row[item]
            if count > target_count:
                better += 1
            elif count == target_count and item_pop > target_pop:
                better += 1
            elif count == target_count and item_pop == target_pop and item_row < target_row:
                better += 1
        return better + 1

    for item in trans_row:
        if item != query and item in id_to_row:
            better += 1

    pops_desc = pop_array[pop_desc_order]
    # First position whose popularity is <= target_pop. Items before it have a higher count of... 
    # pops_desc is descending. Count of pop > target_pop:
    cut = int(np.searchsorted(-pops_desc, -target_pop, side="left"))
    n_greater = cut
    for item in trans_row:
        if item == query or item not in id_to_row:
            continue
        if int(popularity.get(item, 0)) > target_pop:
            n_greater -= 1
    query_row = id_to_row.get(query)
    if query_row is not None and int(pop_array[query_row]) > target_pop:
        n_greater -= 1
    if n_greater < 0:
        n_greater = 0
    better += n_greater

    tie_group = indices_with_pop.get(target_pop, [])
    n_tie = int(np.searchsorted(np.asarray(tie_group, dtype=np.int64), target_row, side="left"))
    for item in trans_row:
        if item == query or item not in id_to_row:
            continue
        item_row = id_to_row[item]
        if int(popularity.get(item, 0)) == target_pop and item_row < target_row:
            n_tie -= 1
    if query_row is not None and int(pop_array[query_row]) == target_pop and query_row < target_row:
        n_tie -= 1
    if n_tie < 0:
        n_tie = 0
    better += n_tie
    return better + 1


def build_covis_helpers(
    catalog_ids: Sequence[str], popularity: dict[str, int]
) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray, dict[str, int]]:
    pop_array = popularity_array(catalog_ids, popularity)
    indices_with_pop: dict[int, list[int]] = {}
    for row, value in enumerate(pop_array.tolist()):
        indices_with_pop.setdefault(int(value), []).append(row)
    order = _pop_greater_cut(pop_array)
    id_to_row = {product_id: row for row, product_id in enumerate(catalog_ids)}
    return pop_array, indices_with_pop, order, id_to_row


def ranks_for_pairs(
    query_ids: Sequence[str],
    target_ids: Sequence[str],
    catalog_ids: Sequence[str],
    popularity: dict[str, int],
    transitions: dict[str, dict[str, int]],
    kind: str,
) -> np.ndarray:
    """Rank every evaluable pair. kind is 'popularity' or 'covisitation'."""
    id_to_row = {product_id: row for row, product_id in enumerate(catalog_ids)}
    ranks = np.full(len(query_ids), -1, dtype=np.int32)
    if kind == "popularity":
        position = popularity_positions(popularity_array(catalog_ids, popularity))
        for index, (query, target) in enumerate(zip(query_ids, target_ids)):
            q_row = id_to_row.get(query)
            t_row = id_to_row.get(target)
            if q_row is None or t_row is None or q_row == t_row:
                continue
            ranks[index] = popularity_rank(q_row, t_row, position)
        return ranks
    if kind != "covisitation":
        raise ValueError(f"Unknown baseline: {kind}")
    pop_array, indices_with_pop, order, id_to_row = build_covis_helpers(catalog_ids, popularity)
    for index, (query, target) in enumerate(zip(query_ids, target_ids)):
        if query not in id_to_row or target not in id_to_row or query == target:
            continue
        ranks[index] = rank_covisitation(
            query,
            target,
            transitions.get(query, {}),
            popularity,
            id_to_row,
            pop_array,
            indices_with_pop,
            order,
        )
    return ranks
