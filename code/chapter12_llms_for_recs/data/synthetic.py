"""
A small synthetic catalog and session log with the same schema as the Amazon KDD loader.

Items belong to categories; a session mostly stays inside one category, so content
similarity carries real signal. Used by the tests and by ``prepare_data.py --synthetic``
for a pipeline check that needs no download. It is not a benchmark.
"""

from __future__ import annotations

import numpy as np
import polars as pl

CATEGORIES = [
    ("phone case", "Accessories"), ("usb cable", "Electronics"), ("running shoe", "Sports"),
    ("yoga mat", "Sports"), ("coffee grinder", "Kitchen"), ("chef knife", "Kitchen"),
    ("face cream", "Beauty"), ("shampoo", "Beauty"), ("lego set", "Toys"), ("puzzle", "Toys"),
    ("desk lamp", "Home"), ("pillow", "Home"),
]
ADJECTIVES = ["classic", "premium", "compact", "eco", "pro", "mini", "deluxe", "travel",
              "smart", "soft", "rugged", "slim"]
BRANDS = ["Acme", "Nordic", "Zenith", "Orbit", "Maple", "Vega"]


def make_synthetic(num_items: int = 600, num_sessions: int = 3000, seed: int = 42,
                   stay_in_category: float = 0.85):
    """Return (sessions, products) frames shaped like ``load_raw_sessions`` / ``load_products``."""
    rng = np.random.default_rng(seed)
    cat_of = rng.integers(0, len(CATEGORIES), size=num_items)
    titles, brands, descs = [], [], []
    for i in range(num_items):
        noun, dept = CATEGORIES[cat_of[i]]
        adj = ADJECTIVES[rng.integers(len(ADJECTIVES))]
        brand = BRANDS[rng.integers(len(BRANDS))]
        titles.append(f"{brand} {adj} {noun} model {i}")
        brands.append(brand)
        descs.append(f"A {adj} {noun} from the {dept} department.")
    item_ids = [f"SYN{i:06d}" for i in range(num_items)]
    products = pl.DataFrame({"item_id": item_ids, "title": titles, "brand": brands, "desc": descs})

    by_cat = [np.flatnonzero(cat_of == c) for c in range(len(CATEGORIES))]
    popularity = rng.zipf(1.6, size=num_items).astype(np.float64)
    histories, targets = [], []
    for _ in range(num_sessions):
        length = int(rng.integers(3, 9))
        cat = int(rng.integers(len(CATEGORIES)))
        seq = []
        for _ in range(length + 1):
            if rng.random() > stay_in_category:
                cat = int(rng.integers(len(CATEGORIES)))
            pool = by_cat[cat]
            weights = popularity[pool] / popularity[pool].sum()
            seq.append(int(rng.choice(pool, p=weights)))
        histories.append([item_ids[j] for j in seq[:-1]])
        targets.append(item_ids[seq[-1]])
    sessions = pl.DataFrame({
        "session_id": np.arange(num_sessions, dtype=np.int64),
        "history": histories,
        "target": targets,
    })
    return sessions, products
