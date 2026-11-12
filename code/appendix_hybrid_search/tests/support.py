"""Tiny ESCI-shaped tables for CPU tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_tiny_esci(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    products = []
    for index in range(1, 9):
        products.append(
            {
                "product_id": f"p{index}",
                "product_title": f"Title {index}",
                "product_description": f"Description {index} with extra words",
                "product_bullet_point": f"Bullet A {index}\nBullet B {index}\nBullet C {index}\nBullet D {index}",
                "product_brand": "Acme" if index % 2 == 0 else "",
                "product_color": "red",
                "product_locale": "us",
            }
        )
    examples = [
        ("q1", "red shoes", "p1", "E", "train"),
        ("q1", "red shoes", "p2", "I", "train"),
        ("q2", "blue hat", "p3", "E", "train"),
        ("q3", "without nuts", "p4", "S", "train"),
        ("q3", "without nuts", "p5", "I", "train"),
        ("q4", "model 10", "p6", "E", "train"),
        ("q5", "phone case", "p1", "E", "test"),
        ("q5", "phone case", "p7", "I", "test"),
        ("q6", "a very long query with six words", "p8", "E", "test"),
        ("q6", "a very long query with six words", "p2", "C", "test"),
        ("q7", "hi", "p3", "I", "test"),
        ("q8", "not red", "p4", "E", "test"),
    ]
    rows = []
    for number, (qid, query, pid, label, split) in enumerate(examples):
        rows.append(
            {
                "example_id": number,
                "query_id": qid,
                "query": query,
                "product_id": pid,
                "product_locale": "us",
                "esci_label": label,
                "small_version": 1,
                "large_version": 1,
                "split": split,
            }
        )
    pd.DataFrame(products).to_parquet(directory / "shopping_queries_dataset_products.parquet")
    pd.DataFrame(rows).to_parquet(directory / "shopping_queries_dataset_examples.parquet")
    return directory
