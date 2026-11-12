"""Exercise 2. Compare a local Qdrant hybrid query with the script fusion.

When dense and sparse matrices from Exercise 1 are present, the comparison
uses those. Otherwise it runs the same check on a small random collection so
the comparison can be inspected without the full catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from qdrant_demo import run_parity


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant parity check.")
    parser.add_argument("--dense", default=None)
    parser.add_argument("--sparse", default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--output", default="outputs/qdrant_parity.json")
    args = parser.parse_args()
    if args.dense and args.sparse and args.ids:
        dense = np.load(args.dense)
        sparse = sp.load_npz(args.sparse)
        product_ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))
        rng = np.random.default_rng(20261110)
        query_dense = dense[:20] if dense.shape[0] >= 20 else dense
        query_sparse = sparse[: query_dense.shape[0]]
        if query_dense.shape[0] < 4:
            query_dense = rng.normal(size=(4, dense.shape[1])).astype(np.float32)
            query_sparse = sp.random(4, sparse.shape[1], density=0.05, format="csr", random_state=1)
    else:
        rng = np.random.default_rng(20261110)
        product_ids = [f"p{i:03d}" for i in range(30)]
        dense = rng.normal(size=(30, 16)).astype(np.float32)
        sparse = sp.random(30, 40, density=0.1, format="csr", random_state=2)
        query_dense = rng.normal(size=(6, 16)).astype(np.float32)
        query_sparse = sp.random(6, 40, density=0.1, format="csr", random_state=3)
    report = run_parity(
        product_ids,
        dense,
        sparse.tocsr(),
        query_dense,
        query_sparse.tocsr(),
        Path(args.output).with_suffix(".qdrant"),
        prefetch_limit=min(1000, len(product_ids)),
        top_k=min(10, len(product_ids)),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Parity pass: {report['pass']}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
