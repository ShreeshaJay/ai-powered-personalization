"""Seeded best-of-N simulation for Figure C.2.

The constants match ``figures_src/appD_figures.py`` (the internal name of the
Appendix C figure script): seed 20261109, 5000 repetitions, K in
{1, 2, 4, 8, 16, 32, 64, 128, 256}, and candidate quality drawn from
Normal(0, 0.3^2). Validation and test noise are independent Normal(0, 1).

For each K the script selects the candidate with the highest validation score
and records that candidate's mean validation score, its mean test score, and
the mean test score of every candidate. Those three series are the lines in
Figure C.2.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from harness.contract import ROOT

SIM_SEED = 20261109
SIM_REPS = 5000
SIM_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SIM_SIGMA_Q = 0.3
COLUMNS = ["K", "val_selected", "test_selected", "test_average_candidate"]


def simulate(seed: int = SIM_SEED) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for k in SIM_KS:
        quality = rng.normal(0.0, SIM_SIGMA_Q, (SIM_REPS, k))
        validation = quality + rng.normal(0.0, 1.0, (SIM_REPS, k))
        test = quality + rng.normal(0.0, 1.0, (SIM_REPS, k))
        best = validation.argmax(axis=1)
        index = np.arange(SIM_REPS)
        rows.append(
            {
                "K": k,
                "val_selected": float(validation[index, best].mean()),
                "test_selected": float(test[index, best].mean()),
                "test_average_candidate": float(test.mean()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the Figure C.2 series")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "best_of_n.csv")
    parser.add_argument("--seed", type=int, default=SIM_SEED)
    args = parser.parse_args(argv)
    rows = simulate(args.seed)
    write_csv(args.output, rows)
    print(f"seed {args.seed}, {SIM_REPS} repetitions, wrote {args.output.name}")
    for row in rows:
        print(
            f"K={int(row['K']):<3}  val={row['val_selected']:.6f}  "
            f"test={row['test_selected']:.6f}  "
            f"average={row['test_average_candidate']:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
