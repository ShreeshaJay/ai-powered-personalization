"""
Per-stage latency percentiles from the orchestrator's request log.

The orchestrator writes one JSON record per request ("request {...}") with
the time each stage took as the orchestrator saw it. This script reads those
records and prints p50, p95 and p99 per stage.

    python scripts/run_local.py --locust --ramp 60:1     # one user for 60 s
    python scripts/stage_latency.py outputs/local_run/<timestamp>/orchestrator.log
    python scripts/stage_latency.py <log> --rung 1   # full-pipeline requests only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

MARKER = " request {"


def read_records(path: Path) -> List[dict]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            position = line.find(MARKER)
            if position >= 0:
                records.append(json.loads(line[position + len(MARKER) - 1:]))
    return records


def stage_percentiles(records: List[dict], rung: Optional[int] = None) -> Dict[str, dict]:
    samples: Dict[str, List[float]] = {}
    for record in records:
        if rung is not None and record.get("rung") != rung:
            continue
        for stage, ms in record.get("timings_ms", {}).items():
            samples.setdefault(stage, []).append(float(ms))
    return {stage: {"count": len(values),
                    **{f"p{q}": round(float(np.percentile(values, q)), 2) for q in (50, 95, 99)}}
            for stage, values in samples.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="orchestrator log written by scripts/run_local.py")
    parser.add_argument("--rung", type=int, default=None, help="only requests served on this rung")
    args = parser.parse_args()
    table = stage_percentiles(read_records(args.log), args.rung)
    if not table:
        print("no request records found", file=sys.stderr)
        return 1
    print(f"{'stage':<20}{'count':>8}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}")
    for stage, row in table.items():
        print(f"{stage:<20}{row['count']:>8}{row['p50']:>10.1f}{row['p95']:>10.1f}{row['p99']:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
