"""
Report helpers: CSV and Markdown writers, run-info files, and tail attribution.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from latency.platform_check import run_check

STAGE_LABELS = {
    "S1": "admission and queueing",
    "S2": "orchestration and request context",
    "S3": "candidate retrieval",
    "S4": "feature retrieval",
    "S5": "model inference",
    "S6": "re-ranking",
    "S7": "response serialization",
    "R": "unattributed residual",
}


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fields})
    return path


def _fmt(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return value


def markdown_table(rows: Sequence[Mapping[str, object]], columns: Optional[Sequence[str]] = None,
                   digits: int = 3) -> str:
    if not rows:
        return "_no rows_\n"
    columns = list(columns or rows[0].keys())

    def cell(v: object) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def thread_settings() -> Dict[str, object]:
    """Thread-pool sizes of the numerical libraries, where they can be read."""
    out: Dict[str, object] = {}
    try:
        import torch
        out["torch_intra_op_threads"] = torch.get_num_threads()
        out["torch_inter_op_threads"] = torch.get_num_interop_threads()
    except Exception:
        pass
    try:
        import faiss
        out["faiss_omp_max_threads"] = faiss.omp_get_max_threads()
    except Exception:
        pass
    return out


def write_run_info(directory: Path, name: str, args: object, extra: Optional[dict] = None,
                   platform_report: Optional[dict] = None) -> Path:
    """Record seed, arguments, platform check, thread settings, and versions."""
    info = {
        "entry_point": name,
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join([Path(sys.executable).name, "-m", name] + sys.argv[1:]),
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "platform": platform_report or run_check(),
        "threads": thread_settings(),
    }
    if extra:
        info.update(extra)
    path = directory / f"{name.split('.')[-1]}_run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------------ attribution
def stage_shares(attributions: Sequence[Mapping[Tuple[str, str], int]],
                 e2e_ns: Sequence[int]) -> List[Dict[str, object]]:
    """Average critical-path milliseconds and share of end-to-end time per (stage, kind).

    ``attributions[i]`` maps ``(stage, kind)`` to critical-path nanoseconds for
    request ``i``. Shares are ratios of sums (total stage time over total
    end-to-end time) so long requests weigh more, matching how the tail is felt.
    """
    n = len(attributions)
    if n == 0:
        return []
    total_e2e = float(np.sum(e2e_ns))
    keys = sorted({k for a in attributions for k in a}, key=lambda k: (k[0], k[1]))
    rows = []
    for stage, kind in keys:
        total = float(sum(a.get((stage, kind), 0) for a in attributions))
        rows.append({
            "stage": stage,
            "stage_name": STAGE_LABELS.get(stage, stage),
            "kind": kind,
            "mean_ms": total / n / 1e6,
            "share": total / total_e2e if total_e2e else math.nan,
        })
    return rows


def select_band(values: np.ndarray, lo_pct: float, hi_pct: float = 100.0) -> np.ndarray:
    """Indices of values between two percentiles of their own distribution."""
    lo = np.percentile(values, lo_pct)
    hi = np.percentile(values, hi_pct)
    return np.flatnonzero((values >= lo) & (values <= hi))
