"""
Platform check: timer granularity, CPU, threads, and package versions.

Exercise 1 Part B simulates remote calls with calibrated sleeps. If the
operating system rounds short sleeps up by several milliseconds, those
simulated delays are distorted, so the script measures sleep and event-loop
timer granularity first and refuses to run when it is too coarse.

Every entry point records this report in its run-info file, so a reader can
tell which machine, thread settings, and library versions produced a result.

Run directly:  python -m latency.platform_check
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Dict

import numpy as np

PACKAGES = ["numpy", "polars", "pyarrow", "scipy", "scikit-learn", "xgboost",
            "faiss-cpu", "torch", "onnx", "onnxruntime"]
THREAD_ENV_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]


def _sleep_overshoot_ms(target_ms: float = 1.0, n: int = 60) -> Dict[str, float]:
    over = []
    for _ in range(n):
        t = time.perf_counter()
        time.sleep(target_ms / 1000)
        over.append((time.perf_counter() - t) * 1000 - target_ms)
    arr = np.array(over)
    return {"target_ms": target_ms, "median_overshoot_ms": float(np.median(arr)),
            "p99_overshoot_ms": float(np.percentile(arr, 99))}


def _asyncio_overshoot_ms(target_ms: float = 1.0, n: int = 40) -> Dict[str, float]:
    async def probe():
        over = []
        for _ in range(n):
            t = time.perf_counter()
            await asyncio.sleep(target_ms / 1000)
            over.append((time.perf_counter() - t) * 1000 - target_ms)
        return over

    arr = np.array(asyncio.run(probe()))
    return {"target_ms": target_ms, "median_overshoot_ms": float(np.median(arr)),
            "p99_overshoot_ms": float(np.percentile(arr, 99))}


def _cpu_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "logical_cpus": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }
    if hasattr(os, "sched_getaffinity"):
        info["usable_cpus"] = len(os.sched_getaffinity(0))
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                info["model_name"] = line.split(":", 1)[1].strip()
                break
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor.exists():
        info["scaling_governor"] = governor.read_text().strip()
    return info


def _versions() -> Dict[str, str]:
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run_check(max_granularity_ms: float = 2.0) -> Dict[str, object]:
    """Measure the platform and return a JSON-serializable report."""
    sleep = _sleep_overshoot_ms()
    loop = _asyncio_overshoot_ms()
    clock = time.get_clock_info("perf_counter")
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": sys.platform,
        "cpu": _cpu_info(),
        "perf_counter_resolution_s": clock.resolution,
        "sleep_1ms": sleep,
        "asyncio_sleep_1ms": loop,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "packages": _versions(),
        "max_granularity_ms": max_granularity_ms,
    }
    report["timer_ok"] = sleep["median_overshoot_ms"] <= max_granularity_ms
    return report


GUIDANCE = (
    "Measured sleep granularity is coarser than {limit} ms (median overshoot {got:.2f} ms).\n"
    "Simulated remote-call delays would be distorted. Options:\n"
    "  * run on Linux or macOS, where 1 ms sleeps are usually accurate;\n"
    "  * on Windows, use Python 3.11 or newer (high-resolution sleep), close\n"
    "    applications that change the system timer, and use a high-performance\n"
    "    power plan;\n"
    "  * pass --force to run anyway and treat the results as indicative only."
)


def main() -> None:
    report = run_check()
    print(json.dumps(report, indent=2))
    if not report["timer_ok"]:
        print(GUIDANCE.format(limit=report["max_granularity_ms"],
                              got=report["sleep_1ms"]["median_overshoot_ms"]))


if __name__ == "__main__":
    main()
