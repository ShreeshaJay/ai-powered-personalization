"""Start local System One servers and run the frozen benchmark on Colab."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


BENCH_ROOT = Path(__file__).resolve().parents[1]
JEVLITE_PORT = 8000
KEV_PORT = 8008


def wait_healthy(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{url}/v1/models", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"{url} did not become healthy: {last_error}")


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def start_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_benchmark(
    python: str,
    model: str,
    model_id: str,
    base_url: str,
    concurrency: int,
    batch_size: int,
    max_items: int,
) -> None:
    env = os.environ.copy()
    env["USE_TF"] = "0"
    command = [
        python,
        str(BENCH_ROOT / "run_benchmark.py"),
        "--model",
        model,
        "--model-id",
        model_id,
        "--device",
        "cuda",
        "--concurrency",
        str(concurrency),
        "--batch-size",
        str(batch_size),
        "--tasks",
        "compatibility,query_segmentation,brand_category,esci",
    ]
    if max_items:
        command.extend(["--max-items", str(max_items)])
    if model == "jevlite":
        env["JEVLITE_BASE_URL"] = base_url
    if model == "kev":
        env["KEV_BASE_URL"] = base_url
    completed = subprocess.run(command, cwd=str(BENCH_ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{model} benchmark exited {completed.returncode}")


def run_jevlite(python: str, jevlite_src: Path, max_items: int) -> None:
    env = os.environ.copy()
    env["USE_TF"] = "0"
    log_path = BENCH_ROOT / "outputs" / "benchmark" / "jevlite_serve.log"
    process = start_logged(
        [python, "serve.py", "--host", "127.0.0.1", "--port", str(JEVLITE_PORT)],
        jevlite_src,
        log_path,
        env,
    )
    try:
        wait_healthy(f"http://127.0.0.1:{JEVLITE_PORT}", 900.0)
        run_benchmark(
            python,
            "jevlite",
            "vagmi/jev-lite",
            f"http://127.0.0.1:{JEVLITE_PORT}",
            concurrency=1,
            batch_size=8,
            max_items=max_items,
        )
    finally:
        stop_process(process)


def run_kev4(python: str, max_items: int) -> None:
    env = os.environ.copy()
    env["USE_TF"] = "0"
    env["KEV_DTYPE"] = "bf16"
    env["KEV_BACKEND"] = "torch"
    env["KEV_CUDA_GRAPHS"] = "1"
    log_path = BENCH_ROOT / "outputs" / "benchmark" / "kev4_serve.log"
    process = start_logged(
        [
            python,
            "-m",
            "kev.serve",
            "--run",
            "jaredpalmer/kev-4b",
            "--host",
            "127.0.0.1",
            "--port",
            str(KEV_PORT),
        ],
        BENCH_ROOT,
        log_path,
        env,
    )
    try:
        wait_healthy(f"http://127.0.0.1:{KEV_PORT}", 900.0)
        run_benchmark(
            python,
            "kev",
            "jaredpalmer/kev-4b",
            f"http://127.0.0.1:{KEV_PORT}",
            concurrency=4,
            batch_size=32,
            max_items=max_items,
        )
    finally:
        stop_process(process)


def write_report(python: str) -> None:
    subprocess.run(
        [python, str(BENCH_ROOT / "write_comparison_report.py")],
        cwd=str(BENCH_ROOT),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jevlite-src", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--models",
        default="jevlite,kev4",
        help="Comma-separated: jevlite,kev4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if "jevlite" in models:
        print("Running JevLite...", flush=True)
        run_jevlite(args.python, args.jevlite_src, args.max_items)
    if "kev4" in models:
        print("Running Kev-4B...", flush=True)
        run_kev4(args.python, args.max_items)
    write_report(args.python)
    print("Wrote outputs/benchmark/ZERO_SHOT_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
