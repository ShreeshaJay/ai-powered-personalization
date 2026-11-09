"""
Start the five services locally with uvicorn (no Docker needed).

    python scripts/run_local.py                    # start, wait for readiness, run until Ctrl+C
    python scripts/run_local.py --smoke            # start, smoke test, stop
    python scripts/run_local.py --smoke --locust   # ... then a short closed-loop Locust ramp, stop

Ports: orchestrator 8000, retrieval 8001, lightweight ranker 8002,
heavy ranker 8003, re-ranker 8004 (--base-port shifts all five). Every
service loads the active bundle (artifacts/ACTIVE, or CH13_BUNDLE_DIR).
Logs and reports go to outputs/local_run/<timestamp>/. The script refuses
to start if a port is already taken, and every process started here is
stopped on exit, including after errors and Ctrl+C.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

CHAPTER_DIR = Path(__file__).resolve().parents[1]
SERVICES = [                                    # name, app, port offset from --base-port
    ("retrieval", "services.retrieval.app:app", 1),
    ("lightweight_ranker", "services.lightweight_ranker.app:app", 2),
    ("heavy_ranker", "services.heavy_ranker.app:app", 3),
    ("reranker", "services.reranker.app:app", 4),
    ("orchestrator", "services.orchestrator.app:app", 0),
]
WINDOWS = sys.platform == "win32"


def stage_urls(host: str, base_port: int) -> dict:
    return {name: f"http://{host}:{base_port + offset}" for name, _, offset in SERVICES}


def ports_in_use(host: str, base_port: int) -> list:
    taken = []
    for name, _, offset in SERVICES:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, base_port + offset))
            except OSError:
                taken.append(f"{name} ({base_port + offset})")
    return taken


def service_env(host: str, base_port: int) -> dict:
    env = os.environ.copy()
    urls = stage_urls(host, base_port)
    env["PYTHONPATH"] = str(CHAPTER_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["RETRIEVAL_URL"] = urls["retrieval"]
    env["LIGHTWEIGHT_RANKER_URL"] = urls["lightweight_ranker"]
    env["HEAVY_RANKER_URL"] = urls["heavy_ranker"]
    env["RERANKER_URL"] = urls["reranker"]
    # Five processes share the machine: one compute thread each.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS", "XGB_NUM_THREADS", "FAISS_NUM_THREADS"):
        env.setdefault(name, "1")
    env.setdefault("TQDM_DISABLE", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def start(host: str, base_port: int, log_dir: Path) -> list:
    processes = []
    env = service_env(host, base_port)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0
    for name, module, offset in SERVICES:
        port = base_port + offset
        log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
        command = [sys.executable, "-m", "uvicorn", module, "--host", host, "--port", str(port),
                   "--workers", "1", "--log-level", "warning"]
        process = subprocess.Popen(command, cwd=CHAPTER_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=flags)
        processes.append((name, port, process, log))
        print(f"  started {name:20s} port {port}  pid {process.pid}")
    return processes


def stop(processes: list) -> None:
    for name, _, process, _ in processes:
        if process.poll() is None:
            try:
                if WINDOWS:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGINT)
            except OSError:
                pass
    deadline = time.time() + 15
    for name, _, process, log in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        print(f"  stopped {name:20s} exit code {process.returncode}")


def wait_ready(processes: list, host: str, timeout_s: float) -> bool:
    pending = {name: port for name, port, _, _ in processes}
    started = time.time()
    with httpx.Client(timeout=2.0) as client:
        while pending and time.time() - started < timeout_s:
            for name, _, process, _ in processes:
                if name in pending and process.poll() is not None:
                    print(f"  {name} exited with code {process.returncode} before becoming ready; see its log")
                    return False
            for name, port in list(pending.items()):
                try:
                    response = client.get(f"http://{host}:{port}/health/ready")
                    if response.status_code == 200:
                        body = response.json()
                        print(f"  ready {name:20s} model_version={body.get('model_version')} "
                              f"load {body.get('load_seconds')} s, warm-up {body.get('warm_up_ms')} ms "
                              f"(after {time.time() - started:.0f} s)")
                        del pending[name]
                except httpx.HTTPError:
                    pass
            time.sleep(1.0)
    if pending:
        print(f"  not ready after {timeout_s:.0f} s: {sorted(pending)}")
    return not pending


def exited(processes: list) -> list:
    return [f"{name} (exit code {process.returncode})" for name, _, process, _ in processes
            if process.poll() is not None]


def run_step(command: list, env: dict) -> int:
    print("  $ " + " ".join(command))
    return subprocess.call(command, cwd=CHAPTER_DIR, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8000,
                        help="orchestrator port; the four stage services use the next four ports")
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--smoke", action="store_true", help="run scripts/smoke_test.py, then stop")
    parser.add_argument("--locust", action="store_true", help="run a headless Locust ramp, then stop")
    parser.add_argument("--ramp", default=os.environ.get("RAMP", "20:2,40:5,60:10,80:20,100:30"),
                        help="Locust closed-loop ramp: end_second:users,... (default about 100 s)")
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/local_run/<timestamp>")
    args = parser.parse_args()

    taken = ports_in_use(args.host, args.base_port)
    if taken:
        print(f"Ports already in use: {', '.join(taken)}. Stop whatever holds them, or pass --base-port "
              f"to use another range.")
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or CHAPTER_DIR / "outputs" / "local_run" / stamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"Logs and reports: {out}")
    urls = stage_urls(args.host, args.base_port)
    processes = start(args.host, args.base_port, out)
    results = {"started_at_utc": stamp, "base_port": args.base_port, "smoke": None, "locust": None,
               "services_exited_early": []}
    try:
        if not wait_ready(processes, args.host, args.ready_timeout_s):
            return 1
        env = service_env(args.host, args.base_port)
        if args.smoke:
            services = [arg for name, url in urls.items() if name != "orchestrator"
                        for arg in ("--service", f"{name}={url}")]
            results["smoke"] = run_step([sys.executable, "scripts/smoke_test.py", "--url", urls["orchestrator"],
                                         *services, "--out", str(out / "smoke.json")], env)
        if args.locust:
            env["RAMP"] = args.ramp
            env["LOCUST_SUMMARY"] = str(out / "locust_summary.json")
            results["locust"] = run_step(
                [sys.executable, "-m", "locust", "-f", "load/locustfile.py", "--headless",
                 "--host", urls["orchestrator"], "--csv", str(out / "locust"), "--only-summary"], env)
        if not (args.smoke or args.locust):
            print(f"All services ready; orchestrator at {urls['orchestrator']}. Press Ctrl+C to stop.")
            while all(p.poll() is None for _, _, p, _ in processes):
                time.sleep(1.0)
            print("A service exited; stopping the rest.")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        results["services_exited_early"] = exited(processes)
        if results["services_exited_early"]:
            print(f"  services that exited before shutdown: {results['services_exited_early']}; "
                  f"the results above may not come from this run's services. See the logs in {out}.")
        print("Stopping services")
        stop(processes)
        (out / "run_local.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [k for k in ("smoke", "locust") if results[k] not in (None, 0)]
    return 1 if failed or results["services_exited_early"] else 0


if __name__ == "__main__":
    sys.exit(main())
