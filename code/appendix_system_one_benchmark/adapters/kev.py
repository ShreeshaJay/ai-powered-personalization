"""Local Kev adapter. Talks to kev.serve over the TypeSafe /v1/systemone contract."""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from adapters.base import ItemPrediction
from adapters.jev import build_systemone_payload, estimate_input_tokens
from adapters.laya import decode_laya_result
from adapters.schemas import compact_state, laya_questions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "jaredpalmer/kev-0.8b"
DEFAULT_MODEL = "kev-latest"
DEFAULT_BASE_URL = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8008")
DEFAULT_PORT = 8008
DEFAULT_PYTHON = ROOT / ".venvs" / "kev" / "Scripts" / "python.exe"


def default_kev_python() -> Path:
    override = (os.environ.get("KEV_PYTHON") or "").strip()
    return Path(override) if override else DEFAULT_PYTHON


def build_kev_payload(
    task: str,
    adjudication_input: dict[str, Any],
    model_id: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    return build_systemone_payload(task, adjudication_input, model_id=model_id)


class KevAdapter:
    name = "kev"

    def __init__(
        self,
        run: str = DEFAULT_RUN,
        model_id: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
        max_retries: int = 5,
        concurrency: int = 2,
        autostart: bool = True,
        python_path: Path | None = None,
    ) -> None:
        self.run = run
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.autostart = autostart
        self.python_path = Path(python_path) if python_path else default_kev_python()
        self._process: subprocess.Popen[str] | None = None
        self._ready = False

    @property
    def cache_model_id(self) -> str:
        return self.run

    def load(self) -> None:
        if self._ready:
            return
        if self._healthy():
            self._ready = True
            return
        if not self.autostart:
            raise RuntimeError(
                f"Kev serve is not reachable at {self.base_url}. "
                "Start `python -m kev.serve --run "
                f"{self.run}` or pass autostart=True."
            )
        self._start_server()
        self._wait_healthy(timeout_seconds=900.0)
        self._ready = True

    def predict_batch(
        self,
        task: str,
        examples: list[dict[str, Any]],
        batch_size: int = 2,
    ) -> list[ItemPrediction]:
        del batch_size
        if not examples:
            return []
        self.load()
        questions = laya_questions(task)
        workers = max(1, min(self.concurrency, len(examples)))
        predictions: dict[str, ItemPrediction] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._predict_one,
                    example["item_id"],
                    task,
                    compact_state(task, example["adjudication_input"]),
                    questions,
                ): example["item_id"]
                for example in examples
            }
            for future in as_completed(futures):
                prediction = future.result()
                predictions[prediction.item_id] = prediction
        return [predictions[example["item_id"]] for example in examples]

    def _predict_one(
        self,
        item_id: str,
        task: str,
        state: dict[str, Any],
        questions: dict[str, Any],
    ) -> ItemPrediction:
        import httpx

        payload = {
            "model": self.model_id,
            "state": state,
            "questions": questions,
        }
        headers = {"Content-Type": "application/json"}
        api_key = (os.environ.get("KEV_API_KEY") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.base_url}/v1/systemone"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503}:
                    time.sleep(min(2**attempt, 20))
                    last_error = RuntimeError(
                        f"Kev HTTP {response.status_code}: {response.text[:300]}"
                    )
                    continue
                response.raise_for_status()
                raw = response.json()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                prediction = decode_laya_result(
                    item_id, task, raw, elapsed_ms=elapsed_ms
                )
                if not prediction.input_tokens:
                    prediction = ItemPrediction(
                        item_id=prediction.item_id,
                        task=prediction.task,
                        fields=prediction.fields,
                        input_tokens=estimate_input_tokens(payload),
                        elapsed_ms=prediction.elapsed_ms,
                        raw=raw,
                    )
                return prediction
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 20))
        raise RuntimeError(f"Kev request failed for {item_id}: {last_error}")

    def _healthy(self) -> bool:
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
            if response.status_code != 200:
                return False
            names = {
                str(model.get("name"))
                for model in (response.json().get("models") or [])
            }
            return bool(names & {"kev-latest", "jev-latest", self.model_id})
        except httpx.HTTPError:
            return False

    def _start_server(self) -> None:
        if not self.python_path.is_file():
            raise RuntimeError(
                f"Kev Python is missing at {self.python_path}. "
                "Create the isolated 3.12 env under .venvs/kev first."
            )
        log_path = ROOT / "outputs" / "benchmark" / "kev_serve.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("USE_TF", "0")
        env.setdefault("KEV_DTYPE", "bf16")
        env.setdefault("KEV_BACKEND", "torch")
        env.setdefault("KEV_CUDA_GRAPHS", "0")
        env.setdefault("KEV_FUSED", "0")
        log_file = log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                str(self.python_path),
                "-m",
                "kev.serve",
                "--run",
                self.run,
                "--host",
                "127.0.0.1",
                "--port",
                str(DEFAULT_PORT),
            ],
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _wait_healthy(self, timeout_seconds: float) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Kev serve exited with code {self._process.returncode}. "
                    "See outputs/benchmark/kev_serve.log."
                )
            if self._healthy():
                return
            time.sleep(2.0)
        raise RuntimeError(
            f"Kev serve did not become healthy at {self.base_url} "
            f"within {timeout_seconds:.0f}s. See outputs/benchmark/kev_serve.log."
        )
