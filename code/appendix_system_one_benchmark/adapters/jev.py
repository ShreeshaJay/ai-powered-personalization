"""TypeSafe Jev API adapter. One item per request; questions run in parallel."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from adapters.base import ItemPrediction
from adapters.laya import decode_laya_result
from adapters.schemas import compact_state, laya_questions


DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
INPUT_USD_PER_MILLION = 0.042
ENV_KEY_NAMES = ("TYPESAFE_API_KEY", "JEV_API_KEY")


def load_typesafe_api_key() -> str:
    for name in ENV_KEY_NAMES:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / ".env",
        root.parent / ".env",
        root.parents[2] / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() in ENV_KEY_NAMES:
                key = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(name.strip(), key)
                    return key
    raise RuntimeError(
        "Missing TypeSafe API key. Set TYPESAFE_API_KEY before running Jev."
    )


def build_systemone_payload(
    task: str,
    adjudication_input: dict[str, Any],
    model_id: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "state": compact_state(task, adjudication_input),
        "questions": laya_questions(task),
    }


def estimate_input_tokens(payload: dict[str, Any]) -> int:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(serialized) + 3) // 4)


def token_cost_usd(input_tokens: int, rate: float = INPUT_USD_PER_MILLION) -> float:
    return input_tokens * rate / 1_000_000


class JevAdapter:
    name = "jev"
    input_usd_per_million = INPUT_USD_PER_MILLION

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        concurrency: int = 8,
        budget_usd: float | None = 2.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.budget_usd = budget_usd
        self.spent_input_tokens = 0
        self._spend_lock = threading.Lock()
        self._api_key: str | None = None

    @property
    def cache_model_id(self) -> str:
        return self.model_id

    def load(self) -> None:
        self._api_key = load_typesafe_api_key()

    def predict_batch(
        self,
        task: str,
        examples: list[dict[str, Any]],
        batch_size: int = 8,
    ) -> list[ItemPrediction]:
        del batch_size
        if not examples:
            return []
        if self._api_key is None:
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
        estimated = estimate_input_tokens(payload)
        self._reserve_tokens(estimated)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
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
                if response.status_code in {429, 529, 500, 502, 503}:
                    retry_after = float(response.headers.get("retry-after") or 0)
                    delay = retry_after if retry_after > 0 else min(2**attempt, 20)
                    time.sleep(delay)
                    last_error = RuntimeError(
                        f"Jev HTTP {response.status_code}: {response.text[:300]}"
                    )
                    continue
                response.raise_for_status()
                raw = response.json()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                prediction = decode_laya_result(
                    item_id, task, raw, elapsed_ms=elapsed_ms
                )
                billed = prediction.input_tokens or estimated
                self._commit_tokens(estimated, billed)
                return prediction
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 20))
        self._release_tokens(estimated)
        raise RuntimeError(f"Jev request failed for {item_id}: {last_error}")

    def _reserve_tokens(self, additional_tokens: int) -> None:
        with self._spend_lock:
            projected = token_cost_usd(
                self.spent_input_tokens + additional_tokens,
                self.input_usd_per_million,
            )
            if self.budget_usd is not None and projected > self.budget_usd:
                raise RuntimeError(
                    f"Jev budget ${self.budget_usd:.4f} would be exceeded "
                    f"(projected ${projected:.4f})."
                )
            self.spent_input_tokens += additional_tokens

    def _commit_tokens(self, reserved: int, billed: int) -> None:
        with self._spend_lock:
            self.spent_input_tokens += billed - reserved

    def _release_tokens(self, reserved: int) -> None:
        with self._spend_lock:
            self.spent_input_tokens = max(0, self.spent_input_tokens - reserved)
