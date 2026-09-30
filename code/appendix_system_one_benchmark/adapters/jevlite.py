"""Local JevLite adapter. Talks to jevlite serve.py over /v1/systemone."""

from __future__ import annotations

import os
from pathlib import Path

from adapters.kev import KevAdapter


DEFAULT_RUN = "vagmi/jev-lite"
DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class JevLiteAdapter(KevAdapter):
    name = "jevlite"

    def __init__(
        self,
        run: str = DEFAULT_RUN,
        model_id: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 5,
        concurrency: int = 1,
        autostart: bool = False,
        python_path: Path | None = None,
    ) -> None:
        super().__init__(
            run=run,
            model_id=model_id,
            base_url=base_url
            or os.environ.get("JEVLITE_BASE_URL", DEFAULT_BASE_URL),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            concurrency=concurrency,
            autostart=autostart,
            python_path=python_path,
        )
