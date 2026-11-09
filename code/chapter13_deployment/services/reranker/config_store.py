"""
Hot-reloadable value-function configuration with validation (Section 13.6).

Layout of the config directory (RERANKER_CONFIG_DIR, default config/reranker):

    vf-default-v1.json    immutable, versioned files; the name equals config_version
    vf-growth-v1.json
    ACTIVE                one line naming the version replicas should serve

Promote or revert with scripts/promote_config.py, which validates the target
before rewriting ACTIVE atomically. Each replica polls ACTIVE (every
RERANKER_CONFIG_POLL_S seconds) and can be told to reload with
POST /v1/config/reload. A file that fails validation is never applied: the
replica keeps its last good config and reports the error in /v1/config and
/health. Replicas pick up a change at slightly different times, so every
response carries the config_version that produced it.

A request may pin another version with policy_version (Section 13.2.1), for
example to keep an experiment arm on one policy while ACTIVE moves. The
pinned file passes the same validation before it is used.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from common import published
from common.artifacts import CHAPTER_DIR

DEFAULT_CONFIG_DIR = CHAPTER_DIR / "config" / "reranker"
MAX_PINNED = 16        # pinned versions kept in memory per replica
logger = logging.getLogger("ch13.reranker")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ValueWeights(_Strict):
    """Chapter 8's ValueFunctionConfig. Task mapping: P(listen)=engagement, E[play]=play_ratio."""
    w_listen: float = Field(ge=0)
    w_like: float = Field(ge=0)
    w_engagement: float = Field(ge=0)
    w_dislike_penalty: float = Field(ge=0)
    min_listen_threshold: float = Field(0.0, ge=0, le=1)
    normalize_scores: bool = True


class DiversitySettings(_Strict):
    """Chapter 8's DiversityConfig (cosine MMR on content embeddings)."""
    enabled: bool = True
    lambda_param: float = Field(0.7, ge=0, le=1)
    candidate_multiplier: int = Field(2, ge=1, le=10, description="MMR selects num_results x this, as Chapter 8 does")


class PacingSettings(_Strict):
    """Chapter 8's PacingConfig (hard business rules)."""
    enabled: bool = True
    max_per_artist: int = Field(2, ge=1, le=100)
    max_per_album: int = Field(1, ge=1, le=100)
    defer_violations: bool = True


class RerankerConfig(_Strict):
    config_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    author: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)
    description: str = ""
    value_function: ValueWeights
    diversity: DiversitySettings = Field(default_factory=DiversitySettings)
    pacing: PacingSettings = Field(default_factory=PacingSettings)

    @model_validator(mode="after")
    def _weights_not_all_zero(self):
        vf = self.value_function
        if vf.w_listen == 0 and vf.w_like == 0 and vf.w_engagement == 0:
            raise ValueError("at least one positive value-function weight must be non-zero")
        return self


@dataclass(frozen=True)
class ActiveConfig:
    config: RerankerConfig
    value_function: object      # published ValueFunction
    diversity: object           # published DiversityConfig
    pacing: object              # published PacingConfig
    loaded_at: float


class ConfigError(ValueError):
    pass


def build_active(config: RerankerConfig) -> ActiveConfig:
    vf_module = published.chapter8("value_function")
    mmr_module = published.chapter8("mmr_diversity")
    rules = published.chapter8("business_rules")
    weights = config.value_function
    return ActiveConfig(
        config=config,
        value_function=vf_module.ValueFunction(vf_module.ValueFunctionConfig(**weights.model_dump())),
        diversity=mmr_module.DiversityConfig(lambda_param=config.diversity.lambda_param, similarity_metric="cosine",
                                             normalize_relevance=True),
        pacing=rules.PacingConfig(max_per_artist=config.pacing.max_per_artist,
                                  max_per_album=config.pacing.max_per_album,
                                  defer_violations=config.pacing.defer_violations),
        loaded_at=time.time(),
    )


def read_config(directory: Path, version: str) -> RerankerConfig:
    path = Path(directory) / f"{version}.json"
    if not path.is_file():
        raise ConfigError(f"config file {path.name} not found")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        config = RerankerConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"{path.name} failed validation: {problems}") from exc
    if config.config_version != version:
        raise ConfigError(f"{path.name} declares config_version {config.config_version}; it must equal the file name")
    return config


def read_pointer(directory: Path) -> str:
    pointer = Path(directory) / "ACTIVE"
    if not pointer.is_file():
        raise ConfigError(f"no ACTIVE pointer in {directory}")
    return pointer.read_text(encoding="utf-8").strip()


class ConfigStore:
    def __init__(self, directory: Path, poll_seconds: Optional[float] = None):
        self.directory = Path(directory)
        self.poll_seconds = float(os.environ.get("RERANKER_CONFIG_POLL_S", "5")) if poll_seconds is None else poll_seconds
        self._lock = threading.Lock()
        self._active: Optional[ActiveConfig] = None
        self.last_error: Optional[str] = None
        self.last_checked: Optional[float] = None
        self.reloads = 0
        self._pinned: Dict[str, ActiveConfig] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def active(self) -> ActiveConfig:
        return self._active

    def resolve(self, policy_version: Optional[str]) -> Tuple[ActiveConfig, str]:
        """
        The configuration a request should use, and the pin status for the
        response: "none" without a pin, "honoured" when the pinned version
        loaded and validated, "unavailable" when it did not and the active
        version serves instead. Versions are immutable files, so a validated
        pin is cached for the life of the process.
        """
        active = self._active
        if policy_version is None:
            return active, "none"
        if policy_version == active.config.config_version:
            return active, "honoured"
        with self._lock:
            pinned = self._pinned.get(policy_version)
            if pinned is None:
                try:
                    pinned = build_active(read_config(self.directory, policy_version))
                except (ConfigError, ValueError) as exc:
                    logger.warning("Pinned policy %s unavailable, serving %s: %s", policy_version,
                                   active.config.config_version, exc)
                    return active, "unavailable"
                if len(self._pinned) >= MAX_PINNED:
                    self._pinned.pop(next(iter(self._pinned)))
                self._pinned[policy_version] = pinned
        return pinned, "honoured"

    def load_initial(self) -> None:
        """At startup an invalid config is fatal: there is no last good config to fall back to."""
        version = read_pointer(self.directory)
        self._active = build_active(read_config(self.directory, version))
        self.last_checked = time.time()

    def reload(self) -> dict:
        """Apply the version named by ACTIVE if it differs and validates; otherwise keep the current one."""
        with self._lock:
            self.last_checked = time.time()
            try:
                version = read_pointer(self.directory)
                if self._active is not None and version == self._active.config.config_version:
                    self.last_error = None
                    return self.status()
                candidate = build_active(read_config(self.directory, version))
            except (ConfigError, ValueError) as exc:
                self.last_error = str(exc)
                logger.warning("Config reload rejected, keeping %s: %s",
                               self._active.config.config_version if self._active else None, exc)
                return self.status()
            previous = self._active.config.config_version if self._active else None
            self._active = candidate
            self.last_error = None
            self.reloads += 1
            logger.info("Value-function config switched from %s to %s", previous, version)
            return self.status()

    def status(self) -> dict:
        active = self._active
        return {
            "config_version": active.config.config_version if active else None,
            "loaded_at": active.loaded_at if active else None,
            "last_checked": self.last_checked,
            "last_error": self.last_error,
            "reloads": self.reloads,
            "poll_seconds": self.poll_seconds,
        }

    def start_polling(self) -> None:
        if self.poll_seconds <= 0 or self._thread is not None:
            return

        def run():
            while not self._stop.wait(self.poll_seconds):
                self.reload()

        self._thread = threading.Thread(target=run, name="config-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
