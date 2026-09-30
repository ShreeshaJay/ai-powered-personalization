"""Budget-capped dual-model adjudication for the three 100-item pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
PILOT_DIR = ROOT / "pilots"
DEFAULT_CACHE = ROOT / "outputs" / "adjudication_cache.sqlite"

TASK_PROMPT_VERSIONS = {
    "compatibility": "reference_rubric_v1",
    "query_segmentation": "reference_rubric_v3",
    "brand_category": "reference_rubric_v1",
}
TASK_RUBRIC_PATHS = {
    "compatibility": ROOT / "rubrics" / "v1.md",
    "query_segmentation": ROOT / "rubrics" / "v3_query_segmentation.md",
    "brand_category": ROOT / "rubrics" / "v1.md",
}

PROVIDERS = {
    "anthropic": {
        "model": "claude-opus-5",
        "input_usd_per_million": 5.0,
        "output_usd_per_million": 25.0,
    },
    "openai": {
        "model": "gpt-5.6-sol",
        "input_usd_per_million": 4.0,
        "output_usd_per_million": 20.0,
    },
}

PILOT_TASK_FILES = {
    "compatibility": PILOT_DIR / "compatibility_pilot_100.jsonl",
    "query_segmentation": PILOT_DIR / "query_segmentation_pilot_100.jsonl",
    "brand_category": PILOT_DIR / "brand_category_pilot_100.jsonl",
}
REFERENCE_TASK_FILES = {
    "compatibility": ROOT / "references" / "compatibility_reference_2000.jsonl",
    "query_segmentation": (
        ROOT / "references" / "query_segmentation_reference_2000.jsonl"
    ),
    "brand_category": ROOT / "references" / "brand_category_reference_2000.jsonl",
}
DATASET_TASK_FILES = {
    "pilot": PILOT_TASK_FILES,
    "reference": REFERENCE_TASK_FILES,
}
# Backward-compatible import used by the pilot consensus exporter.
TASK_FILES = PILOT_TASK_FILES

TASK_INSTRUCTIONS = {
    "compatibility": """
Judge whether the candidate product can appropriately be used with the anchor
product. Compatibility is directional. Choose exactly one:
- incompatible: explicit mismatch or no meaningful use together.
- insufficient_evidence: plausible relationship, but metadata cannot establish fit.
- generic_compatible: usable together without model-specific fit, or general interoperability is established.
- explicit_fit: supplied metadata explicitly establishes model/family/specification fit.
Complementarity and category proximity are not proof of compatibility. Prefer
insufficient_evidence over guessing.
""".strip(),
    "query_segmentation": """
Classify each raw commerce query on four independent axes.
goal:
- known_item_navigation for a named model/title/item/person, brand-store
destination, retailer, or website;
- product_discovery for a generic type, attributes, recipient, occasion, or need;
- comparison_decision for comparisons or recommendations;
- informational_support for learning, instructions, or troubleshooting;
- service_account for login, order, payment, delivery, subscription, repair,
or customer-service workflows;
- other_unclear otherwise.
A brand plus a generic product type is product discovery, not automatically
navigation.
object:
- product for a distinct named item/model/title/identifier, and for an
accessory whose fit is constrained by a named model;
- category for a generic product type or merchandise family, with or without
ordinary brand/color/size/material/audience constraints;
- brand_store when a brand, retailer, or storefront is the destination and no
product type is requested;
- service_content for accounts, service workflows, websites/content,
instructions, support, restaurants, or other non-product targets;
- unclear otherwise.
specificity uses the first matching level in this order:
1 exact_model_or_identifier: exact model/generation/part identifier, including
an accessory constrained to a named model;
2 named_entity_or_title: brand/store only, person/author, media title,
proprietary line, or named item;
3 product_type_with_constraints: product type plus brand, size, color,
audience, compatibility target without an exact model, quantity, or feature;
4 product_type_only: generic product type/category with no explicit constraint;
5 broad_need_or_occasion: broad need, activity, recipient, theme, or occasion;
6 none_or_unclear.
Do not infer an exact identifier from an unexplained number.
commerce_scope: in_scope, out_of_scope, or ambiguous.
""".strip(),
    "brand_category": """
Use only the offered candidate IDs. Brand intent is explicit_brand,
inferred_product_line, no_brand, or ambiguous_brand. Select none_or_other when
the query expresses no defensible offered brand. Select the category path that
best captures the requested product type, not an accessory mentioned only as
context. Select other_or_ambiguous when no offered category is defensible or
the query targets multiple unrelated types. Set multi_target true only for
multiple distinct requested targets.
""".strip(),
}

ENUMS = {
    "compatibility_label": [
        "incompatible",
        "insufficient_evidence",
        "generic_compatible",
        "explicit_fit",
    ],
    "goal": [
        "known_item_navigation",
        "product_discovery",
        "comparison_decision",
        "informational_support",
        "service_account",
        "other_unclear",
    ],
    "object": ["product", "category", "brand_store", "service_content", "unclear"],
    "specificity": [
        "exact_model_or_identifier",
        "named_entity_or_title",
        "product_type_with_constraints",
        "product_type_only",
        "broad_need_or_occasion",
        "none_or_unclear",
    ],
    "commerce_scope": ["in_scope", "out_of_scope", "ambiguous"],
    "brand_intent": [
        "explicit_brand",
        "inferred_product_line",
        "no_brand",
        "ambiguous_brand",
    ],
    "confidence": ["low", "medium", "high"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def connect_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS labels (
            cache_key TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            task TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            label_json TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            elapsed_seconds REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_labels_lookup
            ON labels(item_id, task, provider, model);
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            item_count INTEGER NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost_usd REAL NOT NULL,
            elapsed_seconds REAL NOT NULL,
            status TEXT NOT NULL,
            error TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def output_properties(task: str) -> tuple[dict[str, Any], list[str]]:
    common = {
        "item_id": {"type": "string"},
        "confidence": {"type": "string", "enum": ENUMS["confidence"]},
        "rationale": {"type": "string"},
    }
    if task == "compatibility":
        properties = {
            **common,
            "label": {"type": "string", "enum": ENUMS["compatibility_label"]},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
        }
        required = ["item_id", "label", "confidence", "evidence", "rationale"]
    elif task == "query_segmentation":
        properties = {
            **common,
            "goal": {"type": "string", "enum": ENUMS["goal"]},
            "object": {"type": "string", "enum": ENUMS["object"]},
            "specificity": {"type": "string", "enum": ENUMS["specificity"]},
            "commerce_scope": {
                "type": "string",
                "enum": ENUMS["commerce_scope"],
            },
        }
        required = [
            "item_id",
            "goal",
            "object",
            "specificity",
            "commerce_scope",
            "confidence",
            "rationale",
        ]
    elif task == "brand_category":
        properties = {
            **common,
            "brand_intent": {"type": "string", "enum": ENUMS["brand_intent"]},
            "brand_choice": {"type": "string"},
            "category_choice": {"type": "string"},
            "multi_target": {"type": "boolean"},
        }
        required = [
            "item_id",
            "brand_intent",
            "brand_choice",
            "category_choice",
            "multi_target",
            "confidence",
            "rationale",
        ]
    else:
        raise ValueError(f"Unknown task: {task}")
    return properties, required


def response_schema(task: str, batch_size: int) -> dict[str, Any]:
    properties, required = output_properties(task)
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "description": f"Exactly {batch_size} labels, one per supplied item.",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def prompt_hash(task: str) -> str:
    rubric = TASK_RUBRIC_PATHS[task].read_text(encoding="utf-8")
    return sha256(
        json.dumps(
            {
                "version": TASK_PROMPT_VERSIONS[task],
                "task": task,
                "instructions": TASK_INSTRUCTIONS[task],
                "rubric_sha256": sha256(rubric),
            },
            sort_keys=True,
        )
    )


def item_cache_key(
    row: dict[str, Any], task: str, provider: str, model: str, prompt_digest: str
) -> tuple[str, str]:
    serialized = json.dumps(
        row["adjudication_input"], ensure_ascii=False, sort_keys=True
    )
    input_digest = sha256(serialized)
    key = sha256(
        "|".join(
            [
                row["item_id"],
                task,
                provider,
                model,
                prompt_digest,
                input_digest,
            ]
        )
    )
    return key, input_digest


def build_prompt(task: str, rows: list[dict[str, Any]]) -> str:
    payload = [row["adjudication_input"] for row in rows]
    return (
        TASK_INSTRUCTIONS[task]
        + "\n\nJudge every item independently. Return every item_id exactly once. "
        + "Use only the supplied evidence and candidates. Keep each rationale to "
        + "one concise sentence.\n\nItems:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def resolve_candidate_choice(
    value: Any,
    candidates: list[dict[str, str]],
    escape_id: str,
) -> str | None:
    text = str(value).strip()
    candidate_ids = {candidate["id"] for candidate in candidates}
    if text in candidate_ids:
        return text
    normalized = text.casefold()
    value_to_id = {
        str(candidate["value"]).strip().casefold(): candidate["id"]
        for candidate in candidates
    }
    if normalized in value_to_id:
        return value_to_id[normalized]
    escape_aliases = {
        "none",
        "other",
        "none or other",
        "none_or_other",
        "other or ambiguous",
        "other_or_ambiguous",
    }
    if normalized in escape_aliases:
        return escape_id
    return None


def parse_and_validate(
    task: str, text: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    payload = json.loads(text)
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("Response lacks labels array")
    expected = {row["item_id"]: row for row in rows}
    parsed: dict[str, dict[str, Any]] = {}
    properties, required = output_properties(task)
    del properties

    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("A label is not an object")
        missing = set(required) - set(label)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        item_id = str(label["item_id"])
        if item_id not in expected or item_id in parsed:
            raise ValueError(f"Unexpected or duplicate item_id: {item_id}")
        if label["confidence"] not in ENUMS["confidence"]:
            raise ValueError(f"Invalid confidence for {item_id}")

        if task == "compatibility":
            if label["label"] not in ENUMS["compatibility_label"]:
                raise ValueError(f"Invalid compatibility label for {item_id}")
        elif task == "query_segmentation":
            for field in ("goal", "object", "specificity", "commerce_scope"):
                if label[field] not in ENUMS[field]:
                    raise ValueError(f"Invalid {field} for {item_id}")
        else:
            if label["brand_intent"] not in ENUMS["brand_intent"]:
                raise ValueError(f"Invalid brand intent for {item_id}")
            item_input = expected[item_id]["adjudication_input"]
            brand_choice = resolve_candidate_choice(
                label["brand_choice"],
                item_input["brand_candidates"],
                "none_or_other",
            )
            category_choice = resolve_candidate_choice(
                label["category_choice"],
                item_input["category_candidates"],
                "other_or_ambiguous",
            )
            if brand_choice is None:
                raise ValueError(
                    f"Invalid brand candidate for {item_id}: "
                    f"{label['brand_choice']!r}"
                )
            if category_choice is None:
                raise ValueError(
                    f"Invalid category candidate for {item_id}: "
                    f"{label['category_choice']!r}"
                )
            label["brand_choice"] = brand_choice
            label["category_choice"] = category_choice
            if not isinstance(label["multi_target"], bool):
                raise ValueError(f"Invalid multi_target for {item_id}")
        parsed[item_id] = label

    if set(parsed) != set(expected):
        raise ValueError(f"Missing item IDs: {sorted(set(expected) - set(parsed))}")
    return [parsed[row["item_id"]] for row in rows]


def call_anthropic(
    prompt: str, task: str, batch_size: int, max_output_tokens: int
) -> tuple[str, int, int]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=PROVIDERS["anthropic"]["model"],
        max_tokens=max_output_tokens,
        system=(
            "You are an independent benchmark reference-label judge. "
            "Return only the requested structured output."
        ),
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": response_schema(task, batch_size),
            },
        },
        timeout=180.0,
    )
    text_blocks = [
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ]
    if not text_blocks:
        raise ValueError("Anthropic response has no text block")
    return (
        "".join(text_blocks),
        int(response.usage.input_tokens),
        int(response.usage.output_tokens),
    )


def call_openai(
    prompt: str, task: str, batch_size: int, max_output_tokens: int
) -> tuple[str, int, int]:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=PROVIDERS["openai"]["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an independent benchmark reference-label judge. "
                    "Return only the requested structured output."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        reasoning_effort="low",
        verbosity="low",
        max_completion_tokens=max_output_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": f"{task}_reference_labels",
                "strict": True,
                "schema": response_schema(task, batch_size),
            },
        },
        timeout=180.0,
    )
    choice = response.choices[0].message
    if getattr(choice, "refusal", None):
        raise ValueError(f"OpenAI refusal: {choice.refusal}")
    if not choice.content:
        raise ValueError("OpenAI response has no content")
    usage = response.usage
    return choice.content, int(usage.prompt_tokens), int(usage.completion_tokens)


CALLERS: dict[str, Callable[[str, str, int, int], tuple[str, int, int]]] = {
    "anthropic": call_anthropic,
    "openai": call_openai,
}


def call_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    prices = PROVIDERS[provider]
    return (
        input_tokens * float(prices["input_usd_per_million"])
        + output_tokens * float(prices["output_usd_per_million"])
    ) / 1_000_000


def already_cached(
    connection: sqlite3.Connection,
    rows: list[dict[str, Any]],
    task: str,
    provider: str,
    model: str,
    prompt_digest: str,
) -> set[str]:
    keys = [
        item_cache_key(row, task, provider, model, prompt_digest)[0]
        for row in rows
    ]
    if not keys:
        return set()
    placeholders = ",".join("?" for _ in keys)
    result = connection.execute(
        f"SELECT cache_key FROM labels WHERE cache_key IN ({placeholders})", keys
    )
    return {str(row[0]) for row in result}


def spent_cost(connection: sqlite3.Connection) -> float:
    row = connection.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM calls"
    ).fetchone()
    return float(row[0])


def reserve_cost(
    provider: str, prompt: str, max_output_tokens: int
) -> float:
    estimated_input_tokens = math.ceil(len(prompt) / 3.5)
    return call_cost(provider, estimated_input_tokens, max_output_tokens)


def save_success(
    connection: sqlite3.Connection,
    *,
    task: str,
    provider: str,
    model: str,
    rows: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
    raw_response: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    elapsed_seconds: float,
    prompt_digest: str,
) -> None:
    per_item_cost = cost_usd / len(rows)
    per_item_input = input_tokens // len(rows)
    per_item_output = output_tokens // len(rows)
    timestamp = utc_now()
    for row, label in zip(rows, parsed, strict=True):
        cache_key, input_digest = item_cache_key(
            row, task, provider, model, prompt_digest
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO labels (
                cache_key, item_id, task, provider, model, prompt_version,
                prompt_hash, input_hash, label_json, raw_response,
                input_tokens, output_tokens, cost_usd, elapsed_seconds, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                row["item_id"],
                task,
                provider,
                model,
                TASK_PROMPT_VERSIONS[task],
                prompt_digest,
                input_digest,
                json.dumps(label, ensure_ascii=False, sort_keys=True),
                raw_response,
                per_item_input,
                per_item_output,
                per_item_cost,
                elapsed_seconds / len(rows),
                timestamp,
            ),
        )
    connection.execute(
        """
        INSERT INTO calls (
            task, provider, model, item_count, input_tokens, output_tokens,
            cost_usd, elapsed_seconds, status, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ok', '', ?)
        """,
        (
            task,
            provider,
            model,
            len(rows),
            input_tokens,
            output_tokens,
            cost_usd,
            elapsed_seconds,
            timestamp,
        ),
    )
    connection.commit()


def save_failure(
    connection: sqlite3.Connection,
    *,
    task: str,
    provider: str,
    model: str,
    item_count: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    elapsed_seconds: float,
    error: str,
) -> None:
    connection.execute(
        """
        INSERT INTO calls (
            task, provider, model, item_count, input_tokens, output_tokens,
            cost_usd, elapsed_seconds, status, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'error', ?, ?)
        """,
        (
            task,
            provider,
            model,
            item_count,
            input_tokens,
            output_tokens,
            cost_usd,
            elapsed_seconds,
            error[:2000],
            utc_now(),
        ),
    )
    connection.commit()


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    connection = connect_cache(args.cache)
    planned: list[dict[str, Any]] = []
    completed_items = 0
    errors: list[dict[str, str]] = []
    task_files = DATASET_TASK_FILES[args.dataset]

    for task in args.tasks:
        rows = read_jsonl(task_files[task])[: args.max_items_per_task]
        digest = prompt_hash(task)
        for provider in args.providers:
            model = PROVIDERS[provider]["model"]
            cached = already_cached(
                connection, rows, task, provider, model, digest
            )
            missing = [
                row
                for row in rows
                if item_cache_key(row, task, provider, model, digest)[0]
                not in cached
            ]
            for batch in chunks(missing, args.batch_size):
                prompt = build_prompt(task, batch)
                reserve = reserve_cost(provider, prompt, args.max_output_tokens)
                planned.append(
                    {
                        "task": task,
                        "provider": provider,
                        "model": model,
                        "items": len(batch),
                        "reserved_cost_usd": round(reserve, 6),
                    }
                )
                if not args.execute:
                    continue
                if spent_cost(connection) + reserve > args.budget_usd:
                    raise RuntimeError(
                        f"Budget guard: spent plus reserve exceeds ${args.budget_usd:.2f}"
                    )

                started = time.perf_counter()
                input_tokens = 0
                output_tokens = 0
                incurred_cost = 0.0
                try:
                    raw, input_tokens, output_tokens = CALLERS[provider](
                        prompt, task, len(batch), args.max_output_tokens
                    )
                    elapsed = time.perf_counter() - started
                    incurred_cost = call_cost(
                        provider, input_tokens, output_tokens
                    )
                    parsed = parse_and_validate(task, raw, batch)
                    save_success(
                        connection,
                        task=task,
                        provider=provider,
                        model=model,
                        rows=batch,
                        parsed=parsed,
                        raw_response=raw,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=incurred_cost,
                        elapsed_seconds=elapsed,
                        prompt_digest=digest,
                    )
                    completed_items += len(batch)
                    print(
                        f"{provider} {task}: {len(batch)} items, "
                        f"{input_tokens}+{output_tokens} tokens, "
                        f"${incurred_cost:.4f}, "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    save_failure(
                        connection,
                        task=task,
                        provider=provider,
                        model=model,
                        item_count=len(batch),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=incurred_cost,
                        elapsed_seconds=elapsed,
                        error=repr(exc),
                    )
                    errors.append(
                        {
                            "task": task,
                            "provider": provider,
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"ERROR {provider} {task}: {exc!r}",
                        flush=True,
                    )
                    if args.stop_on_error:
                        connection.close()
                        raise

    result = {
        "execute": args.execute,
        "dataset": args.dataset,
        "tasks": args.tasks,
        "providers": args.providers,
        "max_items_per_task": args.max_items_per_task,
        "batch_size": args.batch_size,
        "planned_calls": len(planned),
        "planned_reserved_cost_usd": round(
            sum(item["reserved_cost_usd"] for item in planned), 4
        ),
        "completed_item_labels": completed_items,
        "spent_cost_usd": round(spent_cost(connection), 6),
        "errors": errors,
        "cache": str(args.cache),
    }
    connection.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(DATASET_TASK_FILES), default="pilot"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_FILES),
        default=list(TASK_FILES),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=tuple(PROVIDERS),
        default=list(PROVIDERS),
    )
    parser.add_argument("--max-items-per-task", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--budget-usd", type=float, default=10.0)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_limit = 100 if args.dataset == "pilot" else 2000
    if args.max_items_per_task < 1 or args.max_items_per_task > dataset_limit:
        raise ValueError(
            f"--max-items-per-task must be between 1 and {dataset_limit}"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be positive")
    for provider, env_name in (
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ):
        if args.execute and provider in args.providers and not os.getenv(env_name):
            raise RuntimeError(f"{env_name} is required for {provider}")

    result = run(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
