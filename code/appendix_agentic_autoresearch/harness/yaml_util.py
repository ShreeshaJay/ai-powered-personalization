"""YAML loader for the two config files in this folder.

PyYAML is used when it is installed. The fallback accepts the indentation,
lists, scalars, and comments used by contract.yaml and search_space.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def load_yaml(source: str | Path) -> Any:
    """Load YAML text or a UTF-8 file."""

    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        text = source
    try:
        import yaml
    except ImportError:
        return _parse_simple(text)
    loaded = yaml.safe_load(text)
    return loaded


def _strip_comment(line: str) -> str:
    quote = None
    for index, char in enumerate(line):
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _scalar(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _prepare(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if "\t" in raw:
            raise ValueError("YAML tabs are not accepted; use spaces")
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    return lines


def _parse_simple(text: str) -> Any:
    lines = _prepare(text)
    if not lines:
        return None
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(f"YAML trailing content: {lines[index][1]}")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    content = lines[index][1]
    if content.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_map(lines, index, indent)


def _parse_map(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict, int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indent before {content}")
        if content.startswith("- "):
            break
        key, separator, rest = content.partition(":")
        if not separator:
            raise ValueError(f"Expected key: value, got {content}")
        key = key.strip()
        rest = rest.strip()
        index += 1
        if rest == "":
            if index < len(lines) and lines[index][0] > line_indent:
                value, index = _parse_block(lines, index, lines[index][0])
            else:
                value = None
        else:
            value = _scalar(rest)
        result[key] = value
    return result, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list, int]:
    items: list[Any] = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent or not content.startswith("- "):
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indent before {content}")
        rest = content[2:].strip()
        index += 1
        if rest == "":
            if index < len(lines) and lines[index][0] > line_indent:
                value, index = _parse_block(lines, index, lines[index][0])
            else:
                value = None
        else:
            value = _scalar(rest)
        items.append(value)
    return items, index
