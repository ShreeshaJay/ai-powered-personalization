"""
Prompts and output parsers for the LLM sections of Chapter 12.

Prompts are plain strings; ``utils.llm`` wraps them in the model's chat template. Parsers
never trust the model's formatting: they report what was valid so that output validity is
measured instead of assumed.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

LABELS = string.ascii_uppercase
MAX_TITLE_CHARS = 100


def short_title(title: str, limit: int = MAX_TITLE_CHARS) -> str:
    title = " ".join(str(title).split())
    return title if len(title) <= limit else title[: limit - 3].rstrip() + "..."


def _history_block(history_titles: Sequence[str]) -> str:
    return "\n".join(f"{i}. {short_title(t)}" for i, t in enumerate(history_titles, start=1))


# ----------------------------------------------------------------------------
# Section 12.5: listwise zero-shot ranking
# ----------------------------------------------------------------------------

def listwise_ranking_prompt(history_titles: Sequence[str], candidate_titles: Sequence[str]) -> str:
    if len(candidate_titles) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} candidates can be labelled")
    candidates = "\n".join(f"[{LABELS[i]}] {short_title(t)}" for i, t in enumerate(candidate_titles))
    last = LABELS[len(candidate_titles) - 1]
    return (
        "A shopper viewed these products in this order during one session:\n"
        f"{_history_block(history_titles)}\n\n"
        "Rank the candidate products by how likely the shopper is to view each one next.\n"
        f"{candidates}\n\n"
        f"Answer with one line that starts with 'Ranking:' followed by all letters A to {last}, "
        "most likely first, separated by commas. Do not explain."
    )


@dataclass
class ParsedRanking:
    order: List[int]                    # candidate indices, complete permutation after filling
    valid_labels: int                   # distinct valid labels the model produced
    duplicate_labels: int
    invalid_labels: int                 # letters outside the slate
    found_ranking_line: bool
    filled: List[int] = field(default_factory=list)  # indices appended because the model omitted them

    @property
    def parsed(self) -> bool:
        return self.valid_labels > 0

    def complete(self, num_candidates: int) -> bool:
        return self.valid_labels == num_candidates


_LABEL_TOKEN = re.compile(r"(?<![A-Za-z])\[?([A-Za-z])\]?(?![A-Za-z])")


def parse_listwise_ranking(output: str, num_candidates: int, rng: np.random.Generator) -> ParsedRanking:
    """Parse 'Ranking: C, A, F, ...'.

    Only the text after 'Ranking:' is read when that marker exists; otherwise only the first
    non-empty line, so letters inside explanations ("I think A ...") are not mistaken for
    labels. Omitted candidates are appended in a seeded random order, which keeps a
    malformed answer from inheriting the display order.
    """
    text = output.strip()
    marker = re.search(r"ranking\s*:", text, flags=re.IGNORECASE)
    if marker:
        segment = text[marker.end():].split("\n", 1)[0]
    else:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        segment = lines[0] if lines else ""
    order: List[int] = []
    duplicates = invalid = 0
    for token in _LABEL_TOKEN.findall(segment):
        idx = LABELS.find(token.upper())
        if idx < 0 or idx >= num_candidates:
            invalid += 1
        elif idx in order:
            duplicates += 1
        else:
            order.append(idx)
    valid = len(order)
    missing = [i for i in range(num_candidates) if i not in order]
    filled = [missing[i] for i in rng.permutation(len(missing))] if missing else []
    return ParsedRanking(order=order + filled, valid_labels=valid, duplicate_labels=duplicates,
                         invalid_labels=invalid, found_ranking_line=bool(marker), filled=filled)


# ----------------------------------------------------------------------------
# Section 12.5: TALLRec-style Yes/No engagement prompt
# ----------------------------------------------------------------------------

YES, NO = "Yes", "No"


def engagement_prompt(history_titles: Sequence[str], candidate_title: str) -> str:
    return (
        "A shopper viewed these products in this order during one session:\n"
        f"{_history_block(history_titles)}\n\n"
        f"Candidate product: {short_title(candidate_title)}\n"
        "Will the shopper view the candidate product next? Answer Yes or No."
    )


# ----------------------------------------------------------------------------
# Section 12.4: offline knowledge generation (KAR-style)
# ----------------------------------------------------------------------------

def item_knowledge_prompt(title: str, brand: str, description: str) -> str:
    desc = " ".join(str(description).split())[:300]
    return (
        "Describe this product for a shopping recommender in two sentences: its category, "
        "what it is used for, and the kind of shopper who buys it. Use only the details given.\n"
        f"Title: {short_title(title, 160)}\n"
        f"Brand: {brand or 'unknown'}\n"
        f"Description: {desc or 'none'}"
    )


def session_intent_prompt(history_titles: Sequence[str]) -> str:
    return (
        "A shopper viewed these products in this order during one session:\n"
        f"{_history_block(history_titles)}\n\n"
        "In one short sentence, state what the shopper is most likely trying to buy."
    )


def clean_generation(text: str, max_chars: int = 600) -> str:
    text = " ".join(text.strip().split())
    return text[:max_chars]
