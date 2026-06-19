from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(value: str) -> str:
    """Lowercase, strip diacritics, collapse non-alphanumeric runs to a single space.

    Used for fuzzy name matching across Czech-language CRM data.
    NFD decomposition separates base letters from combining diacritical marks (category Mn),
    which are then dropped before the final alphanumeric-only collapse.
    """
    lowered = (value or "").strip().lower()
    unaccented = "".join(
        c for c in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()


def safe_text(value: Any) -> str:
    """Return str(value).strip(), or '' for None."""
    if value is None:
        return ""
    return str(value).strip()


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"</?answer>", re.IGNORECASE)
ISO_DATE_RE = re.compile(
    r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?(?!\d)"
)


def strip_think_tags(value: Any) -> str:
    """Remove <think>...</think> reasoning blocks emitted by the LLM."""
    return THINK_TAG_RE.sub("", str(value or "")).strip()


def strip_answer_tags(value: Any) -> str:
    """Remove bare <answer>/</answer> markers from LLM output."""
    return ANSWER_TAG_RE.sub("", str(value or "")).strip()


def format_european_dates(value: Any) -> str:
    """Rewrite ISO dates/datetimes (2026-06-12 14:30) to Czech form (12.6.2026 14:30)."""

    def repl(match: re.Match[str]) -> str:
        year, month, day, hour, minute, second = match.groups()
        formatted = f"{int(day)}.{int(month)}.{year}"
        if hour and minute:
            formatted += f" {hour}:{minute}"
            if second and second != "00":
                formatted += f":{second}"
        return formatted

    return ISO_DATE_RE.sub(repl, str(value or ""))
