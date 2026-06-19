# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from difflib import SequenceMatcher

from app.utils.text import normalize_text

# Czech declension suffixes stripped to approximate a word stem. Order does not
# matter — every matching suffix contributes a variant.
_STEM_SUFFIXES: tuple[str, ...] = (
    "skym",
    "skem",
    "ovou",
    "ove",
    "ova",
    "ovi",
    "ych",
    "ich",
    "ho",
    "mu",
    "ou",
    "em",
    "am",
    "um",
    "om",
    "m",
    "a",
    "u",
    "e",
    "y",
    "i",
)


def expand_fuzzy_token_variants(token: str) -> set[str]:
    """Return the token plus stem variants with Czech declension suffixes removed."""
    variants = {token}
    for suffix in _STEM_SUFFIXES:
        if len(token) <= len(suffix) + 2:
            continue
        if token.endswith(suffix):
            variants.add(token[: -len(suffix)])
    return {item for item in variants if item}


def fuzzy_tokens(value: str) -> set[str]:
    """Tokenize normalized text and expand each token with its stem variants."""
    normalized = normalize_text(value)
    if not normalized:
        return set()
    expanded: set[str] = set()
    for token in normalized.split():
        expanded.update(expand_fuzzy_token_variants(token))
    return expanded


def score_lexical_fuzzy(query: str, candidate: str) -> float:
    """Score 0..1 lexical similarity tolerant to Czech inflection and word order."""
    q = normalize_text(query)
    c = normalize_text(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c:
        return 0.96
    if c in q:
        return 0.85

    q_tokens = fuzzy_tokens(q)
    c_tokens = fuzzy_tokens(c)
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    ratio = SequenceMatcher(None, q, c).ratio()
    score = max(overlap * 0.95, ratio * 0.75)
    if overlap >= 0.66 and ratio >= 0.55:
        score = max(score, 0.82)
    return min(1.0, score)
