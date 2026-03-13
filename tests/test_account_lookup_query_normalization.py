from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import _normalize_account_lookup_query, _score_lexical_fuzzy


def test_normalize_account_lookup_query_strips_leading_company_word() -> None:
    assert _normalize_account_lookup_query("firma Čemat") == "Čemat"
    assert _normalize_account_lookup_query("z firmy ČEMAT trading, spol. s r.o.") == "ČEMAT trading, spol. s r.o."


def test_normalize_account_lookup_query_keeps_company_word_inside_name() -> None:
    assert _normalize_account_lookup_query("Fajn Firma, s.r.o.") == "Fajn Firma, s.r.o."


def test_normalized_account_lookup_query_improves_target_match_score() -> None:
    candidate = "ČEMAT trading, spol. s r.o."
    raw_score = _score_lexical_fuzzy("firma Čemat", candidate)
    normalized_score = _score_lexical_fuzzy(_normalize_account_lookup_query("firma Čemat"), candidate)

    assert normalized_score > raw_score
    assert normalized_score >= 0.9
