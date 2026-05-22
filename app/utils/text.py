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
