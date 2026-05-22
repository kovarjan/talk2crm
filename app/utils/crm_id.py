from __future__ import annotations

import re

# Matches both Coripo/SugarCRM-style 32-char hex IDs and standard UUID4 with dashes.
CRM_ID_RE = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)


def is_valid_crm_id(value: str | None) -> bool:
    text = str(value or "").strip()
    return bool(text and CRM_ID_RE.match(text))
