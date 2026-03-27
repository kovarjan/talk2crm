from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import _extract_account_ids_from_query


ACCOUNT_ID = "6c3284e1-176c-ed80-77f9-652d0ad83b5c"
CONTACT_ID = "1f39facd-bc89-da5c-1f89-67d2bf0ac5dd"


def test_extract_account_ids_from_query_payload() -> None:
    payload = {"Accounts.id": ACCOUNT_ID}
    ids = _extract_account_ids_from_query(payload, context=None)
    assert ids == [ACCOUNT_ID]


def test_extract_account_ids_from_query_expression_string() -> None:
    payload = f"AccountId == '{ACCOUNT_ID}'"
    ids = _extract_account_ids_from_query(payload, context=None)
    assert ids == [ACCOUNT_ID]
