from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import _contact_cards


def test_contact_cards_always_return_table_with_links_for_small_result_set() -> None:
    records = [
        {
            "id": "1f39facd-bc89-da5c-1f89-67d2bf0ac5dd",
            "name": "Karel Vodička",
            "account_name": "ZLINER s.r.o.",
            "email1": "karel@example.com",
            "phone_mobile": "+420123456789",
        },
        {
            "id": "bbf4032e-7fee-8b8c-869e-5f1543450ae6",
            "name": "David Kostka",
            "account_name": "ZLINER s.r.o.",
            "email1": "david@example.com",
            "phone_mobile": "+420987654321",
        },
    ]

    cards = _contact_cards(records, total_count=2, title="Kontakty firmy (2)", force_table=True)
    assert isinstance(cards, list)
    assert len(cards) == 1

    table = cards[0]
    assert table.get("type") == "table"
    assert table.get("tag") == "Contacts"

    rows = table.get("rows")
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[0].get("link", {}).get("url") == "/#detail/Contacts/1f39facd-bc89-da5c-1f89-67d2bf0ac5dd"
    assert rows[1].get("link", {}).get("url") == "/#detail/Contacts/bbf4032e-7fee-8b8c-869e-5f1543450ae6"
