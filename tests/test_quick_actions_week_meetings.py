from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.quick_actions import try_handle_quick_action


class WeekMeetingsCrmClient:
    mode = "coripo_public"

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((module, action, dict(data)))
        if module == "Meetings" and action == "list":
            return {"records": list(self.records)}
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_week_meetings_uses_crm_date_filter_and_returns_week_rows() -> None:
    now = datetime.now().replace(microsecond=0)
    in_week_dt = now
    out_week_dt = now + timedelta(days=14)
    client = WeekMeetingsCrmClient(
        [
            {
                "id": "in-week",
                "name": "Týdenní porada",
                "date_start": in_week_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Planned",
                "assigned_user_id": "crm-user-guid-not-28",
            },
            {
                "id": "out-week",
                "name": "Budoucí schůzka",
                "date_start": out_week_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Planned",
            },
        ]
    )

    result = asyncio.run(
        try_handle_quick_action(
            input_text="jaké máme schůzky na tento týden?",
            crm_client=client,  # type: ignore[arg-type]
            user_id="28",
            action_confirmation=False,
        )
    )

    assert isinstance(result, dict)
    assert result.get("status") == "ok"
    message = str(result.get("message_to_user") or "")
    assert "Schůzky tento týden" in message
    assert "Týdenní porada" in message
    assert "Budoucí schůzka" not in message

    assert client.calls
    module, action, data = client.calls[0]
    assert module == "Meetings"
    assert action == "list"
    assert isinstance(data.get("filter"), dict)
    top = data["filter"]["operands"][0]
    assert top["operator"] == "and"
    assert len(top["operands"]) == 2
    assert top["operands"][0]["field"] == "date_start"
    assert top["operands"][0]["type"] == "moreThanInclude"
    assert top["operands"][1]["field"] == "date_start"
    assert top["operands"][1]["type"] == "lessThanInclude"

