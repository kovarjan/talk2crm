from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import build_tools


class DummyCrmClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"module": module, "action": action, "data": data})
        if module == "Meetings" and action == "list":
            response_fields = data.get("response_fields") if isinstance(data, dict) else None
            has_required_fields = isinstance(response_fields, list) and "name" in response_fields and "date_start" in response_fields
            if not has_required_fields:
                # Simulate sparse CRM projection (ids only) when fields are not explicitly requested.
                return {
                    "records": [
                        {"id": "00000000-0000-0000-0000-000000000001"},
                        {"id": "00000000-0000-0000-0000-000000000002"},
                    ]
                }
            rows = []
            for i in range(1, 8):
                rows.append(
                    {
                        "id": f"00000000-0000-0000-0000-00000000000{i}",
                        "name": f"Schůzka {i}",
                        "date_start": f"2026-03-0{i} 10:00:00",
                        "location": f"Místnost {i}",
                        "status": "Plánováno",
                    }
                )
            return {"records": rows}
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_crm_search_tool_returns_table_cards_for_many_meetings() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="jaké mám schůzky na další týden",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_search_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_search_tool")

    raw = asyncio.run(
        crm_search_tool.ainvoke(
            {
                "query": {
                    "filter": {"field": "start_date", "operator": "between", "value": ["2026-03-07", "2026-03-14"]},
                    "order": {"field": "start_date", "direction": "asc"},
                    "limit": 10,
                },
                "scope": "meetings",
                "limit": 10,
            }
        )
    )
    result = json.loads(raw)

    assert result.get("status") == "ok"
    assert result.get("module") == "Meetings"
    assert result.get("total_count") == 7
    cards = result.get("cards") or []
    assert isinstance(cards, list) and len(cards) == 1
    assert cards[0].get("type") == "table"
    rows = cards[0].get("rows") or []
    assert len(rows) == 7
    assert rows[0].get("cells", {}).get("name") == "Schůzka 1"
    assert rows[0].get("link", {}).get("url") == "/#detail/Meetings/00000000-0000-0000-0000-000000000001"
