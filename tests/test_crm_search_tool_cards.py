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
    mode = "coripo_public"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"module": module, "action": action, "data": data})
        if module == "Meetings" and action == "list":
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


def test_crm_query_tool_returns_cards_and_summary_for_meetings() -> None:
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
    crm_query_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_query_tool")

    raw = asyncio.run(
        crm_query_tool.ainvoke(
            {
                "module": "Meetings",
                "date_from": "2026-03-07",
                "date_to": "2026-03-14",
                "order_by": "date_start:asc",
                "limit": 10,
            }
        )
    )
    result = json.loads(raw)

    assert result.get("status") == "ok"
    assert "cards" in result
    assert "summary" in result
    assert result.get("total") == 7
    cards = result.get("cards") or []
    assert isinstance(cards, list) and len(cards) >= 1
