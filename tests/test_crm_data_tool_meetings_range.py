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
            return {
                "records": [
                    {
                        "id": "c1b620d5-b6e3-4e87-8de8-7e4f43c59f46",
                        "name": "Schůzka - Fakturace projektu",
                        "date_start": "2026-03-08 10:00:00",
                        "location": "Brno",
                        "status": "Plánováno",
                        "assigned_user_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
                    }
                ]
            }
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_crm_data_tool_meetings_explicit_range_overrides_when_user_does_not_ask_next_week() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",  # non-UUID app user id should not filter assigned_user_id
        input_text="ukaž schůzky v období",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": "date >= '2026-03-08' AND date <= '2026-03-15'",
                "query_type": "meetings",
            }
        )
    )
    result = json.loads(raw)

    assert result.get("status") == "ok"
    assert result.get("query_type") == "meetings_range"
    assert result.get("total_count") == 1
    cards = result.get("cards") or []
    assert cards
    first_card = cards[0]
    if first_card.get("type") == "table":
        rows = first_card.get("rows") or []
        assert rows and rows[0].get("cells", {}).get("name") == "Schůzka - Fakturace projektu"
    else:
        assert first_card.get("type") == "record"
        assert first_card.get("title") == "Schůzka - Fakturace projektu"

    meetings_calls = [c for c in client.calls if c.get("module") == "Meetings" and c.get("action") == "list"]
    assert meetings_calls, "Expected Meetings/list call."
    used_filter = meetings_calls[0]["data"].get("filter")
    assert isinstance(used_filter, dict)
    inner = used_filter["operands"][0]["operands"]
    assert inner[0]["field"] == "date_start"
    assert inner[0]["type"] == "moreThanInclude"
    assert inner[0]["value"] == "2026-03-08"
    assert inner[1]["field"] == "date_start"
    assert inner[1]["type"] == "lessThanInclude"
    assert inner[1]["value"] == "2026-03-15"


def test_crm_data_tool_meetings_next_week_intent_wins_over_bad_llm_dates() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="jaké mám schůzky další týden?",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                # Simulate wrong range hallucinated by LLM/tool input.
                "query": "date >= '2026-03-08' AND date <= '2026-03-15'",
                "query_type": "meetings",
            }
        )
    )
    result = json.loads(raw)
    assert result.get("query_type") == "meetings_next_week"
