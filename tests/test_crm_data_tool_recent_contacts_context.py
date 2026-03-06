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
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        self.calls.append({"module": "generic", "action": scope, "data": {"query": query}})
        return {"records": []}


def test_crm_data_tool_prefers_recent_contacts_context_for_name_resolution() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="Naplánuj call s Igorem na pátek ráno.",
        request_context={
            "recent_contacts": [
                {
                    "id": "930c744a-2bc6-d135-e826-5faa689b42e1",
                    "name": "Vladislav Posekaný",
                    "account_name": "Xella CZ, s.r.o.",
                },
                {
                    "id": "13e4f0e2-f4f9-24fa-6b10-5f929ebf2f9d",
                    "name": "Igor Forberger",
                    "account_name": "Xella CZ, s.r.o.",
                },
            ]
        },
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": "Igor",
                "query_type": "contact",
                "limit": 1,
            }
        )
    )
    result = json.loads(raw)

    assert result.get("status") == "ok"
    assert result.get("source") == "recent_contacts_context"
    assert result.get("total_count") == 1

    cards = result.get("cards") or []
    assert cards
    assert cards[0].get("title") == "Igor Forberger"
    assert client.calls == []
