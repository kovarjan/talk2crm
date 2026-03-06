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

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_crm_action_tool_overrides_meeting_module_to_calls_for_call_intent() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="Naplánuj call s Igorem na pátek ráno.",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_action_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_action_tool")

    raw = asyncio.run(
        crm_action_tool.ainvoke(
            {
                "module": "Meetings",
                "action": "create",
                "data_json": "{}",
            }
        )
    )
    result = json.loads(raw)
    pending = result.get("pending_action") or {}

    assert result.get("status") == "confirmation_required"
    assert pending.get("module") == "Calls"
    assert pending.get("effective_module") == "Calls"
    assert pending.get("requested_module") == "Meetings"


def test_crm_action_tool_call_subject_is_used_for_name() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="Naplánuj call s Igorem na pátek ráno.",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_action_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_action_tool")

    raw = asyncio.run(
        crm_action_tool.ainvoke(
            {
                "module": "Meetings",
                "action": "create",
                "data_json": json.dumps({"subject": "Call with Igor"}),
            }
        )
    )
    result = json.loads(raw)
    pending = result.get("pending_action") or {}
    data = pending.get("data") or {}
    fields = data.get("fields") or {}

    assert pending.get("module") == "Calls"
    assert fields.get("name") == "Hovor - Call with Igor"
