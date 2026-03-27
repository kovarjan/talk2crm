from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch

from app.core.config import get_settings
from app.engine.quick_actions import QuickActionResult, try_handle_quick_action
from app.engine.tools import build_tools


class DummyCrmClient:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if module == "Contacts" and action == "list":
            return {
                "records": [
                    {
                        "id": "11111111-2222-3333-4444-555555555555",
                        "first_name": "Karel",
                        "last_name": "Vybíhal",
                        "account_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        "account_name": "ZLINER s.r.o.",
                    }
                ]
            }
        if module == "Accounts" and action == "list":
            return {
                "records": [
                    {
                        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        "name": "ZLINER s.r.o.",
                    }
                ]
            }
        return {"status": "ok"}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}



def _assert_pending_contract(pending: dict[str, Any]) -> None:
    assert pending.get("module")
    assert pending.get("requested_module")
    assert pending.get("effective_module")
    assert pending.get("action")
    assert isinstance(pending.get("data"), dict)
    assert isinstance(pending.get("adjustments"), list)
    assert isinstance(pending.get("ambiguities"), list)
    assert isinstance(pending.get("requires_confirmation"), bool)



def test_pending_action_contract_is_canonical_for_quick_action_and_tool() -> None:
    client = DummyCrmClient()

    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        quick_result = asyncio.run(
            try_handle_quick_action(
                input_text="naplánuj schůzku s Karlem vybíhalem z firmy zliner na středu ráno",
                crm_client=client,  # type: ignore[arg-type]
                user_id="28",
                action_confirmation=False,
            )
        )
    assert isinstance(quick_result, QuickActionResult)
    assert not quick_result.should_fallback
    assert quick_result.data.get("status") == "confirmation_required"
    quick_pending = quick_result.data.get("pending_action") or {}
    _assert_pending_contract(quick_pending)

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
    _assert_pending_contract(pending)
