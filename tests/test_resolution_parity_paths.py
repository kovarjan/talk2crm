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
from app.engine.adjustments import ModuleAdjustmentEngine
from app.engine.quick_actions import QuickActionResult, try_handle_quick_action
from app.engine.tools import build_tools


class DummyCrmClient:
    mode = "coripo_public"

    contact_id = "1f39facd-bc89-da5c-1f89-67d2bf0ac5dd"
    account_id = "6c3284e1-176c-ed80-77f9-652d0ad83b5c"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if module == "Contacts" and action == "list":
            return {
                "records": [
                    {
                        "id": self.contact_id,
                        "first_name": "Karel",
                        "last_name": "Vybíhal",
                        "account_id": self.account_id,
                        "account_name": "ZLINER s.r.o.",
                        "email1": "karel@example.com",
                    }
                ]
            }
        if module == "Accounts" and action == "list":
            return {
                "records": [
                    {
                        "id": self.account_id,
                        "name": "ZLINER s.r.o.",
                    }
                ]
            }
        if module == "Meetings" and action == "create":
            return {"status": "ok", "id": "d9207f5e-0abc-4b66-96d3-bd4e6ec97200"}
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        if scope == "contacts":
            return {
                "records": [
                    {
                        "id": self.contact_id,
                        "first_name": "Karel",
                        "last_name": "Vybíhal",
                        "account_id": self.account_id,
                        "account_name": "ZLINER s.r.o.",
                    }
                ]
            }
        if scope == "accounts":
            return {
                "records": [
                    {
                        "id": self.account_id,
                        "name": "ZLINER s.r.o.",
                    }
                ]
            }
        return {"records": []}



def _outcome_from_quick(result: QuickActionResult) -> str:
    pending = result.data.get("pending_action") if isinstance(result.data.get("pending_action"), dict) else {}
    data = pending.get("data") if isinstance(pending.get("data"), dict) else {}
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    return "resolved" if str(fields.get("parent_type") or "") == "Contacts" else "unresolved"



def _outcome_from_adjustment(result: dict[str, Any]) -> str:
    fields = result.get("fields") if isinstance(result.get("fields"), dict) else {}
    return "resolved" if str(fields.get("parent_type") or "") == "Contacts" else "unresolved"



def _outcome_from_data_tool(result: dict[str, Any]) -> str:
    return "resolved" if int(result.get("total") or 0) > 0 else "unresolved"



def test_resolution_outcome_parity_across_quick_adjustment_and_read_tool() -> None:
    text = "naplánuj schůzku s Karlem vybíhalem z firmy zliner na středu ráno"
    client = DummyCrmClient()

    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        quick_result = asyncio.run(
            try_handle_quick_action(
                input_text=text,
                crm_client=client,  # type: ignore[arg-type]
                user_id="1",
                action_confirmation=False,
            )
        )
    assert isinstance(quick_result, QuickActionResult)
    assert not quick_result.should_fallback

    adjustment_engine = ModuleAdjustmentEngine(
        tenant_id="ai-local",
        user_id="1",
        crm_client=client,  # type: ignore[arg-type]
        input_text=text,
        request_context={},
    )
    adjusted = asyncio.run(
        adjustment_engine.apply(
            module="Meetings",
            action="create",
            data={"fields": {"contact_name": "Karlem Vybíhalem", "account_name": "zliner"}},
        )
    )

    tools = build_tools(
        tenant_id="ai-local",
        user_id="1",
        input_text=text,
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_query_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_query_tool")
    raw = asyncio.run(
        crm_query_tool.ainvoke(
            {
                "module": "Contacts",
                "search": "Karlem Vybíhalem",
                "limit": 1,
            }
        )
    )
    data_result = json.loads(raw)

    outcomes = {
        _outcome_from_quick(quick_result),
        _outcome_from_adjustment(adjusted.data),
        _outcome_from_data_tool(data_result),
    }
    assert outcomes == {"resolved"}
