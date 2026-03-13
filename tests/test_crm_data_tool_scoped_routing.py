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


ACCOUNT_ID = "6c3284e1-176c-ed80-77f9-652d0ad83b5c"
CONTACT_ID = "1f39facd-bc89-da5c-1f89-67d2bf0ac5dd"


class DummyCrmClient:
    def __init__(self) -> None:
        self.list_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.generic_calls: list[tuple[str, str]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.list_calls.append((module, action, dict(data)))
        if module == "Accounts" and action == "list":
            return {
                "records": [
                    {
                        "id": ACCOUNT_ID,
                        "name": "ZLINER s.r.o.",
                    }
                ]
            }
        if module == "Contacts" and action == "list":
            filter_data = data.get("filter") if isinstance(data, dict) else None
            if isinstance(filter_data, dict):
                return {
                    "records": [
                        {
                            "id": CONTACT_ID,
                            "name": "Karel Vybíhal",
                            "account_id": ACCOUNT_ID,
                            "account_name": "ZLINER s.r.o.",
                            "email1": "karel@zliner.cz",
                        }
                    ]
                }
            return {"records": []}
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        self.generic_calls.append((scope, str(query)))
        if scope == "all":
            return {
                "calls": {"records": [{"id": "e449866f-27d4-ef1b-67a3-641c699866e5", "name": "Volal sám"}]},
                "accounts": {"records": [{"id": ACCOUNT_ID, "name": "ZLINER s.r.o."}]},
            }
        return {"records": []}


def test_crm_data_tool_scoped_search_stays_module_pure_for_accounts() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="naplánuj schůzku s Karlem vybíhalem z firmy zliner na středu ráno",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": {"company_name": "zliner"},
                "query_type": "search",
                "scope": "accounts",
                "limit": 1,
            }
        )
    )
    result = json.loads(raw)

    assert result.get("status") == "ok"
    assert result.get("module") == "Accounts"
    assert result.get("query_type") == "scoped_search_accounts"
    assert result.get("total_count") == 1
    assert not any(scope == "all" for scope, _ in client.generic_calls)


def test_crm_data_tool_routes_company_contacts_query_deterministically() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="jaké máme kontakty k firmě zliner?",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": "zliner",
                "query_type": "auto",
                "limit": 10,
            }
        )
    )
    result = json.loads(raw)

    assert result.get("status") == "ok"
    assert result.get("query_type") == "contacts_by_company"
    assert result.get("module") == "Contacts"
    assert result.get("total_count") == 1
    cards = result.get("cards") or []
    assert cards and cards[0].get("type") == "table"
