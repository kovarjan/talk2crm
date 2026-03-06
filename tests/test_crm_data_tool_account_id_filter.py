from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import _extract_account_ids_from_query, build_tools


ACCOUNT_ID = "6c3284e1-176c-ed80-77f9-652d0ad83b5c"
CONTACT_ID = "1f39facd-bc89-da5c-1f89-67d2bf0ac5dd"


class DummyCrmClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"module": module, "action": action, "data": data})
        if module == "Accounts" and action == "list":
            return {"records": [{"id": ACCOUNT_ID, "name": "ZLINER s.r.o."}]}
        if module == "Contacts" and action == "list" and isinstance(data, dict) and isinstance(data.get("filter"), dict):
            return {
                "records": [
                    {
                        "id": CONTACT_ID,
                        "name": "Karel Vodička",
                        "account_name": "ZLINER s.r.o.",
                        "email1": "karel@zliner.cz",
                        "phone_mobile": "+420123456789",
                    }
                ]
            }
        if module == "Contacts" and action == "list":
            return {"records": []}
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_extract_account_ids_from_query_payload() -> None:
    payload = {"Accounts.id": ACCOUNT_ID}
    ids = _extract_account_ids_from_query(payload, context=None)
    assert ids == [ACCOUNT_ID]


def test_extract_account_ids_from_query_expression_string() -> None:
    payload = f"AccountId == '{ACCOUNT_ID}'"
    ids = _extract_account_ids_from_query(payload, context=None)
    assert ids == [ACCOUNT_ID]


def test_crm_data_tool_uses_relate_filter_with_direct_account_id() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="Jaké máme kontakty k firmě Zliner",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": {"Accounts.id": ACCOUNT_ID},
                "query_type": "contacts",
                "limit": 100,
            }
        )
    )
    result = json.loads(raw)

    assert result.get("query_type") == "contacts_by_company"
    assert result.get("resolved_account_ids") == [ACCOUNT_ID]
    assert result.get("total_count") == 1

    cards = result.get("cards") or []
    assert isinstance(cards, list) and cards
    assert cards[0].get("type") == "table"
    rows = cards[0].get("rows") or []
    assert rows and rows[0].get("link", {}).get("url") == f"/#detail/Contacts/{CONTACT_ID}"

    contacts_relation_calls = [
        call
        for call in client.calls
        if call.get("module") == "Contacts" and call.get("action") == "list" and isinstance(call.get("data", {}).get("filter"), dict)
    ]
    assert contacts_relation_calls, "Expected Contacts/list call with relation filter."
    relation_filter = contacts_relation_calls[0]["data"]["filter"]

    first_operand = relation_filter["operands"][0]["operands"][0]
    assert first_operand["module"] == "Accounts"
    assert first_operand["type"] == "relate"
    assert first_operand["name"] == "account_name"
    assert first_operand["filter"]["operator"] == "and"
    nested = first_operand["filter"]["operands"][0]
    assert nested["field"] == "id"
    assert nested["type"] == "eq"
    assert nested["value"] == ACCOUNT_ID


def test_crm_data_tool_uses_relate_filter_with_account_id_expression_string() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="jaké máme kontakty k firmě Zliner",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": f"AccountId == '{ACCOUNT_ID}'",
                "query_type": "contacts",
                "limit": 100,
            }
        )
    )
    result = json.loads(raw)
    assert result.get("resolved_account_ids") == [ACCOUNT_ID]
    assert result.get("total_count") == 1


def test_crm_data_tool_reads_account_id_from_scope_expression() -> None:
    client = DummyCrmClient()
    tools = build_tools(
        tenant_id="ai-local",
        user_id="28",
        input_text="jaké máme kontakty k firmě Zliner",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    crm_data_tool = next(tool for tool in tools if getattr(tool, "name", "") == "crm_data_tool")

    raw = asyncio.run(
        crm_data_tool.ainvoke(
            {
                "query": "Zliner",
                "query_type": "contacts",
                "scope": f"Accounts.id={ACCOUNT_ID}",
                "limit": 100,
            }
        )
    )
    result = json.loads(raw)
    assert result.get("query_type") == "contacts_by_company"
    assert result.get("resolved_account_ids") == [ACCOUNT_ID]
    assert result.get("total_count") == 1
