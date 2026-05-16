from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import build_tools


class FakeCrmClient:
    mode = "coripo_public"

    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.last_call = {"module": module, "action": action, "data": data}
        return {"records": []}


def _make_tools(crm_client=None, crm_mode="on"):
    client = crm_client or FakeCrmClient()
    with patch("app.engine.tools.get_settings") as mock_settings:
        mock_settings.return_value.crm_mode = crm_mode
        mock_settings.return_value.tool_call_logging = False
        tools = build_tools(
            tenant_id="test-tenant",
            user_id="user1",
            input_text="",
            request_context=None,
            crm_client=client,
            rag_service=None,
            action_confirmation=False,
        )
    return tools


def _get_tool(tools, name):
    return next(t for t in tools if getattr(t, "name", "") == name)


def test_build_tools_returns_expected_tools():
    tools = _make_tools()
    names = {getattr(t, "name", "") for t in tools}
    assert names == {
        "crm_action_tool",
        "rag_search_tool",
        "my_meetings_tool",
        "crm_query_tool",
        "get_company_overview",
    }


def test_crm_query_tool_crm_off():
    tools = _make_tools(crm_mode="off")
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Contacts"})))
    assert result["status"] == "crm-disabled"


def test_get_company_overview_crm_off():
    tools = _make_tools(crm_mode="off")
    tool = _get_tool(tools, "get_company_overview")
    result = json.loads(asyncio.run(tool.ainvoke({"account_id": "7d956317-c1ba-0118-1c91-61545f91b878"})))
    assert result["status"] == "crm-disabled"


def test_get_company_overview_invalid_account_id():
    tools = _make_tools()
    tool = _get_tool(tools, "get_company_overview")
    result = json.loads(asyncio.run(tool.ainvoke({"account_id": "not-a-crm-id"})))
    assert result["status"] == "tool_validation_error"
    assert "account_id must be a valid CRM record id" in result["message"]


def test_get_company_overview_calls_company_overview_action():
    class OverviewClient(FakeCrmClient):
        async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
            self.last_call = {"module": module, "action": action, "data": data}
            return {
                "module": "Accounts",
                "record": {"id": data["id"], "name": "Test Account"},
                "activities_summary": {"calls": "1"},
                "activities": {"calls": "1"},
                "related_records": {"Contacts": [{"id": "abc", "name": "Jane Doe"}]},
            }

    client = OverviewClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "get_company_overview")
    account_id = "7d956317-c1ba-0118-1c91-61545f91b878"
    result = json.loads(asyncio.run(tool.ainvoke({"account_id": account_id})))

    assert client.last_call == {
        "module": "Accounts",
        "action": "company_overview",
        "data": {"id": account_id},
    }
    assert result["module"] == "Accounts"
    assert result["record"]["id"] == account_id


def test_crm_query_tool_search_produces_cont_filter():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    asyncio.run(tool.ainvoke({"module": "Contacts", "search": "Novak"}))
    assert client.last_call is not None
    filt = client.last_call["data"]["filter"]
    # search wraps in nested and-group with field="*", type="cont"
    nested = filt["operands"][0]
    assert nested["operands"][0]["field"] == "*"
    assert nested["operands"][0]["type"] == "cont"
    assert nested["operands"][0]["value"] == "Novak"


def test_crm_query_tool_date_range_meetings():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    asyncio.run(tool.ainvoke({"module": "Meetings", "date_from": "2025-03-10", "date_to": "2025-03-14"}))
    filt = client.last_call["data"]["filter"]
    fields = [op["field"] for op in filt["operands"]]
    assert all(f == "date_start" for f in fields)


def test_crm_query_tool_filters_parsed():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    filters_json = json.dumps([{"field": "status", "op": "eq", "value": "Held"}])
    asyncio.run(tool.ainvoke({"module": "Calls", "filters": filters_json}))
    filt = client.last_call["data"]["filter"]
    assert filt["operands"][0]["field"] == "status"
    assert filt["operands"][0]["type"] == "eq"
    assert filt["operands"][0]["value"] == "Held"


def test_crm_query_tool_contacts_account_id_translates_to_relation_filter():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    filters_json = json.dumps([{"field": "account_id", "op": "eq", "value": "a42333d4-c035-2f73-865c-64ee30906163"}])
    asyncio.run(tool.ainvoke({"module": "Contacts", "filters": filters_json}))
    filt = client.last_call["data"]["filter"]
    relation_group = filt["operands"][0]
    relation_operand = relation_group["operands"][0]
    relate_operand = relation_group["operands"][1]

    assert relation_group["operator"] == "or"
    assert relation_operand["field"] == "id"
    assert relation_operand["fieldModule"] == "Contacts"
    assert relation_operand["fieldRel"] == ["accounts"]
    assert relation_operand["type"] == "eq"
    assert relation_operand["value"] == "a42333d4-c035-2f73-865c-64ee30906163"
    assert relate_operand["type"] == "relate"
    assert relate_operand["module"] == "Accounts"
    assert relate_operand["relationship"] == ["accounts"]
    assert relate_operand["filter"]["operands"][0]["field"] == "id"
    assert relate_operand["filter"]["operands"][0]["value"] == "a42333d4-c035-2f73-865c-64ee30906163"


def test_crm_query_tool_meetings_account_id_translates_to_parent_filter():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    filters_json = json.dumps([{"field": "account_id", "op": "eq", "value": "a42333d4-c035-2f73-865c-64ee30906163"}])
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Meetings", "filters": filters_json})))
    assert result["status"] == "ok"
    assert client.last_call is not None
    operand = client.last_call["data"]["filter"]["operands"][0]
    assert operand["field"] == "parent_id"
    assert operand["fieldModule"] == "Meetings"
    assert operand["type"] == "eq"
    assert operand["value"] == "a42333d4-c035-2f73-865c-64ee30906163"
    assert operand["parent_type"] == "Accounts"


def test_crm_query_tool_supports_opportunities_account_filter():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    filters_json = json.dumps([{"field": "account_id", "op": "eq", "value": "1667abd3-3b35-ad3e-bf5e-607fdc94d3da"}])
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Opportunities", "filters": filters_json})))

    assert result["status"] == "ok"
    assert client.last_call is not None
    assert client.last_call["module"] == "Opportunities"
    operand = client.last_call["data"]["filter"]["operands"][0]
    assert operand["field"] == "id"
    assert operand["fieldModule"] == "Opportunities"
    assert operand["fieldRel"] == ["accounts"]
    assert operand["type"] == "eq"
    assert operand["value"] == "1667abd3-3b35-ad3e-bf5e-607fdc94d3da"
    assert operand["relationField"] is None


def test_crm_query_tool_supports_quotes_account_filter():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    filters_json = json.dumps([{"field": "account_id", "op": "eq", "value": "1667abd3-3b35-ad3e-bf5e-607fdc94d3da"}])
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Quotes", "filters": filters_json})))

    assert result["status"] == "ok"
    assert client.last_call is not None
    assert client.last_call["module"] == "Quotes"
    operand = client.last_call["data"]["filter"]["operands"][0]
    assert operand["field"] == "id"
    assert operand["fieldModule"] == "Quotes"
    assert operand["fieldRel"] == ["billing_accounts"]
    assert operand["type"] == "eq"
    assert operand["value"] == "1667abd3-3b35-ad3e-bf5e-607fdc94d3da"
    assert operand["relationField"] is None


def test_crm_query_tool_invalid_filters_json():
    tools = _make_tools()
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Contacts", "filters": "not-valid-json"})))
    assert result["status"] in {"error", "tool_validation_error"}
    assert "Invalid filters JSON" in result["message"]


def test_crm_query_tool_limit_capped_at_500():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    asyncio.run(tool.ainvoke({"module": "Contacts", "limit": 500}))
    assert client.last_call["data"]["limit"] == 500


def test_crm_query_tool_order_by():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    asyncio.run(tool.ainvoke({"module": "Contacts", "order_by": "name:asc"}))
    assert client.last_call["data"]["order"] == [{"field": "name", "sort": "ASC", "module": None}]


def test_crm_query_tool_returns_ok_structure():
    tools = _make_tools()
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Contacts"})))
    assert result["status"] == "ok"
    assert "total" in result
    assert "summary" in result
    assert "cards" in result


def test_my_meetings_tool_builds_login_user_filter_and_returns_table_cards():
    class MeetingsClient(FakeCrmClient):
        async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
            self.last_call = {"module": module, "action": action, "data": data}
            return {
                "records": [
                    {
                        "id": "11111111-2222-3333-4444-555555555555",
                        "name": "Porada",
                        "date_start": "2026-03-13 10:00:00",
                        "status": "Planned",
                        "location": "Brno",
                    }
                ]
            }

    client = MeetingsClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "my_meetings_tool")
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {"date_from": "2026-03-10", "date_to": "2026-03-20", "limit": 50}
            )
        )
    )

    assert client.last_call is not None
    filt = client.last_call["data"]["filter"]
    assert filt["operands"][0]["field"] == "assigned_user_id"
    assert filt["operands"][0]["type"] == "eq"
    assert filt["operands"][0]["value"] == "{%LOGIN_USER%}"
    assert filt["operands"][1]["field"] == "date_start"
    assert filt["operands"][1]["type"] == "moreThanInclude"
    assert filt["operands"][1]["value"] == "2026-03-10"
    assert filt["operands"][2]["field"] == "date_start"
    assert filt["operands"][2]["type"] == "lessThanInclude"
    assert filt["operands"][2]["value"] == "2026-03-20"
    assert result["status"] == "ok"
    assert result["total"] == 1
    assert isinstance(result["cards"], list) and result["cards"]
    assert result["cards"][0]["type"] == "table"


def test_my_meetings_tool_without_date_range_uses_only_login_filter():
    class MeetingsClient(FakeCrmClient):
        async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
            self.last_call = {"module": module, "action": action, "data": data}
            return {"records": []}

    client = MeetingsClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "my_meetings_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"limit": 500})))

    assert client.last_call is not None
    filt = client.last_call["data"]["filter"]
    assert len(filt["operands"]) == 1
    assert filt["operands"][0]["field"] == "assigned_user_id"
    assert filt["operands"][0]["type"] == "eq"
    assert filt["operands"][0]["value"] == "{%LOGIN_USER%}"
    assert client.last_call["data"]["limit"] == 500
    assert result["status"] == "ok"
    assert result["date_from"] is None
    assert result["date_to"] is None


def test_my_meetings_tool_rejects_invalid_timeframe():
    tools = _make_tools()
    tool = _get_tool(tools, "my_meetings_tool")
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {"date_from": "2026-03-20", "date_to": "2026-03-10"}
            )
        )
    )
    assert result["status"] == "error"
    assert "date_from must be <=" in result["message"]
