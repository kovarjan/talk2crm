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


def test_build_tools_returns_three_tools():
    tools = _make_tools()
    names = {getattr(t, "name", "") for t in tools}
    assert names == {"crm_action_tool", "rag_search_tool", "crm_query_tool"}


def test_crm_query_tool_crm_off():
    tools = _make_tools(crm_mode="off")
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Contacts"})))
    assert result["status"] == "crm-disabled"


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


def test_crm_query_tool_invalid_filters_json():
    tools = _make_tools()
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"module": "Contacts", "filters": "not-valid-json"})))
    assert result["status"] == "error"
    assert "Invalid filters JSON" in result["message"]


def test_crm_query_tool_limit_capped_at_100():
    client = FakeCrmClient()
    tools = _make_tools(crm_client=client)
    tool = _get_tool(tools, "crm_query_tool")
    asyncio.run(tool.ainvoke({"module": "Contacts", "limit": 500}))
    assert client.last_call["data"]["limit"] == 100


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
