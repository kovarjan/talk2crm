from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.tools import build_tools


class FakeCrmClient:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def _make_tools():
    with patch("app.engine.tools.get_settings") as mock_settings:
        mock_settings.return_value.crm_mode = "on"
        mock_settings.return_value.tool_call_logging = False
        return build_tools(
            tenant_id="test-tenant",
            user_id="user-1",
            input_text="",
            request_context=None,
            crm_client=FakeCrmClient(),  # type: ignore[arg-type]
            rag_service=None,
            action_confirmation=False,
        )


def _get_tool(tools, name):
    return next(t for t in tools if getattr(t, "name", "") == name)


def test_crm_action_tool_rejects_update_without_record_id() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_action_tool")
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "module": "Meetings",
                    "action": "update",
                    "data_json": json.dumps({"fields": {"name": "X"}}),
                }
            )
        )
    )
    assert result.get("status") == "tool_validation_error"
    assert "requires a valid target record id" in str(result.get("message") or "")


def test_crm_action_tool_rejects_create_with_record_id() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_action_tool")
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "module": "Meetings",
                    "action": "create",
                    "data_json": json.dumps({"id": "11111111-2222-3333-4444-555555555555"}),
                }
            )
        )
    )
    assert result.get("status") == "tool_validation_error"
    assert "cannot include target record id" in str(result.get("message") or "")


def test_crm_query_tool_rejects_invalid_module() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(asyncio.run(tool.ainvoke({"module": "InvalidModule"})))
    assert result.get("status") == "tool_validation_error"
    assert "Unsupported module" in str(result.get("message") or "")


def test_crm_query_tool_rejects_in_operator_for_non_id_field() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_query_tool")
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "module": "Accounts",
                    "filters": json.dumps(
                        [
                            {
                                "field": "name",
                                "op": "in",
                                "value": ["365.bank"],
                            }
                        ]
                    ),
                }
            )
        )
    )
    assert result.get("status") == "tool_validation_error"
    assert "supported only for field 'id'" in str(result.get("message") or "")


def test_crm_action_tool_schema_rejects_unknown_fields() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_action_tool")
    with pytest.raises(Exception):
        asyncio.run(
            tool.ainvoke(
                {
                    "module": "Meetings",
                    "action": "create",
                    "data_json": "{}",
                    "record": "11111111-2222-3333-4444-555555555555",
                }
            )
        )


def test_crm_action_tool_accepts_record_id_alias_for_update() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_action_tool")
    record_id = "11111111-2222-3333-4444-555555555555"
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "module": "Contacts",
                    "action": "update",
                    "data_json": json.dumps({"fields": {"title": "Projektový manažer", "phone_work": ""}}),
                    "record_id": record_id,
                }
            )
        )
    )
    assert result.get("status") == "confirmation_required"
    pending = result.get("pending_action") or {}
    data = pending.get("data") or {}
    assert data.get("id") == record_id


def test_crm_action_tool_normalizes_scoped_record_id_alias_for_update() -> None:
    tools = _make_tools()
    tool = _get_tool(tools, "crm_action_tool")
    record_id = "11111111-2222-3333-4444-555555555555"
    result = json.loads(
        asyncio.run(
            tool.ainvoke(
                {
                    "module": "Contacts",
                    "action": "update",
                    "data_json": json.dumps({"fields": {"title": "Projektový manažer"}}),
                    "record_id": f"Contacts-{record_id}",
                }
            )
        )
    )
    assert result.get("status") == "confirmation_required"
    pending = result.get("pending_action") or {}
    data = pending.get("data") or {}
    assert data.get("id") == record_id
