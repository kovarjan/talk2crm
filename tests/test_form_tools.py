from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine import form_tools
from app.engine.form_tools import build_form_tools
from app.engine.tools import build_tools

SCHEMA = {"module": "Contacts", "sections": [{"group": "g", "fields": [
    {"name": "first_name", "type": "text"},
    {"name": "account_id", "type": "relate", "id_field": "account_id", "target_module": "Accounts"},
]}]}


class StubCrm:
    mode = "coripo_public"

    def __init__(self, schema: dict[str, Any] | Exception = SCHEMA) -> None:
        self.schema = schema
        self.calls: list[tuple[str, str | None]] = []

    async def get_ai_schema(self, module: str, record_id: str | None = None) -> dict[str, Any]:
        self.calls.append((module, record_id))
        if isinstance(self.schema, Exception):
            raise self.schema
        return self.schema

    async def execute_module_action(self, module: str, action: str, data: dict) -> dict:
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict:
        return {"records": []}


def _tool(tools: list, name: str):
    return next(t for t in tools if getattr(t, "name", "") == name)


def test_propose_form_fields_returns_form_patch_with_schema_and_current_values() -> None:
    crm = StubCrm()
    fake = AsyncMock(return_value={"message": "Doplněno.", "fields": {"first_name": "Jan"}})
    with patch.object(form_tools, "extract_fields", new=fake):
        tools = build_form_tools(
            tenant_id="t", user_id="u",
            request_context={"form": {"editable": True, "values": {"first_name": "", "last_name": "Novák"}}},
            crm_client=crm, rag_service=None,
        )
        raw = asyncio.run(_tool(tools, "propose_form_fields_tool").ainvoke(
            {"module": "Contacts", "record_id": None, "instructions": "Jan Novák, Brno"}
        ))
    payload = json.loads(raw)
    assert payload["status"] == "ok"
    assert payload["form_patch"]["fields"] == {"first_name": "Jan"}
    assert payload["form_patch"]["schema"] == {"sections": SCHEMA["sections"]}
    assert payload["form_patch"]["module"] == "Contacts"
    assert payload["message_to_user"] == "Doplněno."
    assert crm.calls == [("Contacts", None)]
    kwargs = fake.call_args.kwargs
    assert kwargs["field_schema"] == SCHEMA
    assert kwargs["current_values"] == {"first_name": "", "last_name": "Novák"}
    assert kwargs["messages"][-1] == {"role": "user", "content": "Jan Novák, Brno"}


def test_research_record_passes_subject_and_returns_sources() -> None:
    crm = StubCrm()
    fake = AsyncMock(return_value={"message": "Nalezeno.", "fields": {"first_name": "X"}, "sources": ["https://a.cz"]})
    with patch.object(form_tools, "research_company", new=fake):
        tools = build_form_tools(tenant_id="t", user_id="u", request_context={}, crm_client=crm, rag_service=None)
        raw = asyncio.run(_tool(tools, "research_record_tool").ainvoke(
            {"module": "ProductTemplates", "record_id": "p1", "subject": "Vrtačka XY", "hint": "Bosch"}
        ))
    payload = json.loads(raw)
    assert payload["status"] == "ok"
    assert payload["form_patch"]["sources"] == ["https://a.cz"]
    assert payload["form_patch"]["record"] == "p1"
    kwargs = fake.call_args.kwargs
    assert kwargs["subject"] == "Vrtačka XY"
    assert kwargs["subject_label"] == "Produkt"
    assert kwargs["hint"] == "Bosch"


def test_schema_failure_is_reported_not_raised() -> None:
    crm = StubCrm(RuntimeError("403"))
    tools = build_form_tools(tenant_id="t", user_id="u", request_context={}, crm_client=crm, rag_service=None)
    raw = asyncio.run(_tool(tools, "propose_form_fields_tool").ainvoke({"module": "Contacts", "record_id": None, "instructions": "x"}))
    payload = json.loads(raw)
    assert payload["status"] == "form_unavailable"
    assert "form_patch" not in payload


def test_build_tools_adds_form_tools_only_with_form_capability() -> None:
    crm = StubCrm()
    with_form = {getattr(t, "name", "") for t in build_tools(
        tenant_id="t", user_id="u", input_text="x", request_context=None,
        crm_client=crm, rag_service=None, capabilities={"crm", "form"},
    )}
    without = {getattr(t, "name", "") for t in build_tools(
        tenant_id="t", user_id="u", input_text="x", request_context=None,
        crm_client=crm, rag_service=None, capabilities={"crm"},
    )}
    assert {"propose_form_fields_tool", "research_record_tool"} <= with_form
    assert not ({"propose_form_fields_tool", "research_record_tool"} & without)


def test_propose_form_fields_passes_lines_into_patch() -> None:
    schema = {"module": "Quotes", "sections": [{"group": "g", "fields": [
        {"name": "name", "type": "text"},
        {"name": "lines", "type": "line_items", "line_module": "Products", "line_fields": [{"name": "name", "type": "text", "editable": True}]},
    ]}]}
    crm = StubCrm(schema)
    fake = AsyncMock(return_value={"message": "ok", "fields": {"name": "N"}, "lines": {"rows": [{"name": "A"}], "dropped": []}})
    with patch.object(form_tools, "extract_fields", new=fake):
        tools = build_form_tools(tenant_id="t", user_id="u", request_context={}, crm_client=crm, rag_service=None)
        raw = asyncio.run(_tool(tools, "propose_form_fields_tool").ainvoke({"module": "Quotes", "record_id": "q1", "instructions": "x"}))
    patch_out = json.loads(raw)["form_patch"]
    assert patch_out["lines"]["rows"] == [{"name": "A"}]
    assert patch_out["lines"]["line_module"] == "Products"


def test_action_on_open_record_is_redirected_to_form_patch() -> None:
    schema = {"module": "ProductTemplates", "sections": [{"group": "g", "fields": [{"name": "description", "type": "textarea"}]}]}
    crm = StubCrm(schema)
    ctx = {"module": "ProductTemplates", "record": "p1", "form": {"editable": True, "values": {}}, "capabilities": ["form"]}
    tools = build_tools(tenant_id="t", user_id="u", input_text="vlož do description", request_context=ctx,
                        crm_client=crm, rag_service=None, capabilities={"crm", "form"})
    action = _tool(tools, "crm_action_tool")
    raw = asyncio.run(action.ainvoke({"module": "ProductTemplates", "action": "update", "record_id": "p1",
                                      "data_json": json.dumps({"fields": {"description": "Pneumatika", "bogus": 1}})}))
    out = json.loads(raw)
    assert out["status"] == "ok" and out["redirected_to_form"] is True
    assert out["form_patch"]["fields"] == {"description": "Pneumatika"}
    assert out["form_patch"]["record"] == "p1"
    assert "bogus" in out["message_to_user"]


def test_action_on_another_record_is_not_redirected() -> None:
    crm = StubCrm()
    ctx = {"module": "ProductTemplates", "record": "p1", "form": {"editable": True, "values": {}}}
    tools = build_tools(tenant_id="t", user_id="u", input_text="x", request_context=ctx,
                        crm_client=crm, rag_service=None, capabilities={"crm", "form"})
    action = _tool(tools, "crm_action_tool")
    raw = asyncio.run(action.ainvoke({"module": "ProductTemplates", "action": "update", "record_id": "other",
                                      "data_json": json.dumps({"fields": {"description": "x"}})}))
    assert json.loads(raw).get("redirected_to_form") is None
