from __future__ import annotations

import copy
import json

import pytest

from _tool_fakes import FakeNativeTool, FakeTransport, error_result, ok_result, sample_manifest
from app.tools.crm_provider import CrmToolProvider
from app.tools.manifest_cache import ManifestCache
from app.tools.resilience import TenantGuards
from app.tools.toolset import ToolSet

MEETING_ID = "0a1b2c3d-1111-2222-3333-444455556666"


def _custom_tool(name: str, capability: str = "crm") -> dict:
    return {
        "name": name,
        "version": "1.0.0",
        "title": name,
        "description": "vlastní nástroj klienta\ndruhý řádek",
        "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "capability": capability, "timeoutSeconds": 10},
    }


async def _toolset(transport: FakeTransport, *, context: dict | None = None, natives: list | None = None, capabilities=None) -> ToolSet:
    provider = CrmToolProvider(tenant_id="t1", transport=transport, cache=ManifestCache(), guards=TenantGuards())
    native_list = natives if natives is not None else [FakeNativeTool("rag_search_tool")]
    return await ToolSet.build(
        tenant_id="t1",
        user_id="u1",
        request_context=context,
        crm_client=None,
        native_tools=lambda enabled: native_list,
        capabilities=capabilities,
        provider=provider,
    )


@pytest.mark.asyncio
async def test_merges_crm_and_native_tools_in_prompt_order() -> None:
    toolset = await _toolset(FakeTransport())
    names = toolset.names()
    assert names[:2] == ["rag_search_tool", "crm_query_tool"]
    assert {"crm_action_tool", "my_meetings_tool", "get_company_overview", "crm_record_detail_tool"} <= set(names)
    assert toolset.describe()["tools"][1]["source"] == "crm"


@pytest.mark.asyncio
async def test_crm_tool_prompt_is_rendered_from_manifest() -> None:
    blocks = "\n".join((await _toolset(FakeTransport())).prompt_blocks())
    assert "crm_query_tool(module: str, filters: list=[], search: str|null=null" in blocks
    assert "data_json: dict={}" in blocks
    assert '   — přesný dotaz do CRM.' in blocks
    assert 'action: "create"|"update"|"delete"' in blocks


@pytest.mark.asyncio
async def test_manual_is_available_without_web_and_returns_readable_source_passages() -> None:
    passage = {"path": "Komplet-CORIPO-manual.md", "title": "Manuál CORIPO",
               "url": "/#wiki/Komplet-CORIPO-manual.md", "start_line": 700, "end_line": 720,
               "content": "Zvolte Import a nahrajte CSV."}
    transport = FakeTransport(responses=[ok_result({"status": "ok", "locale": "cs_CZ", "results": [passage]})])
    toolset = await _toolset(transport, context={"capabilities": []})
    assert "crm_manual_tool" in toolset.names()
    assert "web_search_tool" not in toolset.names()
    assert toolset.is_read_only("crm_manual_tool")
    observation = json.loads(await toolset.invoke("crm_manual_tool", {"query": "import kontakty"}))
    assert observation["results"] == [passage]
    assert "cards" not in observation
    assert "confirmation" not in observation
    assert "oficiální manuály CORIPO" in "\n".join(toolset.prompt_blocks())


@pytest.mark.asyncio
async def test_manifest_tool_cannot_shadow_a_native_tool() -> None:
    manifest = sample_manifest()
    manifest["tools"].append(_custom_tool("web_search_tool"))
    toolset = await _toolset(FakeTransport(manifest), natives=[])
    assert "web_search_tool" not in toolset.names()
    assert toolset.dropped == ["web_search_tool"]


@pytest.mark.asyncio
async def test_client_capability_tools_only_when_enabled() -> None:
    manifest = sample_manifest()
    manifest["tools"].append(_custom_tool("assa_door_config", capability="doors"))
    manifest["capabilities"].append({"id": "doors", "title": "Dveře", "prompt": "REŽIM DVEŘE", "default_on": False})

    off = await _toolset(FakeTransport(manifest), context={"capabilities": []})
    assert "assa_door_config" not in off.names()
    on = await _toolset(FakeTransport(manifest), context={"capabilities": ["doors"]})
    assert "assa_door_config" in on.names()
    assert on.unknown_capabilities == []
    assert "REŽIM DVEŘE" in on.capability_prompt()


@pytest.mark.asyncio
async def test_read_tool_observation_gets_cards_from_records() -> None:
    records = [{"id": MEETING_ID, "name": "Schůzka", "date_start": "2026-10-01 10:00:00", "status": "Planned"}]
    transport = FakeTransport(responses=[ok_result({"status": "ok", "module": "Meetings", "total": 1, "summary": "• Schůzka", "records": records, "display": "table"})])
    toolset = await _toolset(transport)
    observation = json.loads(await toolset.invoke("my_meetings_tool", {"limit": "5"}))
    assert transport.calls[0]["arguments"] == {"limit": 5}  # coerced
    assert observation["cards"][0]["type"] == "table"
    assert observation["records"][0]["id"] == MEETING_ID


@pytest.mark.asyncio
async def test_invalid_arguments_never_leave_the_gateway() -> None:
    transport = FakeTransport()
    toolset = await _toolset(transport)
    observation = json.loads(await toolset.invoke("crm_query_tool", {"module": "Contacts", "limit": 9999}))
    assert observation["status"] == "tool_validation_error"
    assert observation["field_errors"][0]["path"] == "/limit"
    assert transport.calls == []


@pytest.mark.asyncio
async def test_crm_tool_error_reaches_the_model() -> None:
    transport = FakeTransport(responses=[error_result("invalid_arguments", "CRM odmítlo pole", [{"path": "/data_json/fields/x", "problem": "unknown_field"}])])
    toolset = await _toolset(transport)
    observation = json.loads(await toolset.invoke("crm_action_tool", {"module": "Meetings", "action": "create", "data_json": {"fields": {"x": 1}}}))
    assert observation["status"] == "tool_validation_error"
    assert observation["field_errors"][0]["problem"] == "unknown_field"


def _preview_response(token: str = "tok-1") -> dict:
    confirm_args = {"module": "Meetings", "action": "create", "data_json": {"fields": {"name": "Schůzka Nováková - demo", "date_start": "2026-10-01 10:00:00"}}}
    return ok_result(
        {"status": "confirmation_required", "message": "Akce mění CRM data a vyžaduje potvrzení uživatele.", "module": "Meetings", "action": "create",
         "adjustments": ["Resolved contact"], "missing_required": [{"field": "zapis"}]},
        meta={"version": "1.0.0", "confirmation": {"required": True, "token": token, "expires_at": "2026-10-01T10:15:00Z", "arguments": confirm_args}},
    )


@pytest.mark.asyncio
async def test_write_tool_is_previewed_and_becomes_a_pending_tool_call() -> None:
    transport = FakeTransport(responses=[_preview_response()])
    toolset = await _toolset(transport)
    args = {"module": "Meetings", "action": "create", "data_json": json.dumps({"fields": {"contact_name": "Novákovou"}})}
    observation = json.loads(await toolset.invoke("crm_action_tool", args))

    assert transport.calls[0]["mode"] == "preview"
    assert transport.calls[0]["arguments"]["data_json"] == {"fields": {"contact_name": "Novákovou"}}
    assert observation["status"] == "confirmation_required"
    pending = observation["pending_action"]
    assert pending["tool"] == "crm_action_tool" and pending["confirmation_token"] == "tok-1"
    assert pending["arguments"]["data_json"]["fields"]["name"] == "Schůzka Nováková - demo"
    assert pending["data"]["fields"]["date_start"] == "2026-10-01 10:00:00"  # legacy envelope keys still filled
    assert pending["module"] == "Meetings" and pending["action"] == "create"
    assert observation["missing_required"] == [{"field": "zapis"}]


@pytest.mark.asyncio
async def test_execute_pending_sends_token_and_exact_arguments() -> None:
    transport = FakeTransport(responses=[_preview_response(), ok_result({"status": "ok", "id": MEETING_ID, "module": "Meetings", "action": "create", "message_to_user": "Záznam byl vytvořen."})])
    toolset = await _toolset(transport)
    pending = json.loads(await toolset.invoke("crm_action_tool", {"module": "Meetings", "action": "create", "data_json": {}}))["pending_action"]

    observation = await toolset.execute_pending(pending)
    call = transport.calls[1]
    assert call["mode"] == "execute" and call["confirmation_token"] == "tok-1"
    assert call["arguments"] == pending["arguments"]
    assert observation["id"] == MEETING_ID


@pytest.mark.asyncio
async def test_repreview_after_edit_gets_a_new_token() -> None:
    transport = FakeTransport(responses=[_preview_response("tok-1"), _preview_response("tok-2")])
    toolset = await _toolset(transport)
    pending = json.loads(await toolset.invoke("crm_action_tool", {"module": "Meetings", "action": "create", "data_json": {}}))["pending_action"]
    edited = copy.deepcopy(pending["data"])
    edited["fields"]["date_start"] = "2026-10-02 14:00:00"

    observation = await toolset.repreview_pending(pending, edited)
    assert transport.calls[1]["mode"] == "preview"
    assert transport.calls[1]["arguments"]["data_json"]["fields"]["date_start"] == "2026-10-02 14:00:00"
    assert observation["pending_action"]["confirmation_token"] == "tok-2"


@pytest.mark.asyncio
async def test_open_form_write_is_redirected_to_a_form_patch(monkeypatch) -> None:
    async def fake_patch(*, module, record_id, data, crm_client):
        return json.dumps({"status": "ok", "redirected_to_form": True, "form_patch": {"module": module, "fields": data["fields"]}})

    monkeypatch.setattr("app.engine.form_tools.action_as_form_patch", fake_patch)
    transport = FakeTransport()
    context = {"capabilities": ["form"], "form": {"editable": True}, "module": "Accounts", "record_id": MEETING_ID}
    toolset = await _toolset(transport, context=context)
    observation = json.loads(await toolset.invoke("crm_action_tool", {"module": "Accounts", "action": "update", "data_json": {"id": MEETING_ID, "fields": {"description": "x"}}}))
    assert observation["redirected_to_form"] is True
    assert transport.calls == []


@pytest.mark.asyncio
async def test_crm_down_without_manifest_leaves_native_tools() -> None:
    transport = FakeTransport()
    transport.manifest_error = ConnectionError("down")
    toolset = await _toolset(transport)
    assert toolset.names() == ["rag_search_tool"]
    assert toolset.crm_available is False


@pytest.mark.asyncio
async def test_company_overview_card_links_to_the_record() -> None:
    account_id = "671e9914-41f8-56ad-17a1-6347cd682930"
    transport = FakeTransport(responses=[ok_result({
        "status": "ok", "module": "Accounts", "record": {"id": account_id, "name": "acmark"},
        "records": [{"id": account_id, "name": "acmark"}], "display": "record",
        "card": {"title": "acmark", "meta": {"IČ": "98765444", "Město": "Brno"}},
    })])
    toolset = await _toolset(transport)
    observation = json.loads(await toolset.invoke("get_company_overview", {"account_id": account_id}))
    card = observation["cards"][0]
    assert (card["type"], card["title"], card["meta"]) == ("record", "acmark", {"IČ": "98765444", "Město": "Brno"})
    assert card["actions"][0]["url"] == f"/#detail/Accounts/{account_id}"


@pytest.mark.asyncio
async def test_record_card_preserves_typed_fields_for_localized_ui() -> None:
    fields = [{"name": "total", "label": "Total:", "label_key": "LBL_TOTAL",
               "type": "readonly_computed", "source_type": "currency",
               "value": {"amount": "6763.90", "currency_code": "CZK"}}]
    transport = FakeTransport(responses=[ok_result({
        "module": "Quotes", "records": [{"id": "quote-1", "name": "Quote"}], "display": "record",
        "card": {"title": "Quote", "meta": {"Total:": "6763.900000"}, "meta_fields": fields},
    })])
    toolset = await _toolset(transport)
    observation = json.loads(await toolset.invoke("crm_record_detail_tool", {"module": "Quotes", "record_id": "quote-1"}))
    assert observation["cards"][0]["meta_fields"] == fields
    assert observation["cards"][0]["actions"][0]["label_key"] == "LBL_AI_OPEN"


@pytest.mark.asyncio
async def test_ares_is_available_without_web_and_preserves_external_registry_data() -> None:
    data = {"name": "Firma s.r.o.", "ticker_symbol": "00012345", "sic_code": "CZ00012345"}
    transport = FakeTransport(responses=[ok_result({"status": "ok", "ico": "00012345", "company_data": data})])
    toolset = await _toolset(transport, context={"capabilities": []})
    assert "crm_ares_tool" in toolset.names()
    assert toolset.is_read_only("crm_ares_tool")
    assert "web_search_tool" not in toolset.names()
    observation = json.loads(await toolset.invoke("crm_ares_tool", {"ico": "00012345"}))
    assert observation["company_data"] == data
    assert "cards" not in observation
    assert "confirmation" not in observation
    assert "stejnou službu ARES" in "\n".join(toolset.prompt_blocks())
