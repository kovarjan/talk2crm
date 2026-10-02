from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from _tool_fakes import FakeTransport, ok_result
from app.api.models import ProcessInputRequest
from app.domain.contracts import build_pending_action_envelope
from app.engine import pipeline
from app.services.module_catalog import ModuleCatalog
from app.tools.crm_provider import CrmToolProvider
from app.tools.manifest_cache import ManifestCache
from app.tools.resilience import TenantGuards
from app.tools.toolset import ToolSet

ARGS = {"module": "Meetings", "action": "create", "data_json": {"fields": {"name": "Schůzka - demo", "date_start": "2026-10-01 10:00:00", "date_end": "2026-10-01 11:00:00", "duration_hours": 1, "duration_minutes": 0}}}
PENDING = build_pending_action_envelope(
    module="Meetings", action="create", data=ARGS["data_json"],
    tool_call={"tool": "crm_action_tool", "tool_version": "1.0.0", "arguments": ARGS, "confirmation_token": "tok-1"},
)


async def _run(input_text: str, context: dict[str, Any], transport: FakeTransport) -> tuple[dict[str, Any], AsyncMock]:
    chat = SimpleNamespace(id="c1", name="x", tool=None)
    chat_service = SimpleNamespace(
        get_chat_or_404=AsyncMock(return_value=chat),
        create_chat=AsyncMock(return_value=chat),
        load_latest_pending_action=AsyncMock(return_value=PENDING),
        load_latest_crm_record_created_event=AsyncMock(return_value=None),
        load_latest_history_selection=AsyncMock(return_value=None),
        append_message=AsyncMock(),
        load_messages=AsyncMock(return_value=[]),
        chat_message_item_to_agent_history=lambda item: item,
    )

    async def toolset(**kwargs):
        provider = CrmToolProvider(tenant_id="t", transport=transport, cache=ManifestCache(), guards=TenantGuards())
        return await ToolSet.build(tenant_id="t", user_id="u", request_context=kwargs["request_context"], crm_client=None,
                                   native_tools=lambda enabled: [], provider=provider)

    run_agent = AsyncMock(side_effect=AssertionError("the LLM must not run for this turn"))
    with (
        patch.object(pipeline, "TenantManager") as tm,
        patch.object(pipeline, "CoripoClient"),
        patch.object(pipeline, "get_rag_service", return_value=None),
        patch.object(pipeline, "get_module_catalog", new=AsyncMock(return_value=ModuleCatalog.static_fallback())),
        patch.object(pipeline, "chat_service", chat_service),
        patch.object(pipeline, "build_turn_toolset", side_effect=toolset),
        patch.object(pipeline, "run_agent", run_agent),
        patch.object(pipeline, "ensure_chat_name", new=AsyncMock()),
        patch.object(pipeline, "maybe_generate_voice", new=AsyncMock(return_value=(None, None))),
        patch.object(pipeline, "log_llm_trace"),
    ):
        tm.return_value.get_credentials = AsyncMock(return_value=SimpleNamespace(crm_base_url="http://crm", crm_token="tok"))
        result = await pipeline.process_input_core(
            db=SimpleNamespace(commit=AsyncMock()),
            ctx={"tenant_id": "t", "user_id": "u", "user_name": None},
            payload=ProcessInputRequest(input_text=input_text, chat_id="c1", context=context),
            background_tasks=SimpleNamespace(add_task=lambda *a, **k: None),
        )
    return result, run_agent


@pytest.mark.asyncio
async def test_confirm_action_executes_the_pending_tool_call_with_its_token() -> None:
    transport = FakeTransport(responses=[ok_result({"status": "ok", "module": "Meetings", "action": "create", "id": "m1", "message_to_user": "Záznam „Schůzka - demo“ byl vytvořen."})])
    result, run_agent = await _run("ano", {"confirm_action": True}, transport)

    call = transport.calls[0]
    assert (call["name"], call["mode"], call["confirmation_token"]) == ("crm_action_tool", "execute", "tok-1")
    assert call["arguments"] == ARGS
    assert result["message_to_user"] == "Záznam „Schůzka - demo“ byl vytvořen."
    assert result["action_result"].get("pending_action") is None
    run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_edit_of_pending_tool_call_is_previewed_again() -> None:
    confirm_args = json.loads(json.dumps(ARGS))
    confirm_args["data_json"]["fields"].update({"date_start": "2026-10-01 14:00:00", "date_end": "2026-10-01 15:00:00"})
    transport = FakeTransport(responses=[ok_result(
        {"status": "confirmation_required", "module": "Meetings", "action": "create"},
        meta={"confirmation": {"required": True, "token": "tok-2", "arguments": confirm_args}},
    )])
    result, run_agent = await _run("změň to na 14:00", {}, transport)

    call = transport.calls[0]
    assert call["mode"] == "preview"
    sent_fields = call["arguments"]["data_json"]["fields"]
    assert sent_fields["date_start"].endswith("14:00:00")
    assert "date_end" not in sent_fields  # stale derived end dropped; the CRM re-derives it
    pending = result["action_result"]["pending_action"]
    assert pending["confirmation_token"] == "tok-2"
    command = json.loads(result["action_result"]["output"])
    assert command["tool"] == "crm_action_tool" and command["confirmation_token"] == "tok-2"
    run_agent.assert_not_called()
