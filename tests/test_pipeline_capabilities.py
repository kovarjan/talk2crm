from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.models import ProcessInputRequest
from app.engine import pipeline
from app.engine.events import StreamEvent

PATCH = {"module": "Contacts", "record": None, "fields": {"first_name": "Jan"}, "lines": {"mode": "append", "rows": []},
         "invitees": {}, "sources": [], "message": "Doplněno.", "schema": {"sections": []}}


class _Chat(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_pipeline_passes_capabilities_and_emits_form_patch() -> None:
    chat = _Chat(id="c1", name=None, tool=None)
    events: list[StreamEvent] = []

    async def emit(event: StreamEvent) -> None:
        events.append(event)

    agent_result = {
        "output": "<answer>Hotovo.</answer>",
        "intermediate_steps": [{"tool": "propose_form_fields_tool", "observation": {"status": "ok", "form_patch": PATCH}}],
    }
    creds = SimpleNamespace(crm_base_url="http://crm", crm_token="tok")
    chat_service = SimpleNamespace(
        get_chat_or_404=AsyncMock(return_value=chat),
        create_chat=AsyncMock(return_value=chat),
        load_latest_pending_action=AsyncMock(return_value=None),
        load_latest_crm_record_created_event=AsyncMock(return_value=None),
        load_latest_history_selection=AsyncMock(return_value=None),
        append_message=AsyncMock(),
        load_messages=AsyncMock(return_value=[]),
        chat_message_item_to_agent_history=lambda item: item,
    )
    captured: dict[str, Any] = {}

    def fake_build_tools(**kwargs):
        captured["build_capabilities"] = kwargs.get("capabilities")
        return []

    async def fake_run_agent(**kwargs):
        captured["agent_capabilities"] = kwargs.get("capabilities")
        return agent_result

    with (
        patch.object(pipeline, "TenantManager") as tm,
        patch.object(pipeline, "CoripoClient"),
        patch.object(pipeline, "get_rag_service", return_value=None),
        patch.object(pipeline, "chat_service", chat_service),
        patch.object(pipeline, "build_tools", side_effect=fake_build_tools),
        patch.object(pipeline, "run_agent", side_effect=fake_run_agent),
        patch.object(pipeline, "ensure_chat_name", new=AsyncMock()),
        patch.object(pipeline, "maybe_generate_voice", new=AsyncMock(return_value=(None, None))),
        patch.object(pipeline, "log_llm_trace"),
    ):
        tm.return_value.get_credentials = AsyncMock(return_value=creds)
        db = SimpleNamespace(commit=AsyncMock())
        payload = ProcessInputRequest(
            input_text="doplň kontakt Jan Novák",
            chat_id="c1",
            context={"module": "Contacts", "capabilities": ["form", "products"], "form": {"editable": True, "values": {}}},
        )
        result = await pipeline.process_input_core(
            db=db, ctx={"tenant_id": "t", "user_id": "u", "user_name": None},
            payload=payload, background_tasks=SimpleNamespace(add_task=lambda *a, **k: None), emit=emit,
        )

    assert captured["build_capabilities"] == {"crm", "form"}
    assert captured["agent_capabilities"] == {"crm", "form"}
    assert result["capabilities"] == ["crm", "form"]
    assert result["unknown_capabilities"] == ["products"]
    assert result["form_patch"] == PATCH
    assert chat.tool == "form"
    types = [e.type for e in events]
    assert types.index("form_patch") < types.index("result")
    assert events[types.index("form_patch")].data == PATCH
