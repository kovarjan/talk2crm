# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Core text-to-action pipeline behind /process-input/ and /process-audio/.

Flow: pending-action patch → quick action → LLM agent, with chat persistence,
context merging (pending actions, CRM-created records, UI focus) and optional
TTS response generation.
"""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import TenantContext
from app.api.models import ProcessInputRequest
from app.core.audio import synthesize_to_file
from app.core.config import get_settings
from app.core.logging import get_logger, log_llm_trace
from app.engine.agent import run_agent
from app.engine.events import EmitFn, StreamEvent
from app.engine.pending_patch import try_patch_pending_action
from app.engine.quick_actions import QuickActionResult, try_handle_quick_action
from app.engine.rag import get_rag_service
from app.engine.capabilities import resolve_capabilities
from app.engine.tools import build_tools
from app.presentation.agent_result import extract_form_patch_from_agent_result
from app.presentation.agent_result import normalize_agent_result_for_ui, to_user_message
from app.services import chat_service
from app.services.chat_titles import ensure_chat_name
from app.services.crm_client import CoripoClient
from app.services.tenant_manager import TenantManager
from app.utils.modules import canonical_module_name


logger = get_logger(__name__)
settings = get_settings()

_FALLBACK_USER_MESSAGE = (
    "Omlouvám se, tady se mi nepodařilo připravit odpověď. "
    "Zkuste prosím dotaz zopakovat nebo přeformulovat."
)


def first_nonempty_from_map(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def merge_context_with_pending_action(
    context: dict[str, Any] | None,
    pending_action: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(context or {})
    if not isinstance(pending_action, dict):
        return merged

    if "pending_action" not in merged:
        merged["pending_action"] = pending_action

    pending_module = str(
        pending_action.get("module") or pending_action.get("effective_module") or ""
    ).strip()
    if pending_module and not str(merged.get("module") or "").strip():
        merged["module"] = pending_module

    data = pending_action.get("data")
    data_obj = data if isinstance(data, dict) else {}
    fields_raw = data_obj.get("fields")
    fields = fields_raw if isinstance(fields_raw, dict) else {}

    parent_type = str(fields.get("parent_type") or data_obj.get("parent_type") or "").strip()
    parent_id = str(fields.get("parent_id") or data_obj.get("parent_id") or "").strip()
    if parent_id and not str(merged.get("record") or "").strip():
        merged["record"] = parent_id
    if parent_type and not str(merged.get("record_module") or "").strip():
        merged["record_module"] = parent_type

    entities_raw = merged.get("entities")
    entities = dict(entities_raw) if isinstance(entities_raw, dict) else {}

    contact_name = first_nonempty_from_map(
        fields,
        ["contact_name", "related_contact_name", "invite_contact_name", "participant_name"],
    ) or first_nonempty_from_map(
        data_obj,
        ["contact_name", "related_contact_name", "invite_contact_name", "participant_name"],
    )
    account_name = first_nonempty_from_map(
        fields,
        ["account_name", "related_account_name", "company", "company_name"],
    ) or first_nonempty_from_map(
        data_obj,
        ["account_name", "related_account_name", "company", "company_name"],
    )

    if contact_name and not str(entities.get("contact_name") or "").strip():
        entities["contact_name"] = contact_name
    if account_name and not str(entities.get("account_name") or "").strip():
        entities["account_name"] = account_name

    parent_type_norm = parent_type.lower()
    if parent_id and "contact" in parent_type_norm and not str(entities.get("contact_id") or "").strip():
        entities["contact_id"] = parent_id
    if parent_id and "account" in parent_type_norm and not str(entities.get("account_id") or "").strip():
        entities["account_id"] = parent_id

    if entities:
        merged["entities"] = entities

    return merged


def merge_context_with_created_record(
    context: dict[str, Any] | None,
    created_record_event: dict[str, str] | None,
) -> dict[str, Any]:
    merged = dict(context or {})
    if not isinstance(created_record_event, dict):
        return merged

    module = canonical_module_name(created_record_event.get("module"))
    record_id = str(created_record_event.get("record_id") or "").strip()
    record_name = str(created_record_event.get("record_name") or "").strip()

    if module and not str(merged.get("module") or "").strip():
        merged["module"] = module
    if module and not str(merged.get("record_module") or "").strip():
        merged["record_module"] = module
    if record_id and not str(merged.get("record") or "").strip():
        merged["record"] = record_id
    if record_id and not str(merged.get("record_id") or "").strip():
        merged["record_id"] = record_id
    if record_name and not str(merged.get("record_name") or "").strip():
        merged["record_name"] = record_name

    entities_raw = merged.get("entities")
    entities = dict(entities_raw) if isinstance(entities_raw, dict) else {}
    module_lower = module.lower()
    if module_lower == "meetings" and record_id and not str(entities.get("meeting_id") or "").strip():
        entities["meeting_id"] = record_id
    if module_lower == "calls" and record_id and not str(entities.get("call_id") or "").strip():
        entities["call_id"] = record_id
    if entities:
        merged["entities"] = entities

    return merged


def merge_context_with_history_selection(
    context: dict[str, Any] | None,
    selection: dict[str, str] | None,
) -> dict[str, Any]:
    merged = dict(context or {})
    if not isinstance(selection, dict):
        return merged

    module = canonical_module_name(selection.get("module"))
    record_id = str(selection.get("record_id") or "").strip()
    record_name = str(selection.get("record_name") or "").strip()
    selection_label = str(selection.get("selection_label") or record_name or "").strip()
    if not module or not record_id:
        return merged

    merged["module"] = module
    merged["record_module"] = module
    merged["record"] = record_id
    merged["record_id"] = record_id
    if record_name:
        merged["record_name"] = record_name
    if selection_label:
        merged["selected_option_label"] = selection_label
    merged["selection_source"] = "history_selection"

    entities_raw = merged.get("entities")
    entities = dict(entities_raw) if isinstance(entities_raw, dict) else {}
    module_lower = module.lower()
    if module_lower == "accounts":
        entities["account_id"] = record_id
        if record_name:
            entities["account_name"] = record_name
    elif module_lower == "contacts":
        entities["contact_id"] = record_id
        if record_name:
            entities["contact_name"] = record_name
    elif module_lower == "leads":
        entities["lead_id"] = record_id
        if record_name:
            entities["lead_name"] = record_name
    if entities:
        merged["entities"] = entities
    return merged


def is_read_only_data_query(input_text: str) -> bool:
    lowered = (input_text or "").strip().lower()
    normalized = "".join(
        char for char in unicodedata.normalize("NFD", lowered) if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    mutation_tokens = (
        "vytvor",
        "zaloz",
        "naplanuj",
        "pridej",
        "uprav",
        "zmen",
        "smaz",
        "odstran",
        "delete",
        "update",
        "create",
        "patch",
    )
    if any(token in normalized for token in mutation_tokens):
        return False

    question_markers = (
        "kdo ",
        "jake ",
        "jaky ",
        "co ",
        "kde ",
        "kdy ",
        "kolik ",
        "?",
    )
    data_tokens = (
        "schuzk",
        "meeting",
        "kontakt",
        "contact",
        "firma",
        "spolecnost",
        "account",
        "zaznam",
        "records",
    )
    return any(marker in normalized for marker in question_markers) and any(
        token in normalized for token in data_tokens
    )


def build_soft_ui_focus_hint(context: dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict):
        return None

    module = canonical_module_name(
        context.get("record_module")
        or context.get("module")
    )
    record_name = str(context.get("record_name") or "").strip()
    record_id = str(context.get("record_id") or context.get("record") or "").strip()
    if not module and not record_name and not record_id:
        return None

    module_label_map = {
        "meetings": "schůzky",
        "calls": "hovoru",
        "contacts": "kontaktu",
        "accounts": "firmy",
        "tasks": "úkolu",
        "notes": "poznámky",
    }
    module_label = module_label_map.get(module.lower(), "záznamu") if module else "záznamu"

    parts: list[str] = [f"Uživatel má v CRM právě otevřen DetailView {module_label}."]
    if record_name:
        parts.append(f"Název: '{record_name}'.")
    if record_id:
        parts.append(f"ID: {record_id}.")
    parts.append("Ber to jako orientační kontext (nápovědu), ne jako závazný fakt.")
    return " ".join(parts)


def with_soft_ui_focus_hint(context: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(context or {})
    if str(merged.get("ui_focus_hint_cz") or "").strip():
        return merged
    hint = build_soft_ui_focus_hint(merged)
    if hint:
        merged["ui_focus_hint_cz"] = hint
    return merged


async def maybe_generate_voice(
    *,
    text: str,
    return_voice: bool,
    background_tasks: BackgroundTasks,
) -> tuple[str | None, str | None]:
    if not return_voice:
        return None, None

    file_id = uuid.uuid4().hex
    output_path = settings.cache_audio_dir / f"{file_id}.wav"

    background_tasks.add_task(
        synthesize_to_final_path,
        text,
        str(output_path),
        settings.tts_voice,
    )
    return file_id, f"/audio/{file_id}"


async def synthesize_to_final_path(
    text: str,
    final_output_path: str,
    voice: str | None = None,
) -> None:
    final_path = Path(final_output_path)
    temp_path = final_path.with_name(f"{final_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        await synthesize_to_file(text=text, output_path=str(temp_path), voice=voice)
        temp_path.replace(final_path)
    finally:
        temp_path.unlink(missing_ok=True)


async def process_input_core(
    *,
    db: AsyncSession,
    ctx: TenantContext,
    payload: ProcessInputRequest,
    background_tasks: BackgroundTasks,
    emit: EmitFn | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    tenant_id = ctx["tenant_id"]
    user_id = ctx["user_id"]

    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(tenant_id)
    crm_client = CoripoClient(
        credentials.crm_base_url,
        credentials.crm_token,
        user_id=user_id,
        user_name=ctx["user_name"],
    )
    rag_service = get_rag_service()

    if payload.chat_id:
        chat = await chat_service.get_chat_or_404(
            db,
            chat_id=payload.chat_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
    else:
        chat = await chat_service.create_chat(db, tenant_id=tenant_id, user_id=user_id, persist=False)

    if emit:
        await emit(StreamEvent("accepted", {"chat_id": str(chat.id)}))

    latest_pending_action = await chat_service.load_latest_pending_action(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    incoming_context = dict(payload.context or {})
    enabled_capabilities, unknown_capabilities = resolve_capabilities(incoming_context)
    latest_created_record = await chat_service.load_latest_crm_record_created_event(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    incoming_module = canonical_module_name(incoming_context.get("module"))
    created_module = canonical_module_name(
        latest_created_record.get("module") if isinstance(latest_created_record, dict) else ""
    )
    if latest_created_record and (not incoming_module or incoming_module.lower() == created_module.lower()):
        incoming_context = merge_context_with_created_record(incoming_context, latest_created_record)
    action_confirmation = bool(incoming_context.get("confirm_action", False))
    should_merge_pending = bool(latest_pending_action) and (
        action_confirmation or not is_read_only_data_query(payload.input_text)
    )
    if should_merge_pending:
        effective_context = merge_context_with_pending_action(incoming_context, latest_pending_action)
    else:
        effective_context = incoming_context
    latest_history_selection = await chat_service.load_latest_history_selection(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
        input_text=payload.input_text,
    )
    if latest_history_selection:
        effective_context = merge_context_with_history_selection(effective_context, latest_history_selection)
    effective_context = with_soft_ui_focus_hint(effective_context)
    request_context = effective_context or None

    await chat_service.append_message(
        db,
        chat=chat,
        tenant_id=tenant_id,
        user_id=user_id,
        role="user",
        content=payload.input_text,
    )

    action_confirmation = bool(effective_context.get("confirm_action", False))
    execution_mode = "quick_action"
    available_tools: list[str] = []

    pending_for_patch = (
        effective_context.get("pending_action") if isinstance(effective_context.get("pending_action"), dict) else None
    ) or latest_pending_action
    pending_patch_result = None
    if isinstance(pending_for_patch, dict) and not action_confirmation:
        pending_patch_result = try_patch_pending_action(
            input_text=payload.input_text,
            pending_action=pending_for_patch,
        )

    if pending_patch_result is not None:
        execution_mode = "pending_patch"
        agent_result = pending_patch_result
    else:
        quick_result: QuickActionResult | None = None
        if settings.quick_action_enabled:
            quick_result = await try_handle_quick_action(
                input_text=payload.input_text,
                crm_client=crm_client,
                user_id=user_id,
                action_confirmation=action_confirmation,
            )
        if quick_result is not None and not quick_result.should_fallback:
            agent_result = quick_result.data
        else:
            execution_mode = "agent"
            if emit:
                await emit(StreamEvent("pipeline.mode", {"mode": execution_mode}))
            tools = build_tools(
                tenant_id=tenant_id,
                user_id=user_id,
                input_text=payload.input_text,
                request_context=request_context,
                crm_client=crm_client,
                rag_service=rag_service,
                action_confirmation=action_confirmation,
                capabilities=enabled_capabilities,
            )
            available_tools = [str(getattr(tool, "name", "")) for tool in tools if getattr(tool, "name", None)]
            history_for_agent = await chat_service.load_messages(
                db,
                chat_id=chat.id,
                tenant_id=tenant_id,
                user_id=user_id,
                limit=20,
            )
            recent_history_for_agent = [
                chat_service.chat_message_item_to_agent_history(item)
                for item in history_for_agent
            ]
            try:
                agent_result = await run_agent(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    input_text=payload.input_text,
                    context=request_context,
                    tools=tools,
                    chat_history=recent_history_for_agent,
                    emit=emit,
                    db=db,
                    capabilities=enabled_capabilities,
                )
                agent_result = normalize_agent_result_for_ui(agent_result)
            except Exception as exc:
                logger.exception(
                    "Agent execution failed tenant=%s user=%s chat=%s",
                    tenant_id,
                    user_id,
                    chat.id,
                )
                agent_result = {
                    "output": (
                        "Nastala chyba při provedení akce v CRM. "
                        "Akce nebyla provedena. "
                        "Zkontrolujte povinná pole a potvrďte akci znovu."
                    ),
                    "error": str(exc),
                    "status": "agent_error",
                    "message_to_user": (
                        "Nastala chyba při provedení akce v CRM. "
                        "Akce nebyla provedena. "
                        "Zkontrolujte povinná pole a potvrďte akci znovu."
                    ),
                }

    assistant_raw_text = str(
        agent_result.get("message_to_user")
        or agent_result.get("output")
        or ""
    )
    assistant_text = to_user_message(assistant_raw_text)
    if not assistant_text:
        assistant_text = _FALLBACK_USER_MESSAGE
    form_patch = extract_form_patch_from_agent_result(agent_result)
    capability_tag = ",".join(sorted(enabled_capabilities - {"crm"})) or None
    if capability_tag != getattr(chat, "tool", None):
        chat.tool = capability_tag
    await chat_service.append_message(
        db,
        chat=chat,
        tenant_id=tenant_id,
        user_id=user_id,
        role="assistant",
        content=assistant_text,
        metadata={"agent_result": agent_result, "raw_output": assistant_raw_text},
    )

    history = await chat_service.load_messages(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    await ensure_chat_name(
        chat=chat,
        history=history,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    await db.commit()

    file_id, audio_url = await maybe_generate_voice(
        text=assistant_text,
        return_voice=payload.return_voice,
        background_tasks=background_tasks,
    )

    chat_history_dump = [item.model_dump(mode="json") for item in history]
    tool_calls = agent_result.get("intermediate_steps")
    if not isinstance(tool_calls, list):
        tool_calls = []
    cards = agent_result.get("cards")
    if not isinstance(cards, list):
        cards = []
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    log_llm_trace(
        logger,
        tenant_id=tenant_id,
        user_id=user_id,
        chat_id=chat.id,
        request={
            "input_text": payload.input_text,
            "chat_id": payload.chat_id or chat.id,
            "confirm_action": action_confirmation,
            "return_voice": payload.return_voice,
        },
        context=request_context,
        tool_calls=tool_calls,
        outcome={
            "status": str(agent_result.get("status") or "ok"),
            "final_answer": assistant_text,
            "message_to_user": assistant_text,
            "output": agent_result.get("output"),
            "error": agent_result.get("error"),
        },
        resources={
            "execution_mode": execution_mode,
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "crm_mode": settings.crm_mode,
            "rag_available": rag_service is not None,
            "available_tools": available_tools,
            "capabilities": sorted(enabled_capabilities),
            "return_voice": payload.return_voice,
            "audio_generated": bool(file_id),
            "duration_ms": duration_ms,
        },
        chat_history=chat_history_dump,
    )

    result_payload = {
        "action_result": agent_result,
        "message_to_user": assistant_text,
        "chat_id": chat.id,
        "chat_name": chat.name,
        "chat_history": chat_history_dump,
        "cards": cards,
        "audio_file_id": file_id,
        "audio_response_id": file_id,
        "audio_url": audio_url,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "capabilities": sorted(enabled_capabilities),
        "unknown_capabilities": unknown_capabilities,
        "form_patch": form_patch,
    }
    if emit:
        if form_patch:
            await emit(StreamEvent("form_patch", form_patch))
        await emit(StreamEvent("result", {
            **result_payload,
            "chat_id": str(chat.id),
        }))
    return result_payload
