from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import unicodedata

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import case, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError

from app.api.dependencies import TenantContext, get_tenant_context
from app.api.models import (
    BaseResponse,
    ChatCreate,
    ChatHistoryResponse,
    ChatMessageItem,
    ChatReplaceRequest,
    CreateChatResponse,
    ProcessInputRequest,
    RagIngestRequest,
    SearchRequest,
    UserChatsResponse,
)
from app.core.audio import synthesize_to_file, transcribe
from app.core.config import get_settings
from app.core.logging import get_logger, log_llm_trace
from app.domain.contracts import normalize_pending_action_envelope
from app.engine.agent import run_agent
from app.engine.pending_patch import try_patch_pending_action
from app.engine.quick_actions import QuickActionResult, try_handle_quick_action
from app.engine.rag import TenantRAGService
from app.engine.tools import build_tools
from app.services.crm_client import SugarClient
from app.services.tenant_manager import TenantManager
from database.models import Chat, ChatMessage
from database.session import get_db
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()
_rag_service_instance: TenantRAGService | None = None
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_MD_FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)```", re.IGNORECASE)
_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LEADING_SYMBOL_RE = re.compile(r"^[\s\-\u2022>*]+")
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
    flags=re.UNICODE,
)
_CHAT_TITLE_MAX_CHARS = 80
_CHAT_TITLE_MAX_WORDS = 6


def get_rag_service() -> TenantRAGService | None:
    global _rag_service_instance
    if _rag_service_instance is not None:
        return _rag_service_instance
    try:
        _rag_service_instance = TenantRAGService()
        return _rag_service_instance
    except Exception:
        logger.exception("Unable to initialize Qdrant RAG service")
        return None


def _chat_message_to_model(message: ChatMessage) -> ChatMessageItem:
    metadata = message.metadata_json or {}
    if message.role == "assistant" and isinstance(metadata, dict):
        cards = metadata.get("cards")
        if not isinstance(cards, list):
            agent_result = metadata.get("agent_result")
            if isinstance(agent_result, dict) and isinstance(agent_result.get("cards"), list):
                metadata = dict(metadata)
                metadata["cards"] = agent_result["cards"]
    return ChatMessageItem(
        role=message.role,
        content=message.content,
        metadata=metadata,
        created_at=message.created_at,
    )


async def _load_messages(
    db: AsyncSession,
    *,
    chat_id: str,
    tenant_id: str,
    user_id: str,
) -> list[ChatMessageItem]:
    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.user_id == user_id,
        )
        .order_by(
            ChatMessage.created_at.asc(),
            case(
                (ChatMessage.role == "user", 0),
                (ChatMessage.role == "assistant", 1),
                else_=2,
            ).asc(),
            ChatMessage.id.asc(),
        )
    )
    result = await db.execute(stmt)
    return [_chat_message_to_model(item) for item in result.scalars().all()]


async def _get_chat_or_404(
    db: AsyncSession,
    *,
    chat_id: str,
    tenant_id: str,
    user_id: str,
) -> Chat:
    stmt = select(Chat).where(
        Chat.id == chat_id,
        Chat.tenant_id == tenant_id,
        Chat.user_id == user_id,
    )
    result = await db.execute(stmt)
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


async def _create_chat(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    persist: bool = True,
) -> Chat:
    chat = Chat(
        id=uuid.uuid4().hex,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    db.add(chat)
    if not persist:
        return chat

    attempts = 5
    for attempt in range(attempts):
        try:
            await db.commit()
            await db.refresh(chat)
            return chat
        except OperationalError as exc:
            await db.rollback()
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(0.05 * (2**attempt))

    return chat


async def _append_message(
    db: AsyncSession,
    *,
    chat: Chat,
    tenant_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    db.add(
        ChatMessage(
            chat_id=chat.id,
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            content=content,
            metadata_json=metadata or {},
        )
    )
    chat.updated_at = datetime.now(timezone.utc)


async def _maybe_generate_voice(
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
        _synthesize_to_final_path,
        text,
        str(output_path),
        settings.tts_voice,
    )
    return file_id, f"/audio/{file_id}"


async def _synthesize_to_final_path(
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


def _to_user_message(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text or "")
    cleaned = _MD_FENCE_RE.sub(r"\1", cleaned)
    cleaned = _MD_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MD_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _MD_INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = "\n".join(_LEADING_SYMBOL_RE.sub("", line) for line in cleaned.splitlines())
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or (text or "")


def _try_parse_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None




def _looks_like_noise(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lower = value.lower()
    if len(value) > 1400:
        return True
    noisy_patterns = [
        "here's the translation of the menu labels",
        '"row_count"',
        '"menu":',
        "```json",
    ]
    return any(pattern in lower for pattern in noisy_patterns)


def _normalize_agent_result_for_ui(agent_result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(agent_result or {})
    steps = normalized.get("intermediate_steps") or []

    for step in reversed(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = _try_parse_json(observation) or {}
        else:
            obs = {}
        if not obs:
            continue

        status = str(obs.get("status") or "").strip().lower()
        if status in {"confirmation_required", "resolution_required"}:
            pending_raw = obs.get("pending_action") if isinstance(obs.get("pending_action"), dict) else {}
            pending = normalize_pending_action_envelope(pending_raw) or pending_raw
            module = str(
                pending.get("effective_module")
                or pending.get("module")
                or pending.get("requested_module")
                or ""
            ).strip()
            action = str(pending.get("action") or "").strip()
            data = pending.get("data") if isinstance(pending.get("data"), dict) else {}
            message = str(
                obs.get("message")
                or (
                    "Akce je připravena a čeká na vaše potvrzení."
                    if status == "confirmation_required"
                    else "Pro pokračování potřebuji upřesnit cílový záznam."
                )
            ).strip()

            command_payload = {
                "action": action or "create",
                "module": module or "Meetings",
                "data_json": json.dumps(data, ensure_ascii=False),
                "message_to_user": message,
            }
            normalized["status"] = status
            normalized["pending_action"] = pending
            normalized["output"] = json.dumps(command_payload, ensure_ascii=False)
            normalized["message_to_user"] = message
            return normalized

        if status in {"crm_http_error", "crm_action_error"}:
            message = str(
                obs.get("message")
                or "CRM akce selhala. Zkontrolujte mapování polí a povinné hodnoty."
            ).strip()
            normalized["status"] = status
            normalized["message_to_user"] = message
            if not str(normalized.get("output") or "").strip():
                normalized["output"] = message
            return normalized

    for step in reversed(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        tool_name = str(step.get("tool") or "").strip()
        if tool_name not in {"crm_data_tool", "crm_search_tool", "crm_query_tool", "my_meetings_tool"}:
            continue

        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = _try_parse_json(observation) or {}
        else:
            obs = {}
        if not obs:
            continue

        cards = obs.get("cards")
        if isinstance(cards, list):
            normalized["cards"] = cards
        tool_message = _to_user_message(
            str(
                obs.get("message_to_user")
                or obs.get("summary")
                or ""
            ).strip()
        )
        if tool_message:
            normalized["message_to_user"] = tool_message
        tool_status = str(obs.get("status") or "").strip()
        if tool_status:
            normalized["status"] = tool_status
        if not str(normalized.get("output") or "").strip():
            normalized["output"] = json.dumps(obs, ensure_ascii=False)
        return normalized

    output_text = str(normalized.get("output") or "")
    parsed_output = _try_parse_json(output_text)
    if isinstance(parsed_output, dict) and parsed_output.get("message_to_user"):
        normalized["message_to_user"] = _to_user_message(str(parsed_output["message_to_user"]))
        return normalized

    clean = _to_user_message(output_text)
    if not clean or _looks_like_noise(clean):
        clean = (
            "Nerozuměla jsem spolehlivě požadavku. "
            "Upřesněte prosím akci, modul a čas (např. schůzka v úterý 9:30)."
        )
    normalized["message_to_user"] = clean
    return normalized


def _extract_pending_action_from_agent_result(agent_result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(agent_result, dict):
        return None

    direct_pending = agent_result.get("pending_action")
    if isinstance(direct_pending, dict):
        return normalize_pending_action_envelope(direct_pending) or direct_pending

    steps = agent_result.get("intermediate_steps")
    if not isinstance(steps, list):
        return None

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = _try_parse_json(observation) or {}
        else:
            obs = {}
        if not isinstance(obs, dict):
            continue
        status = str(obs.get("status") or "").strip().lower()
        pending = obs.get("pending_action")
        if status in {"confirmation_required", "resolution_required"} and isinstance(pending, dict):
            return normalize_pending_action_envelope(pending) or pending
    return None


def _first_nonempty_from_map(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _merge_context_with_pending_action(
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

    contact_name = _first_nonempty_from_map(
        fields,
        ["contact_name", "related_contact_name", "invite_contact_name", "participant_name"],
    ) or _first_nonempty_from_map(
        data_obj,
        ["contact_name", "related_contact_name", "invite_contact_name", "participant_name"],
    )
    account_name = _first_nonempty_from_map(
        fields,
        ["account_name", "related_account_name", "company", "company_name"],
    ) or _first_nonempty_from_map(
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


def _is_read_only_data_query(input_text: str) -> bool:
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


async def _load_latest_pending_action(
    db: AsyncSession,
    *,
    chat_id: str,
    tenant_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.user_id == user_id,
            ChatMessage.role == "assistant",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    message = result.scalar_one_or_none()
    if message is None or not isinstance(message.metadata_json, dict):
        return None

    agent_result = message.metadata_json.get("agent_result")
    if not isinstance(agent_result, dict):
        return None
    return _extract_pending_action_from_agent_result(agent_result)


def _history_for_title_prompt(history: list[ChatMessageItem]) -> str:
    lines: list[str] = []
    # Keep prompt small and focused on the beginning of conversation topic.
    for item in history[:8]:
        role = (item.role or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _to_user_message(item.content or "")
        if not content:
            continue
        compact = re.sub(r"\s+", " ", content).strip()
        if len(compact) > 220:
            compact = compact[:220].rstrip() + "..."
        prefix = "U" if role == "user" else "A"
        lines.append(f"{prefix}: {compact}")
    return "\n".join(lines)


def _sanitize_chat_name(value: str | None) -> str:
    text = _to_user_message(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("`\"'“”„")
    if not text:
        return ""
    # Keep only the first line if model adds extra explanation.
    text = text.splitlines()[0].strip()
    words = text.split()
    if len(words) > _CHAT_TITLE_MAX_WORDS:
        text = " ".join(words[:_CHAT_TITLE_MAX_WORDS])
    if len(text) > _CHAT_TITLE_MAX_CHARS:
        text = text[:_CHAT_TITLE_MAX_CHARS].rstrip(" ,.;:-")
    return text


def _fallback_chat_name(history: list[ChatMessageItem]) -> str:
    for item in history:
        if (item.role or "").strip().lower() != "user":
            continue
        text = re.sub(r"\s+", " ", (item.content or "")).strip()
        if not text:
            continue
        words = text.split()
        short = " ".join(words[:_CHAT_TITLE_MAX_WORDS])
        return _sanitize_chat_name(short) or "Novy chat"
    return "Novy chat"


async def _generate_chat_name_with_llm(
    *,
    history: list[ChatMessageItem],
    tenant_id: str,
    user_id: str,
) -> str:
    transcript = _history_for_title_prompt(history)
    if not transcript:
        return _fallback_chat_name(history)

    settings = get_settings()
    messages = [
        SystemMessage(content="/nothink"),
        HumanMessage(
            content=(
                "Jsi asistent, který vytváří krátké názvy konverzací v češtině.\n"
                "Pravidla:\n"
                "- vrať pouze název (žádné vysvětlení)\n"
                "- 2 až 6 slov\n"
                "- max 80 znaků\n"
                "- bez uvozovek a bez tečky na konci\n\n"
                f"Konverzace:\n{transcript}\n\název:"
            )
        ),
    ]
    models_to_try = list(dict.fromkeys([settings.llm_title_model, settings.llm_model]))
    for model_name in models_to_try:
        llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=model_name,
            temperature=0,
            max_tokens=200,
        )
        try:
            response = await llm.ainvoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            if isinstance(raw, list):
                raw = " ".join(str(part) for part in raw)
            cleaned = _sanitize_chat_name(str(raw))
            if cleaned:
                return cleaned
        except Exception:
            logger.warning(
                "Chat title generation failed with model=%s tenant=%s user=%s",
                model_name,
                tenant_id,
                user_id,
            )

    return _fallback_chat_name(history)


async def _ensure_chat_name(
    *,
    chat: Chat,
    history: list[ChatMessageItem],
    tenant_id: str,
    user_id: str,
) -> str:
    existing = (chat.name or "").strip()
    if existing:
        return existing

    generated = await _generate_chat_name_with_llm(
        history=history,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    generated = _sanitize_chat_name(generated)
    if not generated:
        generated = _fallback_chat_name(history)
    chat.name = generated[:255]
    chat.updated_at = datetime.now(timezone.utc)
    return chat.name


async def _process_input_core(
    *,
    db: AsyncSession,
    ctx: TenantContext,
    payload: ProcessInputRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    started = time.perf_counter()
    tenant_id = ctx["tenant_id"]
    user_id = ctx["user_id"]

    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(tenant_id)
    crm_client = SugarClient(
        credentials.crm_base_url,
        credentials.crm_token,
        user_id=user_id,
        user_name=ctx["user_name"],
    )
    rag_service = get_rag_service()

    if payload.chat_id:
        chat = await _get_chat_or_404(
            db,
            chat_id=payload.chat_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
    else:
        chat = await _create_chat(db, tenant_id=tenant_id, user_id=user_id, persist=False)

    latest_pending_action = await _load_latest_pending_action(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    incoming_context = dict(payload.context or {})
    action_confirmation = bool(incoming_context.get("confirm_action", False))
    should_merge_pending = bool(latest_pending_action) and (
        action_confirmation or not _is_read_only_data_query(payload.input_text)
    )
    if should_merge_pending:
        effective_context = _merge_context_with_pending_action(incoming_context, latest_pending_action)
    else:
        effective_context = incoming_context
    request_context = effective_context or None

    await _append_message(
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
            tools = build_tools(
                tenant_id=tenant_id,
                user_id=user_id,
                input_text=payload.input_text,
                request_context=request_context,
                crm_client=crm_client,
                rag_service=rag_service,
                action_confirmation=action_confirmation,
            )
            available_tools = [str(getattr(tool, "name", "")) for tool in tools if getattr(tool, "name", None)]
            try:
                agent_result = await run_agent(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    input_text=payload.input_text,
                    context=request_context,
                    tools=tools,
                )
                agent_result = _normalize_agent_result_for_ui(agent_result)
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
    assistant_text = _to_user_message(assistant_raw_text)
    if not assistant_text:
        assistant_text = (
            "Nerozuměla jsem spolehlivě požadavku. "
            "Upřesněte prosím akci, modul a čas (např. schůzka v úterý 9:30)."
        )
    await _append_message(
        db,
        chat=chat,
        tenant_id=tenant_id,
        user_id=user_id,
        role="assistant",
        content=assistant_text,
        metadata={"agent_result": agent_result, "raw_output": assistant_raw_text},
    )

    history = await _load_messages(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    await _ensure_chat_name(
        chat=chat,
        history=history,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    await db.commit()

    file_id, audio_url = await _maybe_generate_voice(
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
            "return_voice": payload.return_voice,
            "audio_generated": bool(file_id),
            "duration_ms": duration_ms,
        },
        chat_history=chat_history_dump,
    )

    return {
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
    }


async def _trigger_ingest(
    *,
    tenant_id: str,
    credentials_base_url: str,
    credentials_token: str,
    module: str,
    user_id: str | None = None,
    user_name: str | None = None,
    record_limit: int | None = None,
    page_size: int | None = None,
    incremental: bool = True,
) -> None:
    rag_service = get_rag_service()
    if rag_service is None:
        logger.warning("Skipping ingest because RAG service is unavailable")
        return

    client = SugarClient(
        credentials_base_url,
        credentials_token,
        user_id=user_id,
        user_name=user_name,
    )
    try:
        ingested = await rag_service.ingest_from_crm(
            tenant_id=tenant_id,
            crm_client=client,
            module=module,
            limit=record_limit,
            page_size=page_size or 500,
            incremental=incremental,
        )
        logger.info(
            "RAG ingest complete tenant=%s module=%s records=%s",
            tenant_id,
            module,
            ingested,
        )
    except Exception:
        logger.exception("RAG ingest failed tenant=%s module=%s", tenant_id, module)


def _normalize_ingest_modules(modules: list[str] | None) -> list[str]:
    if not modules:
        return ["Contacts", "Accounts", "Meetings"]
    canonical_map = {
        "contacts": "Contacts",
        "accounts": "Accounts",
        "meetings": "Meetings",
        "calls": "Calls",
        "tasks": "Tasks",
        "notes": "Notes",
        "opportunities": "Opportunities",
        "leads": "Leads",
        "users": "Users",
        "cases": "Cases",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for module in modules:
        value = (module or "").strip()
        if not value:
            continue
        canonical = canonical_map.get(value.lower(), value)
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        normalized.append(canonical)
    return normalized or ["Contacts", "Accounts", "Meetings"]


@router.get("/", response_model=BaseResponse)
async def health_check() -> BaseResponse:
    return BaseResponse(success=True, response={"status": "ok"})


@router.get("/ping/", response_model=BaseResponse)
async def ping() -> BaseResponse:
    return BaseResponse(success=True, response={"status": "ok"})


@router.post("/chats/", response_model=CreateChatResponse)
async def create_chat(
    body: ChatCreate | None = None,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> CreateChatResponse:
    if body is not None and body.user_id != ctx["user_id"]:
        raise HTTPException(status_code=403, detail="user_id mismatch with auth header")
    chat = await _create_chat(db, tenant_id=ctx["tenant_id"], user_id=ctx["user_id"])
    return CreateChatResponse(chat_id=chat.id)


@router.get("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat(
    chat_id: str,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    chat = await _get_chat_or_404(
        db,
        chat_id=chat_id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    history = await _load_messages(
        db,
        chat_id=chat.id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    response_name = (chat.name or "").strip() or _fallback_chat_name(history)
    return ChatHistoryResponse(
        chat_id=chat.id,
        history=history,
        name=response_name,
        updated_at=chat.updated_at,
    )


@router.get("/chats/user/{user_id_in_path}", response_model=UserChatsResponse)
async def get_user_chats(
    user_id_in_path: str,
    limit: int | None = None,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> UserChatsResponse:
    if user_id_in_path != ctx["user_id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    stmt = (
        select(Chat)
        .where(Chat.tenant_id == ctx["tenant_id"], Chat.user_id == ctx["user_id"])
        .order_by(Chat.updated_at.desc())
    )
    result = await db.execute(stmt)
    chats = result.scalars().all()
    if limit is not None:
        chats = chats[: max(limit, 0)]

    items: list[ChatHistoryResponse] = []
    for chat in chats:
        history = await _load_messages(
            db,
            chat_id=chat.id,
            tenant_id=ctx["tenant_id"],
            user_id=ctx["user_id"],
        )
        response_name = (chat.name or "").strip() or _fallback_chat_name(history)
        items.append(
            ChatHistoryResponse(
                chat_id=chat.id,
                history=history,
                name=response_name,
                updated_at=chat.updated_at,
            )
        )

    return UserChatsResponse(user_id=ctx["user_id"], tenant=ctx["tenant_id"], chats=items)


@router.put("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def put_chat(
    chat_id: str,
    body: ChatReplaceRequest,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    if body.chat_id != chat_id:
        raise HTTPException(status_code=400, detail="chat_id mismatch")

    chat = await _get_chat_or_404(
        db,
        chat_id=chat_id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )

    await db.execute(
        delete(ChatMessage).where(
            ChatMessage.chat_id == chat.id,
            ChatMessage.tenant_id == ctx["tenant_id"],
            ChatMessage.user_id == ctx["user_id"],
        )
    )

    for item in body.history:
        await _append_message(
            db,
            chat=chat,
            tenant_id=ctx["tenant_id"],
            user_id=ctx["user_id"],
            role=item.role,
            content=item.content,
            metadata=item.metadata,
        )

    if body.name is not None:
        chat.name = body.name
    chat.updated_at = datetime.now(timezone.utc)

    await db.commit()

    history = await _load_messages(
        db,
        chat_id=chat.id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    return ChatHistoryResponse(
        chat_id=chat.id,
        history=history,
        name=chat.name,
        updated_at=chat.updated_at,
    )


@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    chat = await _get_chat_or_404(
        db,
        chat_id=chat_id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    await db.delete(chat)
    await db.commit()
    return {"ok": True}


@router.post("/process-input/", response_model=BaseResponse)
async def process_input(
    payload: ProcessInputRequest,
    background_tasks: BackgroundTasks,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    response = await _process_input_core(
        db=db,
        ctx=ctx,
        payload=payload,
        background_tasks=background_tasks,
    )

    if payload.context and payload.context.get("ingest_module"):
        module = str(payload.context["ingest_module"])
        tenant_manager = TenantManager(db)
        credentials = await tenant_manager.get_credentials(ctx["tenant_id"])
        background_tasks.add_task(
            _trigger_ingest,
            tenant_id=ctx["tenant_id"],
            credentials_base_url=credentials.crm_base_url,
            credentials_token=credentials.crm_token,
            module=module,
            user_id=ctx["user_id"],
        )

    return BaseResponse(success=True, response=response)


@router.post("/process-audio/", response_model=BaseResponse)
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chat_id: str | None = Form(None),
    context: str | None = Form(None),
    user_locale: str | None = Form("cs"),
    return_voice: bool = Form(False),
    x_chat_id: str | None = Header(default=None, alias="X-Chat-Id"),
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    effective_chat_id = x_chat_id or chat_id

    parsed_context: dict[str, Any] | None = None
    if context:
        normalized_context = context.strip()
        # Some frontends append raw objects into FormData and send "[object Object]".
        # Treat these placeholders as "no context" instead of failing the whole request.
        if normalized_context.lower() in {"null", "undefined", "[object object]"}:
            logger.warning(
                "Ignoring non-JSON context placeholder for /process-audio tenant=%s user=%s value=%s",
                ctx["tenant_id"],
                ctx["user_id"],
                normalized_context,
            )
        else:
            try:
                parsed_context = json.loads(normalized_context)
                if parsed_context is None:
                    parsed_context = None
                elif not isinstance(parsed_context, dict):
                    raise ValueError("context must be a JSON object")
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid context JSON: expected a JSON object string "
                        "(for example JSON.stringify(context)). "
                        f"Parser error: {exc}"
                    ),
                ) from exc

    suffix = Path(file.filename or "upload.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        payload = await file.read()
        temp.write(payload)
        temp_path = temp.name

    try:
        input_text = transcribe(temp_path, language=user_locale)
        process_payload = ProcessInputRequest(
            input_text=input_text,
            chat_id=effective_chat_id,
            context=parsed_context,
            return_voice=return_voice,
        )
        response = await _process_input_core(
            db=db,
            ctx=ctx,
            payload=process_payload,
            background_tasks=background_tasks,
        )
        response["transcription"] = input_text
        response["transcript"] = input_text
        return BaseResponse(success=True, response=response)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to remove temp file %s", temp_path)


@router.post("/search/", response_model=BaseResponse)
async def search(
    payload: SearchRequest,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(ctx["tenant_id"])
    crm_client = SugarClient(
        credentials.crm_base_url,
        credentials.crm_token,
        user_id=ctx["user_id"],
        user_name=ctx["user_name"],
    )

    query = payload.input_text.strip()
    scope = (payload.scope or "all").lower()
    rag_service = get_rag_service()

    if scope in {"contacts", "accounts", "meetings"}:
        crm = await crm_client.generic_search(query=query, scope=scope)
        rag: list[dict[str, Any]] = []
        if rag_service is not None:
            try:
                rag = rag_service.search(tenant_id=ctx["tenant_id"], query=query, limit=5)
            except Exception:
                logger.exception(
                    "RAG search failed tenant=%s scope=%s",
                    ctx["tenant_id"],
                    scope,
                )
        return BaseResponse(success=True, response={"scope": scope, "crm": crm, "rag": rag})

    contact_results = await crm_client.generic_search(query=query, scope="contacts")
    account_results = await crm_client.generic_search(query=query, scope="accounts")
    meeting_results = await crm_client.generic_search(query=query, scope="meetings")
    rag: list[dict[str, Any]] = []
    if rag_service is not None:
        try:
            rag = rag_service.search(tenant_id=ctx["tenant_id"], query=query, limit=5)
        except Exception:
            logger.exception(
                "RAG search failed tenant=%s scope=all",
                ctx["tenant_id"],
            )

    return BaseResponse(
        success=True,
        response={
            "scope": "all",
            "contacts": contact_results,
            "accounts": account_results,
            "meetings": meeting_results,
            "rag": rag,
        },
    )


@router.post("/rag/ingest/", response_model=BaseResponse)
async def rag_ingest(
    payload: RagIngestRequest,
    background_tasks: BackgroundTasks,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    rag_service = get_rag_service()
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    modules = _normalize_ingest_modules(payload.modules)
    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(ctx["tenant_id"])

    if payload.synchronous:
        client = SugarClient(
            credentials.crm_base_url,
            credentials.crm_token,
            user_id=ctx["user_id"],
            user_name=ctx["user_name"],
        )
        record_limit = payload.record_limit
        page_size = payload.page_size or 500
        results: dict[str, int] = {}
        for module in modules:
            try:
                count = await rag_service.ingest_from_crm(
                    tenant_id=ctx["tenant_id"],
                    crm_client=client,
                    module=module,
                    limit=record_limit,
                    page_size=page_size,
                    incremental=payload.incremental,
                )
                results[module] = count
            except Exception:
                logger.exception(
                    "Synchronous RAG ingest failed tenant=%s module=%s",
                    ctx["tenant_id"],
                    module,
                )
                results[module] = 0
        return BaseResponse(
            success=True,
            response={
                "ingest_impl": "paginated_v3_orderless_erroraware",
                "mode": "synchronous",
                "tenant_id": ctx["tenant_id"],
                "modules": results,
                "record_limit": record_limit,
                "page_size": page_size,
                "incremental": payload.incremental,
            },
        )

    page_size = payload.page_size or 500
    for module in modules:
        background_tasks.add_task(
            _trigger_ingest,
            tenant_id=ctx["tenant_id"],
            credentials_base_url=credentials.crm_base_url,
            credentials_token=credentials.crm_token,
            module=module,
            user_id=ctx["user_id"],
            user_name=ctx["user_name"],
            record_limit=payload.record_limit,
            page_size=page_size,
            incremental=payload.incremental,
        )
    return BaseResponse(
        success=True,
        response={
            "ingest_impl": "paginated_v3_orderless_erroraware",
            "mode": "background",
            "tenant_id": ctx["tenant_id"],
            "scheduled_modules": modules,
            "record_limit": payload.record_limit,
            "page_size": page_size,
            "incremental": payload.incremental,
        },
    )


@router.get("/rag/status/", response_model=BaseResponse)
async def rag_status(
    module: str | None = None,
    ctx: TenantContext = Depends(get_tenant_context),
) -> BaseResponse:
    rag_service = get_rag_service()
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    normalized_module = None
    if module:
        trimmed = module.strip()
        canonical_map = {
            "contacts": "Contacts",
            "accounts": "Accounts",
            "meetings": "Meetings",
            "calls": "Calls",
            "tasks": "Tasks",
            "notes": "Notes",
            "opportunities": "Opportunities",
            "leads": "Leads",
            "users": "Users",
            "cases": "Cases",
        }
        normalized_module = canonical_map.get(trimmed.lower(), trimmed) if trimmed else None
    count = rag_service.count(tenant_id=ctx["tenant_id"], module=normalized_module)
    return BaseResponse(
        success=True,
        response={
            "tenant_id": ctx["tenant_id"],
            "module": normalized_module,
            "vector_count": count,
        },
    )


@router.get("/audio/{file_id}")
async def get_audio(file_id: str) -> FileResponse:
    audio_path = settings.cache_audio_dir / f"{file_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav")
