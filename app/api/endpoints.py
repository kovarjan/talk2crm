from __future__ import annotations

import json
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.core.logging import get_logger
from app.engine.agent import run_agent
from app.engine.quick_actions import try_handle_quick_action
from app.engine.rag import TenantRAGService
from app.engine.tools import build_tools
from app.services.crm_client import SugarClient
from app.services.tenant_manager import TenantManager
from database.models import Chat, ChatMessage
from database.session import get_db


logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()
_rag_service_instance: TenantRAGService | None = None
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


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
    return ChatMessageItem(
        role=message.role,
        content=message.content,
        metadata=message.metadata_json or {},
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
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
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


async def _create_chat(db: AsyncSession, *, tenant_id: str, user_id: str) -> Chat:
    chat = Chat(
        id=uuid.uuid4().hex,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
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
        synthesize_to_file,
        text,
        str(output_path),
        settings.tts_voice,
    )
    return file_id, f"/audio/{file_id}"


def _to_user_message(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text or "").strip()
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
        if status == "confirmation_required":
            pending = obs.get("pending_action") if isinstance(obs.get("pending_action"), dict) else {}
            module = str(pending.get("module") or "").strip()
            action = str(pending.get("action") or "").strip()
            data = pending.get("data") if isinstance(pending.get("data"), dict) else {}
            message = str(
                obs.get("message")
                or "Akce je připravena a čeká na vaše potvrzení."
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

    output_text = str(normalized.get("output") or "")
    parsed_output = _try_parse_json(output_text)
    if isinstance(parsed_output, dict) and parsed_output.get("message_to_user"):
        normalized["message_to_user"] = str(parsed_output["message_to_user"])
        return normalized

    clean = _to_user_message(output_text)
    if _looks_like_noise(clean):
        clean = (
            "Nerozuměl jsem spolehlivě požadavku. "
            "Upřesněte prosím akci, modul a čas (např. schůzka v úterý 9:30)."
        )
    normalized["message_to_user"] = clean
    return normalized


async def _process_input_core(
    *,
    db: AsyncSession,
    ctx: TenantContext,
    payload: ProcessInputRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    tenant_id = ctx["tenant_id"]
    user_id = ctx["user_id"]

    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(tenant_id)
    crm_client = SugarClient(
        credentials.crm_base_url,
        credentials.crm_token,
        user_id=user_id,
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
        chat = await _create_chat(db, tenant_id=tenant_id, user_id=user_id)

    await _append_message(
        db,
        chat=chat,
        tenant_id=tenant_id,
        user_id=user_id,
        role="user",
        content=payload.input_text,
    )

    action_confirmation = bool((payload.context or {}).get("confirm_action", False))

    quick_result = await try_handle_quick_action(
        input_text=payload.input_text,
        crm_client=crm_client,
        user_id=user_id,
        action_confirmation=action_confirmation,
    )
    if quick_result is not None:
        agent_result = quick_result
    else:
        tools = build_tools(
            tenant_id=tenant_id,
            user_id=user_id,
            input_text=payload.input_text,
            request_context=payload.context,
            crm_client=crm_client,
            rag_service=rag_service,
            action_confirmation=action_confirmation,
        )
        try:
            agent_result = await run_agent(
                tenant_id=tenant_id,
                user_id=user_id,
                input_text=payload.input_text,
                context=payload.context,
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
        or agent_result
    )
    assistant_text = _to_user_message(assistant_raw_text)
    await _append_message(
        db,
        chat=chat,
        tenant_id=tenant_id,
        user_id=user_id,
        role="assistant",
        content=assistant_text,
        metadata={"agent_result": agent_result, "raw_output": assistant_raw_text},
    )

    await db.commit()

    history = await _load_messages(
        db,
        chat_id=chat.id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    file_id, audio_url = await _maybe_generate_voice(
        text=assistant_text,
        return_voice=payload.return_voice,
        background_tasks=background_tasks,
    )

    return {
        "action_result": agent_result,
        "message_to_user": assistant_text,
        "chat_id": chat.id,
        "chat_history": [item.model_dump(mode="json") for item in history],
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
    record_limit: int | None = None,
    page_size: int | None = None,
) -> None:
    rag_service = get_rag_service()
    if rag_service is None:
        logger.warning("Skipping ingest because RAG service is unavailable")
        return

    client = SugarClient(
        credentials_base_url,
        credentials_token,
        user_id=user_id,
    )
    try:
        ingested = await rag_service.ingest_from_crm(
            tenant_id=tenant_id,
            crm_client=client,
            module=module,
            limit=record_limit,
            page_size=page_size or 500,
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
    return ChatHistoryResponse(
        chat_id=chat.id,
        history=history,
        name=chat.name,
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
        items.append(
            ChatHistoryResponse(
                chat_id=chat.id,
                history=history,
                name=chat.name,
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
    user_locale: str | None = Form("en"),
    return_voice: bool = Form(False),
    x_chat_id: str | None = Header(default=None, alias="X-Chat-Id"),
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    effective_chat_id = x_chat_id or chat_id

    parsed_context: dict[str, Any] | None = None
    if context:
        try:
            parsed_context = json.loads(context)
            if not isinstance(parsed_context, dict):
                raise ValueError("context must be a JSON object")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid context JSON: {exc}") from exc

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
                "mode": "synchronous",
                "tenant_id": ctx["tenant_id"],
                "modules": results,
                "record_limit": record_limit,
                "page_size": page_size,
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
            record_limit=payload.record_limit,
            page_size=page_size,
        )
    return BaseResponse(
        success=True,
        response={
            "mode": "background",
            "tenant_id": ctx["tenant_id"],
            "scheduled_modules": modules,
            "record_limit": payload.record_limit,
            "page_size": page_size,
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
