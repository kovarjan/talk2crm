from __future__ import annotations

import asyncio
import json
import re
import tempfile
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
    ChatReplaceRequest,
    CrmRecordCreatedEventRequest,
    CreateChatResponse,
    ProcessInputRequest,
    RagIngestRequest,
    SearchRequest,
    UserChatsResponse,
)
from app.core.audio import transcribe
from app.core.config import get_settings
from app.core.logging import get_logger
from app.engine.pipeline import process_input_core
from app.engine.rag import get_rag_service
from app.services import chat_service
from app.services.chat_titles import fallback_chat_name
from app.services.crm_client import CoripoClient
from app.services.tenant_manager import TenantManager
from app.utils.modules import canonical_module_name
from database.models import Chat, ChatMessage
from database.session import get_db


logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()


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

    client = CoripoClient(
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
    normalized: list[str] = []
    seen: set[str] = set()
    for module in modules:
        value = (module or "").strip()
        if not value:
            continue
        canonical = canonical_module_name(value) or value
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
    chat = await chat_service.create_chat(db, tenant_id=ctx["tenant_id"], user_id=ctx["user_id"])
    return CreateChatResponse(chat_id=chat.id)


@router.get("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat(
    chat_id: str,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    chat = await chat_service.get_chat_or_404(
        db,
        chat_id=chat_id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    history = await chat_service.load_messages(
        db,
        chat_id=chat.id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    response_name = (chat.name or "").strip() or fallback_chat_name(history)
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

    # Cap at 50 by default; callers that need more must pass an explicit limit.
    # NOTE: full message history is still loaded per chat (N+1). Acceptable while
    # the per-user chat count is bounded by this SQL-level limit. Move to a
    # joined/batched load if this becomes a bottleneck at scale.
    effective_limit = max(1, min(int(limit), 200)) if limit is not None else 50

    stmt = (
        select(Chat)
        .where(Chat.tenant_id == ctx["tenant_id"], Chat.user_id == ctx["user_id"])
        .order_by(Chat.updated_at.desc())
        .limit(effective_limit)
    )
    result = await db.execute(stmt)
    chats = result.scalars().all()

    items: list[ChatHistoryResponse] = []
    for chat in chats:
        history = await chat_service.load_messages(
            db,
            chat_id=chat.id,
            tenant_id=ctx["tenant_id"],
            user_id=ctx["user_id"],
        )
        response_name = (chat.name or "").strip() or fallback_chat_name(history)
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

    chat = await chat_service.get_chat_or_404(
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
        await chat_service.append_message(
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

    history = await chat_service.load_messages(
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
    chat = await chat_service.get_chat_or_404(
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
    response = await process_input_core(
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


@router.post("/crm/events/record-created/", response_model=BaseResponse)
async def crm_record_created_event(
    payload: CrmRecordCreatedEventRequest,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    record_id = str(payload.record_id or "").strip()
    if not record_id:
        raise HTTPException(status_code=400, detail="record_id is required")
    module = canonical_module_name(payload.module)
    if not module:
        raise HTTPException(status_code=400, detail="module is required")

    chat = await chat_service.get_chat_or_404(
        db,
        chat_id=payload.chat_id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    latest_event = await chat_service.load_latest_crm_record_created_event(
        db,
        chat_id=chat.id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    if (
        latest_event
        and latest_event.get("module", "").lower() == module.lower()
        and latest_event.get("record_id") == record_id
    ):
        return BaseResponse(
            success=True,
            response={
                "chat_id": chat.id,
                "module": module,
                "record_id": record_id,
                "record_name": payload.record_name,
                "deduplicated": True,
                "pending_action_cleared": False,
            },
        )

    latest_pending_action = await chat_service.load_latest_pending_action(
        db,
        chat_id=chat.id,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
    )
    pending_action = str(
        (latest_pending_action or {}).get("action") if isinstance(latest_pending_action, dict) else ""
    ).strip().lower()
    pending_module = canonical_module_name(
        (latest_pending_action or {}).get("effective_module")
        or (latest_pending_action or {}).get("module")
        if isinstance(latest_pending_action, dict)
        else ""
    )
    pending_create_resolved = pending_action == "create" and (
        not pending_module or pending_module.lower() == module.lower()
    )

    message = (payload.user_message or "").strip()
    record_name = str(payload.record_name or "").strip()
    if not message:
        if record_name:
            message = (
                f"Potvrzeno z CRM: záznam '{record_name}' ({module}) byl vytvořen. "
                f"CRM ID: {record_id}."
            )
        else:
            message = f"Potvrzeno z CRM: záznam v modulu {module} byl vytvořen. CRM ID: {record_id}."
        if pending_create_resolved:
            message += " Další úpravy provedu přímo nad tímto CRM ID."

    source = str(payload.source or "crm").strip() or "crm"
    crm_sync_event: dict[str, Any] = {
        "type": "record_created",
        "module": module,
        "record_id": record_id,
        "source": source,
    }
    if record_name:
        crm_sync_event["record_name"] = record_name

    metadata: dict[str, Any] = {
        "crm_sync_event": crm_sync_event,
        "agent_result": {
            "status": "crm_sync_record_created",
            "message_to_user": message,
            "output": message,
            "crm_sync_event": crm_sync_event,
        },
    }
    if isinstance(latest_pending_action, dict):
        metadata["resolved_pending_action"] = latest_pending_action

    await chat_service.append_message(
        db,
        chat=chat,
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
        role="assistant",
        content=message,
        metadata=metadata,
    )
    await db.commit()

    return BaseResponse(
        success=True,
        response={
            "chat_id": chat.id,
            "module": module,
            "record_id": record_id,
            "record_name": record_name or None,
            "source": source,
            "pending_action_cleared": pending_create_resolved,
            "deduplicated": False,
        },
    )


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
        # faster-whisper is CPU-bound and synchronous; keep it off the event loop.
        input_text = await asyncio.to_thread(transcribe, temp_path, language=user_locale)
        process_payload = ProcessInputRequest(
            input_text=input_text,
            chat_id=effective_chat_id,
            context=parsed_context,
            return_voice=return_voice,
        )
        response = await process_input_core(
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
    crm_client = CoripoClient(
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
                rag = await rag_service.search(tenant_id=ctx["tenant_id"], query=query, limit=5)
            except Exception:
                logger.exception(
                    "RAG search failed tenant=%s scope=%s",
                    ctx["tenant_id"],
                    scope,
                )
        return BaseResponse(success=True, response={"scope": scope, "crm": crm, "rag": rag})

    contact_results, account_results, meeting_results = await asyncio.gather(
        crm_client.generic_search(query=query, scope="contacts"),
        crm_client.generic_search(query=query, scope="accounts"),
        crm_client.generic_search(query=query, scope="meetings"),
    )
    rag: list[dict[str, Any]] = []
    if rag_service is not None:
        try:
            rag = await rag_service.search(tenant_id=ctx["tenant_id"], query=query, limit=5)
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
        client = CoripoClient(
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
        normalized_module = canonical_module_name(trimmed) if trimmed else None
    count = await rag_service.count(tenant_id=ctx["tenant_id"], module=normalized_module)
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
    if not re.fullmatch(r"[a-f0-9]{32}", file_id):
        raise HTTPException(status_code=400, detail="Invalid audio file ID")
    audio_path = settings.cache_audio_dir / f"{file_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav")
