from __future__ import annotations

import asyncio
import json
import re
import tempfile
import uuid as _uuid
from collections.abc import AsyncGenerator
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
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import TenantContext, get_stream_token_context, get_tenant_context
from app.api.models import (
    BaseResponse,
    ChatCreate,
    ChatHistoryResponse,
    ChatReplaceRequest,
    CrmRecordCreatedEventRequest,
    CreateChatResponse,
    ExtractFieldsRequest,
    GenerateRequest,
    ProcessInputRequest,
    RagIngestRequest,
    RecommendActionsRequest,
    SearchRequest,
    UserChatsResponse,
)
from app.core.audio import transcribe, transcribe_segments
from app.core.config import get_settings
from app.core.logging import get_logger
from app.engine.events import StreamEvent
from app.engine.llm import get_chat_llm
from app.engine.pipeline import process_input_core
from app.engine.recommendations import recommend_actions
from app.engine.extraction import extract_fields
from langchain_core.messages import HumanMessage, SystemMessage
from app.engine.rag import get_rag_service
from app.domain.skill_contracts import (
    SkillIndexEntry,
    TenantOverlay,
    TenantOverlayCreate,
    UserOverlay,
    UserOverlayCreate,
)
from app.services import chat_service
from app.services.chat_titles import fallback_chat_name
from app.services.crm_client import CoripoClient
from app.services.skill_service import SkillService
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


@router.post("/generate/", response_model=BaseResponse)
async def generate(
    payload: GenerateRequest,
    ctx: TenantContext = Depends(get_tenant_context),
) -> BaseResponse:
    """Direct LLM call — no agent harness, no tools, no chat history.
    Intended for automation and simple text generation where the caller
    supplies all context in the prompts."""
    settings = get_settings()
    llm = get_chat_llm(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=payload.max_tokens or 2048,
    )
    messages: list = []
    if payload.system_prompt:
        messages.append(SystemMessage(content=payload.system_prompt))
    messages.append(HumanMessage(content=payload.prompt))

    result = await llm.ainvoke(messages)
    text = result.content if hasattr(result, "content") else str(result)

    logger.debug(
        "generate tenant=%s user=%s chars=%d",
        ctx["tenant_id"],
        ctx["user_id"],
        len(str(text)),
    )
    return BaseResponse(success=True, response={"text": str(text)})


@router.post("/recommend-actions/", response_model=BaseResponse)
async def recommend_actions_endpoint(
    payload: RecommendActionsRequest,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    """Read-only tool-using agent call. Looks up related CRM context (via
    CRM search/overview tools) to propose follow-up actions, but never
    writes to the CRM itself — the caller turns suggestions into normal
    create-command dry-run cards."""
    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(ctx["tenant_id"])
    crm_client = CoripoClient(
        credentials.crm_base_url,
        credentials.crm_token,
        user_id=ctx["user_id"],
        user_name=ctx["user_name"],
    )
    rag_service = get_rag_service()

    actions = await recommend_actions(
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
        module=payload.module,
        record_id=payload.record_id,
        text=payload.text,
        context=payload.context,
        crm_client=crm_client,
        rag_service=rag_service,
        db=db,
    )

    logger.debug(
        "recommend_actions tenant=%s user=%s module=%s record=%s count=%d",
        ctx["tenant_id"],
        ctx["user_id"],
        payload.module,
        payload.record_id,
        len(actions),
    )
    return BaseResponse(success=True, response={"actions": actions})


@router.post("/extract-fields/", response_model=BaseResponse)
async def extract_fields_endpoint(
    payload: ExtractFieldsRequest,
    ctx: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> BaseResponse:
    """Read-only tool-using agent call for Smart Paste. Extracts field values
    for the given module's live ai_schema from pasted/typed text, resolving
    relate fields via CRM/RAG tools where possible — never writes to the
    CRM. The caller (Coripo PHP/FE) applies the result into the currently
    open form; the user's own Save is what actually persists anything."""
    tenant_manager = TenantManager(db)
    credentials = await tenant_manager.get_credentials(ctx["tenant_id"])
    crm_client = CoripoClient(
        credentials.crm_base_url,
        credentials.crm_token,
        user_id=ctx["user_id"],
        user_name=ctx["user_name"],
    )
    rag_service = get_rag_service()

    result = await extract_fields(
        tenant_id=ctx["tenant_id"],
        user_id=ctx["user_id"],
        module=payload.module,
        record_id=payload.record_id,
        field_schema=payload.field_schema,
        current_values=payload.current_values,
        messages=payload.messages,
        crm_client=crm_client,
        rag_service=rag_service,
        db=db,
    )

    logger.debug(
        "extract_fields tenant=%s user=%s module=%s record=%s fields=%d",
        ctx["tenant_id"],
        ctx["user_id"],
        payload.module,
        payload.record_id,
        len(result.get("fields") or {}),
    )
    return BaseResponse(success=True, response=result)


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


@router.post("/process-input/stream")
async def process_input_stream(
    payload: ProcessInputRequest,
    background_tasks: BackgroundTasks,
    ctx: TenantContext = Depends(get_stream_token_context),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

    async def emit(event: StreamEvent) -> None:
        await queue.put(event)

    async def run_pipeline() -> None:
        try:
            await process_input_core(
                db=db,
                ctx=ctx,
                payload=payload,
                background_tasks=background_tasks,
                emit=emit,
            )
        except Exception:
            logger.exception(
                "Stream pipeline error tenant=%s user=%s",
                ctx["tenant_id"],
                ctx["user_id"],
            )
            await queue.put(StreamEvent(
                "error",
                {"status": 500, "message_to_user": "Nastala chyba při zpracování požadavku."},
            ))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_pipeline())

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                if event is None:
                    break
                yield event.to_sse()
                if event.type in ("result", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/process-audio/stream")
async def process_audio_stream(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chat_id: str | None = Form(None),
    context: str | None = Form(None),
    user_locale: str | None = Form("cs"),
    return_voice: bool = Form(False),
    x_chat_id: str | None = Header(default=None, alias="X-Chat-Id"),
    ctx: TenantContext = Depends(get_stream_token_context),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    effective_chat_id = x_chat_id or chat_id
    parsed_context: dict[str, Any] | None = None
    if context:
        normalized_context = context.strip()
        if normalized_context.lower() not in {"null", "undefined", "[object object]"}:
            try:
                parsed_context = json.loads(normalized_context)
                if not isinstance(parsed_context, dict):
                    raise ValueError("context must be a JSON object")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid context JSON: {exc}")

    suffix = Path(file.filename or "upload.webm").suffix or ".webm"
    raw_bytes = await file.read()

    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

    async def emit(event: StreamEvent) -> None:
        await queue.put(event)

    async def run_audio_pipeline() -> None:
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            full_text_parts: list[str] = []
            seg_index = 0
            loop = asyncio.get_event_loop()

            def iter_segments() -> None:
                nonlocal seg_index
                for seg_text in transcribe_segments(tmp_path, language=user_locale):
                    full_text_parts.append(seg_text)
                    asyncio.run_coroutine_threadsafe(
                        emit(StreamEvent(
                            "transcription.partial",
                            {"text": seg_text, "segment_index": seg_index},
                        )),
                        loop,
                    )
                    seg_index += 1

            await asyncio.to_thread(iter_segments)
            input_text = " ".join(full_text_parts).strip()
            await emit(StreamEvent("transcription.done", {"text": input_text}))

            process_payload = ProcessInputRequest(
                input_text=input_text,
                chat_id=effective_chat_id,
                context=parsed_context,
                return_voice=return_voice,
            )
            await process_input_core(
                db=db,
                ctx=ctx,
                payload=process_payload,
                background_tasks=background_tasks,
                emit=emit,
            )
        except Exception:
            logger.exception(
                "Stream audio pipeline error tenant=%s user=%s",
                ctx["tenant_id"],
                ctx["user_id"],
            )
            await queue.put(StreamEvent(
                "error",
                {"status": 500, "message_to_user": "Nastala chyba při zpracování audia."},
            ))
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            await queue.put(None)

    task = asyncio.create_task(run_audio_pipeline())

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                if event is None:
                    break
                yield event.to_sse()
                if event.type in ("result", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


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


# ---------------------------------------------------------------------------
# Skill management endpoints
# ---------------------------------------------------------------------------


def _require_skill_modification_enabled() -> None:
    if not settings.skills_modification_enabled:
        raise HTTPException(status_code=403, detail="Skill modification is disabled")


@router.get("/skills/", response_model=list[SkillIndexEntry])
async def list_skill_index(
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> list[SkillIndexEntry]:
    svc = SkillService(db, settings)
    return await svc.load_index(x_tenant, x_user_id)


@router.get("/skills/tenant/pending/", response_model=list[TenantOverlay])
async def list_pending_tenant_skills(
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> list[TenantOverlay]:
    svc = SkillService(db, settings)
    return await svc.list_pending_tenant_overlays(x_tenant)


@router.post("/skills/tenant/{name}/approve", response_model=TenantOverlay)
async def approve_tenant_skill(
    name: str,
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> TenantOverlay:
    _require_skill_modification_enabled()
    svc = SkillService(db, settings)
    overlay = await svc.approve_tenant_overlay(x_tenant, name)
    if overlay is None:
        raise HTTPException(status_code=404, detail="Skill overlay not found")
    return overlay


@router.post("/skills/tenant/", response_model=TenantOverlay)
async def upsert_tenant_skill(
    data: TenantOverlayCreate,
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> TenantOverlay:
    _require_skill_modification_enabled()
    data.created_by = x_user_id
    svc = SkillService(db, settings)
    overlay = await svc.upsert_tenant_overlay(x_tenant, data)
    if overlay is None:
        raise HTTPException(status_code=403, detail="Skill modification is disabled")
    return overlay


@router.delete("/skills/tenant/{name}")
async def deactivate_tenant_skill(
    name: str,
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    _require_skill_modification_enabled()
    svc = SkillService(db, settings)
    ok = await svc.deactivate_tenant_overlay(x_tenant, name)
    if not ok:
        raise HTTPException(status_code=404, detail="Skill overlay not found")
    return {"deleted": True}


@router.post("/skills/user/", response_model=UserOverlay)
async def upsert_user_skill(
    data: UserOverlayCreate,
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> UserOverlay:
    _require_skill_modification_enabled()
    svc = SkillService(db, settings)
    overlay = await svc.upsert_user_overlay(x_tenant, x_user_id, data)
    if overlay is None:
        raise HTTPException(status_code=403, detail="Skill modification is disabled")
    return overlay


@router.delete("/skills/user/{name}")
async def deactivate_user_skill(
    name: str,
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    _require_skill_modification_enabled()
    svc = SkillService(db, settings)
    ok = await svc.deactivate_user_overlay(x_tenant, x_user_id, name)
    if not ok:
        raise HTTPException(status_code=404, detail="Skill overlay not found")
    return {"deleted": True}


@router.get("/audio/{file_id}")
async def get_audio(file_id: str) -> FileResponse:
    if not re.fullmatch(r"[a-f0-9]{32}", file_id):
        raise HTTPException(status_code=400, detail="Invalid audio file ID")
    audio_path = settings.cache_audio_dir / f"{file_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav")
