from __future__ import annotations

import json
import os
import re
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import anyio

from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException, Form, Header
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import CACHE_DIR
from core.pipelines.command_pipeline import run_command_pipeline
from core.services.stt import transcribe_audio
from core.services.llm import query_llm
from core.utils.chat import ChatSession
from core.adapters.crm_api import (call_crm_api, process_crm_response)
from core.services.chat_store import (
    make_redis,
    create_chat,
    chat_exists,
    get_history,
    get_chat_meta,
    set_history,
    update_chat_meta,
    append_messages,
    delete_chat,
    get_user_chats,
)

from core.services.hybrid_search import search_contacts, search_accounts, search_meetings

app = FastAPI(
    title="Talk2API",
    description="API for chat, audio processing, and CRM/search actions.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/openapi.json", include_in_schema=False)
async def openapi_json():
    return app.openapi()

@app.get("/swagger", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Schemas -----------------------------

class CreateChatResponse(BaseModel):
    chat_id: str

class ChatHistoryResponse(BaseModel):
    chat_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    name: Optional[str] = None
    updated_at: Optional[str] = None

class UserChatsResponse(BaseModel):
    user_id: str
    tenant: str
    chats: List[ChatHistoryResponse] = Field(default_factory=list)

def _build_chat_title_prompt(
    history: List[Dict[str, Any]],
    max_messages: int = 5,
) -> Optional[str]:
    user_messages = [
        msg.get("content", "")
        for msg in history
        if msg.get("role") == "user" and msg.get("content")
    ]
    if not user_messages:
        return None
    tail = user_messages[-max_messages:]
    joined = "\n".join(f"- {msg}" for msg in tail)
    return (
        "Create a concise chat title (max 6 words). "
        "Use the user's language. Output only the title, no quotes.\n\n"
        f"User messages:\n{joined}"
    )

def _extract_message_json(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _filter_chat_history_for_user(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for msg in history:
        role = msg.get("role")
        if role == "user":
            filtered.append(msg)
            continue
        if role != "assistant":
            continue
        content = msg.get("content")
        data = _extract_message_json(content) if isinstance(content, str) else None
        if data and data.get("message_to_user"):
            filtered.append({"role": "assistant", "content": data["message_to_user"]})
    return filtered

def _get_last_user_message(history: List[Dict[str, Any]]) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            return str(msg.get("content") or "").strip()
    return ""

def _parse_updated_at(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed

async def _generate_chat_name(
    chat_id: str,
    tenant: Optional[str],
    user_id: Optional[str],
    history: List[Dict[str, Any]],
) -> None:
    if not tenant or not user_id:
        return
    r = make_redis()
    meta = await get_chat_meta(chat_id, r, tenant, user_id)
    message_count = len(history)
    last_user_message = _get_last_user_message(history)
    if meta:
        if (
            meta.get("message_count") == message_count
            and meta.get("last_user_message") == last_user_message
        ):
            return
    prompt = _build_chat_title_prompt(history)
    if not prompt:
        return
    chat_history = ChatSession(init=False)
    chat_history.add_system("You create short, specific chat titles. In czech language.")
    result = await anyio.to_thread.run_sync(
        lambda: query_llm(
            chat_history,
            prompt,
            temperature=0.2,
            add_history=False,
            returnJson=False,
            no_thinking=False,
        )
    )
    title = (result.get("text") or "").strip().strip("\"'")
    if not title:
        return
    title = title.splitlines()[0].strip()
    if len(title) > 80:
        title = title[:80].rstrip()
    await update_chat_meta(
        chat_id,
        {
            "name": title,
            "message_count": message_count,
            "last_user_message": last_user_message,
        },
        r,
        tenant,
        user_id,
    )

class ProcessInputPayload(BaseModel):
    input_text: Optional[str] = None
    chat_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None  # legacy support
    replace_last_user: bool = False
    context: Optional[Dict[str, Any]] = None
    tenant: Optional[str] = None
    user_id: Optional[str] = None
    return_voice: bool = False

class ProcessAudioResponse(BaseModel):
    success: bool
    response: Dict[str, Any]
    chat_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None

class ProcessInputResponse(BaseModel):
    success: bool
    response: Dict[str, Any]
    chat_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None  # for debugging/legacy
    context: Optional[Dict[str, Any]] = None

# --------------------------- Chat endpoints ---------------------------

# Generate a new chat ID
@app.post("/chats/", response_model=CreateChatResponse)
async def api_create_chat(tenant: Optional[str] = None, user_id: Optional[str] = None):
    r = make_redis()
    chat_id = await create_chat(r, tenant=tenant, user_id=user_id)
    return {"chat_id": chat_id}

# Get chat history by chat ID
@app.get("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def api_get_chat(chat_id: str, tenant: Optional[str] = None, user_id: Optional[str] = None):
    r = make_redis()
    if not await chat_exists(chat_id, r, tenant, user_id):
        raise HTTPException(404, "Chat not found")
    history = await get_history(chat_id, r, tenant, user_id)
    meta = await get_chat_meta(chat_id, r, tenant, user_id)
    updated_at = meta.get("updated_at")
    if updated_at is not None:
        updated_at = str(updated_at)
    return {
        "chat_id": chat_id,
        "history": history,
        "name": meta.get("name"),
        "updated_at": updated_at,
    }

@app.get("/chats/user/{user_id}", response_model=UserChatsResponse)
async def api_get_chats_by_user(
    user_id: str,
    tenant: str,
    limit: Optional[int] = None,
    debug: bool = False,
):
    r = make_redis()
    chat_ids = await get_user_chats(tenant, user_id, r)
    chats: List[ChatHistoryResponse] = []
    for chat_id in chat_ids:
        history = await get_history(chat_id, r, tenant, user_id)
        if not debug:
            history = _filter_chat_history_for_user(history)
        meta = await get_chat_meta(chat_id, r, tenant, user_id)
        updated_at = meta.get("updated_at")
        if updated_at is not None:
            updated_at = str(updated_at)
        chats.append(
            ChatHistoryResponse(
                chat_id=chat_id,
                history=history,
                name=meta.get("name"),
                updated_at=updated_at,
            )
        )
    chats.sort(key=lambda chat: _parse_updated_at(chat.updated_at), reverse=True)
    if limit is not None:
        chats = chats[: max(limit, 0)]
    return {"user_id": user_id, "tenant": tenant, "chats": chats}

# Update chat history by chat ID
@app.put("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def api_put_chat(
    chat_id: str,
    body: ChatHistoryResponse,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
):
    r = make_redis()
    if chat_id != body.chat_id:
        raise HTTPException(400, "chat_id mismatch")
    await set_history(chat_id, body.history, r, tenant, user_id)
    if background_tasks:
        background_tasks.add_task(
            _generate_chat_name,
            chat_id,
            tenant,
            user_id,
            body.history,
        )
    meta = await get_chat_meta(chat_id, r, tenant, user_id)
    updated_at = meta.get("updated_at")
    if updated_at is not None:
        updated_at = str(updated_at)
    return {
        "chat_id": chat_id,
        "history": body.history,
        "name": meta.get("name"),
        "updated_at": updated_at,
    }

# Delete chat by chat ID
@app.delete("/chats/{chat_id}")
async def api_delete_chat(chat_id: str, tenant: Optional[str] = None, user_id: Optional[str] = None):
    r = make_redis()
    await delete_chat(chat_id, r, tenant, user_id)
    return {"ok": True}

# ------------------------ Root (keep existing) ------------------------

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ping/")
def ping():
    return {"status": "ok"}

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    """
    Retrieves a cached audio file by its ID.
    """
    audio_path = os.path.join(CACHE_DIR, "audio", f"{file_id}.wav")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(audio_path, media_type="audio/wav")

# ---------------------- Audio processing (opt) -----------------------

@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...), 
    chat_id: Optional[str] = None, 
    x_chat_id: Optional[str] = Header(None),
    tenant: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    user_locale: Optional[str] = Form("cs-CZ"),
    return_voice: Optional[bool] = Form(False),
    background_tasks: BackgroundTasks = None,
):
    # Prefer chat_id from header if provided
    chat_id = x_chat_id or chat_id


    print(f"Received audio file: {file.filename}, chat_id: {chat_id}")
    print(f"Tenant: {tenant}, User ID: {user_id}, Locale: {user_locale}, Return Voice: {return_voice}, Context: {context}")

    # persist upload to tmp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        r = make_redis()
        tenant = tenant or "ai-local"  # TODO: make dynamic per user/client
        user_id = user_id or "28"  # TODO: dynamic

        # create or load chat
        if chat_id and await chat_exists(chat_id, r, tenant, user_id):
            print("LOADING existing chat history in Redis...")
            history_list = await get_history(chat_id, r, tenant, user_id)
        else:
            print("CREATING new chat...")
            chat_id = await create_chat(r, tenant=tenant, user_id=user_id)
            history_list = []

        chat_history = ChatSession(True)

        if history_list:
            chat_history.reset()
            chat_history.load_history(history_list)

        context_obj = None
        if context:
            try:
                context_obj = json.loads(context)
            except Exception:
                print("Invalid context payload for audio request, ignoring.")

        if context_obj:
            chat_history.inject_context(context_obj)

        input_text = transcribe_audio(temp_path, language=user_locale) or ""
        if history_list:
            input_text = chat_history.compose_following_user_message(input_text)

        if not tenant or tenant == "none":
            return {
                "success": False,
                "response": {"action": "question", "message_to_user": "Omlouvám se, ale nemohu pokračovat bez platného tenantu vaší instance.\nKontaktujte administrátora aby vám povolil využívání AI funkcí."},
                "chat_id": chat_id,
                "chat_history": chat_history.get_messages(),
            }

        if not user_id or user_id == "none":
            return {
                "success": False,
                "response": {"action": "question", "message_to_user": "Omlouvám se, ale nemohu pokračovat bez platného uživatele."},
                "chat_id": chat_id,
                "chat_history": chat_history.get_messages(),
            }

        extra_context = {
            "tenant": tenant,
            "current_user_id": user_id,
            "timezone": "Europe/Prague",
        }
        # pipeline expects (voice_path, raw_text, chat_history)
        response = run_command_pipeline(None, input_text, chat_history, tenant, extra_context, return_voice=return_voice)

        crm_response = None
        if response.get("action") in ("create", "update", "delete"):
            try:
                print("Skip creating debug disabled params:", response)
                # print("Calling CRM API with:", response)
                # crm_response = call_crm_api(response)
                # print("CRM API response:", crm_response)
            except Exception as e:
                print("Error calling CRM API:", str(e))
                crm_response = {"error": str(e)}

            if crm_response:
                chat_history.add_assistant(f"CRM API response: \n{process_crm_response(crm_response)}")

        # save back
        await set_history(chat_id, chat_history.get_messages(), r, tenant, user_id)
        if background_tasks:
            background_tasks.add_task(
                _generate_chat_name,
                chat_id,
                tenant,
                user_id,
                chat_history.get_messages(),
            )

        return {
            "success": True,
            "response": response,
            "chat_id": chat_id,
            "chat_history": chat_history.get_messages(),  # Debug only
            "crm_response": crm_response,  # Debug only
        }
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

# ---------------------- Text processing (preferred) ----------------------

@app.post("/process-input/", response_model=ProcessInputResponse)
async def process_input(payload: ProcessInputPayload, background_tasks: BackgroundTasks = None):
    """
    Preferred usage:
      FE sends: { "chat_id": "...", "input_text": "Naplánuj..." }
    Legacy still supported:
      FE sends: { "chat_history": [...], "input_text": "..." }
    """
    r = make_redis()

    # Resolve chat_id and existing history
    chat_id = payload.chat_id
    print(">>> chat_id: ", chat_id)

    print(">>> payload: ")
    print(payload)

    if chat_id and await chat_exists(chat_id, r, payload.tenant, payload.user_id):
        print("LOADING existing chat history in Redis...")
        history_list = await get_history(chat_id, r, payload.tenant, payload.user_id)
    elif payload.chat_history:
        # Legacy path: create a chat and seed it
        print("Creating new chat with provided history (legacy path)...")
        chat_id = await create_chat(
            r,
            initial_history=payload.chat_history,
            tenant=payload.tenant,
            user_id=payload.user_id,
        )
        history_list = payload.chat_history
    else:
        # Fresh chat
        print("Creating new chat (fresh start)...")
        chat_id = await create_chat(r, tenant=payload.tenant, user_id=payload.user_id)
        history_list = []

    # Build ChatSession
    chat_history = ChatSession(True)
    
    # Compose following user message if we already have history
    input_text = payload.input_text or ""
    if history_list:
        chat_history.reset()
        chat_history.load_history(history_list)
        input_text = chat_history.compose_following_user_message(input_text)

    # Add context to chat history
    if payload.context:
        chat_history.inject_context(payload.context)


    if not payload.tenant or payload.tenant == 'none':
        return {
            "success": False,
            "response": {"action": "question", "message_to_user": "Omlouvám se, ale nemohu pokračovat bez platného tenantu vaší instance.\nKontaktujte administrátora aby vám povolil využívání AI funkcí."},
            "chat_id": chat_id,
            "chat_history": chat_history.get_messages(),
        }

    if not payload.user_id or payload.user_id == 'none':
        return {
            "success": False,
            "response": {"action": "question", "message_to_user": "Omlouvám se, ale nemohu pokračovat bez platného uživatele."},
            "chat_id": chat_id,
            "chat_history": chat_history.get_messages(),
        }

    # tenant = "ai-local"  # TODO: make dynamic per user/client
    extra_context = {
        "tenant": payload.tenant,
        "current_user_id": payload.user_id,
        "timezone": "Europe/Prague",
    }
    # Run pipeline
    response = run_command_pipeline(None, input_text, chat_history, payload.tenant, extra_context, return_voice=payload.return_voice)


    crm_response = None

    # call to coripo API 
    if response.get("action") == "create" or response.get("action") == "update" or response.get("action") == "delete":
        try:
            print("Skip creating debug disabled params:", response)
            # print("Calling CRM API with:", response)
            # crm_response = call_crm_api(response)
            # print("CRM API response:", crm_response)
        except Exception as e:
            print("Error calling CRM API:", str(e))
            crm_response = {"error": str(e)}

        # add to chat history
        if crm_response:
            chat_history.add_assistant(f"CRM API response: \n{process_crm_response(crm_response)}")

    # Persist history
    await set_history(chat_id, chat_history.get_messages(), r, payload.tenant, payload.user_id)
    if background_tasks:
        background_tasks.add_task(
            _generate_chat_name,
            chat_id,
            payload.tenant,
            payload.user_id,
            chat_history.get_messages(),
        )

    print("-----------------RECAP-----------------")
    chat_history.pretty_print()

    # Return lean payload (no bulky history needed for FE unless you want it)
    return {
        "success": True, 
        "response": response, 
        "chat_id": chat_id,
        "chat_history": chat_history.get_messages(),  # Debug only
        "crm_response": crm_response  # Debug only
    }

# ---------------------- Search ----------------------

@app.post("/search/", response_model=ProcessInputResponse)
def search(payload: ProcessInputPayload):
    """
    Simple search endpoint for testing.
    """
    print(">>> payload: ")
    print(payload)


    if not payload.tenant or payload.tenant == 'none':
        return {"success": False, "response": {"action": "question", "message_to_user": "Omlouvám se, ale nemohu pokračovat bez platného tenantu vaší instance.\nKontaktujte administrátora aby vám povolil využívání AI funkcí."}}

    if not payload.user_id or payload.user_id == 'none':
        return {"success": False, "response": {"action": "question", "message_to_user": "Omlouvám se, ale nemohu pokračovat bez platného uživatele."}}

    tenant = payload.tenant.strip()
    query = (payload.input_text or "").strip()
    scope = (getattr(payload, "scope", "") or "").lower()  # optional: "contacts"|"accounts"|"meetings"|"all"
    top_k = getattr(payload, "top_k", 5) or 5

    try:
        if scope == "contacts":
            items = search_contacts(query, tenant=tenant, top_k=top_k)
            return {"success": True, "response": {"items": items}}
        elif scope == "accounts":
            items = search_accounts(query, tenant=tenant, top_k=top_k)
            return {"success": True, "response": {"items": items}}
        elif scope == "meetings":
            items = search_meetings(query, tenant=tenant, top_k=top_k)
            return {"success": True, "response": {"items": items}}
        else:
            return {
                "success": True,
                "response": {
                    "contacts": search_contacts(query, tenant=tenant, top_k=top_k, overview_only=True),
                    "accounts": search_accounts(query, tenant=tenant, top_k=top_k, overview_only=True),
                    "meetings": search_meetings(query, tenant=tenant, top_k=top_k, overview_only=True),
                },
            }
    except FileNotFoundError as e:
        return JSONResponse(content={"success": False, "response": {"action": "error", "message_to_user": f"Chybí index pro tento tenant ({tenant}). {e}"}})
    except Exception as e:
        return JSONResponse(content={"success": False, "response": {"action": "error", "message_to_user": f"Nepodařilo se provést vyhledávání: {e}"}})
    
