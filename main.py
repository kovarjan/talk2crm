from __future__ import annotations

import os
import tempfile
import shutil
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.pipelines.command_pipeline import run_command_pipeline
from core.utils.chat import ChatSession
from fastapi import Header
from core.adapters.crm_api import (call_crm_api, process_crm_response)
from core.services.chat_store import (
    make_redis, create_chat, chat_exists, get_history, set_history, append_messages, delete_chat
)

from core.services.hybrid_search import search_contacts, search_accounts, search_meetings

app = FastAPI()

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

class ProcessInputPayload(BaseModel):
    input_text: Optional[str] = None
    chat_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None  # legacy support
    replace_last_user: bool = False
    context: Optional[Dict[str, Any]] = None
    tenant: Optional[str] = None
    user_id: Optional[str] = None

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

@app.post("/chats/", response_model=CreateChatResponse)
async def api_create_chat():
    r = make_redis()
    chat_id = await create_chat(r)
    return {"chat_id": chat_id}

@app.get("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def api_get_chat(chat_id: str):
    r = make_redis()
    if not await chat_exists(chat_id, r):
        raise HTTPException(404, "Chat not found")
    history = await get_history(chat_id, r)
    return {"chat_id": chat_id, "history": history}

@app.put("/chats/{chat_id}", response_model=ChatHistoryResponse)
async def api_put_chat(chat_id: str, body: ChatHistoryResponse):
    r = make_redis()
    if chat_id != body.chat_id:
        raise HTTPException(400, "chat_id mismatch")
    await set_history(chat_id, body.history, r)
    return {"chat_id": chat_id, "history": body.history}

@app.delete("/chats/{chat_id}")
async def api_delete_chat(chat_id: str):
    r = make_redis()
    await delete_chat(chat_id, r)
    return {"ok": True}

# ------------------------ Root (keep existing) ------------------------

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ping/")
def ping():
    return {"status": "ok"}

# ---------------------- Audio processing (opt) -----------------------

@app.post("/process-audio/", response_model=ProcessAudioResponse)
async def process_audio(
    file: UploadFile = File(...), 
    chat_id: Optional[str] = None, 
    x_chat_id: Optional[str] = Header(None)
):
    # Prefer chat_id from header if provided
    chat_id = x_chat_id or chat_id

    print(f"Received audio file: {file.filename}, chat_id: {chat_id}")

    # persist upload to tmp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        r = make_redis()
        # create or load chat
        if chat_id and await chat_exists(chat_id, r):
            print("LOADING existing chat history in Redis...")
            history_list = await get_history(chat_id, r)
        else:
            print("CREATING new chat...")
            chat_id = await create_chat(r)
            history_list = []

        chat_history = ChatSession(True)

        if history_list:
            chat_history.reset()
            chat_history.load_history(history_list)
            # TODO: add composing user prompts from history like in text input
            # input_text = chat_history.compose_following_user_message(input_text)

        tenant = "ai-local"  # TODO: make dynamic per user/client
        extra_context = {
            "tenant": tenant,
            "current_user_id": "28",  # TODO: dynamic
            "timezone": "Europe/Prague",
        }
        # pipeline expects (voice_path, raw_text, chat_history)
        response = run_command_pipeline(temp_path, None, chat_history, tenant, extra_context)

        # save back
        await set_history(chat_id, chat_history.get_messages(), r)

        return {
            "success": True,
            "response": response,
            "chat_id": chat_id,
            "chat_history": chat_history.get_messages()  # Debug only
        }
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

# ---------------------- Text processing (preferred) ----------------------

@app.post("/process-input/", response_model=ProcessInputResponse)
async def process_input(payload: ProcessInputPayload):
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

    if chat_id and await chat_exists(chat_id, r):
        print("LOADING existing chat history in Redis...")
        history_list = await get_history(chat_id, r)
    elif payload.chat_history:
        # Legacy path: create a chat and seed it
        print("Creating new chat with provided history (legacy path)...")
        chat_id = await create_chat(r, initial_history=payload.chat_history)
        history_list = payload.chat_history
    else:
        # Fresh chat
        print("Creating new chat (fresh start)...")
        chat_id = await create_chat(r)
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
    response = run_command_pipeline(None, input_text, chat_history, payload.tenant, extra_context)


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
    await set_history(chat_id, chat_history.get_messages(), r)

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
    