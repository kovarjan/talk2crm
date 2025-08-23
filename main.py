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
from core.services.chat_store import (
    make_redis, create_chat, chat_exists, get_history, set_history, append_messages, delete_chat
)

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

class ProcessAudioResponse(BaseModel):
    success: bool
    response: Dict[str, Any]
    chat_id: Optional[str] = None

class ProcessInputResponse(BaseModel):
    success: bool
    response: Dict[str, Any]
    chat_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None  # for debugging/legacy

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

# ---------------------- Audio processing (opt) -----------------------

@app.post("/process-audio/", response_model=ProcessAudioResponse)
async def process_audio(file: UploadFile = File(...), chat_id: Optional[str] = None):
    # persist upload to tmp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        r = make_redis()
        # create or load chat
        if chat_id and await chat_exists(chat_id, r):
            history_list = await get_history(chat_id, r)
        else:
            chat_id = await create_chat(r)
            history_list = []

        chat_history = ChatSession(True)
        chat_history.load_history(history_list)

        # pipeline expects (voice_path, raw_text, chat_history)
        response = run_command_pipeline(temp_path, None, chat_history)

        # save back
        await set_history(chat_id, chat_history.get_messages(), r)

        return {"success": True, "response": response, "chat_id": chat_id}
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

    # Run pipeline
    response = run_command_pipeline(None, input_text, chat_history)

    # Persist history
    await set_history(chat_id, chat_history.get_messages(), r)

    # Return lean payload (no bulky history needed for FE unless you want it)
    return {
        "success": True, 
        "response": response, 
        "chat_id": chat_id,
        "chat_history": chat_history.get_messages()  # Debug only
    }
