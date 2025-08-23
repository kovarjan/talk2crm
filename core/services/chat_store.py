# core/services/chat_store.py
from __future__ import annotations

import os
import json
import uuid
from typing import List, Dict, Any, Optional

from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHAT_KEY_PREFIX = os.getenv("CHAT_KEY_PREFIX", "chat:")
CHAT_TTL_SECONDS = int(os.getenv("CHAT_TTL_SECONDS", "604800"))  # 7 days

def _key(chat_id: str) -> str:
    return f"{CHAT_KEY_PREFIX}{chat_id}"

def make_redis() -> Redis:
    return Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

async def create_chat(r: Optional[Redis] = None, initial_history: Optional[List[Dict[str, Any]]] = None) -> str:
    if r is None:
        r = make_redis()
    chat_id = str(uuid.uuid4())
    history = initial_history or []
    await r.set(_key(chat_id), json.dumps(history, ensure_ascii=False))
    await r.expire(_key(chat_id), CHAT_TTL_SECONDS)
    return chat_id

async def chat_exists(chat_id: str, r: Optional[Redis] = None) -> bool:
    if r is None:
        r = make_redis()
    return bool(await r.exists(_key(chat_id)))

async def get_history(chat_id: str, r: Optional[Redis] = None) -> List[Dict[str, Any]]:
    if r is None:
        r = make_redis()
    raw = await r.get(_key(chat_id))
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []

async def set_history(chat_id: str, history: List[Dict[str, Any]], r: Optional[Redis] = None) -> None:
    if r is None:
        r = make_redis()
    await r.set(_key(chat_id), json.dumps(history, ensure_ascii=False))
    await r.expire(_key(chat_id), CHAT_TTL_SECONDS)

async def append_messages(chat_id: str, messages: List[Dict[str, Any]], r: Optional[Redis] = None) -> None:
    history = await get_history(chat_id, r)
    history.extend(messages)
    await set_history(chat_id, history, r)

async def delete_chat(chat_id: str, r: Optional[Redis] = None) -> None:
    if r is None:
        r = make_redis()
    await r.delete(_key(chat_id))
