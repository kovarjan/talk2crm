# core/services/chat_store.py
from __future__ import annotations

import os
import json
import uuid
from typing import List, Dict, Any, Optional

from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHAT_KEY_PREFIX = os.getenv("CHAT_KEY_PREFIX", "chat:")
CHAT_USER_INDEX_PREFIX = os.getenv("CHAT_USER_INDEX_PREFIX", "chat_user:")
CHAT_META_PREFIX = os.getenv("CHAT_META_PREFIX", "chat_meta:")
CHAT_TTL_SECONDS = int(os.getenv("CHAT_TTL_SECONDS", "604800"))  # 7 days

def _key(chat_id: str, tenant: Optional[str] = None, user_id: Optional[str] = None) -> str:
    if tenant and user_id:
        return f"{CHAT_KEY_PREFIX}{tenant}:{user_id}:{chat_id}"
    return f"{CHAT_KEY_PREFIX}{chat_id}"

def _user_index_key(tenant: str, user_id: str) -> str:
    return f"{CHAT_USER_INDEX_PREFIX}{tenant}:{user_id}"

def _meta_key(chat_id: str, tenant: Optional[str] = None, user_id: Optional[str] = None) -> str:
    if tenant and user_id:
        return f"{CHAT_META_PREFIX}{tenant}:{user_id}:{chat_id}"
    return f"{CHAT_META_PREFIX}{chat_id}"

async def _add_to_user_index(
    chat_id: str,
    tenant: Optional[str],
    user_id: Optional[str],
    r: Redis,
) -> None:
    if not tenant or not user_id:
        return
    index_key = _user_index_key(tenant, user_id)
    raw = await r.get(index_key)
    try:
        chat_ids = json.loads(raw) if raw else []
    except Exception:
        chat_ids = []
    if chat_id in chat_ids:
        return
    chat_ids.insert(0, chat_id)
    await r.set(index_key, json.dumps(chat_ids, ensure_ascii=False))
    await r.expire(index_key, CHAT_TTL_SECONDS)

async def _remove_from_user_index(
    chat_id: str,
    tenant: Optional[str],
    user_id: Optional[str],
    r: Redis,
) -> None:
    if not tenant or not user_id:
        return
    index_key = _user_index_key(tenant, user_id)
    raw = await r.get(index_key)
    if not raw:
        return
    try:
        chat_ids = json.loads(raw)
    except Exception:
        return
    if chat_id not in chat_ids:
        return
    chat_ids = [cid for cid in chat_ids if cid != chat_id]
    await r.set(index_key, json.dumps(chat_ids, ensure_ascii=False))
    await r.expire(index_key, CHAT_TTL_SECONDS)

def make_redis() -> Redis:
    return Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

async def create_chat(
    r: Optional[Redis] = None,
    initial_history: Optional[List[Dict[str, Any]]] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    if r is None:
        r = make_redis()
    chat_id = str(uuid.uuid4())
    history = initial_history or []
    await r.set(_key(chat_id, tenant, user_id), json.dumps(history, ensure_ascii=False))
    await r.expire(_key(chat_id, tenant, user_id), CHAT_TTL_SECONDS)
    await _add_to_user_index(chat_id, tenant, user_id, r)
    return chat_id

async def chat_exists(
    chat_id: str,
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> bool:
    if r is None:
        r = make_redis()
    return bool(await r.exists(_key(chat_id, tenant, user_id)))

async def get_history(
    chat_id: str,
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if r is None:
        r = make_redis()
    raw = await r.get(_key(chat_id, tenant, user_id))
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []

async def set_history(
    chat_id: str,
    history: List[Dict[str, Any]],
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    if r is None:
        r = make_redis()
    await r.set(_key(chat_id, tenant, user_id), json.dumps(history, ensure_ascii=False))
    await r.expire(_key(chat_id, tenant, user_id), CHAT_TTL_SECONDS)
    await _add_to_user_index(chat_id, tenant, user_id, r)

async def set_chat_meta(
    chat_id: str,
    meta: Dict[str, Any],
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    if r is None:
        r = make_redis()
    if not meta:
        return
    await r.set(_meta_key(chat_id, tenant, user_id), json.dumps(meta, ensure_ascii=False))
    await r.expire(_meta_key(chat_id, tenant, user_id), CHAT_TTL_SECONDS)

async def get_chat_meta(
    chat_id: str,
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if r is None:
        r = make_redis()
    raw = await r.get(_meta_key(chat_id, tenant, user_id))
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
        return meta if isinstance(meta, dict) else {"name": str(meta)}
    except Exception:
        return {"name": raw}

async def set_chat_name(
    chat_id: str,
    name: str,
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    if not name:
        return
    meta = {"name": name}
    await set_chat_meta(chat_id, meta, r, tenant, user_id)

async def get_chat_name(
    chat_id: str,
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    meta = await get_chat_meta(chat_id, r, tenant, user_id)
    return meta.get("name")

async def append_messages(
    chat_id: str,
    messages: List[Dict[str, Any]],
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    history = await get_history(chat_id, r, tenant, user_id)
    history.extend(messages)
    await set_history(chat_id, history, r, tenant, user_id)

async def delete_chat(
    chat_id: str,
    r: Optional[Redis] = None,
    tenant: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    if r is None:
        r = make_redis()
    await r.delete(_key(chat_id, tenant, user_id))
    await r.delete(_meta_key(chat_id, tenant, user_id))
    await _remove_from_user_index(chat_id, tenant, user_id, r)

async def get_user_chats(
    tenant: str,
    user_id: str,
    r: Optional[Redis] = None,
    limit: Optional[int] = None,
) -> List[str]:
    if r is None:
        r = make_redis()
    raw = await r.get(_user_index_key(tenant, user_id))
    if not raw:
        return []
    try:
        chat_ids = json.loads(raw)
    except Exception:
        return []
    if limit is None:
        return chat_ids
    return chat_ids[: max(limit, 0)]
