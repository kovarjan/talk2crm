# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Chat and chat-message persistence helpers shared by routes and the pipeline."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import case, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.models import ChatMessageItem
from app.presentation.agent_result import extract_pending_action_from_agent_result
from app.utils.modules import canonical_module_name
from database.models import Chat, ChatMessage


def chat_message_to_model(message: ChatMessage) -> ChatMessageItem:
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


async def load_messages(
    db: AsyncSession,
    *,
    chat_id: str,
    tenant_id: str,
    user_id: str,
    limit: int | None = None,
) -> list[ChatMessageItem]:
    role_order = case(
        (ChatMessage.role == "user", 0),
        (ChatMessage.role == "assistant", 1),
        else_=2,
    )
    stmt = select(ChatMessage).where(
        ChatMessage.chat_id == chat_id,
        ChatMessage.tenant_id == tenant_id,
        ChatMessage.user_id == user_id,
    )
    if limit is None:
        stmt = stmt.order_by(ChatMessage.created_at.asc(), role_order.asc(), ChatMessage.id.asc())
        result = await db.execute(stmt)
        items = list(result.scalars().all())
    else:
        # Fetch only the newest N rows at SQL level, then restore ascending order.
        stmt = (
            stmt.order_by(ChatMessage.created_at.desc(), role_order.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        items = list(reversed(result.scalars().all()))
    return [chat_message_to_model(item) for item in items]


async def get_chat_or_404(
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


async def create_chat(
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
        except OperationalError:
            await db.rollback()
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(0.05 * (2**attempt))

    return chat


async def append_message(
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


def extract_crm_record_created_event(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(metadata, dict):
        return None

    containers: list[dict[str, Any]] = [metadata]
    agent_result = metadata.get("agent_result")
    if isinstance(agent_result, dict):
        containers.append(agent_result)

    for container in containers:
        raw_event = container.get("crm_sync_event")
        if not isinstance(raw_event, dict):
            continue
        event_type = str(raw_event.get("type") or "").strip().lower()
        if event_type and event_type != "record_created":
            continue
        module = canonical_module_name(raw_event.get("module"))
        record_id = str(raw_event.get("record_id") or "").strip()
        if not module or not record_id:
            continue
        normalized_event: dict[str, str] = {
            "type": "record_created",
            "module": module,
            "record_id": record_id,
        }
        record_name = str(raw_event.get("record_name") or "").strip()
        if record_name:
            normalized_event["record_name"] = record_name
        source = str(raw_event.get("source") or "").strip()
        if source:
            normalized_event["source"] = source
        return normalized_event
    return None


async def load_latest_pending_action(
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
    return extract_pending_action_from_agent_result(agent_result)


async def load_latest_crm_record_created_event(
    db: AsyncSession,
    *,
    chat_id: str,
    tenant_id: str,
    user_id: str,
) -> dict[str, str] | None:
    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.user_id == user_id,
            ChatMessage.role == "assistant",
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(30)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()
    for message in messages:
        metadata = message.metadata_json if isinstance(message.metadata_json, dict) else {}
        event = extract_crm_record_created_event(metadata)
        if event is not None:
            return event
    return None
