# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Chat and chat-message persistence helpers shared by routes and the pipeline."""

from __future__ import annotations

import asyncio
import json
import re
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

_SELECTION_RE = re.compile(r"^\s*(?:moznost\s*)?(\d{1,2})(?:\s|[).]|$)", re.IGNORECASE)
_DETAIL_LINK_RE = re.compile(r"/#detail/(?P<module>[^/]+)/(?P<record_id>[^/?#]+)")


def _selection_index(input_text: str) -> int | None:
    match = _SELECTION_RE.match(str(input_text or ""))
    if not match:
        return None
    try:
        index = int(match.group(1))
    except ValueError:
        return None
    return index if index >= 1 else None


def _parse_detail_link(url: Any) -> tuple[str, str] | tuple[None, None]:
    match = _DETAIL_LINK_RE.search(str(url or ""))
    if not match:
        return None, None
    module = canonical_module_name(match.group("module"))
    record_id = str(match.group("record_id") or "").strip()
    if not module or not record_id:
        return None, None
    return module, record_id


def _extract_selection_candidates_from_cards(cards: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_type = str(card.get("type") or "").strip().lower()
        if card_type == "record":
            actions = card.get("actions") if isinstance(card.get("actions"), list) else []
            link_url = ""
            for action in actions:
                if isinstance(action, dict) and str(action.get("action") or "").strip().lower() == "link":
                    link_url = str(action.get("url") or "").strip()
                    break
            module, record_id = _parse_detail_link(link_url)
            name = str(card.get("title") or "").strip()
            if module and record_id and name:
                candidates.append(
                    {
                        "module": module,
                        "record_id": record_id,
                        "record_name": name,
                        "selection_label": name,
                    }
                )
            continue

        if card_type != "table":
            continue

        default_module = canonical_module_name(card.get("tag"))
        rows = card.get("rows") if isinstance(card.get("rows"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            link = row.get("link") if isinstance(row.get("link"), dict) else {}
            module, record_id = _parse_detail_link(link.get("url"))
            module = module or default_module
            cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
            name = str(cells.get("name") or row.get("title") or "").strip()
            if module and record_id and name:
                candidates.append(
                    {
                        "module": module,
                        "record_id": record_id,
                        "record_name": name,
                        "selection_label": name,
                    }
                )
    return candidates


def _extract_selection_candidates_from_agent_result(agent_result: dict[str, Any]) -> list[dict[str, str]]:
    steps = agent_result.get("intermediate_steps")
    if not isinstance(steps, list):
        return []

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if str(step.get("tool") or "").strip() != "rag_search_tool":
            continue

        observation = step.get("observation")
        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except Exception:
                continue
        if not isinstance(observation, list):
            continue

        candidates: list[dict[str, str]] = []
        for item in observation:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
            module = canonical_module_name(payload.get("module"))
            record_id = str(payload.get("record_id") or record.get("id") or "").strip()
            record_name = str(record.get("name") or "").strip()
            if not module or not record_id or not record_name:
                continue

            label_parts = [record_name]
            city = str(record.get("billing_address_city") or record.get("primary_address_city") or "").strip()
            street = str(record.get("billing_address_street") or record.get("primary_address_street") or "").strip()
            website = str(record.get("website") or "").strip()
            if city:
                label_parts.append(city)
            if street:
                label_parts.append(street)
            if website:
                label_parts.append(website)

            candidates.append(
                {
                    "module": module,
                    "record_id": record_id,
                    "record_name": record_name,
                    "selection_label": " - ".join(label_parts[:2]) if len(label_parts) >= 2 else record_name,
                }
            )
        if candidates:
            return candidates
    return []


def extract_history_selection(metadata: dict[str, Any] | None, input_text: str) -> dict[str, str] | None:
    index = _selection_index(input_text)
    if index is None or not isinstance(metadata, dict):
        return None

    cards = metadata.get("cards")
    if not isinstance(cards, list):
        agent_result = metadata.get("agent_result")
        if isinstance(agent_result, dict) and isinstance(agent_result.get("cards"), list):
            cards = agent_result["cards"]

    candidates: list[dict[str, str]] = []
    if isinstance(cards, list):
        candidates = _extract_selection_candidates_from_cards(cards)

    if not candidates:
        agent_result = metadata.get("agent_result")
        if isinstance(agent_result, dict):
            candidates = _extract_selection_candidates_from_agent_result(agent_result)

    if index < 1 or index > len(candidates):
        return None
    return candidates[index - 1]


def _observation_value(observation: Any) -> Any:
    if isinstance(observation, str):
        try:
            return json.loads(observation)
        except Exception:
            return None
    return observation


def _record_label(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("name") or "").strip(),
        str(record.get("billing_address_city") or record.get("primary_address_city") or "").strip(),
        str(record.get("billing_address_street") or record.get("primary_address_street") or "").strip(),
    ]
    return ", ".join(part for part in parts if part)


def _summarize_rag_observation(observation: Any, *, limit: int = 5) -> list[str]:
    items = observation if isinstance(observation, list) else []
    lines: list[str] = []
    for idx, item in enumerate(items[:limit], start=1):
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
        module = canonical_module_name(payload.get("module"))
        record_id = str(payload.get("record_id") or record.get("id") or "").strip()
        label = _record_label(record)
        if module and record_id and label:
            lines.append(f"{idx}) {module} id={record_id} name={label}")
    return lines


def _summarize_records_observation(observation: Any, *, limit: int = 5) -> list[str]:
    if not isinstance(observation, dict):
        return []
    module = canonical_module_name(observation.get("module"))
    records = observation.get("records")
    if not isinstance(records, list):
        record = observation.get("record")
        records = [record] if isinstance(record, dict) else []

    lines: list[str] = []
    for idx, record in enumerate(records[:limit], start=1):
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id") or record.get("record_id") or "").strip()
        label = _record_label(record)
        if record_id and label:
            lines.append(f"{idx}) {module or 'CRM'} id={record_id} name={label}")
    return lines


def summarize_agent_memory(agent_result: dict[str, Any] | None, *, limit: int = 5) -> str:
    if not isinstance(agent_result, dict):
        return ""
    steps = agent_result.get("intermediate_steps")
    if not isinstance(steps, list):
        return ""

    blocks: list[str] = []
    for step in steps[-4:]:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "").strip()
        observation = _observation_value(step.get("observation"))
        if tool == "rag_search_tool":
            lines = _summarize_rag_observation(observation, limit=limit)
        elif tool in {"crm_query_tool", "my_meetings_tool", "get_company_overview"}:
            lines = _summarize_records_observation(observation, limit=limit)
        else:
            lines = []
        if lines:
            blocks.append(f"{tool}: " + "; ".join(lines))
    return "\n".join(blocks)


def chat_message_item_to_agent_history(message: ChatMessageItem) -> dict[str, str]:
    content = str(message.content or "")
    if message.role == "assistant" and isinstance(message.metadata, dict):
        memory = summarize_agent_memory(message.metadata.get("agent_result"))
        if memory:
            content = f"{content}\nRelevantni CRM pamet z predchozich nastroju:\n{memory}"
    return {"role": message.role, "content": content}


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


async def load_latest_history_selection(
    db: AsyncSession,
    *,
    chat_id: str,
    tenant_id: str,
    user_id: str,
    input_text: str,
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
        .limit(10)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()
    for message in messages:
        metadata = message.metadata_json if isinstance(message.metadata_json, dict) else {}
        selection = extract_history_selection(metadata, input_text)
        if selection is not None:
            return selection
    return None
