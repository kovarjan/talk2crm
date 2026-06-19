# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Chat title generation (LLM with deterministic fallback)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.api.models import ChatMessageItem
from app.core.config import get_settings
from app.core.logging import get_logger
from app.engine.llm import get_chat_llm
from app.presentation.agent_result import to_user_message
from database.models import Chat


logger = get_logger(__name__)

CHAT_TITLE_MAX_CHARS = 80
CHAT_TITLE_MAX_WORDS = 6


def history_for_title_prompt(history: list[ChatMessageItem]) -> str:
    lines: list[str] = []
    # Keep prompt small and focused on the beginning of conversation topic.
    for item in history[:8]:
        role = (item.role or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = to_user_message(item.content or "")
        if not content:
            continue
        compact = re.sub(r"\s+", " ", content).strip()
        if len(compact) > 220:
            compact = compact[:220].rstrip() + "..."
        prefix = "U" if role == "user" else "A"
        lines.append(f"{prefix}: {compact}")
    return "\n".join(lines)


def sanitize_chat_name(value: str | None) -> str:
    text = to_user_message(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("`\"'“”„")
    if not text:
        return ""
    # Keep only the first line if model adds extra explanation.
    text = text.splitlines()[0].strip()
    words = text.split()
    if len(words) > CHAT_TITLE_MAX_WORDS:
        text = " ".join(words[:CHAT_TITLE_MAX_WORDS])
    if len(text) > CHAT_TITLE_MAX_CHARS:
        text = text[:CHAT_TITLE_MAX_CHARS].rstrip(" ,.;:-")
    return text


def fallback_chat_name(history: list[ChatMessageItem]) -> str:
    for item in history:
        if (item.role or "").strip().lower() != "user":
            continue
        text = re.sub(r"\s+", " ", (item.content or "")).strip()
        if not text:
            continue
        words = text.split()
        short = " ".join(words[:CHAT_TITLE_MAX_WORDS])
        return sanitize_chat_name(short) or "Novy chat"
    return "Novy chat"


async def generate_chat_name_with_llm(
    *,
    history: list[ChatMessageItem],
    tenant_id: str,
    user_id: str,
) -> str:
    transcript = history_for_title_prompt(history)
    if not transcript:
        return fallback_chat_name(history)

    settings = get_settings()
    messages = [
        SystemMessage(content="/nothink"),
        HumanMessage(
            content=(
                "Jsi asistent, který vytváří krátké názvy konverzací v češtině.\n"
                "Pravidla:\n"
                "- vrať pouze název (žádné vysvětlení)\n"
                "- 2 až 6 slov\n"
                "- max 80 znaků\n"
                "- bez uvozovek a bez tečky na konci\n\n"
                f"Konverzace:\n{transcript}\n\název:"
            )
        ),
    ]
    models_to_try = list(dict.fromkeys([settings.llm_title_model, settings.llm_model]))
    for model_name in models_to_try:
        llm = get_chat_llm(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=model_name,
            temperature=0.0,
            max_tokens=200,
        )
        try:
            response = await llm.ainvoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            if isinstance(raw, list):
                raw = " ".join(str(part) for part in raw)
            cleaned = sanitize_chat_name(str(raw))
            if cleaned:
                return cleaned
        except Exception:
            logger.warning(
                "Chat title generation failed with model=%s tenant=%s user=%s",
                model_name,
                tenant_id,
                user_id,
            )

    return fallback_chat_name(history)


async def ensure_chat_name(
    *,
    chat: Chat,
    history: list[ChatMessageItem],
    tenant_id: str,
    user_id: str,
) -> str:
    existing = (chat.name or "").strip()
    if existing:
        return existing

    generated = await generate_chat_name_with_llm(
        history=history,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    generated = sanitize_chat_name(generated)
    if not generated:
        generated = fallback_chat_name(history)
    chat.name = generated[:255]
    chat.updated_at = datetime.now(timezone.utc)
    return chat.name
