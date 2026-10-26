# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import get_settings


@lru_cache(maxsize=8)
def get_chat_llm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    """Process-wide ChatOpenAI instances, one per (base_url, model, params).

    ChatOpenAI is stateless across invocations but owns an HTTP connection
    pool; constructing it per request discards pooled connections. Requests
    time out and retry per LLM_REQUEST_TIMEOUT_SECONDS / LLM_MAX_RETRIES.
    """
    settings = get_settings()
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else settings.llm_request_timeout_seconds,
        max_retries=max_retries if max_retries is not None else settings.llm_max_retries,
    )
