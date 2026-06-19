# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI


@lru_cache(maxsize=8)
def get_chat_llm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    """Process-wide ChatOpenAI instances, one per (base_url, model, params).

    ChatOpenAI is stateless across invocations but owns an HTTP connection
    pool; constructing it per request discards pooled connections.
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
