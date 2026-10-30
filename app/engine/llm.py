# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Any

import openai
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# (base_url, model) pairs that rejected a temperature (reasoning models such as
# gpt-5/gpt-6/o-series only accept the default). Learned from the first 400 and kept
# for the process lifetime; models that accept temperature keep receiving it.
_NO_TEMPERATURE: set[tuple[str, str]] = set()


class _AdaptiveChatOpenAI(ChatOpenAI):
    """ChatOpenAI that stops sending ``temperature`` once an endpoint rejects it."""

    def _temperature_key(self) -> tuple[str, str]:
        return (str(self.openai_api_base or ""), str(self.model_name))

    def _get_request_payload(self, *args: Any, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(*args, **kwargs)
        if self._temperature_key() in _NO_TEMPERATURE:
            payload.pop("temperature", None)
        return payload

    def _learn_no_temperature(self, exc: openai.BadRequestError) -> bool:
        key = self._temperature_key()
        if self.temperature is None or key in _NO_TEMPERATURE or "temperature" not in str(exc).lower():
            return False
        _NO_TEMPERATURE.add(key)
        logger.info("LLM %s rejects temperature; retrying without it (base_url=%s)", key[1], key[0])
        return True

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await super()._agenerate(*args, **kwargs)
        except openai.BadRequestError as exc:
            if not self._learn_no_temperature(exc):
                raise
        return await super()._agenerate(*args, **kwargs)

    async def _astream(self, *args: Any, **kwargs: Any):
        started = False
        try:
            async for chunk in super()._astream(*args, **kwargs):
                started = True
                yield chunk
            return
        except openai.BadRequestError as exc:
            if started or not self._learn_no_temperature(exc):
                raise
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk


@lru_cache(maxsize=64)
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
    return _AdaptiveChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout if timeout is not None else settings.llm_request_timeout_seconds,
        max_retries=max_retries if max_retries is not None else settings.llm_max_retries,
    )
