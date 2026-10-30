# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import httpx
import openai
import pytest
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from app.engine import llm as llm_module
from app.engine.llm import get_chat_llm

TEMP_ERROR = (
    "litellm.UnsupportedParamsError: gpt-6-luna doesn't support temperature=0.0 while reasoning is active"
)


def _bad_request(message: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


@pytest.fixture(autouse=True)
def _reset():
    llm_module._NO_TEMPERATURE.clear()
    get_chat_llm.cache_clear()
    yield
    llm_module._NO_TEMPERATURE.clear()
    get_chat_llm.cache_clear()


def _llm(model="gpt-6-luna"):
    return get_chat_llm(base_url="http://x/v1", api_key="k", model=model, temperature=0.0)


def test_temperature_is_sent_until_the_endpoint_rejects_it():
    assert _llm()._get_request_payload([("human", "hi")])["temperature"] == 0.0
    llm_module._NO_TEMPERATURE.add(("http://x/v1", "gpt-6-luna"))
    assert "temperature" not in _llm()._get_request_payload([("human", "hi")])
    other = _llm("qwen3")  # a different model on the same endpoint still gets it
    assert other._get_request_payload([("human", "hi")])["temperature"] == 0.0


@pytest.mark.asyncio
async def test_agenerate_retries_once_without_temperature_and_remembers():
    ok = ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
    payloads: list[dict] = []
    llm = _llm()

    async def fake_agenerate(self, *args, **kwargs):
        payload = self._get_request_payload(*args, **kwargs)
        payloads.append(payload)
        if "temperature" in payload:
            raise _bad_request(TEMP_ERROR)
        return ok

    with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
        result = await llm.ainvoke([("human", "hi")])
        assert result.content == "ok"
        await llm.ainvoke([("human", "again")])

    assert ["temperature" in p for p in payloads] == [True, False, False]


@pytest.mark.asyncio
async def test_astream_retries_when_rejected_before_first_chunk():
    llm = _llm()
    calls: list[bool] = []

    async def fake_astream(self, *args, **kwargs):
        payload = self._get_request_payload(*args, **kwargs)
        calls.append("temperature" in payload)
        if "temperature" in payload:
            raise _bad_request(TEMP_ERROR)
        yield ChatGenerationChunk(message=AIMessageChunk(content="hi"))

    with patch.object(ChatOpenAI, "_astream", fake_astream):
        chunks = [c async for c in llm.astream([("human", "hi")])]

    assert "".join(c.content for c in chunks) == "hi"
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_unrelated_bad_request_is_not_swallowed():
    llm = _llm()

    async def fake_agenerate(self, *args, **kwargs):
        raise _bad_request("context length exceeded")

    with patch.object(ChatOpenAI, "_agenerate", fake_agenerate):
        with pytest.raises(openai.BadRequestError):
            await llm.ainvoke([("human", "hi")])
    assert not llm_module._NO_TEMPERATURE
