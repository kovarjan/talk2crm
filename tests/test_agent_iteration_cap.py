# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.engine import agent
from app.services.tenant_llm import LLMConfig

TOOL_CALL = '<tool_call>{"name":"crm_query_tool","args":{"module":"Tasks"}}</tool_call>'


class FakeTool:
    name = "crm_query_tool"

    async def ainvoke(self, args):
        return '{"status": "ok", "summary": "Žádné záznamy.", "records": []}'


def _settings(limit: int):
    return SimpleNamespace(
        agent_max_iterations=limit, skills_enabled=False, skills_capture_enabled=False, llm_temperature=0.0
    )


async def _run(llm, limit=3):
    config = LLMConfig(base_url="http://x/v1", api_key="k", model="m", title_model="m")
    with (
        patch.object(agent, "get_settings", return_value=_settings(limit)),
        patch.object(agent, "get_chat_llm", return_value=llm),
        patch.object(agent, "resolve_llm_config", new=AsyncMock(return_value=config)),
        patch.object(agent, "_build_system_prompt", return_value="system"),
    ):
        return await agent.run_agent(
            tenant_id="t", user_id="u", input_text="briefing", context=None,
            tools=[FakeTool()], capture_skills=False,
        )


@pytest.mark.asyncio
async def test_cap_reached_forces_an_answer_from_gathered_data():
    replies = [TOOL_CALL] * 3 + ["<answer>Meetings: 11 held. Tasks: none found.</answer>"]
    llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=[SimpleNamespace(content=r) for r in replies]))
    result = await _run(llm, limit=3)
    assert result["output"] == "Meetings: 11 held. Tasks: none found."
    assert len(result["intermediate_steps"]) == 3
    assert llm.ainvoke.await_count == 4
    assert "Limit volání nástrojů" in llm.ainvoke.await_args.args[0][-1].content


@pytest.mark.asyncio
async def test_cap_reached_and_model_still_calls_tools_falls_back_to_last_summary():
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=TOOL_CALL)))
    result = await _run(llm, limit=2)
    assert result["output"] == "Žádné záznamy."
    assert llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_normal_answer_before_the_cap_is_untouched():
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="<answer>Hotovo.</answer>")))
    result = await _run(llm, limit=3)
    assert result["output"] == "Hotovo."
    assert llm.ainvoke.await_count == 1
