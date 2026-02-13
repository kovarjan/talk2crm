from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def get_agent_executor(tenant_id: str, tools: list) -> AgentExecutor:
    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for tenant '{tenant_id}'. "
                "Use tools for CRM actions and knowledge lookup. "
                "Always be explicit about action, result, and assumptions. "
                "Any mutating CRM action (create/update/patch/delete) must be "
                "explicitly confirmed by user before execution.",
            ),
            (
                "human",
                "User input: {input}\n"
                "Context: {context}\n"
                "Current user id: {user_id}\n"
                "Reply with a concise JSON-like summary in plain text.",
            ),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.debug,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )


async def run_agent(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    context: dict[str, Any] | None,
    tools: list,
) -> dict[str, Any]:
    executor = get_agent_executor(tenant_id=tenant_id, tools=tools)
    result = await executor.ainvoke(
        {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "input": input_text,
            "context": context or {},
        }
    )

    steps = result.get("intermediate_steps")
    if isinstance(steps, list):
        serialized_steps: list[dict[str, Any]] = []
        for item in steps:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            action, observation = item
            tool_name = getattr(action, "tool", None)
            tool_input = getattr(action, "tool_input", None)
            log_text = getattr(action, "log", None)

            serialized_steps.append(
                {
                    "tool": str(tool_name) if tool_name is not None else None,
                    "tool_input": tool_input,
                    "observation": observation,
                    "log": str(log_text) if log_text is not None else None,
                }
            )
        result["intermediate_steps"] = serialized_steps
    else:
        result["intermediate_steps"] = []

    return result
