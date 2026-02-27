from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import (
    LLM_STEP_AGENT_ACTION,
    LLM_STEP_FINAL_RESPONSE,
    LLM_STEP_TOOL_RESULT,
    LLM_STEP_USER_INPUT,
    get_logger,
    log_llm_step,
)


logger = get_logger(__name__)


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
                "Jsi CRM asistent pro tenant '{tenant_id}'. "
                "Odpovidej cesky, pokud uzivatel vyslovne nechce jiny jazyk. "
                "Pouzivej nastroje pro CRM akce a vyhledavani. "
                "Vzdy bud explicitni ohledne akce, vysledku a predpokladu. "
                "Mutacni CRM akce (create/update/patch/delete) musi byt pred provedenim potvrzena uzivatelem. "
                "Pokud context obsahuje pending_action a uzivatel posila opravu nebo doplneni, "
                "navaz na tuto pending_action a uprav data misto obecnych doplnujicich dotazu.",
            ),
            (
                "human",
                "Uzivatelsky vstup: {input}\n"
                "Kontext: {context}\n"
                "Aktualni uzivatel id: {user_id}\n"
                "Pouzij nastroje, kdyz jsou potreba, a vrat strucnou odpoved pro uzivatele.",
            ),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        # LangChain verbose prints dense plaintext tool invocation lines.
        # We keep this off and emit structured step logs ourselves.
        verbose=False,
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
    log_llm_step(
        logger,
        LLM_STEP_USER_INPUT,
        {"input_text": input_text, "context": context or {}, "tool_count": len(tools)},
    )
    log_llm_step(
        logger,
        LLM_STEP_AGENT_ACTION,
        {"event": "agent_execution_started"},
    )

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
            log_llm_step(
                logger,
                LLM_STEP_AGENT_ACTION,
                {
                    "tool": str(tool_name) if tool_name is not None else None,
                    "tool_input": tool_input,
                },
            )
            log_llm_step(
                logger,
                LLM_STEP_TOOL_RESULT,
                {
                    "tool": str(tool_name) if tool_name is not None else None,
                    "observation": observation,
                },
            )
        result["intermediate_steps"] = serialized_steps
    else:
        result["intermediate_steps"] = []

    log_llm_step(
        logger,
        LLM_STEP_FINAL_RESPONSE,
        result.get("output") if isinstance(result, dict) else result,
    )

    return result
