from __future__ import annotations

import re
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime

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
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _extract_final_answer_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    without_think = _THINK_TAG_RE.sub("", text).strip()
    return without_think or text


def get_agent_executor(tenant_id: str, tools: list) -> AgentExecutor:
    settings = get_settings()
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        # temperature=settings.llm_temperature,
    )
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d %H:%M:%S")
    current_day = now.strftime("%A")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "/nothink\n"
                "Jsi CRM asistent pro Coripo (tenant: {tenant_id}). "
                "Vždy odpovídej česky, pokud uživatel nepožádá jinak. "
                "Odpovídej stručně, bez markdown a bez emoji. "
                "Aktuální datum: {current_date}, den: {current_day}.\n\n"

                "KRITICKÉ PRAVIDLO: Jakýkoliv dotaz na data z CRM (kontakty, firmy, schůzky, záznamy) "
                "MUSÍŠ vyřídit voláním nástrojů. Nikdy neodpovídej z paměti nebo bez volání nástroje.\n\n"

                "PRAVIDLA:\n"
                "- Hledáš firmu → zavolej rag_search_tool(module='accounts') pro získání ID.\n"
                "- Hledáš kontakty firmy → rag_search_tool pro ID, pak crm_query_tool(module='Contacts', filters account_id).\n"
                "- Mutační akce (vytvoř, uprav, smaž) → nejprve připrav a požádej o potvrzení, proveď až po souhlasu.\n"
                "- Pokud RAG vrátí jiný název než uživatel zadal, upozorni na to v odpovědi.\n"
                "- Pokud kontext obsahuje pending_action a uživatel potvrzuje, použij crm_action_tool k provedení.\n"

                "VZORY:\n"
                "  'jaké kontakty má firma ABB' → rag_search_tool(query='ABB', module='accounts') "
                "→ crm_query_tool(module='Contacts', filters=[account_id=<id>])\n"
                "  'jaké mám schůzky příští týden' → my_meetings_tool(date_from='YYYY-MM-DD', date_to='YYYY-MM-DD')\n"
                "  'Naplánuj schůzku s Novákem ze Zlíneru' → rag_search_tool('Zlíner','accounts') "
                "+ rag_search_tool('Novák','contacts') → odpověď s potvrzením\n"
            ),
            (
                "human",
                "Uzivatelsky vstup: {input}\n"
                "Kontext: {context}\n"
                "Aktualni uzivatel id: {user_id}\n"
                # "Pouzij nastroje, kdyz jsou potreba, a vrat strucnou odpoved pro uzivatele.",
                "Postupuj presne podle pravidel, pouzij nastroje pokud jsou potreba, a vrat jasnou a strucnou odpoved."
            ),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(current_date=current_date, current_day=current_day)

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        # LangChain verbose prints dense plaintext tool invocation lines.
        # We keep this off and emit structured step logs ourselves.
        verbose=True,
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

    final_answer = result.get("output", "").strip()
    if not final_answer:
        logger.warning(
            "tenant=%s Agent returned empty output. intermediate_steps=%s",
            tenant_id,
            result.get("intermediate_steps"),
        )
        result["output"] = "Omlouvám se, nepodařilo se zpracovat dotaz. Zkuste ho přeformulovat."

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
        (
            {
                "final_answer": _extract_final_answer_text(result.get("output")),
                "raw_output": result.get("output"),
            }
            if isinstance(result, dict)
            else result
        ),
    )

    return result
