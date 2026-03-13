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
        temperature=settings.llm_temperature,
    )
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d %H:%M:%S")
    current_day = now.strftime("%A")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Jsi coripo CRM asistent pro tenant '{tenant_id}'. Odpovídej vždy česky, pokud uživatel výslovně nepožádá o jiný jazyk.\n\n"
                
                "ZÁKLADNÍ PRAVIDLA:\n"
                "1. FORMÁTOVÁNÍ: Odpovědi piš jako prostý text. Je PŘÍSNĚ ZAKÁZÁNO používat markdown (žádné tabulky, žádné tučné písmo) a nepoužívej žádné emoji.\n"
                "2. KONTEXT: Vždy zkontroluj předchozí zprávy v historii chatu. Pokud uživatel použije zájmeno (např. 's ním', 'tamto'), dohledej entitu v předchozí konverzaci.\n"
                "3. POTVRZENÍ: Mutační akce v CRM (create/update/patch/delete) nesmíš provést bez výslovného souhlasu uživatele. Pokud context obsahuje 'pending_action' a dotaz směřuje na mutaci dat, navaž na ni. Jinak ji ignoruj.\n"
                "4. UPŘÍMNOST: Vždy explicitně popiš, jakou akci jsi provedl, jaký je výsledek a z jakých předpokladů vycházíš.\n\n"

                "POUŽITÍ NÁSTROJŮ A POSTUPY (CHECKLISTS):\n"

                "A. Vždy první krok - Identifikace záměru:\n"
                "- Krok 1: Urči, zda je dotaz informativní (např. 'Kolik mám kontaktů?') nebo akční (např. 'Vytvoř mi kontakt Jan Novak').\n"
                "- Krok 2: Pokud je dotaz akční, zkontroluj 'pending_action' v kontextu. Pokud tam je relevantní akce, použij 'crm_action_tool' pro její provedení, ale NEŽ ji provedeš, vždy požádej uživatele o potvrzení.\n\n"
                
                "B. Vyhledávání kontaktů a firem (rag_search_tool & crm_query_tool):\n"
                "- Krok 1: Pokud se uživatel dotazuje na kontakt z dané firmy, vyhledej první firmu a potom v crm vyhledej kontakty spojené s touto firmou a tímto jménem kontaktu.\n"
                "- Krok 2: Pro hledání firmy použij 'rag_search_tool' pro získání 'account_id' (fuzzy hledání).\n"
                "- Krok 3: Pro hledání kontaktů pod firmou použij 'crm_query_tool' s module='Contacts' a filtruj přes relate filtr na 'Accounts.id'.\n"
                "- Krok 4: Pokud 'rag_search_tool' nevrátí nic, NIKDY neodpovídej, že záznam neexistuje, dokud ho neověříš přímo přes 'crm_query_tool' nebo 'crm_action_tool'.\n\n"
                
                "C. Plánování schůzek a hovorů (my_meetings_tool):\n"
                "- Krok 1: Identifikuj účastníka - modul a id. (Pokud chybí, hledej v kontextu chatu. Pokud ho neznáš, zeptej se uživatele).\n"
                "- Krok 2: Identifikuj časové okno. Použij 'my_meetings_tool(date_from, date_to)' pro kontrolu volného času a případných kolizí.\n"
                "- Krok 3: Zkontroluj výsledek. Pokud je volno, navrhni čas a požádej uživatele o potvrzení naplánování.\n\n"

                "D. Všeobecné dotazy a mutace v CRM:\n"
                "- 1: Pokud je dotaz informativní (např. 'Kolik mám kontaktů?'), použij 'crm_query_tool' pro získání odpovědi.\n"

                "Aktuální datum a čas: {current_date}, den v týdnu: {current_day}."
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
