from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _clean(text: Any) -> str:
    return _THINK_TAG_RE.sub("", str(text or "")).strip()


def _build_date_context(now: datetime) -> str:
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    next_week_start = week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)
    month_start = today.replace(day=1)
    month_end = ((month_start.replace(day=28) + timedelta(days=4)).replace(day=1)) - timedelta(days=1)
    fmt = "%Y-%m-%d"
    return (
        f"dnes={today.strftime(fmt)} "
        f"zitra={tomorrow.strftime(fmt)} "
        f"tento_tyden={week_start.strftime(fmt)}/{week_end.strftime(fmt)} "
        f"pristi_tyden={next_week_start.strftime(fmt)}/{next_week_end.strftime(fmt)} "
        f"tento_mesic={month_start.strftime(fmt)}/{month_end.strftime(fmt)}"
    )


def _parse_tool_call_json(raw: str) -> dict | None:
    """Parse the JSON inside a <tool_call> block, using json_repair as fallback."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        repaired = repair_json(raw, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    return None


def _sanitize_data_json_string(value: str) -> str:
    """Best-effort repair of a data_json string the model passed pre-encoded."""
    try:
        json.loads(value)
        return value
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        return repair_json(value)
    except Exception:
        return value


def _build_system_prompt(tenant_id: str) -> str:
    now = datetime.now()
    date_ctx = _build_date_context(now)
    return f"""/nothink
Jsi CRM asistent (tenant: {tenant_id}). Odpovídej česky. Stručně, bez markdown.
Datum: {now.strftime("%Y-%m-%d")} ({now.strftime("%A")}). Rozsahy: {date_ctx}

PRAVIDLO: Vždy zavolej nástroj. Nikdy neodpovídej z paměti.

DOSTUPNÉ NÁSTROJE — volaj přes <tool_call> tag:
1. rag_search_tool(query: str, module: str="", limit: int=5)
   — sémantické/fuzzy hledání v RAG indexu. Použij pro získání account_id/contact_id.

2. crm_query_tool(module: str, filters: str="[]", search: str=null, limit: int=20)
   — přesný dotaz do CRM. Pro přesné lookupy jména osoby/firmy použij nejdřív search.
     filters je JSON pole [{{"field":"...","op":"eq","value":"..."}}]

3. my_meetings_tool(date_from: str, date_to: str, limit: int=100)
   — schůzky uživatele v období (ISO daty YYYY-MM-DD)

4. crm_action_tool(module: str, action: str, data_json: str="{{}}")
   — mutace: create/update/delete. Pouze po potvrzení uživatele.
   — pro update/delete vždy pošli cílové ID do data_json.id (record_id je jen kompatibilní fallback).
   POZOR: pokud vrátí {{"status": "confirmation_required"}}, OKAMŽITĚ dej <answer> s textem z "message_to_user". Nevolej žádný další nástroj.

FORMÁT ODPOVĚDI:
- Pokud chceš zavolat nástroj: <tool_call>{{"name": "jmeno_nastroje", "args": {{"param": "hodnota"}}}}</tool_call>
- Pokud máš finální odpověď pro uživatele: <answer>Tvá odpověď česky</answer>
- Pokud užvatel nezadá dostatečně přesný dotaz, doptej se na upřesnění.

VZORY:
Dotaz: "kontakty firmy Zlíner"
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Zlíner", "module": "accounts"}}}}</tool_call>
Po výsledku RAG (account_id=XYZ):
→ <tool_call>{{"name": "crm_query_tool", "args": {{"module": "Contacts", "filters": "[{{\"field\":\"account_id\",\"op\":\"eq\",\"value\":\"XYZ\"}}]"}}}}</tool_call>

Dotaz: "schůzky příští týden"
→ <tool_call>{{"name": "my_meetings_tool", "args": {{"date_from": "pristi_tyden_start", "date_to": "pristi_tyden_end"}}}}</tool_call>

Dotaz: "naplánuj schůzku s Liborem Adamcem na úterý ráno"
Krok 1 — najdi kontakt:
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Libor Adamec", "module": "contacts"}}}}</tool_call>
Krok 2 — vytvoř schůzku s contact_id z výsledku (data_json piš jako objekt, NE jako string):
→ <tool_call>{{"name": "crm_action_tool", "args": {{"module": "Meetings", "action": "create", "data_json": {{"fields": {{"contact_name": "Libor Adamec", "contact_id": "abc-123", "date_start": "2026-04-08 09:00:00", "description": "..."}}}}}}}}}}</tool_call>
Krok 3 — crm_action_tool vrátí {{"status":"confirmation_required"}}. Okamžitě:
→ <answer>Připraveno: schůzka s Liborem Adamcem v úterý 8.4. v 9:00. Potvrďte prosím provedení.</answer>

Dotaz: "Naplánuj schůzku s Petrem na zítra"
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Petr", "module": "contacts"}}}}</tool_call>
Model: "Našel jsem 3 kontakty jménem Petr. Potřebuji upřesnit, o kterého Petera se jedná. Můžeš mi dát více informací, jako je název firmy, pozice, nebo jiné detaily?"
"""


async def run_agent(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    context: dict[str, Any] | None,
    tools: list,
) -> dict[str, Any]:
    log_llm_step(logger, LLM_STEP_USER_INPUT, {"input_text": input_text, "context": context or {}, "tool_count": len(tools)})
    log_llm_step(logger, LLM_STEP_AGENT_ACTION, {"event": "agent_execution_started"})

    settings = get_settings()
    # No bind_tools — LiteLLM strips tool parameter schemas when forwarding to Ollama,
    # so the model sees tools with empty schemas and can't generate tool calls.
    # We describe tools in the system prompt and parse <tool_call> tags from text output.
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
    )

    tools_by_name: dict[str, Any] = {getattr(t, "name", ""): t for t in tools}
    system_prompt = _build_system_prompt(tenant_id)
    if context and context.get("module"):
        human_text = (
            f"Vstup: {input_text}\n"
            f"Kontext: {json.dumps(context, ensure_ascii=False)}\n"
            f"User ID: {user_id}"
        )
    else:
        human_text = (
            f"Vstup: {input_text}\n"
            f"User ID: {user_id}"
        )

    messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
    intermediate_steps: list[dict[str, Any]] = []
    final_answer = ""

    for iteration in range(6):
        try:
            response: AIMessage = await llm.ainvoke(messages)
        except Exception as exc:
            logger.warning("tenant=%s LLM call failed at iteration=%d: %s", tenant_id, iteration, exc)
            break

        raw_content = str(getattr(response, "content", "") or "")
        content = _clean(raw_content)

        if not content:
            logger.warning("tenant=%s Empty/thinking-only response at iteration=%d", tenant_id, iteration)
            if iteration < 2:
                # Push the model to output something concrete
                messages.append(AIMessage(content=raw_content or ""))
                messages.append(HumanMessage(content="Zavolej nyní příslušný nástroj pomocí <tool_call> tagu."))
                continue
            break

        # Check for tool call in the text
        tool_match = _TOOL_CALL_RE.search(content)
        if tool_match:
            raw_json = tool_match.group(1).strip()
            call = _parse_tool_call_json(raw_json)
            if call is None:
                logger.warning("tenant=%s Invalid tool_call JSON at iteration=%d | raw=%s", tenant_id, iteration, raw_json)
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content="Chyba JSON v tool_call. Použij data_json jako objekt (ne string). Zkus znovu."))
                continue

            t_name = str(call.get("name", ""))
            t_args = call.get("args", {}) or {}

            # data_json must reach the tool as a JSON string.
            # Models reliably produce a native object — serialize it.
            # If the model passed a string, sanitize it (strip stray newlines, attempt repair).
            if isinstance(t_args.get("data_json"), (dict, list)):
                t_args["data_json"] = json.dumps(t_args["data_json"], ensure_ascii=False)
            elif isinstance(t_args.get("data_json"), str):
                t_args["data_json"] = _sanitize_data_json_string(t_args["data_json"])

            log_llm_step(logger, LLM_STEP_AGENT_ACTION, {"tool": t_name, "tool_input": t_args})

            tool_obj = tools_by_name.get(t_name)
            if tool_obj is None:
                t_result = json.dumps({"error": f"Nástroj '{t_name}' neexistuje. Dostupné: {list(tools_by_name)}"}, ensure_ascii=False)
            else:
                try:
                    t_result = str(await tool_obj.ainvoke(t_args))
                except Exception as exc:
                    t_result = json.dumps({"error": str(exc)}, ensure_ascii=False)

            log_llm_step(logger, LLM_STEP_TOOL_RESULT, {"tool": t_name, "observation": t_result})
            intermediate_steps.append({"tool": t_name, "tool_input": t_args, "observation": t_result, "log": None})

            # Detect confirmation_required and force an immediate answer — no more tool calls.
            try:
                _obs = json.loads(t_result) if isinstance(t_result, str) else t_result
            except Exception:
                _obs = {}
            if isinstance(_obs, dict) and _obs.get("status") == "confirmation_required":
                _msg = str(_obs.get("message_to_user") or "Akce vyžaduje potvrzení uživatele.")
                final_answer = _msg
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=f"Výsledek nástroje {t_name}:\n{t_result}"))
                break

            messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=f"Výsledek nástroje {t_name}:\n{t_result}\n\nPokračuj: zavolej další nástroj nebo dej <answer>."))
            continue

        # Check for explicit answer tag
        answer_match = _ANSWER_RE.search(content)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            break

        # Plain text response with no tags — use as final answer
        final_answer = content
        break

    if not final_answer:
        # Synthesize from last tool result if available
        if intermediate_steps:
            last_obs = intermediate_steps[-1].get("observation", "")
            try:
                parsed = json.loads(last_obs) if isinstance(last_obs, str) else last_obs
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                final_answer = str(parsed.get("message_to_user") or parsed.get("summary") or "").strip()
            if not final_answer:
                final_answer = str(last_obs)[:500]

    if not final_answer:
        logger.warning("tenant=%s Agent returned empty output after %d steps", tenant_id, len(intermediate_steps))
        final_answer = "Omlouvám se, nepodařilo se zpracovat dotaz. Zkuste ho přeformulovat."

    result = {
        "output": final_answer,
        "intermediate_steps": intermediate_steps,
    }

    log_llm_step(logger, LLM_STEP_FINAL_RESPONSE, {"final_answer": final_answer, "raw_output": final_answer})
    return result
