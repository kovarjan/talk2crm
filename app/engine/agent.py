from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.domain.skill_contracts import BaseSkillInfo, RuntimeSkillContext
from app.engine.answer_stream import AnswerStreamState
from app.engine.base_skills import BASE_SKILLS
from app.engine.events import EmitFn, StreamEvent, tool_label_cz
from app.engine.llm import get_chat_llm
from app.utils.text import format_european_dates, strip_answer_tags, strip_think_tags
from app.core.logging import (
    LLM_STEP_AGENT_ACTION,
    LLM_STEP_FINAL_RESPONSE,
    LLM_STEP_TOOL_RESULT,
    LLM_STEP_USER_INPUT,
    get_logger,
    log_llm_step,
)


logger = get_logger(__name__)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _normalize_final_answer_text(text: Any) -> str:
    return format_european_dates(strip_answer_tags(text)).strip()


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


def _history_to_messages(
    chat_history: list[dict[str, str]] | None, *, limit: int = 20
) -> list[Any]:
    """
    Converts stored chat history into real alternating chat turns.
    Small models attend to structured turns far better than to a transcript
    embedded in the user message, which is critical for multi-turn reference
    resolution ("ten kód", "ta firma", ...).
    """
    if not chat_history:
        return []
    messages: list[Any] = []
    for item in chat_history[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 1500:
            content = content[:1500].rstrip() + "..."
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def _build_system_prompt(tenant_id: str, tool_names: set[str]) -> str:
    now = datetime.now()
    date_ctx = _build_date_context(now)
    prefix = get_settings().llm_system_prompt_prefix
    prefix_block = f"{prefix}\n" if prefix else ""
    aggregate_tool_block = ""
    aggregate_rules = ""
    if "crm_aggregate_tool" in tool_names:
        aggregate_tool_block = """
6. crm_aggregate_tool(module: str, operation: str, metric: str="amount", account_id: str|null=null, date_from: str|null=null, date_to: str|null=null, date_field: str="date_issued", statuses: list[str]|null=null, exclude_cancelled: bool=true)
   — pro součty, průměry, minima, maxima a počty CRM záznamů.
   — použij pro reporty nad fakturami, nabídkami a obchodními případy.
   — moduly: invoices, invoice_items, quotes, quote_items, opportunities.

7. math_tool(numbers: list[number], operation: str)
   — matematika nad čísly zadanými přímo uživatelem, ne nad CRM daty.
"""
        aggregate_rules = """
- Pro CRM reporty a finanční součty vždy použij crm_aggregate_tool; nepočítej je ručně z textu ani z výsledků crm_query_tool.
"""
    return f"""{prefix_block}Jsi CRM asistent (muž) (tenant: {tenant_id}). Odpovídej česky. Stručně, bez markdown.
Používej mužský rod v odpovědích (např. "našel jsem", "připravil jsem").
Datum: {now.strftime("%Y-%m-%d")} ({now.strftime("%A")}). Rozsahy: {date_ctx}

PRAVIDLO: Pro CRM data vždy zavolej nástroj — o obsahu CRM neodpovídej z paměti modelu.
  VÝJIMKA: Pokud je odpověď už v předchozích zprávách konverzace nebo v pending_action (např. "jaký byl ten kód?", "jak se jmenovala ta firma?"), odpověz rovnou <answer> bez volání nástroje.
PRAVIDLO: crm_action_tool volej POUZE když uživatel žádá vytvoření, úpravu nebo smazání záznamu. Otázky ("jednali jsme...?", "kolik...?", "jaký byl...?") jsou ČTECÍ — nikdy na ně nereaguj mutací.
PRAVIDLO: Data do crm_action_tool musí vycházet pouze z aktuálního vstupu, předchozích zpráv konverzace a výsledků nástrojů. Nikdy necopy-paste hodnoty z ukázek.
PRAVIDLO PENDING AKCE: Kontext může obsahovat pending_action — návrh akce čekající na potvrzení.
  Pokud se uživatel jen na něco PTÁ, odpověz na otázku <answer> — NIKDY znovu nevolej crm_action_tool se stejnými daty.
  Pokud pending_action.action == "create": záznam v CRM JEŠTĚ NEEXISTUJE, žádné CRM id není k dispozici.
  Pro úpravu polí (datum, čas, popis, ...): zavolej crm_action_tool s action="create" a KOMPLETNÍMI poli — zkopíruj všechna pole z pending_action.data.fields a přepiš to, co uživatel mění.
  NIKDY nevolej action="update"/"delete" na záznam s pending_action.action="create".
  NIKDY nefiltruj Meetings/Calls podle contact_id — použij parametr search.
PRAVIDLO POVINNÁ POLE: Při vytváření záznamu odvoď sám vše, co jde odvodit z požadavku a kontextu — name (např. "Schůzka - Fakturace projektu"), description i zapis (zápis = totéž co description, pokud uživatel neřekl jinak). Bez zadané délky trvá schůzka 1 hodinu.
  NIKDY se neptej uživatele na hodnoty, které lze odvodit nebo domyslet. Ptej se jen na informaci, která opravdu chybí (např. termín, nebo který ze dvou kontaktů).
  Pokud crm_action_tool vrátí "needs_more_info", oprav data_json sám podle field_errors (doplň nebo odstraň pole) a zavolej nástroj znovu; uživatele se ptej až když hodnotu nelze odvodit.
  Pole v "missing_required" jsou jen varování — vytvoření neblokují. Kontakt u schůzky/hovoru patří do invitees (contact_id stačí uvést, nástroj ho tam přesune); "Týká se" (parent) je firma kontaktu.

DOSTUPNÉ NÁSTROJE — volaj přes <tool_call> tag:
1. rag_search_tool(query: str, module: str="", limit: int=5)
   — sémantické/fuzzy hledání v RAG indexu. Použij pro získání account_id/contact_id.
   — module může být také "opportunities", "quotes" nebo "acm_invoices", pokud hledáš obchodní případy, nabídky nebo faktury.

2. crm_query_tool(module: str, filters: str="[]", search: str=null, limit: int=20)
   — přesný dotaz do CRM. Pro přesné lookupy jména osoby/firmy použij nejdřív search.
   — podporované moduly pro čtení: Accounts, Contacts, Meetings, Calls, Tasks, Notes, Leads, Opportunities, Quotes, acm_invoices.
     filters je JSON pole [{{"field":"...","op":"eq","value":"..."}}]
   — fields id/account_id/contact_id a také *.id nebo *|id musí mít jako value jen skutečné CRM UUID, nikdy název firmy/kontaktu to nic nenajde.
   — pro více konkrétních záznamů můžeš použít pouze {{ "field": "id", "op": "in", "value": ["<CRM_ID_1>", "<CRM_ID_2>"] }}

3. my_meetings_tool(date_from: str|null=null, date_to: str|null=null, limit: int=100)
   — moje schůzky (assigned_user_id = login user)
   — date_from/date_to jsou volitelné; bez datumu vrací nejnovější schůzky podle limitu

4. crm_action_tool(module: str, action: str, data_json: str="{{}}")
   — mutace: create/update/delete. Pouze po potvrzení uživatele.
   — pro update/delete vždy pošli cílové ID do data_json.id (record_id je jen kompatibilní fallback).
   POZOR: pokud vrátí {{"status": "confirmation_required"}}, OKAMŽITĚ dej <answer> s textem z "message_to_user". Nevolej žádný další nástroj.

5. get_company_overview(account_id: str)
   — vrátí kompaktní AI detail firmy (Accounts) + related_records ze subpanelů.
   — activities i každý related_records subpanel je ve výchozím stavu omezen na 10 nejnovějších záznamů.
   — používej pro detail firmy, když máš account_id.
{aggregate_tool_block}

FORMÁT ODPOVĚDI:
- Pokud chceš zavolat nástroj: <tool_call>{{"name": "jmeno_nastroje", "args": {{"param": "hodnota"}}}}</tool_call>
- Pokud máš finální odpověď pro uživatele: <answer>Tvá odpověď česky</answer>
- Pokud uživatel nezadá dostatečně přesný dotaz, doptej se na upřesnění.
{aggregate_rules}

UI KONTEXTU: Kontext může obsahovat ui_focus_hint_cz a pole module/record/record_name/record_id z CRM obrazovky.
  Ber to jako orientační nápovědu (co má uživatel pravděpodobně otevřené), NE jako závazný fakt.
  Přednost má uživatelský vstup nebo výsledek nástroje před UI kontextem.

VZORY:
Dotaz: "kontakty firmy Zlíner"
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Zlíner", "module": "accounts"}}}}</tool_call>
Po výsledku RAG (skutečné account_id ze záznamu, např. account_id="[REAL_ACCOUNT_ID_Z_VYSLEDKU]"):
→ <tool_call>{{"name": "crm_query_tool", "args": {{"module": "Contacts", "filters": "[{{\"field\":\"account_id\",\"op\":\"eq\",\"value\":\"[REAL_ACCOUNT_ID_Z_VYSLEDKU]\"}}]"}}}}</tool_call>

Dotaz: "schůzky příští týden"
→ <tool_call>{{"name": "my_meetings_tool", "args": {{"date_from": "pristi_tyden_start", "date_to": "pristi_tyden_end"}}}}</tool_call>

Dotaz: "jednali jsme někdy s firmou [Firma]?" (ČTECÍ dotaz — žádná mutace!)
Krok 1 — najdi firmu:
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "[Firma]", "module": "accounts"}}}}</tool_call>
Krok 2 — zjisti aktivity firmy (account_id z výsledku):
→ <tool_call>{{"name": "crm_query_tool", "args": {{"module": "Meetings", "filters": "[{{\"field\":\"account_id\",\"op\":\"eq\",\"value\":\"[REAL_ACCOUNT_ID_Z_VYSLEDKU]\"}}]"}}}}</tool_call>
→ <answer>Shrnutí nalezených schůzek/aktivit — NIKDY nenavrhuj vytvoření schůzky, když se uživatel jen ptá.</answer>

Dotaz: "jaký byl ten kód, co jsem psal?" (odpověď je v konverzaci — žádný nástroj)
→ <answer>Kód z předchozí zprávy: ...</answer>

Dotaz: "naplánuj schůzku s [Jméno] na úterý ráno"
Krok 1 — najdi kontakt:
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "[Jméno]", "module": "contacts"}}}}</tool_call>
Krok 2 — vytvoř schůzku s contact_id z výsledku (data_json piš jako objekt, NE jako string):
→ <tool_call>{{"name": "crm_action_tool", "args": {{"module": "Meetings", "action": "create", "data_json": {{"fields": {{"contact_name": "[Jméno z výsledku]", "contact_id": "[REAL_CONTACT_ID_Z_VYSLEDKU]", "name": "Schůzka - [téma]", "date_start": "[REAL_DATETIME]", "description": "[téma]", "zapis": "[téma]"}}}}}}}}}}</tool_call>
Krok 3 — crm_action_tool vrátí {{"status":"confirmation_required"}}. Okamžitě:
→ <answer>Připraveno: schůzka je připravena. Potvrďte prosím provedení.</answer>

Dotaz: "Naplánuj schůzku s Petrem na zítra"
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Petr", "module": "contacts"}}}}</tool_call>
Model: "Našel jsem více podobných kontaktů:
1) Petr Novák — ABC s.r.o., obchodník, Brno, +420...
2) Petr Novák — XYZ a.s., servisní technik, Praha, +420...
Napiš prosím číslo možnosti nebo název firmy."

Dotaz: "Naplánuj schůzku s Lucií Kovářovou na čtvrtek"
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Lucie Kovářová", "module": "contacts"}}}}</tool_call>
Po výsledku RAG jsou 2 identické kontakty se stejným jménem:
Model interně vybere první kontakt ze seznamu a pokračuje vytvořením schůzky bez dalšího doptávání.

PRAVIDLO: Data do crm_action_tool musí vycházet pouze z aktuálního vstupu, předchozích zpráv konverzace a výsledků nástrojů. Nikdy necopy-paste hodnoty z ukázek.
"""


def _fallback_base_skills() -> list[BaseSkillInfo]:
    return [BaseSkillInfo(**skill) for skill in BASE_SKILLS]


# Strong references to in-flight capture tasks so they are not garbage-collected
_capture_tasks: set[asyncio.Task] = set()


async def _capture_skill_in_background(
    user_message: str,
    agent_response: str,
    tenant_id: str,
    user_id: str,
    request_id: Optional[str],
    settings: Settings,
) -> None:
    # Runs after the response is delivered — the request-scoped session may
    # already be closed, so open a dedicated one.
    try:
        from database.session import AsyncSessionLocal
        from app.services.skill_service import SkillService
        from app.services.skill_capture_service import SkillCaptureService

        async with AsyncSessionLocal() as db:
            skill_svc = SkillService(db, settings)
            capture_svc = SkillCaptureService(
                skill_service=skill_svc,
                settings=settings,
                llm_base_url=settings.llm_base_url,
                llm_model=settings.llm_model,
                llm_api_key=settings.llm_api_key,
            )
            await capture_svc.maybe_capture(
                user_message=user_message,
                agent_response=agent_response,
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
            )
    except Exception:
        logger.exception(
            "tenant=%s Skill capture background task failed", tenant_id
        )


async def run_agent(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    context: dict[str, Any] | None,
    tools: list,
    chat_history: list[dict[str, str]] | None = None,
    emit: EmitFn | None = None,
    db: Optional[AsyncSession] = None,
    system_prompt_override: str | None = None,
    capture_skills: bool = True,
) -> dict[str, Any]:
    log_llm_step(
        logger,
        LLM_STEP_USER_INPUT,
        {
            "input_text": input_text,
            "context": context or {},
            "history_count": len(chat_history or []),
            "tool_count": len(tools),
        },
    )
    log_llm_step(logger, LLM_STEP_AGENT_ACTION, {"event": "agent_execution_started"})
    seq = 0
    if emit:
        seq += 1
        await emit(StreamEvent("agent.started", {"iteration_limit": 6}, seq=seq))

    settings = get_settings()
    # No bind_tools — LiteLLM strips tool parameter schemas when forwarding to Ollama,
    # so the model sees tools with empty schemas and can't generate tool calls.
    # We describe tools in the system prompt and parse <tool_call> tags from text output.
    llm = get_chat_llm(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=4096,
    )

    tools_by_name: dict[str, Any] = {getattr(t, "name", ""): t for t in tools}

    if system_prompt_override is not None:
        # Purpose-built callers (e.g. recommend_actions) supply their own
        # contract and don't want the conversational CRM-command protocol
        # or per-user/tenant skill overlays mixed in.
        system_prompt = system_prompt_override
    else:
        system_prompt = _build_system_prompt(tenant_id, set(tools_by_name))

        skill_block = ""
        if db is not None and settings.skills_enabled:
            try:
                from app.services.skill_service import SkillService
                skill_svc = SkillService(db, settings)
                skill_ctx = await skill_svc.assemble_context(tenant_id, user_id)
                if not skill_ctx.base_skills:
                    skill_ctx.base_skills = _fallback_base_skills()
                skill_block = skill_ctx.to_prompt_block()
            except Exception:
                logger.exception("tenant=%s Failed to load skill context", tenant_id)
        if not skill_block:
            # Skills disabled, no DB session, or load failure — base rules must
            # still reach the prompt so agent behavior never degrades.
            skill_block = RuntimeSkillContext(
                tenant_id=tenant_id, user_id=user_id,
                base_skills=_fallback_base_skills(),
                tenant_overlays=[], user_overlays=[],
            ).to_prompt_block()
        if skill_block:
            system_prompt = system_prompt + "\n\n" + skill_block

    history_messages = _history_to_messages(chat_history)
    if context:
        ui_focus_hint = str(context.get("ui_focus_hint_cz") or "").strip()
        ui_focus_block = (
            f"KONTEXT UI (nezávazná nápověda): {ui_focus_hint}\n"
            if ui_focus_hint
            else ""
        )
        human_text = (
            f"Vstup: {input_text}\n"
            f"{ui_focus_block}"
            f"Kontext (neautoritativní metadata): {json.dumps(context, ensure_ascii=False)}\n"
            f"User ID: {user_id}"
        )
    else:
        human_text = (
            f"Vstup: {input_text}\n"
            f"User ID: {user_id}"
        )

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        *history_messages,
        HumanMessage(content=human_text),
    ]
    intermediate_steps: list[dict[str, Any]] = []
    final_answer = ""

    # 6 iterations matches LangChain AgentExecutor's default; prevents runaway
    # tool-call loops when the LLM fails to emit a final answer.
    for iteration in range(6):
        try:
            if emit:
                _stream_state = AnswerStreamState()
                _raw_parts: list[str] = []
                async for _chunk in llm.astream(messages):
                    _chunk_text = str(getattr(_chunk, "content", "") or "")
                    _raw_parts.append(_chunk_text)
                    _delta = _stream_state.feed(_chunk_text)
                    if _delta:
                        seq += 1
                        await emit(StreamEvent("answer.delta", {"text": _delta}, seq=seq))
                _tail = _stream_state.flush()
                if _tail and not _stream_state.is_suppressed:
                    seq += 1
                    await emit(StreamEvent("answer.delta", {"text": _tail}, seq=seq))
                raw_content = "".join(_raw_parts)
            else:
                response: AIMessage = await llm.ainvoke(messages)
                raw_content = str(getattr(response, "content", "") or "")
        except Exception as exc:
            logger.warning("tenant=%s LLM call failed at iteration=%d: %s", tenant_id, iteration, exc)
            break
        content = strip_think_tags(raw_content)

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
            if emit:
                seq += 1
                await emit(StreamEvent(
                    "agent.tool_call",
                    {"seq": seq, "tool": t_name, "label_cz": tool_label_cz(t_name, t_args)},
                    seq=seq,
                ))

            tool_obj = tools_by_name.get(t_name)
            if tool_obj is None:
                t_result = json.dumps({"error": f"Nástroj '{t_name}' neexistuje. Dostupné: {list(tools_by_name)}"}, ensure_ascii=False)
            else:
                try:
                    t_result = str(await tool_obj.ainvoke(t_args))
                except Exception as exc:
                    t_result = json.dumps({"error": str(exc)}, ensure_ascii=False)

            log_llm_step(logger, LLM_STEP_TOOL_RESULT, {"tool": t_name, "observation": t_result})
            if emit:
                seq += 1
                try:
                    _obs_for_status = json.loads(t_result) if isinstance(t_result, str) else t_result
                    _status = str((_obs_for_status or {}).get("status", "ok")) if isinstance(_obs_for_status, dict) else "ok"
                except Exception:
                    _status = "ok"
                await emit(StreamEvent(
                    "agent.tool_result",
                    {"seq": seq, "tool": t_name, "status": _status},
                    seq=seq,
                ))
            intermediate_steps.append({"tool": t_name, "tool_input": t_args, "observation": t_result, "log": None})

            # Detect confirmation_required and force an immediate answer — no more tool calls.
            try:
                _obs = json.loads(t_result) if isinstance(t_result, str) else t_result
            except Exception:
                _obs = {}
            if isinstance(_obs, dict) and _obs.get("status") == "confirmation_required":
                _msg = str(
                    _obs.get("message_to_user")
                    or _obs.get("message")
                    or "Akce vyžaduje potvrzení uživatele."
                )
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

    final_answer = _normalize_final_answer_text(final_answer)

    if capture_skills and settings.skills_enabled and settings.skills_capture_enabled:
        task = asyncio.create_task(
            _capture_skill_in_background(
                user_message=input_text,
                agent_response=final_answer,
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=None,
                settings=settings,
            )
        )
        _capture_tasks.add(task)
        task.add_done_callback(_capture_tasks.discard)

    if emit and final_answer:
        seq += 1
        await emit(StreamEvent("answer.done", {"text": final_answer}, seq=seq))

    if not final_answer:
        logger.warning("tenant=%s Agent returned empty output after %d steps", tenant_id, len(intermediate_steps))
        final_answer = "Omlouvám se, nepodařilo se zpracovat dotaz. Zkuste ho přeformulovat."

    result = {
        "output": final_answer,
        "intermediate_steps": intermediate_steps,
    }

    log_llm_step(logger, LLM_STEP_FINAL_RESPONSE, {"final_answer": final_answer, "raw_output": final_answer})
    return result
