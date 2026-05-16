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
_ANSWER_TAG_RE = re.compile(r"</?answer>", re.IGNORECASE)
_ISO_DATE_RE = re.compile(
    r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?(?!\d)"
)


def _clean(text: Any) -> str:
    return _THINK_TAG_RE.sub("", str(text or "")).strip()


def _strip_answer_tags(text: Any) -> str:
    return _ANSWER_TAG_RE.sub("", str(text or "")).strip()


def _format_european_dates(text: Any) -> str:
    def repl(match: re.Match[str]) -> str:
        year, month, day, hour, minute, second = match.groups()
        formatted = f"{int(day)}.{int(month)}.{year}"
        if hour and minute:
            formatted += f" {hour}:{minute}"
            if second and second != "00":
                formatted += f":{second}"
        return formatted

    return _ISO_DATE_RE.sub(repl, str(text or ""))


def _normalize_final_answer_text(text: Any) -> str:
    return _format_european_dates(_strip_answer_tags(text)).strip()


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


def _format_recent_history(chat_history: list[dict[str, str]] | None, *, limit: int = 10) -> str:
    if not chat_history:
        return ""
    lines: list[str] = []
    for item in chat_history[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > 500:
            content = content[:500].rstrip() + "..."
        if role not in {"user", "assistant", "system"}:
            role = "user"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_system_prompt(tenant_id: str) -> str:
    now = datetime.now()
    date_ctx = _build_date_context(now)
    prefix = get_settings().llm_system_prompt_prefix
    prefix_block = f"{prefix}\n" if prefix else ""
    return f"""{prefix_block}Jsi CRM asistent (muž) (tenant: {tenant_id}). Odpovídej česky. Stručně, bez markdown.
Používej mužský rod v odpovědích (např. "našel jsem", "připravil jsem").
Datum: {now.strftime("%Y-%m-%d")} ({now.strftime("%A")}). Rozsahy: {date_ctx}

PRAVIDLO: Vždy zavolej nástroj. Nikdy neodpovídej z paměti.
PRAVIDLO: Data do crm_action_tool musí vycházet pouze z aktuálního vstupu, KONVERZAČNÍ HISTORIE a výsledků nástrojů. Nikdy necopy-paste hodnoty z ukázek.
PRAVIDLO PENDING AKCE: Kontext může obsahovat pending_action — návrh akce čekající na potvrzení.
  Pokud pending_action.action == "create": záznam v CRM JEŠTĚ NEEXISTUJE, žádné CRM id není k dispozici.
  Pro úpravu polí (datum, čas, popis, ...): zavolej crm_action_tool s action="create" a KOMPLETNÍMI poli — zkopíruj všechna pole z pending_action.data.fields a přepiš to, co uživatel mění.
  NIKDY nevolej action="update"/"delete" na záznam s pending_action.action="create".
  NIKDY nefiltruj Meetings/Calls podle contact_id — použij parametr search.

DOSTUPNÉ NÁSTROJE — volaj přes <tool_call> tag:
1. rag_search_tool(query: str, module: str="", limit: int=5)
   — sémantické/fuzzy hledání v RAG indexu. Použij pro získání account_id/contact_id.
   — module může být také "opportunities" nebo "quotes", pokud hledáš obchodní případy nebo nabídky.

2. crm_query_tool(module: str, filters: str="[]", search: str=null, limit: int=20)
   — přesný dotaz do CRM. Pro přesné lookupy jména osoby/firmy použij nejdřív search.
   — podporované moduly pro čtení: Accounts, Contacts, Meetings, Calls, Tasks, Notes, Leads, Opportunities, Quotes.
   — pro obchodní případy používej module="Opportunities"; pro nabídky používej module="Quotes".
     filters je JSON pole [{{"field":"...","op":"eq","value":"..."}}]

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
   — měna částek je uvedena v default_currency.iso4217, výchozí je vždy Kč (CZK) pokud není uvedeno jinak; platí i pro sloupec amount_usdollar.
   — related_records.Opportunities jsou obchodní případy (opportunities), NE nabídky (quotes).
   — používej pro detail firmy, když máš account_id.

FORMÁT ODPOVĚDI:
- Pokud chceš zavolat nástroj: <tool_call>{{"name": "jmeno_nastroje", "args": {{"param": "hodnota"}}}}</tool_call>
- Pokud máš finální odpověď pro uživatele: <answer>Tvá odpověď česky</answer>
- Pokud uživatel nezadá dostatečně přesný dotaz, doptej se na upřesnění.
- Zobrazuj uživateli jenom přeložené názvy modulů jako "Nabídky" ne "Quotes" ani "Nabídky (Quotes)" a podobně.

PRAVIDLA VÝBĚRU KONTAKTU/FIRMY:
- Při výsledku z rag_search_tool vždy nejdřív posuď, zda jde o přesné shody, nebo jen podobné kandidáty.
- Pokud najdeš více IDENTICKÝCH kontaktů (stejné celé jméno a firma), automaticky vyber první záznam ve výsledcích a pokračuj bez doptávání.
- Pokud najdeš více PODOBNÝCH kandidátů a není jasná 1 volba, doptej se a vypiš max 3 konkrétní možnosti.
- U každé možnosti uveď dostupné rozlišující údaje: firma (account_name), pozice (title), město/adresa, telefon, email.
- Nepiš obecné "upřesni prosím". Vždy dej konkrétní výběr možností, aby uživatel mohl odpovědět jednou větou.
- Když uživatel upřesní firmu nebo město, preferuj výběr z už nalezených kandidátů.

PRAVIDLA PRO SCHŮZKY:
- Dotaz "poslední schůzky" bez explicitního období: zavolej my_meetings_tool BEZ datumu,
  s vysokým limitem (300–500). Nepoužívej sérii úzkých denních dotazů.
- Pokud je aktivní kontext firmy, schůzky té firmy zjišťuj přes crm_query_tool
  (module="Meetings", filter na account_id) — ne přes my_meetings_tool.

UI KONTEXTU: Kontext může obsahovat ui_focus_hint_cz a pole module/record/record_name/record_id z CRM obrazovky.
  Ber to jako orientační nápovědu (co má uživatel pravděpodobně otevřené), NE jako závazný fakt.
  Přednost má uživatelský vstup nebo výsledek nástroje před UI kontextem.

VZORY:
Dotaz: "kontakty firmy Zlíner"
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "Zlíner", "module": "accounts"}}}}</tool_call>
Po výsledku RAG (account_id=XYZ):
→ <tool_call>{{"name": "crm_query_tool", "args": {{"module": "Contacts", "filters": "[{{\"field\":\"account_id\",\"op\":\"eq\",\"value\":\"XYZ\"}}]"}}}}</tool_call>

Dotaz: "schůzky příští týden"
→ <tool_call>{{"name": "my_meetings_tool", "args": {{"date_from": "pristi_tyden_start", "date_to": "pristi_tyden_end"}}}}</tool_call>

Dotaz: "naplánuj schůzku s [Jméno] na úterý ráno"
Krok 1 — najdi kontakt:
→ <tool_call>{{"name": "rag_search_tool", "args": {{"query": "[Jméno]", "module": "contacts"}}}}</tool_call>
Krok 2 — vytvoř schůzku s contact_id z výsledku (data_json piš jako objekt, NE jako string):
→ <tool_call>{{"name": "crm_action_tool", "args": {{"module": "Meetings", "action": "create", "data_json": {{"fields": {{"contact_name": "[Jméno z výsledku]", "contact_id": "[REAL_CONTACT_ID_Z_VYSLEDKU]", "date_start": "[REAL_DATETIME]", "description": "..."}}}}}}}}}}</tool_call>
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
"""


async def run_agent(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    context: dict[str, Any] | None,
    tools: list,
    chat_history: list[dict[str, str]] | None = None,
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
    history_block = _format_recent_history(chat_history)
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
            f"KONVERZAČNÍ HISTORIE (nejnovější dole):\n{history_block or '(prázdná)'}\n"
            f"User ID: {user_id}"
        )
    else:
        human_text = (
            f"Vstup: {input_text}\n"
            f"KONVERZAČNÍ HISTORIE (nejnovější dole):\n{history_block or '(prázdná)'}\n"
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

    if not final_answer:
        logger.warning("tenant=%s Agent returned empty output after %d steps", tenant_id, len(intermediate_steps))
        final_answer = "Omlouvám se, nepodařilo se zpracovat dotaz. Zkuste ho přeformulovat."

    result = {
        "output": final_answer,
        "intermediate_steps": intermediate_steps,
    }

    log_llm_step(logger, LLM_STEP_FINAL_RESPONSE, {"final_answer": final_answer, "raw_output": final_answer})
    return result
