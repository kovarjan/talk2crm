# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Tool-using follow-up action recommendations.

Unlike /generate/ (plain LLM call, no tools) and /process-input/ (full
conversational agent that can write to the CRM), this runs a read-only
tool loop: the model may look up the related account/contact's existing
open tasks, opportunities, etc. before proposing follow-ups, so it can
avoid suggesting duplicates — but it can never create or modify records
itself. The caller (Coripo/PHP) turns each suggestion into a normal
create-command dry-run card using the existing confirm/create flow.
"""

from __future__ import annotations

import json
from typing import Any

from app.engine.agent import run_agent
from app.engine.tools import build_tools
from app.services.crm_client import CoripoClient

RECOMMEND_ACTIONS_SYSTEM_PROMPT = """\
Jsi obchodní asistent, který po skončení schůzky nebo hovoru navrhuje konkrétní
následné kroky (úkoly, telefonáty) obchodníkovi v CRM.

K dispozici máš READ-ONLY nástroje pro CRM (vyhledávání záznamů, přehled firmy,
souhrn schůzek). NEMÁŠ k dispozici žádný nástroj pro zápis — nikdy nic
nevytváříš ani needituješ, pouze navrhuješ.

Nástroje volej pomocí tagu:
<tool_call>{"name": "nazev_nastroje", "args": {...}}</tool_call>

Použij nástroje (typicky get_company_overview nebo crm_query_tool na související
firmu/kontakt), abys zjistil, jestli už neexistuje otevřený úkol nebo nabídka
pokrývající stejnou věc — ať nenavrhuješ duplicity.

Až budeš mít dost informací, vrať KONEČNOU odpověď výhradně v tagu <answer>
obsahujícím JSON pole (0 až 4 položky), v tomto formátu:
[
  {
    "type": "task" | "call",
    "title": "krátký název, max 8 slov",
    "description": "1-2 věty, proč tuto akci navrhuješ",
    "due_in_days": číslo dní od dnes (celé číslo, např. 30 pro 'za měsíc'),
    "reasoning": "krátké zdůvodnění na základě přepisu/kontextu"
  }
]

Pravidla:
- Piš v češtině, žádný markdown.
- Vracej pouze konkrétní, akční návrhy podložené přepisem/poznámkou ze schůzky
  (např. přislíbil poslat nabídku, domluvit další schůzku, zavolat zpět).
- Pokud přepis neobsahuje nic, co by vyžadovalo následnou akci, vrať prázdné pole [].
- Nikdy nevolej nástroj pro zápis (žádný takový k dispozici není) a nikdy sám
  nic v CRM nevytvářej.
"""


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list):
            return [item for item in parsed["actions"] if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            from json_repair import repair_json

            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, list):
                return [item for item in repaired if isinstance(item, dict)]
        except Exception:
            pass

    return []


def _normalize_action(item: dict[str, Any]) -> dict[str, Any] | None:
    action_type = str(item.get("type") or "task").strip().lower()
    if action_type not in {"task", "call"}:
        action_type = "task"

    title = str(item.get("title") or "").strip()
    if not title:
        return None

    due_in_days = item.get("due_in_days")
    try:
        due_in_days = int(due_in_days)
    except (TypeError, ValueError):
        due_in_days = None

    return {
        "type": action_type,
        "title": title,
        "description": str(item.get("description") or "").strip(),
        "due_in_days": due_in_days,
        "reasoning": str(item.get("reasoning") or "").strip(),
    }


async def recommend_actions(
    *,
    tenant_id: str,
    user_id: str,
    module: str,
    record_id: str,
    text: str,
    context: dict[str, Any] | None,
    crm_client: CoripoClient,
    rag_service: Any | None,
    db: Any | None = None,
) -> list[dict[str, Any]]:
    """Run a read-only tool-using agent and return a list of suggested
    follow-up actions. Never mutates the CRM."""

    input_text = (
        f"Zdrojový záznam: {module} ({record_id}).\n"
        f"Přepis / poznámka ze schůzky nebo hovoru:\n{text.strip()}"
    )

    tools = [
        tool
        for tool in build_tools(
            tenant_id=tenant_id,
            user_id=user_id,
            input_text=input_text,
            request_context=context,
            crm_client=crm_client,
            rag_service=rag_service,
            action_confirmation=False,
        )
        if getattr(tool, "name", "") != "crm_action_tool"
    ]

    result = await run_agent(
        tenant_id=tenant_id,
        user_id=user_id,
        input_text=input_text,
        context=context,
        tools=tools,
        chat_history=None,
        emit=None,
        db=db,
        system_prompt_override=RECOMMEND_ACTIONS_SYSTEM_PROMPT,
        capture_skills=False,
    )

    raw_output = str(result.get("output") or "")
    actions = [
        normalized
        for item in _extract_json_array(raw_output)
        if (normalized := _normalize_action(item)) is not None
    ]
    return actions[:4]
