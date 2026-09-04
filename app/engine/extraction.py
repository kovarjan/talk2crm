# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Smart Paste field extraction.

Read-only, tool-using agent call (same family as recommend_actions): given a
module's live ai_schema field list, the record's current field values, and a
chat history culminating in the user's pasted/typed text, propose values for
those fields. Never writes to the CRM — the caller (Coripo FE) applies the
result into the currently-open EditView form's own live state; the user's own
Save is what actually persists anything.

Relate fields may be resolved to a real CRM id using the read-only CRM/RAG
tools available to this agent; when no confident match exists, only the
guessed name is returned (no id), so the caller shows it as a manual-pick
hint rather than linking to a fabricated record.
"""

from __future__ import annotations

import json
from typing import Any

from app.engine.agent import run_agent
from app.engine.tools import build_tools
from app.services.crm_client import CoripoClient

EXTRACT_FIELDS_ALLOWED_TOOLS = {"rag_search_tool", "crm_query_tool", "get_company_overview"}

EXTRACT_FIELDS_SYSTEM_PROMPT = """\
Jsi asistent, který z vloženého textu (např. e-mailu od klienta) vyplňuje
pole formuláře v CRM. Nikdy nic v CRM nevytváříš ani needituješ — pouze
navrhuješ hodnoty polí, které uživatel sám zkontroluje a uloží.

Dostaneš:
- schéma modulu: seznam polí s "name", "type", "label", případně "options"
  (pro type "enum"/"multi_enum") nebo "target_module" (pro type "relate")
- aktuální hodnoty polí ve formuláři
- historii konverzace, poslední zpráva je to, co má být zpracováno teď

Pravidla:
- Vracej POUZE pole, která existují ve schématu (přesný "name"). Cokoliv
  jiného bude stejně zahozeno, takže si nevymýšlej pole navíc.
- Pole, pro která text neobsahuje novou informaci a ani jsi je needitoval(a)
  v této zprávě, nech beze změny — vrať aktuální hodnotu, ne prázdnou.
- Pro pole typu "enum"/"multi_enum" vracej pouze hodnotu ze seznamu "options"
  (přesně tak, jak je tam napsaná), jinak pole vynech.
- Pro pole typu "relate" máš k dispozici READ-ONLY nástroje pro CRM
  (typicky crm_query_tool nebo get_company_overview) — použij je, abys
  zkusil(a) najít existující záznam odpovídající jménu z textu v cílovém
  modulu ("target_module"). Pokud najdeš jednoznačnou shodu, vrať
  {"id": "...", "name": "..."}. Pokud ne, vrať pouze {"name": "..."} bez
  "id" — NIKDY si nevymýšlej id.
- Piš v jazyce vstupního textu, výchozí je čeština.

Až budeš mít dost informací (po použití nástrojů, pokud je to potřeba),
vrať KONEČNOU odpověď výhradně v tagu <answer> obsahujícím JSON objekt:
{
  "message": "krátká věta pro uživatele o tom, co jsi vyplnil(a) nebo co chybí",
  "fields": {"nazev_pole": hodnota, ...}
}
"""


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            from json_repair import repair_json

            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

    return {}


def _field_type_map(field_schema: dict[str, Any]) -> dict[str, str]:
    types: dict[str, str] = {}
    for section in field_schema.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for field in section.get("fields") or []:
            if not isinstance(field, dict):
                continue
            name = field.get("name")
            field_type = field.get("type")
            if name and field_type:
                types[str(name)] = str(field_type)
    return types


def _coerce_field_value(field_type: str, raw_value: Any) -> Any | None:
    if field_type in {"relate", "polymorphic_relate"}:
        if not isinstance(raw_value, dict):
            return None
        name = raw_value.get("name")
        if not name:
            return None
        coerced: dict[str, str] = {"name": str(name)}
        record_id = raw_value.get("id")
        if record_id:
            coerced["id"] = str(record_id)
        return coerced

    # Every other standard type in this schema (text/textarea/email/phone/
    # enum/multi_enum/date/datetime/boolean/number_int/number_decimal) is a
    # plain scalar value on the wire; a dict/list here means the model
    # hallucinated structure for a field that doesn't have any.
    if isinstance(raw_value, (dict, list)):
        return None
    return raw_value


def _validate_fields_against_schema(raw_fields: Any, field_schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_fields, dict):
        return {}

    types = _field_type_map(field_schema)
    result: dict[str, Any] = {}
    for name, value in raw_fields.items():
        field_type = types.get(str(name))
        if field_type is None:
            continue  # not a real field on this schema -> dropped, never passed through
        coerced = _coerce_field_value(field_type, value)
        if coerced is None:
            continue
        result[str(name)] = coerced
    return result


async def extract_fields(
    *,
    tenant_id: str,
    user_id: str,
    module: str,
    record_id: str | None,
    field_schema: dict[str, Any],
    current_values: dict[str, Any],
    messages: list[dict[str, str]],
    crm_client: CoripoClient,
    rag_service: Any | None,
    db: Any | None = None,
) -> dict[str, Any]:
    """Run the read-only extraction agent and return validated field values.
    Never mutates the CRM — see module docstring."""

    schema_json = json.dumps(field_schema, ensure_ascii=False)
    current_values_json = json.dumps(current_values or {}, ensure_ascii=False)
    input_text = (
        f"Modul: {module}. ID záznamu: {record_id or '(nový záznam)'}.\n"
        f"Schéma polí (JSON): {schema_json}\n"
        f"Aktuální hodnoty polí (JSON): {current_values_json}"
    )

    chat_history = [
        {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
        for item in (messages or [])
        if isinstance(item, dict) and item.get("content")
    ]

    tools = [
        tool
        for tool in build_tools(
            tenant_id=tenant_id,
            user_id=user_id,
            input_text=input_text,
            request_context=None,
            crm_client=crm_client,
            rag_service=rag_service,
            action_confirmation=False,
        )
        if getattr(tool, "name", "") in EXTRACT_FIELDS_ALLOWED_TOOLS
    ]

    result = await run_agent(
        tenant_id=tenant_id,
        user_id=user_id,
        input_text=input_text,
        context=None,
        tools=tools,
        chat_history=chat_history,
        emit=None,
        db=db,
        system_prompt_override=EXTRACT_FIELDS_SYSTEM_PROMPT,
        capture_skills=False,
    )

    raw_output = str(result.get("output") or "")
    parsed = _extract_json_object(raw_output)
    fields = _validate_fields_against_schema(parsed.get("fields"), field_schema)
    message = str(parsed.get("message") or "").strip()
    if not message:
        message = "Aktualizoval(a) jsem pole podle vloženého textu."

    return {"message": message, "fields": fields}
