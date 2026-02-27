from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any

import httpx
from langchain_core.tools import tool

from app.core.config import get_settings
from app.engine.adjustments import ModuleAdjustmentEngine
from app.engine.rag import TenantRAGService
from app.services.crm_client import SugarClient

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    unaccented = "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _infer_module_from_intent(input_text: str, requested_module: str) -> str:
    normalized = _normalize_text(input_text)
    module_norm = (requested_module or "").strip().lower()
    module_aliases = {
        "meeting": "Meetings",
        "meetings": "Meetings",
        "schuzka": "Meetings",
        "schuzky": "Meetings",
        "call": "Calls",
        "calls": "Calls",
        "hovor": "Calls",
        "task": "Tasks",
        "tasks": "Tasks",
        "ukol": "Tasks",
        "note": "Notes",
        "notes": "Notes",
        "poznamka": "Notes",
        "poznamky": "Notes",
    }

    # Respect explicit module provided by the agent/user and only infer when module is missing/unknown.
    explicit_module = module_aliases.get(module_norm)
    if explicit_module:
        return explicit_module
    if module_norm:
        return requested_module

    call_tokens = ("hovor", "telefonat", "zavolej", "volat", "call")
    task_tokens = ("ukol", "task", "todo", "pripomen", "follow up")
    note_tokens = ("poznamk", "zapis", "zapisek", "note")
    meeting_tokens = ("schuzk", "meeting")

    has_call = any(token in normalized for token in call_tokens)
    has_task = any(token in normalized for token in task_tokens)
    has_note = any(token in normalized for token in note_tokens)
    has_meeting = any(token in normalized for token in meeting_tokens)

    if has_task:
        return "Tasks"
    if has_call:
        return "Calls"
    if has_meeting:
        return "Meetings"
    if has_note:
        return "Notes"

    return "Meetings"


def _parse_datetime(value: Any) -> datetime | None:
    raw = _safe_text(value)
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except Exception:
        return None


def _extract_name_value_scalar(raw: Any) -> Any:
    if isinstance(raw, dict):
        if "value" in raw:
            return raw.get("value")
        if "name" in raw and len(raw) == 1:
            return raw.get("name")
    return raw


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    flattened = dict(record)
    name_value_list = flattened.get("name_value_list")
    if isinstance(name_value_list, dict):
        for key, value in name_value_list.items():
            if key not in flattened:
                flattened[key] = _extract_name_value_scalar(value)

    attrs = flattened.get("attributes")
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            if key not in flattened:
                flattened[key] = _extract_name_value_scalar(value)

    return flattened


def _extract_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_list_keys = {"records", "entry_list", "items", "results"}
    ignored_parent_keys = {
        "fields",
        "field_defs",
        "defs",
        "columns",
        "column_defs",
        "labels",
        "module_defs",
        "layout",
        "metadata",
        "menu",
    }

    def is_record_candidate(item: dict[str, Any]) -> bool:
        record_id = _safe_text(item.get("id"))
        if record_id and _UUID_RE.match(record_id):
            return True

        # Fallback for some list/search responses that omit id but still hold entity row data.
        has_person_shape = bool(_safe_text(item.get("first_name"))) and bool(_safe_text(item.get("last_name")))
        has_company_shape = bool(_safe_text(item.get("account_name") or item.get("company")))
        has_crm_dates = bool(_safe_text(item.get("date_modified") or item.get("date_entered")))
        is_field_def_shape = "vname" in item and "type" in item and ("name" in item or "rname" in item)
        return (has_person_shape and (has_company_shape or has_crm_dates)) and not is_field_def_shape

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, list):
            if parent_key in record_list_keys:
                for item in node:
                    if isinstance(item, dict) and is_record_candidate(item):
                        records.append(_flatten_record(item))
            for item in node:
                walk(item, parent_key=parent_key)
            return

        if not isinstance(node, dict):
            return

        flattened = _flatten_record(node)
        if (
            parent_key in record_list_keys
            and is_record_candidate(flattened)
            and parent_key not in ignored_parent_keys
        ):
            records.append(flattened)

        for key, child in node.items():
            walk(child, parent_key=key)

    walk(value)

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in records:
        record_id = _safe_text(item.get("id"))
        if record_id:
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
        deduped.append(item)
    return deduped


def _filter_valid_contact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in records:
        record_id = _safe_text(row.get("id"))
        if not _UUID_RE.match(record_id):
            continue

        name = _safe_text(row.get("name"))
        first = _safe_text(row.get("first_name"))
        last = _safe_text(row.get("last_name"))
        email = _safe_text(row.get("email1"))
        phone = _safe_text(row.get("phone_mobile") or row.get("phone_work"))
        company = _safe_text(row.get("account_name") or row.get("company"))
        if not any([name, first, last, email, phone, company]):
            continue
        valid.append(row)
    return valid


def _parse_date_token(value: str) -> datetime | None:
    raw = _safe_text(value)
    if not raw:
        return None

    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_date_range_from_text(text: str) -> tuple[datetime, datetime] | None:
    source = _safe_text(text)
    if not source:
        return None

    tokens = re.findall(r"\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}", source)
    parsed: list[datetime] = []
    for token in tokens:
        dt = _parse_date_token(token)
        if dt is not None:
            parsed.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

    if len(parsed) < 2:
        return None
    start = min(parsed)
    end = max(parsed) + timedelta(days=1)
    return start, end


def _record_name(record: dict[str, Any]) -> str:
    name = _safe_text(record.get("name"))
    if name:
        return name
    first = _safe_text(record.get("first_name"))
    last = _safe_text(record.get("last_name"))
    full = f"{first} {last}".strip()
    if full:
        return full
    return "(bez nazvu)"


def _best_record_match(records: list[dict[str, Any]], search: str) -> dict[str, Any] | None:
    if not records:
        return None
    search_norm = _normalize_text(search)
    if not search_norm:
        return None
    best: tuple[int, dict[str, Any]] | None = None
    for record in records:
        candidate = _normalize_text(_record_name(record))
        if not candidate:
            continue
        score = 0
        if candidate == search_norm:
            score = 100
        elif candidate.startswith(search_norm):
            score = 85
        elif search_norm in candidate:
            score = 75
        elif candidate in search_norm:
            score = 60
        if best is None or score > best[0]:
            best = (score, record)
    if best and best[0] >= 70:
        return best[1]
    return None


def _crm_detail_link(module: str, record_id: str) -> str:
    clean_module = _safe_text(module) or "Home"
    clean_id = _safe_text(record_id)
    return f"/#detail/{clean_module}/{clean_id}" if clean_id else "/#"


def _format_date(value: Any) -> str:
    dt = _parse_datetime(value)
    if dt is None:
        return _safe_text(value)
    return dt.strftime("%d.%m.%Y %H:%M")


def _non_empty_meta(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        text = _safe_text(value)
        if text:
            clean[key] = text
    return clean


def _record_card(module: str, title: str, record_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{module}-{record_id}" if record_id else f"{module}-row",
        "type": "record",
        "title": title,
        "tag": module,
        "meta": _non_empty_meta(meta),
        "actions": [
            {
                "label": "Otevrit v CRM",
                "intent": "primary",
                "action": "link",
                "url": _crm_detail_link(module, record_id),
            }
        ]
        if record_id
        else [],
    }


def _table_card(
    *,
    title: str,
    tag: str,
    columns: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"table-{_normalize_text(title) or 'crm'}",
        "type": "table",
        "title": title,
        "tag": tag,
        "columns": columns,
        "rows": rows,
    }


def _extract_person_name(query: str, context: dict[str, Any] | None) -> str:
    patterns = [
        r"\bkdo\s+je\s+([^?.!,]+)",
        r"\bkontakt\s+([^?.!,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return _safe_text(match.group(1)).strip(" '\"")

    entities = context.get("entities") if isinstance(context, dict) else {}
    if isinstance(entities, dict):
        candidate = _safe_text(entities.get("contact_name"))
        if candidate:
            return candidate
    return _safe_text(query)


def _extract_company_name(query: str, context: dict[str, Any] | None) -> str:
    patterns = [
        r"\b(?:k|ke)\s+firme\s+([^?.!,]+)",
        r"\b(?:k|ke)\s+firm[eě]\s+([^?.!,]+)",
        r"\bspolecnosti\s+([^?.!,]+)",
        r"\bfirma\s+([^?.!,]+)",
        r"\bfirmy\s+([^?.!,]+)",
        r"\bcompany\s+([^?.!,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return _safe_text(match.group(1)).strip(" '\"")

    entities = context.get("entities") if isinstance(context, dict) else {}
    if isinstance(entities, dict):
        candidate = _safe_text(entities.get("account_name"))
        if candidate:
            return candidate
    return _safe_text(query)


def _infer_data_query_type(query: str, requested: str = "auto") -> str:
    explicit = _safe_text(requested).lower()
    normalized = _normalize_text(query)
    has_contact = any(token in normalized for token in ("kontakt", "contacts", "contact"))
    has_company = any(token in normalized for token in ("firma", "spolecnost", "company", "firmy", "firme"))

    # LLM often sends query_type="contacts" even when user asked contacts by company.
    # Prefer user-intent heuristic over broad explicit aliases.
    if explicit in {"contacts", "contact"} and has_contact and has_company:
        return "contacts_by_company"

    explicit_aliases = {
        "meetings_next_week": "meetings_next_week",
        "meetings_range": "meetings_range",
        "contact": "contact_by_name",
        "contacts": "contact_by_name",
        "contact_by_name": "contact_by_name",
        "contacts_by_company": "contacts_by_company",
        "company_contacts": "contacts_by_company",
        "generic": "generic_search",
        "generic_search": "generic_search",
    }
    if explicit in explicit_aliases:
        return explicit_aliases[explicit]
    if explicit in {"meetings", "meeting", "schuzky", "schuzka"}:
        normalized_explicit_query = _normalize_text(query)
        if "pristi tyden" in normalized_explicit_query or "next week" in normalized_explicit_query:
            return "meetings_next_week"
        return "meetings_range"
    if explicit in {"meetings_next_week", "contact_by_name", "contacts_by_company", "generic_search"}:
        return explicit

    if "kdo je" in normalized:
        return "contact_by_name"

    has_meeting = any(token in normalized for token in ("schuzk", "meeting"))
    has_next_week = "pristi tyden" in normalized or "next week" in normalized
    if has_meeting and has_next_week:
        return "meetings_next_week"
    if has_meeting and ("obdobi" in normalized or "od " in normalized or "do " in normalized):
        return "meetings_range"
    if has_meeting and _extract_date_range_from_text(query):
        return "meetings_range"

    if has_contact and has_company:
        return "contacts_by_company"

    if has_contact:
        return "contact_by_name"

    return "generic_search"


def _filter_records_by_company(records: list[dict[str, Any]], company_name: str) -> list[dict[str, Any]]:
    normalized_company = _normalize_text(company_name)
    if not normalized_company:
        return records

    filtered: list[dict[str, Any]] = []
    for record in records:
        account = _normalize_text(_safe_text(record.get("account_name") or record.get("company")))
        if normalized_company in account or account in normalized_company:
            filtered.append(record)
    return filtered or records


def _dedupe_and_limit(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_id = _safe_text(record.get("id"))
        name = _normalize_text(_record_name(record))
        fingerprint = f"{record_id}|{name}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
        if len(deduped) >= limit:
            break
    return deduped


def _merge_records_by_id(*record_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rows in record_sets:
        for row in rows:
            rec_id = _safe_text(row.get("id"))
            if not rec_id or rec_id in seen:
                continue
            seen.add(rec_id)
            merged.append(row)
    return merged


def _meeting_cards(records: list[dict[str, Any]], total_count: int) -> list[dict[str, Any]]:
    if not records:
        return []
    if total_count <= 4:
        cards = []
        for row in records:
            cards.append(
                _record_card(
                    "Meetings",
                    _record_name(row),
                    _safe_text(row.get("id")),
                    {
                        "Datum": _format_date(row.get("date_start")),
                        "Misto": _safe_text(row.get("location")),
                        "Stav": _safe_text(row.get("status")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        rec_id = _safe_text(row.get("id"))
        rows.append(
            {
                "id": rec_id,
                "cells": {
                    "date_start": _format_date(row.get("date_start")),
                    "name": _record_name(row),
                    "location": _safe_text(row.get("location")),
                    "status": _safe_text(row.get("status")),
                },
                "link": {
                    "label": "Detail",
                    "url": _crm_detail_link("Meetings", rec_id),
                },
            }
        )

    return [
        _table_card(
            title=f"Schuzky ({total_count})",
            tag="Meetings",
            columns=[
                {"key": "date_start", "label": "Datum"},
                {"key": "name", "label": "Nazev"},
                {"key": "location", "label": "Misto"},
                {"key": "status", "label": "Stav"},
            ],
            rows=rows,
        )
    ]


def _contact_cards(records: list[dict[str, Any]], total_count: int, title: str) -> list[dict[str, Any]]:
    if not records:
        return []

    if total_count <= 4:
        cards = []
        for row in records:
            cards.append(
                _record_card(
                    "Contacts",
                    _record_name(row),
                    _safe_text(row.get("id")),
                    {
                        "Firma": _safe_text(row.get("account_name") or row.get("company")),
                        "E-mail": _safe_text(row.get("email1")),
                        "Telefon": _safe_text(row.get("phone_mobile") or row.get("phone_work")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        rec_id = _safe_text(row.get("id"))
        rows.append(
            {
                "id": rec_id,
                "cells": {
                    "name": _record_name(row),
                    "company": _safe_text(row.get("account_name") or row.get("company")),
                    "email": _safe_text(row.get("email1")),
                    "phone": _safe_text(row.get("phone_mobile") or row.get("phone_work")),
                },
                "link": {
                    "label": "Detail",
                    "url": _crm_detail_link("Contacts", rec_id),
                },
            }
        )

    return [
        _table_card(
            title=title,
            tag="Contacts",
            columns=[
                {"key": "name", "label": "Jmeno"},
                {"key": "company", "label": "Firma"},
                {"key": "email", "label": "E-mail"},
                {"key": "phone", "label": "Telefon"},
            ],
            rows=rows,
        )
    ]


def _generic_cards(records: list[dict[str, Any]], total_count: int) -> list[dict[str, Any]]:
    if not records:
        return []
    if total_count <= 4:
        cards = []
        for row in records:
            module = _safe_text(row.get("_module_hint")) or "CRM"
            cards.append(
                _record_card(
                    module,
                    _record_name(row),
                    _safe_text(row.get("id")),
                    {
                        "Modul": module,
                        "Stav": _safe_text(row.get("status")),
                        "Upraveno": _format_date(row.get("date_modified")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        module = _safe_text(row.get("_module_hint")) or "CRM"
        rec_id = _safe_text(row.get("id"))
        rows.append(
            {
                "id": f"{module}-{rec_id}" if rec_id else module,
                "cells": {
                    "module": module,
                    "name": _record_name(row),
                    "status": _safe_text(row.get("status")),
                },
                "link": {
                    "label": "Detail",
                    "url": _crm_detail_link(module, rec_id),
                }
                if rec_id
                else None,
            }
        )

    return [
        _table_card(
            title=f"Vysledky ({total_count})",
            tag="CRM",
            columns=[
                {"key": "module", "label": "Modul"},
                {"key": "name", "label": "Nazev"},
                {"key": "status", "label": "Stav"},
            ],
            rows=rows,
        )
    ]


def _next_week_range(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.now()
    days_to_next_monday = 7 - current.weekday()
    start = (current + timedelta(days=days_to_next_monday)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = start + timedelta(days=7)
    return start, end


def _sort_records_by_datetime(records: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: _parse_datetime(row.get(field_name)) or datetime.max,
    )


def build_tools(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    request_context: dict[str, Any] | None,
    crm_client: SugarClient,
    rag_service: TenantRAGService | None,
    action_confirmation: bool = False,
) -> list:
    settings = get_settings()
    adjustment_engine = ModuleAdjustmentEngine(
        tenant_id=tenant_id,
        user_id=user_id,
        crm_client=crm_client,
        input_text=input_text,
        request_context=request_context,
    )

    @tool("crm_action_tool")
    async def crm_action_tool(module: str, action: str, data_json: str = "{}") -> str:
        """Execute a SugarCRM module action using module/action/data JSON."""
        if settings.crm_mode.lower() == "off":
            return json.dumps(
                {
                    "status": "crm-disabled",
                    "message": "CRM mode is off, action skipped",
                },
                ensure_ascii=False,
            )

        try:
            data: dict[str, Any] = json.loads(data_json) if data_json else {}
            if not isinstance(data, dict):
                raise ValueError("data_json must decode to an object")
        except Exception as exc:
            return json.dumps({"error": f"Invalid data_json: {exc}"}, ensure_ascii=False)

        normalized_action = (action or "").strip().lower()
        mutating_actions = {"create", "update", "patch", "delete"}
        effective_module = module
        if normalized_action in {"create", "update", "patch"}:
            effective_module = _infer_module_from_intent(input_text=input_text, requested_module=module)

        adjustment = await adjustment_engine.apply(
            module=effective_module,
            action=normalized_action,
            data=data,
        )
        data = adjustment.data

        if normalized_action in mutating_actions and not action_confirmation:
            return json.dumps(
                {
                    "status": "confirmation_required",
                    "message": (
                        "Akce mění CRM data a vyžaduje potvrzení uživatele. "
                        # "Akce mění CRM data a vyžaduje explicitní potvrzení uživatele. "
                        # "Pro provedení zopakujte požadavek s context.confirm_action=true."
                    ),
                    "pending_action": {
                        "module": module,
                        "effective_module": effective_module,
                        "action": normalized_action,
                        "data": data,
                    },
                    "adjustments": adjustment.notes,
                },
                ensure_ascii=False,
            )

        data.setdefault("requested_by_user_id", user_id)
        try:
            result = await crm_client.execute_module_action(
                module=effective_module,
                action=action,
                data=data,
            )
            if normalized_action in mutating_actions and rag_service is not None:
                try:
                    ingested = await rag_service.ingest_from_crm(
                        tenant_id=tenant_id,
                        crm_client=crm_client,
                        module=effective_module,
                    )
                    if isinstance(result, dict):
                        result["_rag_ingested"] = ingested
                except Exception:
                    if isinstance(result, dict):
                        result["_rag_ingest_error"] = "failed"
            if adjustment.notes and isinstance(result, dict):
                result["_adjustments"] = adjustment.notes
            return json.dumps(result, ensure_ascii=False)
        except httpx.HTTPStatusError as exc:
            body_preview = ""
            try:
                body_preview = exc.response.text[:1000]
            except Exception:
                body_preview = ""
            return json.dumps(
                {
                    "status": "crm_http_error",
                    "http_status": exc.response.status_code,
                    "url": str(exc.request.url),
                    "message": (
                        "CRM rejected the action. Confirm field mapping/required "
                        "values for this module."
                    ),
                    "response_body": body_preview,
                    "failed_action": {
                        "module": module,
                        "effective_module": effective_module,
                        "action": normalized_action,
                        "data": data,
                    },
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "crm_action_error",
                    "message": str(exc),
                    "failed_action": {
                        "module": module,
                        "effective_module": effective_module,
                        "action": normalized_action,
                        "data": data,
                    },
                },
                ensure_ascii=False,
            )

    @tool("rag_search_tool")
    def rag_search_tool(query: str, limit: int = 5) -> str:
        """Search tenant-scoped knowledge in Qdrant using semantic retrieval."""
        if rag_service is None:
            return json.dumps(
                {"warning": "RAG unavailable", "results": []},
                ensure_ascii=False,
            )
        results = rag_service.search(tenant_id=tenant_id, query=query, limit=limit)
        return json.dumps(results, ensure_ascii=False)

    @tool("crm_search_tool")
    async def crm_search_tool(query: str, scope: str = "all") -> str:
        """Run SugarCRM global search for contacts/accounts/meetings/all."""
        result = await crm_client.generic_search(query=query, scope=scope)
        return json.dumps(result, ensure_ascii=False)

    @tool("crm_data_tool")
    async def crm_data_tool(query: str, query_type: str = "auto", limit: int = 20) -> str:
        """Read CRM data for user questions and return UI-friendly cards/table payload."""
        if settings.crm_mode.lower() == "off":
            return json.dumps(
                {
                    "status": "crm-disabled",
                    "message_to_user": "CRM je vypnuté, data teď nemohu načíst.",
                    "cards": [],
                },
                ensure_ascii=False,
            )

        safe_limit = max(1, min(int(limit or 20), 30))
        user_query = _safe_text(input_text)
        effective_query = _safe_text(query) or user_query
        # Prefer original user text for intent classification to avoid LLM-rewritten date mistakes.
        resolved_type = _infer_data_query_type(query=user_query or effective_query, requested=query_type)
        explicit_range = _extract_date_range_from_text(effective_query)

        try:
            if resolved_type in {"meetings_next_week", "meetings_range"}:
                if resolved_type == "meetings_next_week":
                    range_start, range_end = _next_week_range()
                elif explicit_range:
                    range_start, range_end = explicit_range
                else:
                    range_start, range_end = _next_week_range()

                result = await crm_client.execute_module_action(
                    module="Meetings",
                    action="list",
                    data={"query": "", "max_results": 250},
                )
                all_records = _extract_records(result)

                selected: list[dict[str, Any]] = []
                for row in all_records:
                    start_value = row.get("date_start") or row.get("date_entered")
                    dt = _parse_datetime(start_value)
                    if dt is None or not (range_start <= dt < range_end):
                        continue
                    assigned_user = _safe_text(row.get("assigned_user_id") or row.get("users_id_c"))
                    if assigned_user and assigned_user != _safe_text(user_id):
                        continue
                    selected.append(row)

                selected = _sort_records_by_datetime(selected, "date_start")
                limited = _dedupe_and_limit(selected, safe_limit)
                total_count = len(selected)
                cards = _meeting_cards(limited, total_count)

                if total_count == 0:
                    message = (
                        "Na příští týden nemáte žádné schůzky. "
                        f"Kontrolované období: {range_start.strftime('%d.%m.%Y')} - {range_end.strftime('%d.%m.%Y')}."
                    )
                else:
                    message = (
                        f"Na příští týden jsem našla {total_count} schůzek. "
                        f"Období: {range_start.strftime('%d.%m.%Y')} - {range_end.strftime('%d.%m.%Y')}."
                    )

                return json.dumps(
                    {
                        "status": "ok",
                        "query_type": resolved_type,
                        "module": "Meetings",
                        "total_count": total_count,
                        "cards": cards,
                        "message_to_user": message,
                    },
                    ensure_ascii=False,
                )

            if resolved_type == "contact_by_name":
                person_name = _extract_person_name(user_query or effective_query, request_context)
                list_result = await crm_client.execute_module_action(
                    module="Contacts",
                    action="list",
                    data={"query": person_name, "q": person_name, "max_results": max(50, safe_limit)},
                )
                search_result = await crm_client.generic_search(query=person_name, scope="contacts")
                all_records = _merge_records_by_id(
                    _filter_valid_contact_records(_extract_records(list_result)),
                    _filter_valid_contact_records(_extract_records(search_result)),
                )
                match = _best_record_match(all_records, person_name)
                selected = [match] if match else _dedupe_and_limit(all_records, safe_limit)
                total_count = len(selected)
                cards = _contact_cards(selected, total_count, title=f"Kontakty ({total_count})")

                if total_count == 0:
                    message = f"Kontakt '{person_name}' jsem nenašla."
                elif match:
                    message = f"Kontakt '{_record_name(match)}' jsem našla."
                else:
                    message = f"Našla jsem {total_count} kontaktů k dotazu '{person_name}'."

                return json.dumps(
                    {
                        "status": "ok",
                        "query_type": resolved_type,
                        "module": "Contacts",
                        "total_count": total_count,
                        "cards": cards,
                        "message_to_user": message,
                    },
                    ensure_ascii=False,
                )

            if resolved_type == "contacts_by_company":
                company_name = _extract_company_name(user_query or effective_query, request_context)
                contacts_list_result = await crm_client.execute_module_action(
                    module="Contacts",
                    action="list",
                    data={"query": company_name, "q": company_name, "max_results": 200},
                )
                contacts_search_result = await crm_client.generic_search(query=company_name, scope="contacts")
                contacts = _merge_records_by_id(
                    _filter_valid_contact_records(_extract_records(contacts_list_result)),
                    _filter_valid_contact_records(_extract_records(contacts_search_result)),
                )
                filtered_contacts = _filter_records_by_company(contacts, company_name)
                selected = _dedupe_and_limit(filtered_contacts, safe_limit)
                total_count = len(filtered_contacts)
                cards = _contact_cards(selected, total_count, title=f"Kontakty firmy ({total_count})")

                if total_count == 0:
                    message = f"Kontakty k firmě '{company_name}' jsem nenašla."
                else:
                    message = f"K firmě '{company_name}' jsem našla {total_count} kontaktů."

                return json.dumps(
                    {
                        "status": "ok",
                        "query_type": resolved_type,
                        "module": "Contacts",
                        "total_count": total_count,
                        "cards": cards,
                        "message_to_user": message,
                    },
                    ensure_ascii=False,
                )

            aggregate = await crm_client.generic_search(query=effective_query, scope="all")
            merged: list[dict[str, Any]] = []
            if isinstance(aggregate, dict):
                module_map = {
                    "contacts": "Contacts",
                    "accounts": "Accounts",
                    "meetings": "Meetings",
                }
                for module_key, payload in aggregate.items():
                    module_name = module_map.get(str(module_key).lower(), str(module_key))
                    for row in _extract_records(payload):
                        row["_module_hint"] = module_name
                        merged.append(row)
            merged = _dedupe_and_limit(merged, safe_limit)
            total_count = len(merged)
            cards = _generic_cards(merged, total_count)
            message = (
                f"Našla jsem {total_count} záznamů pro dotaz '{effective_query}'."
                if total_count
                else f"K dotazu '{effective_query}' jsem nic nenašla."
            )
            return json.dumps(
                {
                    "status": "ok",
                    "query_type": "generic_search",
                    "module": "CRM",
                    "total_count": total_count,
                    "cards": cards,
                    "message_to_user": message,
                },
                ensure_ascii=False,
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            return json.dumps(
                {
                    "status": "crm_http_error",
                    "query_type": resolved_type,
                    "http_status": status_code,
                    "message_to_user": "CRM data se nepodařilo načíst. Zkuste to prosím znovu.",
                    "cards": [],
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "crm_data_error",
                    "query_type": resolved_type,
                    "message_to_user": "Nastala chyba při načítání CRM dat.",
                    "error": str(exc),
                    "cards": [],
                },
                ensure_ascii=False,
            )

    return [crm_action_tool, rag_search_tool, crm_search_tool, crm_data_tool]
