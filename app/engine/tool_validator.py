from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

_CRM_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_WRITE_MODULES = {"meetings", "calls", "tasks", "notes", "contacts", "accounts", "leads"}
_READ_MODULES = {"meetings", "calls", "tasks", "notes", "contacts", "accounts", "leads"}
_RAG_MODULES = {"", "contacts", "accounts", "meetings", "calls", "tasks", "notes", "leads"}

class CrmActionToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module: str
    action: str
    data_json: str = "{}"
    record_id: str | None = None


class RagSearchToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    limit: int = 5
    module: str = ""


class MyMeetingsToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date_from: str
    date_to: str
    limit: int = 100


class CrmQueryToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module: str
    search: str | None = None
    filters: str = "[]"
    date_from: str | None = None
    date_to: str | None = None
    order_by: str | None = None
    limit: int = 20


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_valid_crm_id(value: Any) -> bool:
    return bool(_CRM_ID_RE.match(_safe_text(value)))


def _first_record_id(payload: dict[str, Any]) -> str:
    candidates = [
        payload.get("id"),
        payload.get("record_id"),
        payload.get("recordId"),
    ]
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    candidates.extend(
        [
            fields.get("id"),
            fields.get("record_id"),
            fields.get("recordId"),
        ]
    )
    for value in candidates:
        text = _safe_text(value)
        if text:
            return text
    return ""


def _load_data_json(data_json: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = _safe_text(data_json) or "{}"
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, f"Invalid data_json: {exc}"
    if not isinstance(parsed, dict):
        return None, "Invalid data_json: expected a JSON object."
    return parsed, None


def validate_crm_action_call(*, module: str, action: str, data_json: str) -> str | None:
    module_norm = _safe_text(module).lower()
    action_norm = _safe_text(action).lower()
    if module_norm not in _WRITE_MODULES:
        return f"Unsupported module '{module}'."
    if action_norm not in {"create", "update", "patch", "delete"}:
        return f"Unsupported action '{action}'."

    payload, data_err = _load_data_json(data_json)
    if data_err:
        return data_err
    if payload is None:
        return "Invalid data_json."

    record_id = _first_record_id(payload)
    if action_norm in {"update", "patch", "delete"}:
        if not _is_valid_crm_id(record_id):
            return "Update/Patch/Delete requires a valid target record id."
    if action_norm == "create" and record_id:
        return "Create action cannot include target record id."

    if action_norm in {"update", "patch"} and module_norm in {"meetings", "calls", "tasks", "notes"}:
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        related_ids = {
            _safe_text(payload.get("contact_id")),
            _safe_text(payload.get("account_id")),
            _safe_text(fields.get("contact_id")),
            _safe_text(fields.get("account_id")),
        }
        parent_type = _safe_text(fields.get("parent_type")).lower()
        parent_id = _safe_text(fields.get("parent_id"))
        if parent_id and parent_type in {"contacts", "accounts"}:
            related_ids.add(parent_id)
        related_ids.discard("")
        if record_id and record_id in related_ids:
            return "Target record id collides with related entity id."

    return None


def validate_rag_search_call(*, query: str, limit: int, module: str) -> str | None:
    if not _safe_text(query):
        return "Query is required."
    if int(limit) < 1 or int(limit) > 100:
        return "Limit must be between 1 and 100."
    if _safe_text(module).lower() not in _RAG_MODULES:
        return f"Unsupported module filter '{module}'."
    return None


def validate_my_meetings_call(*, date_from: str, date_to: str, limit: int) -> str | None:
    if not _DATE_RE.match(_safe_text(date_from)):
        return "date_from must be in YYYY-MM-DD format."
    if not _DATE_RE.match(_safe_text(date_to)):
        return "date_to must be in YYYY-MM-DD format."
    if int(limit) < 1 or int(limit) > 500:
        return "Limit must be between 1 and 500."
    return None


def validate_crm_query_call(
    *,
    module: str,
    filters: str,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    order_by: str | None,
) -> str | None:
    if _safe_text(module).lower() not in _READ_MODULES:
        return f"Unsupported module '{module}'."
    if int(limit) < 1 or int(limit) > 500:
        return "Limit must be between 1 and 500."

    try:
        parsed_filters = json.loads(_safe_text(filters) or "[]")
    except Exception as exc:
        return f"Invalid filters JSON: {exc}"
    if not isinstance(parsed_filters, list):
        return "Invalid filters JSON: expected an array."
    for item in parsed_filters:
        if not isinstance(item, dict):
            return "Invalid filters JSON: every filter must be an object."

    if date_from and not _DATE_RE.match(_safe_text(date_from)):
        return "date_from must be in YYYY-MM-DD format."
    if date_to and not _DATE_RE.match(_safe_text(date_to)):
        return "date_to must be in YYYY-MM-DD format."

    order = _safe_text(order_by)
    if order:
        parts = order.split(":")
        if len(parts) != 2:
            return "order_by must follow field:asc|desc format."
        direction = parts[1].strip().lower()
        if direction not in {"asc", "desc"}:
            return "order_by direction must be asc or desc."
    return None
