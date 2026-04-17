# app/engine/filter_builder.py
"""
Builds Coripo REST filter JSON from typed Python parameters.
The LLM never has to know about fieldModule, fieldRel, relationField,
moreThanInclude, etc. — those are Coripo internals.
"""
from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel


class FilterSpec(BaseModel):
    """One field condition the LLM can express."""
    field: str
    op: Literal["eq", "neq", "cont", "starts", "nnull", "null", "gte", "lte", "gt", "lt"]
    value: str | None = None


# Map our simple op names to Coripo API type strings
_OP_MAP: dict[str, str] = {
    "eq":     "eq",
    "neq":    "neq",
    "cont":   "cont",
    "starts": "starts",
    "nnull":  "nnull",
    "null":   "null",
    "gte":    "moreThanInclude",
    "lte":    "lessThanInclude",
    "gt":     "moreThan",
    "lt":     "lessThan",
}

# Which date field to use for each module when date_from/date_to are given
_DATE_FIELD: dict[str, str] = {
    "meetings": "date_start",
    "calls":    "date_start",
    "tasks":    "date_due",
}

_ACTIVITY_MODULES = {"meetings", "calls", "tasks", "notes"}

_MODULE_CANONICAL: dict[str, str] = {
    "meetings": "Meetings",
    "calls": "Calls",
    "tasks": "Tasks",
    "notes": "Notes",
}

_CRM_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)

_ACTIVITY_PARENT_TYPE_ALIASES: dict[str, set[str]] = {
    "Accounts": {"account_id", "accounts.id", "accounts|id", "account_name", "accounts.name", "company"},
    "Contacts": {"contact_id", "contacts.id", "contacts|id", "contact_name", "contacts.name"},
    "Leads": {"lead_id", "leads.id", "leads|id", "lead_name", "leads.name"},
    "Opportunities": {
        "opportunity_id",
        "opportunities.id",
        "opportunities|id",
        "opportunites_id",
        "opportunites.id",
        "opportunites|id",
    },
    "Quotes": {"quote_id", "quotes.id", "quotes|id"},
}

_ACTIVITY_ID_ALIASES: set[str] = {
    "account_id",
    "accounts.id",
    "accounts|id",
    "contact_id",
    "contacts.id",
    "contacts|id",
    "lead_id",
    "leads.id",
    "leads|id",
    "opportunity_id",
    "opportunities.id",
    "opportunities|id",
    "opportunites_id",
    "opportunites.id",
    "opportunites|id",
    "quote_id",
    "quotes.id",
    "quotes|id",
}


def _make_operand(field: str, op_type: str, value: str | None) -> dict[str, Any]:
    return {
        "field": field,
        "fieldModule": None,
        "fieldRel": None,
        "type": op_type,
        "value": value,
        "relationField": None,
    }


def _is_crm_id(value: str | None) -> bool:
    return bool(_CRM_ID_RE.match(str(value or "").strip()))


def _make_activity_parent_operand(
    *,
    module_lower: str,
    field_name: str,
    op: str,
    value: str | None,
) -> dict[str, Any] | None:
    if module_lower not in _ACTIVITY_MODULES:
        return None

    field_norm = (field_name or "").strip().lower()
    parent_type = ""
    for module_name, aliases in _ACTIVITY_PARENT_TYPE_ALIASES.items():
        if field_norm in aliases:
            parent_type = module_name
            break
    if not parent_type:
        return None

    target_field = "parent_id"
    if field_norm not in _ACTIVITY_ID_ALIASES or not _is_crm_id(value):
        target_field = "parent_name"

    # If an ID alias carries a non-ID value, prefer fuzzy company/person match.
    target_op = "cont" if target_field == "parent_name" and op == "eq" else op
    operand = _make_operand(target_field, target_op, value)
    operand["fieldModule"] = _MODULE_CANONICAL.get(module_lower)
    operand["parent_type"] = parent_type
    return operand


def _make_contacts_account_relation_operand(op_type: str, value: str | None) -> dict[str, Any]:
    # Contacts->Accounts relation: keep fieldRel variant and add relate fallback
    # because some Coripo builds only return data for the relate form.
    operands: list[dict[str, Any]] = [
        {
            "field": "id",
            "fieldModule": "Contacts",
            "fieldRel": ["accounts"],
            "type": op_type,
            "value": value,
            "relationField": None,
        }
    ]
    if op_type == "eq" and value:
        operands.append(
            {
                "module": "Accounts",
                "type": "relate",
                "name": "account_name",
                "relationship": ["accounts"],
                "filter": {
                    "operator": "and",
                    "operands": [{"field": "id", "type": "eq", "value": value}],
                },
            }
        )
    return {"operator": "or", "operands": operands}


def build_filter(
    *,
    module: str = "",
    search: str | None = None,
    filters: list[FilterSpec] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """
    Compose a Coripo filter dict from high-level parameters.

    Args:
        module:   CRM module name (used to pick the right date field)
        search:   Full-text search string → maps to field="*", type="cont"
        filters:  List of FilterSpec conditions (AND-ed together)
        date_from: ISO date "YYYY-MM-DD" (inclusive lower bound)
        date_to:   ISO date "YYYY-MM-DD" (inclusive upper bound)

    Returns:
        {"operator": "and", "operands": [...]} ready for the CRM API
    """
    operands: list[dict[str, Any]] = []
    module_lower = (module or "").strip().lower()

    if search:
        operands.append({
            "operator": "and",
            "operands": [_make_operand("*", "cont", search)],
        })

    date_field = _DATE_FIELD.get(module_lower, "date_entered")
    if date_from:
        operands.append(_make_operand(date_field, "moreThanInclude", date_from))
    if date_to:
        operands.append(_make_operand(date_field, "lessThanInclude", date_to))

    account_id_aliases = {"account_id", "accounts.id", "accounts|id"}

    for spec in (filters or []):
        coripo_op = _OP_MAP[spec.op]
        activity_parent_operand = _make_activity_parent_operand(
            module_lower=module_lower,
            field_name=spec.field,
            op=coripo_op,
            value=spec.value,
        )
        if activity_parent_operand is not None:
            operands.append(activity_parent_operand)
            continue
        if module_lower == "contacts" and (spec.field or "").strip().lower() in account_id_aliases:
            operands.append(_make_contacts_account_relation_operand(coripo_op, spec.value))
            continue
        operands.append(_make_operand(spec.field, coripo_op, spec.value))

    return {"operator": "and", "operands": operands}


def build_order(order_by: str | None) -> list[dict[str, Any]]:
    """
    Parse "field:asc" or "field:desc" into Coripo order list.
    Returns [] if order_by is None or malformed.
    """
    if not order_by:
        return []
    parts = order_by.strip().split(":", 1)
    field = parts[0].strip()
    direction = parts[1].strip().upper() if len(parts) > 1 else "ASC"
    if direction not in ("ASC", "DESC"):
        direction = "ASC"
    if not field:
        return []
    return [{"field": field, "sort": direction, "module": None}]
