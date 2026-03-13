# app/engine/filter_builder.py
"""
Builds Coripo REST filter JSON from typed Python parameters.
The LLM never has to know about fieldModule, fieldRel, relationField,
moreThanInclude, etc. — those are Coripo internals.
"""
from __future__ import annotations

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


def _make_operand(field: str, op_type: str, value: str | None) -> dict[str, Any]:
    return {
        "field": field,
        "fieldModule": None,
        "fieldRel": None,
        "type": op_type,
        "value": value,
        "relationField": None,
    }


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

    for spec in (filters or []):
        coripo_op = _OP_MAP.get(spec.op, "cont")
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
