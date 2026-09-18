# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Single-record detail with line items, for the ``crm`` capability.

Labels and types come from the live ai_schema so they match the UI; numeric
line columns are summed here so the model never adds numbers by hand.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.presentation.cards import record_card
from app.utils.modules import canonical_module_name

logger = logging.getLogger(__name__)

MAX_ROWS = 200
_NUMERIC_TYPES = {"number_int", "number_decimal", "currency", "readonly_computed"}


class RecordDetailArgs(BaseModel):
    module: str = Field(description="CRM modul, např. Quotes, acm_invoices, acm_orders, Opportunities")
    record_id: str = Field(description="Skutečné CRM id záznamu")
    include_lines: bool = Field(default=True, description="Zahrnout položky (řádky)")


def _to_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def summarize_lines(line_field: dict[str, Any], max_rows: int = MAX_ROWS) -> dict[str, Any]:
    columns = [
        {"name": str(f.get("name")), "label": str(f.get("label") or f.get("name")), "type": str(f.get("type") or "text")}
        for f in (line_field.get("line_fields") or []) if isinstance(f, dict) and f.get("name")
    ]
    rows_all = [r for r in (line_field.get("current_value") or []) if isinstance(r, dict)]
    sums: dict[str, float] = {}
    for column in columns:
        if column["type"] not in _NUMERIC_TYPES:
            continue
        total = 0.0
        seen = False
        for row in rows_all:
            value = _to_float(row.get(column["name"]))
            if value is not None:
                total += value
                seen = True
        if seen:
            sums[column["name"]] = round(total, 2)
    return {
        "line_module": line_field.get("line_module"),
        "columns": columns,
        "rows": rows_all[:max_rows],
        "totals": {"count": len(rows_all), "sum": sums},
        "truncated": len(rows_all) > max_rows,
    }


def _fields_from_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    fields: dict[str, Any] = {}
    line_field: dict[str, Any] | None = None
    for section in schema.get("sections") or []:
        for field in (section.get("fields") or []) if isinstance(section, dict) else []:
            if not isinstance(field, dict) or not field.get("name"):
                continue
            if field.get("type") == "line_items":
                line_field = field
                continue
            fields[str(field["name"])] = {"label": str(field.get("label") or field["name"]), "value": field.get("current_value")}
    return fields, line_field


def build_record_detail_tools(*, tenant_id: str, user_id: str, crm_client: Any) -> list:
    @tool("crm_record_detail_tool", args_schema=RecordDetailArgs)
    async def crm_record_detail_tool(module: str, record_id: str, include_lines: bool = True) -> str:
        """Detail jednoho CRM záznamu včetně položek a jejich součtů."""
        module_name = canonical_module_name(module) or module
        try:
            schema = await crm_client.get_ai_schema(module_name, record_id)
        except Exception as exc:  # noqa: BLE001 - reported to the model, never raised
            logger.warning("record detail unavailable module=%s id=%s error=%s", module_name, record_id, f"{type(exc).__name__}: {exc}")
            return json.dumps({"status": "record_unavailable", "module": module_name, "record_id": record_id,
                               "message": f"Záznam {module_name}/{record_id} není dostupný ({type(exc).__name__}: {exc})."}, ensure_ascii=False)
        fields, line_field = _fields_from_schema(schema if isinstance(schema, dict) else {})
        name = str((fields.get("name") or {}).get("value") or record_id)
        lines = summarize_lines(line_field) if (include_lines and line_field) else None
        meta = {k: v["value"] for k, v in list(fields.items())[:8] if v.get("value") not in (None, "")}
        payload = {
            "status": "ok", "module": module_name, "record_id": record_id, "name": name,
            "fields": fields, "lines": lines,
            "cards": [record_card(module_name, name, record_id, meta)],
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    return [crm_record_detail_tool]
