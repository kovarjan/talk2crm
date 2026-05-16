from __future__ import annotations

from datetime import datetime
from typing import Any



def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_datetime(value: Any) -> datetime | None:
    raw = safe_text(value)
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


def record_name(record: dict[str, Any]) -> str:
    name = safe_text(record.get("name"))
    if name:
        return name
    first = safe_text(record.get("first_name"))
    last = safe_text(record.get("last_name"))
    full = f"{first} {last}".strip()
    if full:
        return full
    return "(bez nazvu)"


def crm_detail_link(module: str, record_id: str) -> str:
    clean_module = safe_text(module) or "Home"
    clean_id = safe_text(record_id)
    return f"/#detail/{clean_module}/{clean_id}" if clean_id else "/#"


def format_date(value: Any) -> str:
    dt = parse_datetime(value)
    if dt is None:
        return safe_text(value)
    return dt.strftime("%d.%m.%Y %H:%M")


def _non_empty_meta(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        text = safe_text(value)
        if text:
            clean[key] = text
    return clean


def record_card(module: str, title: str, record_id: str, meta: dict[str, Any]) -> dict[str, Any]:
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
                "url": crm_detail_link(module, record_id),
            }
        ]
        if record_id
        else [],
    }


def table_card(*, title: str, tag: str, columns: list[dict[str, str]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"table-{title.lower().replace(' ', '-') or 'crm'}",
        "type": "table",
        "title": title,
        "tag": tag,
        "columns": columns,
        "rows": rows,
    }


def meeting_cards(records: list[dict[str, Any]], total_count: int) -> list[dict[str, Any]]:
    if not records:
        return []
    if total_count <= 4:
        cards = []
        for row in records:
            cards.append(
                record_card(
                    "Meetings",
                    record_name(row),
                    safe_text(row.get("id")),
                    {
                        "Datum": format_date(row.get("date_start")),
                        "Misto": safe_text(row.get("location")),
                        "Stav": safe_text(row.get("status")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        rec_id = safe_text(row.get("id"))
        rows.append(
            {
                "id": rec_id,
                "cells": {
                    "date_start": format_date(row.get("date_start")),
                    "name": record_name(row),
                    "location": safe_text(row.get("location")),
                    "status": safe_text(row.get("status")),
                },
                "link": {
                    "label": "Detail",
                    "url": crm_detail_link("Meetings", rec_id),
                },
            }
        )

    return [
        table_card(
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


def contact_cards(
    records: list[dict[str, Any]],
    total_count: int,
    title: str,
    *,
    force_table: bool = False,
) -> list[dict[str, Any]]:
    if not records:
        return []

    if total_count <= 4 and not force_table:
        cards = []
        for row in records:
            cards.append(
                record_card(
                    "Contacts",
                    record_name(row),
                    safe_text(row.get("id")),
                    {
                        "Firma": safe_text(row.get("account_name") or row.get("company")),
                        "E-mail": safe_text(row.get("email1")),
                        "Telefon": safe_text(row.get("phone_mobile") or row.get("phone_work")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        rec_id = safe_text(row.get("id"))
        rows.append(
            {
                "id": rec_id,
                "cells": {
                    "name": record_name(row),
                    "company": safe_text(row.get("account_name") or row.get("company")),
                    "email": safe_text(row.get("email1")),
                    "phone": safe_text(row.get("phone_mobile") or row.get("phone_work")),
                },
                "link": {
                    "label": "Detail",
                    "url": crm_detail_link("Contacts", rec_id),
                },
            }
        )

    return [
        table_card(
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


def record_module_hint(row: dict[str, Any], default_module: str = "CRM") -> str:
    for key in ("_module_hint", "_module", "module", "record_module"):
        value = safe_text(row.get(key))
        if value:
            return value
    return safe_text(default_module) or "CRM"


def generic_cards(records: list[dict[str, Any]], total_count: int, default_module: str = "CRM") -> list[dict[str, Any]]:
    if not records:
        return []
    if total_count <= 4:
        cards = []
        for row in records:
            module = record_module_hint(row, default_module)
            cards.append(
                record_card(
                    module,
                    record_name(row),
                    safe_text(row.get("id")),
                    {
                        "Modul": module,
                        "Stav": safe_text(row.get("status")),
                        "Upraveno": format_date(row.get("date_modified")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    modules: list[str] = []
    for row in records:
        module = record_module_hint(row, default_module)
        if module not in modules:
            modules.append(module)
        rec_id = safe_text(row.get("id"))
        rows.append(
            {
                "id": f"{module}-{rec_id}" if rec_id else module,
                "cells": {
                    "module": module,
                    "name": record_name(row),
                    "status": safe_text(row.get("status")),
                },
                "link": {
                    "label": "Detail",
                    "url": crm_detail_link(module, rec_id),
                }
                if rec_id
                else None,
            }
        )

    return [
        table_card(
            title=f"Vysledky ({total_count})",
            tag=modules[0] if len(modules) == 1 else "CRM",
            columns=[
                {"key": "module", "label": "Modul"},
                {"key": "name", "label": "Nazev"},
                {"key": "status", "label": "Stav"},
            ],
            rows=rows,
        )
    ]
