from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.nlu.command_parser import CommandParser
from app.services.crm_write_service import CRMWriteService

if TYPE_CHECKING:
    from app.services.crm_client import SugarClient


def _norm(text: str) -> str:
    lowered = (text or "").strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def _meetings_date_range(text: str) -> tuple[str, str] | None:
    """Return (date_from, date_to) ISO strings if text is a meeting list query, else None."""
    n = _norm(text)

    # Must mention meetings
    if not any(k in n for k in ("schuzk", "meeting", "schuzku")):
        return None

    # Must be a query (not a create/update)
    mutation = ("vytvor", "zaloz", "naplanuj", "pridej", "uprav", "zmen", "smaz")
    if any(k in n for k in mutation):
        return None

    now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if any(k in n for k in ("pristi tyd", "pristi tyde", "next week")):
        days_to_monday = (7 - now.weekday()) % 7 or 7
        start = now + timedelta(days=days_to_monday)
        end = start + timedelta(days=6)
    elif any(k in n for k in ("tento tyd", "tento tyde", "this week", "tento tyden")):
        start = now - timedelta(days=now.weekday())
        end = start + timedelta(days=6)
    elif any(k in n for k in ("zitra", "tomorrow")):
        start = now + timedelta(days=1)
        end = start
    elif any(k in n for k in ("dnes", "dnesni", "today")):
        start = now
        end = now
    elif any(k in n for k in ("tento mesic", "this month")):
        start = now.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = next_month - timedelta(days=1)
    else:
        # default: next 14 days
        start = now
        end = now + timedelta(days=14)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


async def _handle_list_meetings(
    date_from: str,
    date_to: str,
    crm_client: "SugarClient",
    user_id: str,
) -> dict[str, Any]:
    from app.engine.tools import (
        _build_login_user_meetings_window_filter,
        _extract_records,
        _format_date,
        _meeting_cards,
        _record_name,
        _safe_text,
    )

    crm_filter = _build_login_user_meetings_window_filter(date_from, date_to)
    payload = {
        "limit": 100,
        "offset": 0,
        "filter": crm_filter,
        "include_field_names": False,
        "response_fields": ["id", "name", "date_start", "status", "location", "assigned_user_name", "parent_name"],
        "order": [{"field": "date_start", "sort": "ASC", "module": "Meetings"}],
    }
    try:
        raw = await crm_client.execute_module_action(module="Meetings", action="list", data=payload)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "cards": [], "output": str(exc)}

    records = _extract_records(raw)
    total = len(records)
    cards = _meeting_cards(records, total, force_table=True)
    summary_lines = [
        f"• {_format_date(r.get('date_start'))} — {_record_name(r)} ({_safe_text(r.get('status'))})"
        for r in records[:20]
    ]
    summary = "\n".join(summary_lines) if summary_lines else "Žádné schůzky v daném období."
    return {
        "status": "ok",
        "module": "Meetings",
        "date_from": date_from,
        "date_to": date_to,
        "total": total,
        "summary": summary,
        "message_to_user": summary,
        "cards": cards,
        "output": json.dumps({"status": "ok", "total": total, "summary": summary}, ensure_ascii=False),
    }


async def try_handle_quick_action(
    *,
    input_text: str,
    crm_client: "SugarClient",
    user_id: str,
    action_confirmation: bool,
) -> dict[str, Any] | None:
    # Fast-path: meeting list queries
    date_range = _meetings_date_range(input_text)
    if date_range is not None:
        date_from, date_to = date_range
        return await _handle_list_meetings(date_from, date_to, crm_client, user_id)

    # Fast-path: create_contact / create_meeting (NLU-parsed)
    parser = CommandParser()
    command = parser.parse(input_text)
    if command is None:
        return None

    if command.intent not in {"create_contact", "create_meeting"}:
        return None

    write_service = CRMWriteService(
        tenant_id="quick-action",
        user_id=user_id,
        input_text=input_text,
        request_context={},
        crm_client=crm_client,
        rag_service=None,
        action_confirmation=action_confirmation,
    )
    return await write_service.execute_quick_command(command)
