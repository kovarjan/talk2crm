from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.nlu.command_parser import CommandParser
from app.services.crm_write_service import CRMWriteService

if TYPE_CHECKING:
    from app.services.crm_client import SugarClient



def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    unaccented = "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()



def _extract_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_list_keys = {"records", "entry_list", "items"}

    def is_record_candidate(item: dict[str, Any]) -> bool:
        if "vname" in item and "type" in item and "name" in item and "id" not in item:
            return False
        return bool(str(item.get("id") or "").strip())

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, list):
            if parent_key in record_list_keys:
                for item in node:
                    if isinstance(item, dict) and is_record_candidate(item):
                        records.append(item)
            for item in node:
                walk(item, parent_key="")
            return

        if not isinstance(node, dict):
            return

        for key, child in node.items():
            walk(child, parent_key=key)

    walk(value)

    if not records and isinstance(value, dict):
        if bool(str(value.get("id") or "").strip()):
            records.append(value)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        rec_id = str(item.get("id") or "").strip()
        if not rec_id or rec_id in seen:
            continue
        seen.add(rec_id)
        deduped.append(item)
    return deduped



def _parse_datetime(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None



def _format_contact_name(item: dict[str, Any]) -> str:
    first = str(item.get("first_name") or "").strip()
    last = str(item.get("last_name") or "").strip()
    full = f"{first} {last}".strip()
    return full or str(item.get("name") or "(bez jména)").strip()


async def try_handle_quick_action(
    *,
    input_text: str,
    crm_client: "SugarClient",
    user_id: str,
    action_confirmation: bool,
) -> dict[str, Any] | None:
    parser = CommandParser()
    command = parser.parse(input_text)
    if command is None:
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

    if command.intent in {"create_contact", "create_meeting"}:
        return await write_service.execute_quick_command(command)

    normalized = _normalize_text(input_text)

    if command.intent == "latest_leads":
        result = await crm_client.execute_module_action(
            "Leads",
            "list",
            {"query": "", "max_results": 10},
        )
        leads = _extract_records(result)
        if not leads:
            return {
                "status": "ok",
                "output": json.dumps(result, ensure_ascii=False),
                "message_to_user": "Nenašel jsem žádné zájemce.",
            }

        lines = ["Nejnovější zájemci:"]
        for idx, lead in enumerate(leads[:10], start=1):
            name = _format_contact_name(lead)
            company = str(lead.get("account_name") or lead.get("company") or "").strip()
            status = str(lead.get("status") or "").strip()
            modified = str(lead.get("date_modified") or lead.get("date_entered") or "").strip()
            detail = f"{idx}. {name}"
            if company:
                detail += f" ({company})"
            if status:
                detail += f" - {status}"
            if modified:
                detail += f" [{modified}]"
            lines.append(detail)

        return {
            "status": "ok",
            "output": json.dumps(result, ensure_ascii=False),
            "message_to_user": "\n".join(lines),
        }

    if command.intent == "week_meetings":
        result = await crm_client.execute_module_action(
            "Meetings",
            "list",
            {"query": "", "max_results": 100},
        )
        meetings = _extract_records(result)

        now = datetime.now()
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)

        selected: list[dict[str, Any]] = []
        for item in meetings:
            dt = _parse_datetime(str(item.get("date_start") or ""))
            if dt is None or not (week_start <= dt < week_end):
                continue

            assigned_id = str(item.get("assigned_user_id") or "").strip()
            if assigned_id and assigned_id != str(user_id):
                continue
            selected.append(item)

        selected.sort(key=lambda row: _parse_datetime(str(row.get("date_start") or "")) or datetime.max)

        if not selected:
            return {
                "status": "ok",
                "output": json.dumps(result, ensure_ascii=False),
                "message_to_user": "Na tento týden nemáte naplánované žádné schůzky.",
            }

        lines = ["Schůzky tento týden:"]
        for idx, meeting in enumerate(selected[:15], start=1):
            name = str(meeting.get("name") or "(bez názvu)").strip()
            when = str(meeting.get("date_start") or "").strip()
            status = str(meeting.get("status") or "").strip()
            place = str(meeting.get("location") or "").strip()
            detail = f"{idx}. {when} - {name}"
            if status:
                detail += f" ({status})"
            if place:
                detail += f" [{place}]"
            lines.append(detail)

        return {
            "status": "ok",
            "output": json.dumps(result, ensure_ascii=False),
            "message_to_user": "\n".join(lines),
        }

    return None
