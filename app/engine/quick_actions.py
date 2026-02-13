from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.crm_client import SugarClient

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+420\s*)?(\d[\d\s]{7,}\d)")



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
        # Exclude field-definition dictionaries and similar metadata objects.
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

    # Fallback for edge payloads that embed a single record directly.
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



def _pick_best_match_by_name(records: list[dict[str, Any]], search: str) -> dict[str, Any] | None:
    if not records:
        return None

    search_norm = _normalize_text(search)
    best: tuple[int, dict[str, Any]] | None = None

    for record in records:
        name = str(record.get("name") or "").strip()
        if not name:
            first = str(record.get("first_name") or "").strip()
            last = str(record.get("last_name") or "").strip()
            name = f"{first} {last}".strip()
        if not name:
            continue

        candidate_norm = _normalize_text(name)
        score = 0
        if candidate_norm == search_norm:
            score = 100
        elif candidate_norm.startswith(search_norm):
            score = 80
        elif search_norm in candidate_norm:
            score = 70
        elif candidate_norm in search_norm:
            score = 60

        if best is None or score > best[0]:
            best = (score, record)

    if best and best[0] >= 70:
        return best[1]
    return None



def _extract_create_contact_payload(text: str) -> dict[str, str] | None:
    normalized = _normalize_text(text)
    if "kontakt" not in normalized:
        return None
    if not any(trigger in normalized for trigger in ("vytvor", "zaloz", "pridej kontakt", "novy kontakt")):
        return None

    name_match = re.search(
        r"vytvo[rř]\s+kontakt\s+(.+?)(?=\s+(?:tel|telefon|mail|email|e-mail|pridej|přidej|ke\s+spole[cč]nosti)\b|$)",
        text,
        re.IGNORECASE,
    )
    if not name_match:
        name_match = re.search(
            r"zalo[zž]\s+kontakt\s+(.+?)(?=\s+(?:tel|telefon|mail|email|e-mail|pridej|přidej|ke\s+spole[cč]nosti)\b|$)",
            text,
            re.IGNORECASE,
        )

    full_name = ""
    if name_match:
        full_name = re.sub(r"\s+", " ", name_match.group(1)).strip(" ,.;")

    if not full_name:
        return None

    parts = full_name.split()
    first_name = parts[0]
    last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)
    company_match = re.search(
        r"(?:ke\s+spole[cč]nosti|k\s+firm[eě]|do\s+spole[cč]nosti)\s+([^,.;:]+)",
        text,
        re.IGNORECASE,
    )

    payload: dict[str, str] = {
        "first_name": first_name,
        "last_name": last_name,
    }
    if email_match:
        payload["email1"] = email_match.group(0).strip()
    if phone_match:
        payload["phone_mobile"] = re.sub(r"\s+", "", phone_match.group(1))
    if company_match:
        payload["company_name_hint"] = company_match.group(1).strip(" ,.;")
    return payload



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



def _is_latest_leads_query(normalized: str) -> bool:
    if "zajemc" not in normalized and "lead" not in normalized:
        return False
    return any(trigger in normalized for trigger in ("nejnovejs", "nove", "posledn"))



def _is_week_meetings_query(normalized: str) -> bool:
    if "tento tyden" not in normalized and "tenhle tyden" not in normalized:
        return False
    return any(token in normalized for token in ("schuzk", "scuzk", "plan", "meeting"))


async def try_handle_quick_action(
    *,
    input_text: str,
    crm_client: "SugarClient",
    user_id: str,
    action_confirmation: bool,
) -> dict[str, Any] | None:
    normalized = _normalize_text(input_text)

    create_contact = _extract_create_contact_payload(input_text)
    if create_contact is not None:
        company_hint = create_contact.pop("company_name_hint", "")
        data: dict[str, Any] = dict(create_contact)

        if company_hint:
            account_search = await crm_client.generic_search(query=company_hint, scope="accounts")
            account_records = _extract_records(account_search)
            account = _pick_best_match_by_name(account_records, company_hint)
            if account:
                account_id = str(account.get("id") or "").strip()
                account_name = str(account.get("name") or company_hint).strip()
                if account_id:
                    data["account_id"] = account_id
                if account_name:
                    data["account_name"] = account_name

        full_name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
        confirmation_text = f"Připraveno: vytvořit kontakt {full_name}."
        if data.get("phone_mobile"):
            confirmation_text += f" Tel: {data['phone_mobile']}."
        if data.get("email1"):
            confirmation_text += f" Email: {data['email1']}."
        if data.get("account_name"):
            confirmation_text += f" Společnost: {data['account_name']}."
        confirmation_text += " Potvrďte prosím provedení."

        pending_action = {
            "module": "Contacts",
            "action": "create",
            "data": data,
        }

        if not action_confirmation:
            return {
                "status": "confirmation_required",
                "pending_action": pending_action,
                "output": json.dumps(
                    {
                        "action": "create",
                        "module": "Contacts",
                        "data_json": json.dumps(data, ensure_ascii=False),
                        "message_to_user": confirmation_text,
                    },
                    ensure_ascii=False,
                ),
                "message_to_user": confirmation_text,
            }

        result = await crm_client.execute_module_action("Contacts", "create", data)
        created_id = str(result.get("id") or result.get("record_id") or "").strip()
        if created_id:
            message = f"Hotovo. Kontakt {full_name} byl vytvořen (ID: {created_id})."
        else:
            message = f"Hotovo. Kontakt {full_name} byl vytvořen."
        return {
            "status": "ok",
            "output": json.dumps(result, ensure_ascii=False),
            "message_to_user": message,
        }

    if _is_latest_leads_query(normalized):
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

    if _is_week_meetings_query(normalized):
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
