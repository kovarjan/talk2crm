from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from typing import Any

from app.utils.text import normalize_text as _normalize_text
from app.domain.contracts import build_pending_action_envelope, normalize_pending_action_envelope
from app.presentation.cards import parse_datetime

_TIME_TOKEN_RE = re.compile(r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b")
_HOURS_RE = re.compile(r"\b(?:na\s+)?(?P<hours>\d{1,2})\s*(?:h|hod|hodin|hodiny)\b", re.IGNORECASE)
_MINUTES_RE = re.compile(r"\b(?:na\s+)?(?P<minutes>\d{1,3})\s*(?:m|min|minut|minuty)\b", re.IGNORECASE)
_DESCRIPTION_RE = re.compile(r"\bpopis\s+(.+)", re.IGNORECASE | re.DOTALL)


def _is_confirmation_only(input_text: str) -> bool:
    normalized = _normalize_text(input_text)
    if not normalized:
        return False
    if any(token in normalized for token in ("zmen", "presun", "preplanuj", "uprav")):
        return False
    if _TIME_TOKEN_RE.search(input_text or ""):
        return False
    confirmations = {
        "ano",
        "jo",
        "ok",
        "potvrzuji",
        "potvrdzuji",
        "potvrd",
        "potvrdit",
        "provest",
        "proved",
        "schvaluji",
    }
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return False
    return all(token in confirmations for token in tokens)


def _looks_like_pending_edit(input_text: str) -> bool:
    normalized = _normalize_text(input_text)
    if not normalized:
        return False
    if any(token in normalized for token in ("kolik", "kdy", "kde", "kdo", "co", "jak")):
        return False
    edit_tokens = (
        "zmen",
        "presun",
        "preplanuj",
        "uprav",
        "dej",
        "misto",
        "bez",
        "na ",
        "popis ",
    )
    if any(token in normalized for token in edit_tokens):
        return True
    if _TIME_TOKEN_RE.search(input_text or ""):
        return True
    return False


def _extract_duration_patch(input_text: str) -> tuple[int, int] | None:
    match_hours = _HOURS_RE.search(input_text or "")
    match_minutes = _MINUTES_RE.search(input_text or "")
    if match_hours:
        hours = max(0, min(12, int(match_hours.group("hours"))))
        return hours, 0
    if match_minutes:
        minutes = max(0, min(59, int(match_minutes.group("minutes"))))
        return 0, minutes
    return None


# _weekday_from_text is intentionally disabled.
# Day-of-week resolution from phrases like "na úterý v pondělí nemůžu" is
# ambiguous when multiple weekdays are present; the LLM receives the current
# date in its system prompt and resolves relative weekdays correctly on its own.


def _resolve_datetime_patch(*, input_text: str, current_value: str, now: datetime | None) -> str | None:
    current_dt = parse_datetime(current_value)
    anchor = current_dt or now or datetime.now()

    time_match = _TIME_TOKEN_RE.search(input_text or "")
    normalized = _normalize_text(input_text)
    explicit_time = time_match is not None
    part_of_day = False

    hour = anchor.hour
    minute = anchor.minute
    if time_match:
        hour = max(0, min(23, int(time_match.group("hour"))))
        minute = max(0, min(59, int(time_match.group("minute"))))
    elif "po poledni" in normalized:
        hour, minute = 13, 0
        part_of_day = True
    elif "odpoledne" in normalized:
        hour, minute = 14, 0
        part_of_day = True
    elif "hned rano" in normalized or "rano" in normalized:
        hour, minute = 8, 0
        part_of_day = True

    # Only patch when there is an explicit time signal; weekday/day changes are
    # handled by the LLM which knows the current date from its system prompt.
    if not explicit_time and not part_of_day:
        return None

    target = anchor.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target.strftime("%Y-%m-%d %H:%M:%S")


def _first_text(fields: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(fields.get(key) or "").strip()
        if value:
            return value
    return ""


def _meeting_patch_summary(fields: dict[str, Any]) -> str:
    contact_name = _first_text(fields, ["contact_name", "related_contact_name", "participant_name"])
    account_name = _first_text(fields, ["account_name", "related_account_name", "company", "company_name"])
    description = _first_text(fields, ["description", "note", "subject"])
    date_start = str(fields.get("date_start") or "").strip()
    dt = parse_datetime(date_start)
    weekday_cz = {
        0: "v pondělí",
        1: "v úterý",
        2: "ve středu",
        3: "ve čtvrtek",
        4: "v pátek",
        5: "v sobotu",
        6: "v neděli",
    }

    message = "Upraveno: vytvořit schůzku"
    if contact_name:
        message += f" s {contact_name}"
    if account_name:
        message += f" z firmy {account_name}"
    if dt is not None:
        message += f" {weekday_cz.get(dt.weekday(), 'v')} {dt.strftime('%d.%m.%Y')} v {dt.strftime('%H:%M')}"
    elif date_start:
        message += f" na {date_start}"
    if description:
        message += f" ohledně {description}"
    message += ". Potvrďte prosím provedení."
    return message


def _generic_patch_summary(*, module: str, action: str) -> str:
    action_label = {
        "create": "vytvořit",
        "update": "upravit",
        "patch": "upravit",
        "delete": "smazat",
    }.get(action, action)
    return f"Upraveno: {action_label} záznam v modulu {module}. Potvrďte prosím provedení."


def try_patch_pending_action(
    *,
    input_text: str,
    pending_action: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any] | None:
    pending = normalize_pending_action_envelope(pending_action)
    if pending is None:
        return None

    action = str(pending.get("action") or "").strip().lower()
    if action not in {"create", "update", "patch", "delete"}:
        return None
    if not bool(pending.get("requires_confirmation", True)):
        return None
    if _is_confirmation_only(input_text):
        return None
    if not _looks_like_pending_edit(input_text):
        return None

    module = str(pending.get("effective_module") or pending.get("module") or "").strip() or "Meetings"
    data = copy.deepcopy(pending.get("data") if isinstance(pending.get("data"), dict) else {})
    fields_raw = data.get("fields")
    fields = copy.deepcopy(fields_raw) if isinstance(fields_raw, dict) else {}
    notes = [str(item) for item in pending.get("adjustments") or [] if isinstance(item, str)]

    changed = False
    module_norm = module.lower()

    if module_norm in {"meetings", "calls"}:
        before = str(fields.get("date_start") or "").strip()
        patched_datetime = _resolve_datetime_patch(
            input_text=input_text,
            current_value=before,
            now=now,
        )
        if patched_datetime and patched_datetime != before:
            fields["date_start"] = patched_datetime
            changed = True
            notes.append(f"Patched pending datetime -> {patched_datetime}")

        duration = _extract_duration_patch(input_text)
        if duration is not None:
            new_hours, new_minutes = duration
            if int(fields.get("duration_hours") or 0) != new_hours or int(fields.get("duration_minutes") or 0) != new_minutes:
                fields["duration_hours"] = new_hours
                fields["duration_minutes"] = new_minutes
                changed = True
                notes.append(f"Patched pending duration -> {new_hours}h {new_minutes}m")

    desc_match = _DESCRIPTION_RE.search(input_text or "")
    if desc_match:
        new_desc = desc_match.group(1).strip().rstrip(".")
        if new_desc and new_desc != str(fields.get("description") or "").strip():
            fields["description"] = new_desc
            changed = True
            notes.append(f"Patched pending description -> {new_desc[:60]}")

    if not changed:
        return None

    data["fields"] = fields
    patched_pending = build_pending_action_envelope(
        module=module,
        requested_module=str(pending.get("requested_module") or module).strip() or module,
        effective_module=module,
        action=action,
        data=data,
        adjustments=notes,
        ambiguities=[item for item in pending.get("ambiguities") or [] if isinstance(item, dict)],
        requires_confirmation=True,
    )

    if module_norm == "meetings":
        message = _meeting_patch_summary(fields)
    else:
        message = _generic_patch_summary(module=module, action=action)

    command_payload = {
        "action": action,
        "module": module,
        "data_json": json.dumps(data, ensure_ascii=False),
        "message_to_user": message,
    }
    return {
        "status": "confirmation_required",
        "pending_action": patched_pending,
        "output": json.dumps(command_payload, ensure_ascii=False),
        "message_to_user": message,
    }
