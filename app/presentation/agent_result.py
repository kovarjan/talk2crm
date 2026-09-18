# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Normalization of raw agent results into UI-facing messages and payloads."""

from __future__ import annotations

import json
import re
from typing import Any

from app.domain.contracts import normalize_pending_action_envelope
from app.utils.text import format_european_dates, strip_answer_tags, strip_think_tags


_MD_FENCE_RE = re.compile(r"```(?:\w+)?\s*([\s\S]*?)```", re.IGNORECASE)
_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LEADING_SYMBOL_RE = re.compile(r"^[\s\-•>*]+")
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
    flags=re.UNICODE,
)


def to_user_message(text: str) -> str:
    """Strip LLM markup (think/answer tags, markdown, emoji) into plain Czech text."""
    cleaned = strip_think_tags(text)
    cleaned = strip_answer_tags(cleaned)
    cleaned = format_european_dates(cleaned)
    cleaned = _MD_FENCE_RE.sub(r"\1", cleaned)
    cleaned = _MD_BOLD_RE.sub(r"\1", cleaned)
    cleaned = _MD_ITALIC_RE.sub(r"\1", cleaned)
    cleaned = _MD_INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = "\n".join(_LEADING_SYMBOL_RE.sub("", line) for line in cleaned.splitlines())
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or (text or "")


def try_parse_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def looks_like_noise(text: str) -> bool:
    """
    Detect raw machine output (JSON blobs, menu dumps) that must never be shown
    as the assistant message. Length alone is NOT noise — long analytical
    answers are legitimate final responses.
    """
    value = (text or "").strip()
    if not value:
        return False
    lower = value.lower()
    if value.startswith("{") or value.startswith("["):
        return True
    noisy_patterns = [
        "here's the translation of the menu labels",
        '"row_count"',
        '"menu":',
        "```json",
    ]
    return any(pattern in lower for pattern in noisy_patterns)


_MODULE_LABELS_CZ = {
    "meetings": "schůzka",
    "calls": "hovor",
    "tasks": "úkol",
    "notes": "poznámka",
    "contacts": "kontakt",
    "accounts": "firma",
    "leads": "lead",
    "opportunities": "obchodní případ",
    "quotes": "nabídka",
}

_ACTION_LABELS_CZ = {
    "create": "vytvoření",
    "update": "úprava",
    "delete": "smazání",
}

_PENDING_SUMMARY_FIELDS = (
    ("name", None),
    ("subject", None),
    ("note_content", "text"),
    ("description", "popis"),
    ("date_start", "od"),
    ("duration_hours", "délka (hod)"),
    ("contact_name", "kontakt"),
    ("parent_name", "týká se"),
)


def describe_pending_action_cz(module: str, action: str, data: dict[str, Any]) -> str:
    """
    Human-readable summary of a pending mutation. This text is persisted as the
    assistant chat message, so the LLM can recall in later turns WHAT it
    proposed — the generic confirmation sentence carries no information.
    """
    module_label = _MODULE_LABELS_CZ.get(module.strip().lower(), module or "záznam")
    action_label = _ACTION_LABELS_CZ.get(action.strip().lower(), action or "akce")
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}

    parts: list[str] = []
    for key, label in _PENDING_SUMMARY_FIELDS:
        value = str(fields.get(key) or "").strip()
        if not value:
            continue
        if len(value) > 120:
            value = value[:120].rstrip() + "..."
        parts.append(f"{label}: {value}" if label else value)

    summary = f"Připravil jsem k potvrzení: {action_label} ({module_label})"
    if parts:
        summary += " — " + "; ".join(parts)
    return summary + ". Potvrďte prosím provedení."


def normalize_agent_result_for_ui(agent_result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(agent_result or {})
    steps = normalized.get("intermediate_steps") or []
    output_text = str(normalized.get("output") or "")
    parsed_output = try_parse_json(output_text)
    if isinstance(parsed_output, dict):
        output_message = to_user_message(str(parsed_output.get("message_to_user") or "").strip())
    else:
        output_message = to_user_message(output_text)
    output_message_usable = bool(output_message) and not looks_like_noise(output_message)

    for step in reversed(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = try_parse_json(observation) or {}
        else:
            obs = {}
        if not obs:
            continue

        status = str(obs.get("status") or "").strip().lower()
        if status in {"confirmation_required", "resolution_required"}:
            pending_raw = obs.get("pending_action") if isinstance(obs.get("pending_action"), dict) else {}
            pending = normalize_pending_action_envelope(pending_raw) or pending_raw
            module = str(
                pending.get("effective_module")
                or pending.get("module")
                or pending.get("requested_module")
                or ""
            ).strip()
            action = str(pending.get("action") or "").strip()
            data = pending.get("data") if isinstance(pending.get("data"), dict) else {}
            if status == "confirmation_required":
                message = describe_pending_action_cz(module, action, data)
            else:
                message = str(
                    obs.get("message")
                    or "Pro pokračování potřebuji upřesnit cílový záznam."
                ).strip()

            command_payload = {
                "action": action or "create",
                "module": module or "Meetings",
                "data_json": json.dumps(data, ensure_ascii=False),
                "message_to_user": message,
            }
            normalized["status"] = status
            normalized["pending_action"] = pending
            normalized["output"] = json.dumps(command_payload, ensure_ascii=False)
            normalized["message_to_user"] = message
            return normalized

        if status in {"crm_http_error", "crm_action_error"}:
            message = str(
                obs.get("message")
                or "CRM akce selhala. Zkontrolujte mapování polí a povinné hodnoty."
            ).strip()
            normalized["status"] = status
            normalized["message_to_user"] = message
            if not str(normalized.get("output") or "").strip():
                normalized["output"] = message
            return normalized

    for step in reversed(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        tool_name = str(step.get("tool") or "").strip()
        if tool_name not in {"crm_data_tool", "crm_search_tool", "crm_query_tool", "my_meetings_tool", "propose_form_fields_tool", "research_record_tool", "crm_record_detail_tool", "product_lookup_tool", "daily_briefing_tool"}:
            continue

        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = try_parse_json(observation) or {}
        else:
            obs = {}
        if not obs:
            continue

        cards = obs.get("cards")
        if isinstance(cards, list):
            normalized["cards"] = cards
        tool_message = to_user_message(
            str(
                obs.get("message_to_user")
                or obs.get("summary")
                or ""
            ).strip()
        )
        # Keep the model's final natural-language answer when available.
        # Tool summary is only a fallback.
        if output_message_usable:
            normalized["message_to_user"] = output_message
        elif tool_message:
            normalized["message_to_user"] = tool_message
        tool_status = str(obs.get("status") or "").strip()
        if tool_status:
            normalized["status"] = tool_status
        if not str(normalized.get("output") or "").strip():
            normalized["output"] = json.dumps(obs, ensure_ascii=False)
        return normalized

    if isinstance(parsed_output, dict) and parsed_output.get("message_to_user"):
        normalized["message_to_user"] = to_user_message(str(parsed_output["message_to_user"]))
        return normalized

    clean = to_user_message(output_text)
    if not clean or looks_like_noise(clean):
        clean = (
            "Omlouvám se, tady se mi nepodařilo připravit odpověď. "
            "Zkuste prosím dotaz zopakovat nebo přeformulovat."
        )
    normalized["message_to_user"] = clean
    return normalized


def extract_pending_action_from_agent_result(agent_result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(agent_result, dict):
        return None

    direct_pending = agent_result.get("pending_action")
    if isinstance(direct_pending, dict):
        return normalize_pending_action_envelope(direct_pending) or direct_pending

    steps = agent_result.get("intermediate_steps")
    if not isinstance(steps, list):
        return None

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = try_parse_json(observation) or {}
        else:
            obs = {}
        if not isinstance(obs, dict):
            continue
        status = str(obs.get("status") or "").strip().lower()
        pending = obs.get("pending_action")
        if status in {"confirmation_required", "resolution_required"} and isinstance(pending, dict):
            return normalize_pending_action_envelope(pending) or pending
    return None


def extract_form_patch_from_agent_result(agent_result: dict[str, Any]) -> dict[str, Any] | None:
    steps = (agent_result or {}).get("intermediate_steps") or []
    for step in reversed(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            obs = observation
        elif isinstance(observation, str):
            obs = try_parse_json(observation) or {}
        else:
            obs = {}
        patch = obs.get("form_patch") if isinstance(obs, dict) else None
        if isinstance(patch, dict) and isinstance(patch.get("fields"), dict):
            return patch
    return None
