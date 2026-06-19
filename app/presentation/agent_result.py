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
    value = (text or "").strip()
    if not value:
        return False
    lower = value.lower()
    if len(value) > 1400:
        return True
    noisy_patterns = [
        "here's the translation of the menu labels",
        '"row_count"',
        '"menu":',
        "```json",
    ]
    return any(pattern in lower for pattern in noisy_patterns)


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
            message = str(
                obs.get("message")
                or (
                    "Akce je připravena a čeká na vaše potvrzení."
                    if status == "confirmation_required"
                    else "Pro pokračování potřebuji upřesnit cílový záznam."
                )
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
        if tool_name not in {"crm_data_tool", "crm_search_tool", "crm_query_tool", "my_meetings_tool"}:
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
            "Nerozuměla jsem spolehlivě požadavku. "
            "Upřesněte prosím akci, modul a čas (např. schůzka v úterý 9:30)."
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
