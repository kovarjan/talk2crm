# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""ToolCallResult → the observation dict the agent loop and presentation layer consume.

Generic over tools: it looks at result shape, never at tool names, so a client's custom
CRM tool gets cards, confirmation handling and error feedback the same way core tools do.

Well-known structuredContent keys: status, summary, message_to_user, records + module
(+ display: "table" | "record", + card: {title, meta{label: value}} for display "record"),
cards, form_patch. A confirmation-gated preview becomes
status="confirmation_required" with a pending_action envelope carrying the tool call.
"""
from __future__ import annotations

from typing import Any

from app.domain.contracts import build_pending_action_envelope
from app.presentation.cards import contact_cards, generic_cards, meeting_cards, record_card, record_name
from app.tools.contracts import ToolCallResult, ToolSpec
from app.utils.text import safe_text

_ERROR_STATUS = {
    "invalid_arguments": "tool_validation_error",
    "unavailable": "crm_unavailable",
    "access_denied": "access_denied",
    "not_found": "not_found",
}


def to_observation(spec: ToolSpec, args: dict[str, Any], result: ToolCallResult) -> dict[str, Any]:
    if result.is_error:
        return error_observation(result)
    if result.confirmation is not None:
        return confirmation_observation(spec, args, result)
    observation = dict(result.structured)
    observation.setdefault("status", "ok")
    if not observation.get("summary") and result.text:
        observation["summary"] = result.text
    if not isinstance(observation.get("cards"), list):
        cards = cards_for(observation)
        if cards:
            observation["cards"] = cards
    return observation


def error_observation(result: ToolCallResult) -> dict[str, Any]:
    error = result.structured.get("error") if isinstance(result.structured.get("error"), dict) else {}
    code = result.error_code or "internal"
    observation: dict[str, Any] = {
        "status": _ERROR_STATUS.get(code, "error"),
        "error_code": code,
        "message": result.text or str(error.get("message") or ""),
    }
    details = error.get("details")
    if isinstance(details, list) and details:
        observation["field_errors"] = details
    return observation


def confirmation_observation(spec: ToolSpec, args: dict[str, Any], result: ToolCallResult) -> dict[str, Any]:
    confirmation = result.confirmation or {}
    structured = result.structured
    confirm_args = confirmation.get("arguments") if isinstance(confirmation.get("arguments"), dict) else args
    data = confirm_args.get("data_json") if isinstance(confirm_args.get("data_json"), dict) else confirm_args
    pending = build_pending_action_envelope(
        module=str(structured.get("module") or confirm_args.get("module") or ""),
        action=str(structured.get("action") or confirm_args.get("action") or "execute"),
        data=data,
        adjustments=[str(n) for n in structured.get("adjustments") or []],
        ambiguities=[a for a in structured.get("ambiguities") or [] if isinstance(a, dict)],
        tool_call={
            "tool": spec.name,
            "tool_version": str(result.meta.get("version") or spec.version),
            "arguments": confirm_args,
            "confirmation_token": confirmation.get("token"),
            "expires_at": confirmation.get("expires_at"),
        },
    )
    observation: dict[str, Any] = {
        "status": "confirmation_required",
        "message": str(structured.get("message") or result.text or "Akce vyžaduje potvrzení uživatele."),
        "pending_action": pending,
        "adjustments": pending["adjustments"],
    }
    for key in ("ambiguities", "missing_required", "record_name"):
        if structured.get(key):
            observation[key] = structured[key]
    return observation


def cards_for(observation: dict[str, Any]) -> list[dict[str, Any]]:
    records = observation.get("records")
    if not isinstance(records, list) or not records:
        return []
    rows = [r for r in records if isinstance(r, dict)]
    module = safe_text(observation.get("module")) or "CRM"
    total = int(observation.get("total") or len(rows))
    display = safe_text(observation.get("display"))
    if display == "record":
        # The tool may label the card itself (`card`: {title, meta{label: value}}) — it
        # knows the CRM's field labels; otherwise the row's own fields are shown.
        row = rows[0]
        card = observation.get("card") if isinstance(observation.get("card"), dict) else {}
        meta = card.get("meta") if isinstance(card.get("meta"), dict) else {
            k: v for k, v in row.items() if k not in {"id", "name"} and safe_text(v)
        }
        title = safe_text(card.get("title")) or record_name(row)
        rendered = record_card(module, title, safe_text(row.get("id")), dict(list(meta.items())[:8]))
        if isinstance(card.get("meta_fields"), list):
            rendered["meta_fields"] = [field for field in card["meta_fields"] if isinstance(field, dict)][:8]
        return [rendered]
    module_lower = module.lower()
    if module_lower == "meetings":
        return meeting_cards(rows, total, force_table=display == "table")
    if module_lower == "contacts":
        return contact_cards(rows, total, f"Kontakty ({total})", force_table=display == "table" or total > 4)
    return generic_cards(rows, total, default_module=module)
