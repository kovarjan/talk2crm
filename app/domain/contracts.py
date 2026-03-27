from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedCommand:
    intent: str
    module: str | None = None
    action: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "unknown"


@dataclass
class EntityResolution:
    status: str
    selected: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


@dataclass
class TemporalResolution:
    status: str
    datetime_value: str | None = None
    inferred_fields: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_confirmation: bool = False


@dataclass
class WritePreparation:
    effective_module: str
    action: str
    payload: dict[str, Any]
    adjustments: list[str] = field(default_factory=list)
    ambiguities: list[dict[str, Any]] = field(default_factory=list)
    confirmation_required: bool = False


def build_pending_action_envelope(
    *,
    module: str,
    action: str,
    data: dict[str, Any],
    requested_module: str | None = None,
    effective_module: str | None = None,
    adjustments: list[str] | None = None,
    ambiguities: list[dict[str, Any]] | None = None,
    requires_confirmation: bool = True,
) -> dict[str, Any]:
    target_module = str(effective_module or module or "").strip() or str(module or "").strip()
    requested = str(requested_module or module or target_module).strip() or target_module
    normalized_action = str(action or "").strip().lower()
    return {
        "module": target_module,
        "requested_module": requested,
        "effective_module": target_module,
        "action": normalized_action,
        "data": data if isinstance(data, dict) else {},
        "adjustments": list(adjustments or []),
        "ambiguities": list(ambiguities or []),
        "requires_confirmation": bool(requires_confirmation),
    }


def normalize_pending_action_envelope(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    module = str(
        value.get("effective_module")
        or value.get("module")
        or value.get("requested_module")
        or ""
    ).strip()
    action = str(value.get("action") or "").strip().lower()
    if not module or not action:
        return None

    data = value.get("data") if isinstance(value.get("data"), dict) else {}
    adjustments = value.get("adjustments") if isinstance(value.get("adjustments"), list) else []
    ambiguities = value.get("ambiguities") if isinstance(value.get("ambiguities"), list) else []
    requires_confirmation = bool(value.get("requires_confirmation", True))

    return build_pending_action_envelope(
        module=module,
        action=action,
        data=data,
        requested_module=str(value.get("requested_module") or module).strip() or module,
        effective_module=module,
        adjustments=[str(item) for item in adjustments],
        ambiguities=[item for item in ambiguities if isinstance(item, dict)],
        requires_confirmation=requires_confirmation,
    )
