from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.pending_patch import try_patch_pending_action


def _pending_meeting() -> dict:
    return {
        "module": "Meetings",
        "requested_module": "Meetings",
        "effective_module": "Meetings",
        "action": "create",
        "data": {
            "fields": {
                "contact_name": "Mimrovou",
                "account_name": "Panas",
                "description": "projednání změn strategie prodeje",
                "date_start": "2026-03-18 14:00:00",
                "duration_hours": 1,
                "duration_minutes": 0,
            }
        },
        "adjustments": [],
        "ambiguities": [],
        "requires_confirmation": True,
    }


def test_pending_patch_updates_time_without_tool_call() -> None:
    result = try_patch_pending_action(
        input_text="přesuň ji na 18.00",
        pending_action=_pending_meeting(),
        now=datetime(2026, 3, 13, 10, 0, 0),
    )
    assert isinstance(result, dict)
    assert result.get("status") == "confirmation_required"
    pending = result.get("pending_action") or {}
    fields = (pending.get("data") or {}).get("fields") or {}
    assert fields.get("date_start") == "2026-03-18 18:00:00"
    assert "18:00" in str(result.get("message_to_user") or "")


def test_pending_patch_updates_weekday_keeps_time() -> None:
    result = try_patch_pending_action(
        input_text="změň ji na čtvrtek",
        pending_action=_pending_meeting(),
        now=datetime(2026, 3, 13, 10, 0, 0),
    )
    assert isinstance(result, dict)
    pending = result.get("pending_action") or {}
    fields = (pending.get("data") or {}).get("fields") or {}
    assert fields.get("date_start") == "2026-03-19 14:00:00"


def test_pending_patch_skips_confirmation_only_input() -> None:
    result = try_patch_pending_action(
        input_text="ano",
        pending_action=_pending_meeting(),
        now=datetime(2026, 3, 13, 10, 0, 0),
    )
    assert result is None


def test_pending_patch_output_contains_serialized_payload() -> None:
    result = try_patch_pending_action(
        input_text="na 16:30",
        pending_action=_pending_meeting(),
        now=datetime(2026, 3, 13, 10, 0, 0),
    )
    assert isinstance(result, dict)
    output = json.loads(str(result.get("output") or "{}"))
    assert output.get("action") == "create"
    assert output.get("module") == "Meetings"


def test_pending_patch_detects_long_weekday_edit_phrase() -> None:
    result = try_patch_pending_action(
        input_text="až to další pondělí ne teď 12.",
        pending_action={
            "module": "Meetings",
            "requested_module": "Meetings",
            "effective_module": "Meetings",
            "action": "create",
            "data": {
                "fields": {
                    "date_start": "2026-04-12 09:00:00",
                    "duration_hours": 1,
                    "duration_minutes": 0,
                }
            },
            "adjustments": [],
            "ambiguities": [],
            "requires_confirmation": True,
        },
        now=datetime(2026, 4, 10, 10, 0, 0),
    )
    assert isinstance(result, dict)
    pending = result.get("pending_action") or {}
    fields = (pending.get("data") or {}).get("fields") or {}
    assert fields.get("date_start") == "2026-04-20 09:00:00"
