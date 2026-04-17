from __future__ import annotations

from app.api.endpoints import (
    _build_soft_ui_focus_hint,
    _extract_crm_record_created_event,
    _merge_context_with_created_record,
    _with_soft_ui_focus_hint,
)


def test_extract_crm_record_created_event_from_top_level_metadata() -> None:
    metadata = {
        "crm_sync_event": {
            "type": "record_created",
            "module": "meetings",
            "record_id": "a1b2c3",
            "record_name": "Schuzka s klientem",
            "source": "crm",
        }
    }
    event = _extract_crm_record_created_event(metadata)
    assert event == {
        "type": "record_created",
        "module": "Meetings",
        "record_id": "a1b2c3",
        "record_name": "Schuzka s klientem",
        "source": "crm",
    }


def test_extract_crm_record_created_event_from_agent_result_metadata() -> None:
    metadata = {
        "agent_result": {
            "status": "crm_sync_record_created",
            "crm_sync_event": {
                "module": "Calls",
                "record_id": "xyz-42",
            },
        }
    }
    event = _extract_crm_record_created_event(metadata)
    assert event == {
        "type": "record_created",
        "module": "Calls",
        "record_id": "xyz-42",
    }


def test_merge_context_with_created_record_populates_record_hints() -> None:
    merged = _merge_context_with_created_record(
        context={"module": "Meetings", "record": None, "entities": {}},
        created_record_event={
            "type": "record_created",
            "module": "Meetings",
            "record_id": "meeting-777",
            "record_name": "Planovani sprintu",
        },
    )
    assert merged["record"] == "meeting-777"
    assert merged["record_id"] == "meeting-777"
    assert merged["record_module"] == "Meetings"
    assert merged["record_name"] == "Planovani sprintu"
    assert merged["entities"]["meeting_id"] == "meeting-777"


def test_merge_context_with_created_record_does_not_override_existing_values() -> None:
    merged = _merge_context_with_created_record(
        context={
            "module": "Meetings",
            "record": "existing-id",
            "record_name": "Puvodni",
            "entities": {"meeting_id": "existing-id"},
        },
        created_record_event={
            "type": "record_created",
            "module": "Meetings",
            "record_id": "new-id",
            "record_name": "Nova schuzka",
        },
    )
    assert merged["record"] == "existing-id"
    assert merged["record_name"] == "Puvodni"
    assert merged["entities"]["meeting_id"] == "existing-id"


def test_build_soft_ui_focus_hint_for_meeting_detail_view() -> None:
    hint = _build_soft_ui_focus_hint(
        {
            "module": "Meetings",
            "record": "cf3ae462-8449-4d70-9a63-30489f33e0f7",
            "record_name": "Schůzka Antonín - Úrokové sazby na úver",
        }
    )
    assert hint is not None
    assert "DetailView schůzky" in hint
    assert "Schůzka Antonín - Úrokové sazby na úver" in hint
    assert "nezávazný fakt" not in hint
    assert "závazný fakt" in hint


def test_with_soft_ui_focus_hint_adds_hint_once() -> None:
    context = {
        "module": "Meetings",
        "record_id": "meeting-42",
        "record_name": "Kontrolní schůzka",
    }
    merged = _with_soft_ui_focus_hint(context)
    assert "ui_focus_hint_cz" in merged
    assert "DetailView schůzky" in str(merged["ui_focus_hint_cz"])

    original_hint = "Uživatel má v CRM otevřen detail schůzky."
    merged_with_existing = _with_soft_ui_focus_hint({**context, "ui_focus_hint_cz": original_hint})
    assert merged_with_existing["ui_focus_hint_cz"] == original_hint
