from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.models import ChatMessageItem
from app.engine.pipeline import (
    build_soft_ui_focus_hint as _build_soft_ui_focus_hint,
    merge_context_with_created_record as _merge_context_with_created_record,
    merge_context_with_history_selection as _merge_context_with_history_selection,
    with_soft_ui_focus_hint as _with_soft_ui_focus_hint,
)
from app.services.chat_service import (
    chat_message_item_to_agent_history as _chat_message_item_to_agent_history,
    extract_crm_record_created_event as _extract_crm_record_created_event,
    extract_history_selection as _extract_history_selection,
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


def test_extract_history_selection_from_rag_search_options() -> None:
    metadata = {
        "agent_result": {
            "intermediate_steps": [
                {
                    "tool": "rag_search_tool",
                    "observation": [
                        {
                            "payload": {
                                "module": "Accounts",
                                "record_id": "11111111-2222-3333-4444-555555555555",
                                "record": {
                                    "id": "11111111-2222-3333-4444-555555555555",
                                    "name": "Baumit SK",
                                    "billing_address_city": "Bratislava",
                                },
                            }
                        },
                        {
                            "payload": {
                                "module": "Accounts",
                                "record_id": "22222222-2222-3333-4444-555555555555",
                                "record": {
                                    "id": "22222222-2222-3333-4444-555555555555",
                                    "name": "Baumit CZ",
                                    "billing_address_city": "Brandys nad Labem",
                                },
                            }
                        },
                    ],
                }
            ]
        }
    }

    selection = _extract_history_selection(metadata, "2")

    assert selection == {
        "module": "Accounts",
        "record_id": "22222222-2222-3333-4444-555555555555",
        "record_name": "Baumit CZ",
        "selection_label": "Baumit CZ - Brandys nad Labem",
    }


def test_extract_history_selection_accepts_number_with_text() -> None:
    metadata = {
        "agent_result": {
            "intermediate_steps": [
                {
                    "tool": "rag_search_tool",
                    "observation": [
                        {
                            "payload": {
                                "module": "Accounts",
                                "record_id": "11111111-2222-3333-4444-555555555555",
                                "record": {"name": "Baumit SK"},
                            }
                        },
                        {
                            "payload": {
                                "module": "Accounts",
                                "record_id": "22222222-2222-3333-4444-555555555555",
                                "record": {"name": "Baumit GmbH"},
                            }
                        },
                        {
                            "payload": {
                                "module": "Accounts",
                                "record_id": "7be50eea-d101-ff49-3fa9-5b87dfad2115",
                                "record": {
                                    "name": "BAUMIT, spol. s r.o.",
                                    "billing_address_city": "Brandys nad Labem",
                                },
                            }
                        },
                    ],
                }
            ]
        }
    }

    selection = _extract_history_selection(metadata, "3 baumit cz")

    assert selection is not None
    assert selection["record_id"] == "7be50eea-d101-ff49-3fa9-5b87dfad2115"
    assert selection["record_name"] == "BAUMIT, spol. s r.o."


def test_merge_context_with_history_selection_populates_account_entities() -> None:
    merged = _merge_context_with_history_selection(
        context={"module": None, "record": None, "entities": {}},
        selection={
            "module": "Accounts",
            "record_id": "22222222-2222-3333-4444-555555555555",
            "record_name": "Baumit CZ",
            "selection_label": "Baumit CZ - Brandys nad Labem",
        },
    )

    assert merged["module"] == "Accounts"
    assert merged["record_id"] == "22222222-2222-3333-4444-555555555555"
    assert merged["record_name"] == "Baumit CZ"
    assert merged["selected_option_label"] == "Baumit CZ - Brandys nad Labem"
    assert merged["selection_source"] == "history_selection"
    assert merged["entities"]["account_id"] == "22222222-2222-3333-4444-555555555555"
    assert merged["entities"]["account_name"] == "Baumit CZ"


def test_agent_history_includes_compact_tool_memory() -> None:
    message = ChatMessageItem(
        role="assistant",
        content="Nasel jsem vice zaznamu Baumit.",
        metadata={
            "agent_result": {
                "intermediate_steps": [
                    {
                        "tool": "rag_search_tool",
                        "observation": [
                            {
                                "payload": {
                                    "module": "Accounts",
                                    "record_id": "7be50eea-d101-ff49-3fa9-5b87dfad2115",
                                    "record": {
                                        "id": "7be50eea-d101-ff49-3fa9-5b87dfad2115",
                                        "name": "BAUMIT, spol. s r.o.",
                                        "billing_address_city": "Brandys nad Labem",
                                    },
                                }
                            }
                        ],
                    }
                ]
            }
        },
    )

    history = _chat_message_item_to_agent_history(message)

    assert history["role"] == "assistant"
    assert "Relevantni CRM pamet" in history["content"]
    assert "rag_search_tool" in history["content"]
    assert "id=7be50eea-d101-ff49-3fa9-5b87dfad2115" in history["content"]
    assert "BAUMIT, spol. s r.o." in history["content"]
