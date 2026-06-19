from __future__ import annotations

from types import SimpleNamespace

from app.services.chat_service import chat_message_to_model as _chat_message_to_model


def test_chat_message_to_model_flattens_cards_from_agent_result() -> None:
    message = SimpleNamespace(
        role="assistant",
        content="K firmě jsem našla kontakty.",
        metadata_json={
            "agent_result": {
                "cards": [{"id": "table-contacts", "type": "table", "rows": []}],
            },
        },
        created_at=None,
    )

    model = _chat_message_to_model(message)
    assert isinstance(model.metadata.get("cards"), list)
    assert model.metadata["cards"][0]["id"] == "table-contacts"


def test_chat_message_to_model_keeps_existing_cards() -> None:
    message = SimpleNamespace(
        role="assistant",
        content="K firmě jsem našla kontakty.",
        metadata_json={
            "cards": [{"id": "table-direct", "type": "table", "rows": []}],
            "agent_result": {
                "cards": [{"id": "table-nested", "type": "table", "rows": []}],
            },
        },
        created_at=None,
    )

    model = _chat_message_to_model(message)
    assert model.metadata["cards"][0]["id"] == "table-direct"
