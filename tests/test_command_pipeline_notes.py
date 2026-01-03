from core.pipelines.command_pipeline import run_command_pipeline
from core.utils.chat import ChatSession


def test_command_pipeline_routes_notes(monkeypatch):
    captured = {}

    def _fake_notes_agent(command_text, chat_history, action="", module_context=None, tenant="unknown"):
        captured["command_text"] = command_text
        captured["action"] = action
        captured["tenant"] = tenant
        return {
            "action": "create",
            "module": "notes",
            "parameters": {"name": "Test note"},
            "metadata": {},
            "message_to_user": "OK",
        }

    monkeypatch.setattr(
        "core.pipelines.command_pipeline.NotesAgent",
        _fake_notes_agent,
    )

    history = ChatSession(init=False)
    history.add_assistant(
        "ModuleDataExtractor - user request context:\n"
        "Module: notes\n"
        "Action: create\n"
        "{}"
    )

    result = run_command_pipeline(
        raw_text="Vytvor poznamku k zakaznikovi",
        history=history,
        tenant="test-tenant",
    )

    assert result["module"] == "notes"
    assert captured["action"] == "create"
    assert captured["tenant"] == "test-tenant"
