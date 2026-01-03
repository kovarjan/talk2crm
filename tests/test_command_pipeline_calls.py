from core.pipelines.command_pipeline import run_command_pipeline
from core.utils.chat import ChatSession


def test_command_pipeline_routes_calls(monkeypatch):
    captured = {}

    def _fake_calls_agent(command_text, chat_history, action="", module_context=None, tenant="unknown"):
        captured["command_text"] = command_text
        captured["action"] = action
        captured["tenant"] = tenant
        return {
            "action": "create",
            "module": "calls",
            "parameters": {"name": "Test call"},
            "metadata": {},
            "message_to_user": "OK",
        }

    monkeypatch.setattr(
        "core.pipelines.command_pipeline.CallsAgent",
        _fake_calls_agent,
    )

    history = ChatSession(init=False)
    history.add_assistant(
        "ModuleDataExtractor - user request context:\n"
        "Module: calls\n"
        "Action: create\n"
        "{}"
    )

    result = run_command_pipeline(
        raw_text="Naplánuj hovor se zákazníkem",
        history=history,
        tenant="test-tenant",
    )

    assert result["module"] == "calls"
    assert captured["action"] == "create"
    assert captured["tenant"] == "test-tenant"
