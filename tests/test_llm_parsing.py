# tests/test_llm_parsing.py
import json
import pytest

# Import the functions under test from your module
from core.services.llm import (
    _extract_balanced_json,
    _extract_final_json,
    parse_agent_output,
    _normalize_meeting_command,
)

# ----------------------- _extract_balanced_json -----------------------

def test_extract_balanced_json_simple_object():
    s = 'noise { "a": 1, "b": [2,3] } tail'
    obj = _extract_balanced_json(s)
    assert obj == {"a": 1, "b": [2, 3]}

def test_extract_balanced_json_ignores_think_and_fences():
    s = """<think>secret chain of thought</think>
    ```json
    { "x": "y", "z": [{"k":1},{"k":2}] }
    ```"""
    obj = _extract_balanced_json(s)
    assert obj["x"] == "y"
    assert obj["z"][1]["k"] == 2

# def test_extract_balanced_json_tolerates_trailing_commas():
#     s = '{ "a": 1, "b": [2,3,], }'
#     obj = _extract_balanced_json(s)
#     assert obj == {"a": 1, "b": [2, 3]}

def test_extract_balanced_json_unbalanced_raises():
    with pytest.raises(ValueError):
        _extract_balanced_json('{ "a": 1, ')

def test_extract_balanced_json_hoists_message_to_user_when_nested():
    s = '{ "action":"create","module":"meetings",' \
        '"parameters":{"name":"X","message_to_user":"Hi"}, "metadata":{} }'
    obj = _extract_balanced_json(s)
    # the helper hoists nested message_to_user to root (if present)
    assert obj["message_to_user"] == "Hi"
    assert "message_to_user" not in obj["parameters"]

# ----------------------- _extract_final_json -----------------------

def test_extract_final_json_takes_last_marker_and_parses():
    s = """
    Final Answer: { "action":"error","module":"x" }
    some chatter
    Final Answer:
    { "action":"create", "module":"meetings", "parameters":{}, "metadata":{} }
    """
    obj = _extract_final_json(s)
    assert obj["action"] == "create"
    assert obj["module"] == "meetings"

def test_extract_final_json_trims_junk_before_brace():
    s = "Final Answer:  \n\n   >>> CODE >>> {\"a\":1}"
    obj = _extract_final_json(s)
    assert obj == {"a": 1}

def test_extract_final_json_handles_fences_and_think():
    s = 'Final Answer: ```json <think>blah</think> { "k": 9 } ```'
    obj = _extract_final_json(s)
    assert obj == {"k": 9}

def test_extract_final_json_none_when_no_marker():
    assert _extract_final_json("no marker here") is None

# ----------------------- parse_agent_output -----------------------

def test_parse_agent_output_accepts_dict_output_with_final_answer_tail():
    out = {
        "output": """
        Observation: ok
        Final Answer: { "action":"create","module":"meetings",
          "parameters":{"related_module":"company"},
          "metadata":{"time":"13:00:45","participants":["a","a","b"]}
        }
        """
    }
    obj = parse_agent_output(out)
    # normalization applied
    assert obj["module"] == "meetings"
    assert obj["parameters"]["related_module"] == "accounts"  # company -> accounts
    assert obj["metadata"]["time"] == "13:00"
    assert obj["metadata"]["participants"] == ["a", "b"]

def test_parse_agent_output_accepts_pure_json_string():
    s = '{ "action":"create","module":"meetings","parameters":{},"metadata":{} }'
    obj = parse_agent_output(s)
    assert obj["action"] == "create"

def test_parse_agent_output_with_tool_noise_then_final_answer():
    s = '''
Action: get_user_agenda
Action Input: {"day":"2025-10-20","user_id":"U"}{"success": true, "agenda": []} Nyní vytvořím ...
Final Answer: {
  "action": "create",
  "module": "meetings",
  "parameters": {
    "name": "Schůzka s Karolínou Mimrovou",
    "related_to": "Karolína Mimrová",
    "related_to_id": "a7987",
    "related_module": "contacts"
  },
  "metadata": {
    "date": "2025-10-20",
    "time": "13:00",
    "duration": 60,
    "participants": ["a7987"]
  }
}
'''
    obj = parse_agent_output(s)
    assert obj["action"] == "create"
    assert obj["module"] == "meetings"
    assert obj["parameters"]["related_module"] == "contacts"
    assert obj["metadata"]["participants"] == ["a7987"]

def test_parse_agent_output_falls_back_to_balanced_json_when_no_marker():
    s = 'Header\n\n{ "action":"create","module":"meetings","parameters":{},"metadata":{} } tail'
    obj = parse_agent_output(s)
    assert obj["module"] == "meetings"

def test_parse_agent_output_returns_error_on_unparseable():
    # no JSON anywhere
    obj = parse_agent_output("just words and no braces")
    assert obj["action"] == "error"
    assert "message_to_user" in obj

# ----------------------- _normalize_meeting_command -----------------------

def test_normalize_meeting_command_company_to_accounts_and_time_trim_and_dedupe():
    cmd = {
        "action": "create",
        "module": "meetings",
        "parameters": {"related_module": "companies", "related_to_id": "acc1"},
        "metadata": {"time": "09:30:15", "participants": ["x", "x", "y"], "participant_ids": ["a","a","b"]},
    }
    out = _normalize_meeting_command(cmd)
    assert out["parameters"]["related_module"] == "accounts"
    assert out["metadata"]["time"] == "09:30"
    assert out["metadata"]["participants"] == ["x", "y"]
    assert out["metadata"]["participant_ids"] == ["a", "b"]

def test_normalize_meeting_command_leaves_non_meetings_unchanged():
    other = {"action": "create", "module": "tasks", "parameters": {"related_module": "companies"}}
    assert _normalize_meeting_command(other) is other
