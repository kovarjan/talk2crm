from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.form_patch import build_form_patch
from app.presentation.agent_result import extract_form_patch_from_agent_result

SCHEMA = {"module": "Contacts", "sections": [{"group": "x", "fields": [{"name": "first_name", "type": "text"}]}]}


def test_build_form_patch_has_fixed_shape() -> None:
    patch = build_form_patch(module="Contacts", record_id=None, fields={"first_name": "Jan"}, schema=SCHEMA, message="ok")
    assert patch == {
        "module": "Contacts",
        "record": None,
        "fields": {"first_name": "Jan"},
        "lines": {"mode": "append", "rows": []},
        "invitees": {},
        "sources": [],
        "message": "ok",
        "schema": {"sections": SCHEMA["sections"]},
    }


def test_build_form_patch_caps_sources_and_keeps_record() -> None:
    patch = build_form_patch(module="Accounts", record_id="abc", fields={}, schema=SCHEMA, sources=[f"https://s{i}" for i in range(15)])
    assert patch["record"] == "abc"
    assert len(patch["sources"]) == 10


def test_extract_from_dict_and_json_observations() -> None:
    patch = build_form_patch(module="Contacts", record_id=None, fields={"first_name": "Jan"}, schema=SCHEMA)
    result = {"intermediate_steps": [
        {"tool": "crm_query_tool", "observation": json.dumps({"records": []})},
        {"tool": "propose_form_fields_tool", "observation": json.dumps({"status": "ok", "form_patch": patch})},
        {"tool": "rag_search_tool", "observation": {"records": []}},
    ]}
    assert extract_form_patch_from_agent_result(result) == patch


def test_extract_returns_none_when_absent() -> None:
    assert extract_form_patch_from_agent_result({"intermediate_steps": []}) is None
    assert extract_form_patch_from_agent_result({}) is None
    assert extract_form_patch_from_agent_result({"intermediate_steps": [{"observation": "not json"}]}) is None
