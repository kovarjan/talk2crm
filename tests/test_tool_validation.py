from __future__ import annotations

from _tool_fakes import sample_manifest
from app.tools.contracts import Manifest
from app.tools.validation import validate_arguments

SPECS = {t.name: t for t in Manifest.parse(sample_manifest()).tools}


def test_llm_slips_are_coerced() -> None:
    args, errors = validate_arguments(
        {"module": "Contacts", "filters": '[{"field": "name", "op": "cont", "value": "Novák"}]', "limit": "10", "search": None},
        SPECS["crm_query_tool"].input_schema,
    )
    assert errors == []
    assert args["filters"][0]["value"] == "Novák"
    assert args["limit"] == 10
    assert args["search"] is None  # nullable: kept


def test_object_argument_given_as_json_string() -> None:
    args, errors = validate_arguments(
        {"module": "Meetings", "action": "create", "data_json": '{"fields": {"name": "x"}}'},
        SPECS["crm_action_tool"].input_schema,
    )
    assert errors == []
    assert args["data_json"] == {"fields": {"name": "x"}}


def test_errors_carry_json_pointer_paths() -> None:
    _, errors = validate_arguments(
        {"module": "Contacts", "filters": [{"field": "name", "op": "like"}], "limit": 0, "bogus": 1},
        SPECS["crm_query_tool"].input_schema,
    )
    paths = {e["path"] for e in errors}
    assert {"/filters/0/op", "/limit", "/"} <= paths


def test_missing_required_argument() -> None:
    _, errors = validate_arguments({}, SPECS["crm_record_detail_tool"].input_schema)
    assert errors and all(e["path"] == "/" for e in errors)
