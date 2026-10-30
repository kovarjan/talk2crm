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


def test_order_by_slips_are_normalised_to_field_colon_direction() -> None:
    schema = SPECS["crm_query_tool"].input_schema
    for given, expected in [
        ("date_start DESC", "date_start:desc"),
        ("date_start desc", "date_start:desc"),
        ("-date_start", "date_start:desc"),
        ("name", "name:asc"),
        ("date_start:desc", "date_start:desc"),
    ]:
        args, errors = validate_arguments({"module": "Meetings", "order_by": given}, schema)
        assert errors == [], given
        assert args["order_by"] == expected


def test_unparseable_order_by_is_still_reported() -> None:
    _, errors = validate_arguments({"module": "Meetings", "order_by": "date_start; DROP"}, SPECS["crm_query_tool"].input_schema)
    assert errors and errors[0]["path"] == "/order_by"
