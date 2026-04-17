from __future__ import annotations

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.filter_builder import FilterSpec, build_filter, build_order


def test_build_filter_search_only():
    f = build_filter(search="Novak")
    assert f["operator"] == "and"
    assert len(f["operands"]) == 1
    nested = f["operands"][0]
    assert nested["operands"][0]["field"] == "*"
    assert nested["operands"][0]["type"] == "cont"
    assert nested["operands"][0]["value"] == "Novak"


def test_build_filter_date_range_meetings():
    f = build_filter(module="Meetings", date_from="2025-03-10", date_to="2025-03-14")
    fields = [op["field"] for op in f["operands"]]
    assert all(field == "date_start" for field in fields)
    types = [op["type"] for op in f["operands"]]
    assert "moreThanInclude" in types
    assert "lessThanInclude" in types


def test_build_filter_date_range_tasks():
    f = build_filter(module="Tasks", date_from="2025-03-10")
    assert f["operands"][0]["field"] == "date_due"


def test_build_filter_date_range_unknown_module():
    f = build_filter(module="Notes", date_from="2025-03-10")
    assert f["operands"][0]["field"] == "date_entered"


def test_build_filter_combined():
    specs = [FilterSpec(field="status", op="eq", value="Held")]
    f = build_filter(search="review", filters=specs, date_from="2025-03-01")
    assert len(f["operands"]) == 3  # search + date_from + status filter


def test_build_filter_eq_operator():
    specs = [FilterSpec(field="status", op="eq", value="Held")]
    f = build_filter(filters=specs)
    assert f["operands"][0]["type"] == "eq"


def test_build_filter_gte_maps_to_more_than_include():
    specs = [FilterSpec(field="amount", op="gte", value="100")]
    f = build_filter(filters=specs)
    assert f["operands"][0]["type"] == "moreThanInclude"


def test_build_filter_empty():
    f = build_filter()
    assert f == {"operator": "and", "operands": []}


def test_build_filter_contacts_account_id_uses_relation_filter():
    specs = [FilterSpec(field="account_id", op="eq", value="a42333d4-c035-2f73-865c-64ee30906163")]
    f = build_filter(module="Contacts", filters=specs)
    relation_group = f["operands"][0]
    relation_operand = relation_group["operands"][0]
    relate_operand = relation_group["operands"][1]

    print(json.dumps(relation_group, indent=2))

    assert relation_group["operator"] == "or"
    assert relation_operand["field"] == "id"
    assert relation_operand["fieldModule"] == "Contacts"
    assert relation_operand["fieldRel"] == ["accounts"]
    assert relation_operand["type"] == "eq"
    assert relation_operand["value"] == "a42333d4-c035-2f73-865c-64ee30906163"
    assert relate_operand["type"] == "relate"
    assert relate_operand["module"] == "Accounts"
    assert relate_operand["relationship"] == ["accounts"]
    assert relate_operand["filter"]["operands"][0]["field"] == "id"
    assert relate_operand["filter"]["operands"][0]["value"] == "a42333d4-c035-2f73-865c-64ee30906163"


def test_build_filter_meetings_account_id_uses_parent_id_and_parent_type():
    specs = [FilterSpec(field="account_id", op="eq", value="a42333d4-c035-2f73-865c-64ee30906163")]
    f = build_filter(module="Meetings", filters=specs)
    operand = f["operands"][0]
    assert operand["field"] == "parent_id"
    assert operand["fieldModule"] == "Meetings"
    assert operand["type"] == "eq"
    assert operand["value"] == "a42333d4-c035-2f73-865c-64ee30906163"
    assert operand["parent_type"] == "Accounts"


def test_build_filter_meetings_account_id_with_name_falls_back_to_parent_name():
    specs = [FilterSpec(field="account_id", op="eq", value="365.bank")]
    f = build_filter(module="Meetings", filters=specs)
    operand = f["operands"][0]
    assert operand["field"] == "parent_name"
    assert operand["fieldModule"] == "Meetings"
    assert operand["type"] == "cont"
    assert operand["value"] == "365.bank"
    assert operand["parent_type"] == "Accounts"


def test_build_order_asc():
    result = build_order("date_start:asc")
    assert result == [{"field": "date_start", "sort": "ASC", "module": None}]


def test_build_order_desc():
    result = build_order("name:desc")
    assert result == [{"field": "name", "sort": "DESC", "module": None}]


def test_build_order_default_asc():
    result = build_order("name")
    assert result == [{"field": "name", "sort": "ASC", "module": None}]


def test_build_order_none():
    assert build_order(None) == []


def test_build_order_empty():
    assert build_order("") == []
