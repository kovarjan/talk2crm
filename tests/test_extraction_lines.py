from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.form_patch import build_form_patch
from app.engine.extraction import EXTRACT_FIELDS_ALLOWED_TOOLS, EXTRACT_FIELDS_SYSTEM_PROMPT, validate_line_rows

LINE_FIELD = {"name": "lines", "type": "line_items", "line_module": "Products", "line_fields": [
    {"name": "name", "type": "text", "editable": True},
    {"name": "product_template_id", "type": "relate", "editable": True, "target_module": "ProductTemplates"},
    {"name": "quantity", "type": "number_decimal", "editable": True},
    {"name": "deal_tot", "type": "readonly_computed", "editable": False},
]}


def test_rows_keep_editable_keys_and_drop_computed() -> None:
    rows, dropped = validate_line_rows([
        {"name": "Vrtačka", "product_template_id": {"id": "p1", "name": "Vrtačka"}, "quantity": "2", "deal_tot": 999, "bogus": 1},
    ], LINE_FIELD)
    assert rows == [{"name": "Vrtačka", "product_template_id": {"id": "p1", "name": "Vrtačka"}, "quantity": "2"}]
    assert dropped == []


def test_rows_without_identity_or_bad_shape_are_dropped_with_reason() -> None:
    rows, dropped = validate_line_rows([
        {"quantity": 1},
        "not a dict",
        {"name": "OK", "quantity": {"nested": True}},
    ], LINE_FIELD)
    assert rows == [{"name": "OK"}]
    assert [d["row"] for d in dropped] == [0, 1]
    assert "name" in dropped[0]["reason"]


def test_unresolved_product_keeps_name_only() -> None:
    rows, _ = validate_line_rows([{"product_template_id": {"name": "Neznámý"}, "quantity": 1}], LINE_FIELD)
    assert rows == [{"product_template_id": {"name": "Neznámý"}, "quantity": 1}]


def test_form_patch_carries_lines() -> None:
    patch = build_form_patch(module="Quotes", record_id="q1", fields={}, schema={"sections": []},
                             lines={"rows": [{"name": "A"}], "dropped": []}, line_module="Products")
    assert patch["lines"] == {"mode": "append", "line_module": "Products", "rows": [{"name": "A"}], "dropped": []}
    patch = build_form_patch(module="Contacts", record_id=None, fields={}, schema={"sections": []})
    assert patch["lines"] == {"mode": "append", "line_module": None, "rows": [], "dropped": []}


def test_prompt_and_tools_mention_lines() -> None:
    assert '"lines"' in EXTRACT_FIELDS_SYSTEM_PROMPT
    assert "product_lookup_tool" in EXTRACT_FIELDS_ALLOWED_TOOLS
