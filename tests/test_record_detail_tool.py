from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.capabilities import CAPABILITIES
from app.engine.record_detail_tool import build_record_detail_tools, summarize_lines

LINE_FIELD = {
    "name": "lines", "type": "line_items", "line_module": "Products",
    "line_fields": [
        {"name": "name", "label": "Název", "type": "text", "editable": True},
        {"name": "quantity", "label": "Množství", "type": "number_decimal", "editable": True},
        {"name": "deal_tot", "label": "Celkem", "type": "readonly_computed", "editable": False},
    ],
    "current_value": [
        {"name": "A", "quantity": 2, "deal_tot": "100.50"},
        {"name": "B", "quantity": 1, "deal_tot": 49.5},
        {"name": "C", "quantity": None, "deal_tot": "x"},
    ],
}
SCHEMA = {"module": "Quotes", "record_id": "q1", "mode": "existing", "sections": [
    {"group": "Základ", "fields": [{"name": "name", "label": "Název", "type": "text", "current_value": "Nabídka 1"},
                                   {"name": "total", "label": "Celkem", "type": "currency", "current_value": "150.00"}]},
    {"group": "Položky", "fields": [LINE_FIELD]},
]}


class _Crm:
    mode = "coripo_public"

    def __init__(self, schema=SCHEMA, fail=False) -> None:
        self.schema, self.fail = schema, fail
        self.calls = []

    async def get_ai_schema(self, module, record_id=None, **kw):
        self.calls.append(("schema", module, record_id))
        if self.fail:
            raise RuntimeError("403")
        return self.schema


def test_summarize_lines_sums_numeric_columns_and_counts() -> None:
    out = summarize_lines(LINE_FIELD)
    assert out["line_module"] == "Products"
    assert [c["name"] for c in out["columns"]] == ["name", "quantity", "deal_tot"]
    assert out["totals"]["count"] == 3
    assert out["totals"]["sum"]["quantity"] == 3.0
    assert out["totals"]["sum"]["deal_tot"] == 150.0
    assert out["truncated"] is False


def test_summarize_lines_truncates() -> None:
    field = {**LINE_FIELD, "current_value": [{"name": str(i), "quantity": 1} for i in range(250)]}
    out = summarize_lines(field, max_rows=200)
    assert len(out["rows"]) == 200 and out["truncated"] is True and out["totals"]["count"] == 250


def test_tool_merges_schema_fields_and_lines() -> None:
    crm = _Crm()
    tool = build_record_detail_tools(tenant_id="t", user_id="u", crm_client=crm)[0]
    out = json.loads(asyncio.run(tool.ainvoke({"module": "quotes", "record_id": "q1"})))
    assert out["status"] == "ok" and out["module"] == "Quotes" and out["name"] == "Nabídka 1"
    assert out["fields"]["total"] == {"label": "Celkem", "value": "150.00"}
    assert out["lines"]["totals"]["sum"]["deal_tot"] == 150.0
    assert out["cards"][0]["type"] == "record"
    assert ("schema", "Quotes", "q1") in crm.calls


def test_tool_without_lines_and_on_error() -> None:
    tool = build_record_detail_tools(tenant_id="t", user_id="u", crm_client=_Crm())[0]
    out = json.loads(asyncio.run(tool.ainvoke({"module": "Quotes", "record_id": "q1", "include_lines": False})))
    assert out["lines"] is None
    tool = build_record_detail_tools(tenant_id="t", user_id="u", crm_client=_Crm(fail=True))[0]
    out = json.loads(asyncio.run(tool.ainvoke({"module": "Quotes", "record_id": "q1"})))
    assert out["status"] == "record_unavailable"


def test_tool_belongs_to_crm_capability() -> None:
    assert "crm_record_detail_tool" in CAPABILITIES["crm"].tool_names
