from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.capabilities import CAPABILITIES, resolve_capabilities
from app.engine.product_tools import build_product_tools
from app.engine.tools import build_tools


def _hit(id_: str, name: str, part: str, price: float, score: float = 0.5) -> dict[str, Any]:
    return {"score": score, "payload": {"module": "ProductTemplates", "record_id": id_, "name": name,
            "record": {"id": id_, "name": name, "mft_part_num": part, "manufacturer_name": "Bosch",
                       "category_name": "Nářadí", "list_price": price, "cost_price": price * 0.7, "currency_id": "-99"}}}


class _Rag:
    def __init__(self, hits=None, fail=False) -> None:
        self.hits = hits or []
        self.fail = fail

    async def search(self, *, tenant_id, query, limit=5, modules=None):
        if self.fail:
            raise RuntimeError("qdrant down")
        return self.hits


class _Crm:
    mode = "coripo_public"

    def __init__(self) -> None:
        self.calls = []

    async def execute_module_action(self, module, action, data):
        self.calls.append((module, action, data))
        return {"records": [{"id": "p9", "name": "Vrtačka Z", "mft_part_num": "Z-9", "list_price": 500}]}

    async def generic_search(self, query, scope="all"):
        return {"records": []}


def _run(tools, args):
    tool = next(t for t in tools if getattr(t, "name", "") == "product_lookup_tool")
    return json.loads(asyncio.run(tool.ainvoke(args)))


def test_exact_part_number_wins_over_vector_score() -> None:
    rag = _Rag([_hit("p1", "Vrtačka XY", "XY-100", 1200, 0.9), _hit("p2", "Vrtačka ABC", "AB-7", 900, 0.95)])
    out = _run(build_product_tools(tenant_id="t", user_id="u", crm_client=_Crm(), rag_service=rag), {"query": "AB-7"})
    assert out["status"] == "ok"
    assert out["results"][0]["id"] == "p2"
    assert out["results"][0]["price_source"] == "catalog"
    assert out["results"][0]["list_price"] == 900
    assert out["cards"][0]["type"] == "table"


def test_degrades_to_crm_search_when_qdrant_fails() -> None:
    crm = _Crm()
    out = _run(build_product_tools(tenant_id="t", user_id="u", crm_client=crm, rag_service=_Rag(fail=True)), {"query": "vrtačka"})
    assert out["status"] == "degraded"
    assert out["results"][0]["id"] == "p9"
    assert crm.calls[0][0] == "ProductTemplates"


def test_products_capability_is_default_on_and_registers_tool() -> None:
    enabled, _ = resolve_capabilities({})
    assert "products" in enabled
    assert CAPABILITIES["products"].tool_names == frozenset({"product_lookup_tool"})
    names = {getattr(t, "name", "") for t in build_tools(
        tenant_id="t", user_id="u", input_text="x", request_context=None,
        crm_client=_Crm(), rag_service=_Rag(), capabilities={"crm", "products"})}
    assert "product_lookup_tool" in names
    names = {getattr(t, "name", "") for t in build_tools(
        tenant_id="t", user_id="u", input_text="x", request_context=None,
        crm_client=_Crm(), rag_service=_Rag(), capabilities={"crm"})}
    assert "product_lookup_tool" not in names
