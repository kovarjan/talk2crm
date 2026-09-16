from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.endpoints import _normalize_ingest_modules
from app.engine.tool_validator import validate_rag_search_call
from app.engine.tools import build_tools
from app.utils.modules import canonical_module_name


class _Crm:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict) -> dict:
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict:
        return {"records": []}


class _Rag:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def search(self, *, tenant_id: str, query: str, limit: int = 5, modules=None) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "modules": list(modules or [])})
        return []

    async def search_entities(self, **kwargs) -> list[dict[str, Any]]:
        return []


def test_product_aliases_canonicalize() -> None:
    for alias in ("product", "products", "producttemplate", "producttemplates", "ProductTemplates"):
        assert canonical_module_name(alias) == "ProductTemplates"


def test_ingest_accepts_product_templates() -> None:
    assert _normalize_ingest_modules(["products", "Contacts"]) == ["ProductTemplates", "Contacts"]
    assert _normalize_ingest_modules(None) == ["Contacts", "Accounts", "Meetings"]


def test_rag_validator_allows_product_templates() -> None:
    assert validate_rag_search_call(query="vrtačka", limit=5, module="producttemplates") is None


def test_rag_search_tool_filters_to_product_templates() -> None:
    rag = _Rag()
    tools = build_tools(tenant_id="t", user_id="u", input_text="x", request_context=None,
                        crm_client=_Crm(), rag_service=rag, capabilities={"crm"})
    tool = next(t for t in tools if getattr(t, "name", "") == "rag_search_tool")
    asyncio.run(tool.ainvoke({"query": "vrtačka", "module": "producttemplates", "limit": 3}))
    assert rag.calls and rag.calls[0]["modules"] == ["ProductTemplates"]
