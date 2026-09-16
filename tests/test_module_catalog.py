from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.agent import _build_system_prompt
from app.engine.tool_validator import validate_crm_action_call, validate_crm_query_call, validate_rag_search_call
from app.services.module_catalog import ModuleCatalog, get_module_catalog, reset_module_catalog_cache

MODULES = [
    {"name": "Quotes", "label": "Nabídky", "read": True, "write": True, "has_schema": True, "line_module": "Products", "rag": True},
    {"name": "acm_custom_thing", "label": "Věc", "read": True, "write": False, "has_schema": True, "line_module": None, "rag": False},
]


class _Crm:
    def __init__(self, fail=False) -> None:
        self.fail = fail
        self.calls = 0

    async def get_ai_modules(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return MODULES


def _query(module: str, catalog=None):
    return validate_crm_query_call(module=module, filters="[]", date_from=None, date_to=None, limit=5, order_by=None, catalog=catalog)


def test_catalog_sets_and_line_module() -> None:
    catalog = ModuleCatalog(MODULES)
    assert catalog.readable() == {"quotes", "acm_custom_thing"}
    assert catalog.writable() == {"quotes"}
    assert catalog.rag() == {"quotes"}
    assert catalog.line_module("quotes") == "Products"
    assert not catalog.is_empty


def test_validators_use_catalog_when_given() -> None:
    catalog = ModuleCatalog(MODULES)
    assert _query("acm_custom_thing", catalog) is None
    assert _query("Contacts", catalog) is not None
    assert validate_crm_action_call(module="acm_custom_thing", action="create", data_json="{}", catalog=catalog) is not None
    assert validate_rag_search_call(query="x", limit=5, module="quotes", catalog=catalog) is None
    assert validate_rag_search_call(query="x", limit=5, module="", catalog=catalog) is None
    # legacy behaviour without a catalog is unchanged
    assert _query("Contacts") is None


def test_get_module_catalog_caches_and_falls_back() -> None:
    reset_module_catalog_cache()
    crm = _Crm()
    a = asyncio.run(get_module_catalog(crm, tenant_id="t", user_id="u"))
    b = asyncio.run(get_module_catalog(crm, tenant_id="t", user_id="u"))
    assert a is b and crm.calls == 1
    reset_module_catalog_cache()
    fallback = asyncio.run(get_module_catalog(_Crm(fail=True), tenant_id="t", user_id="u"))
    assert "contacts" in fallback.readable() and fallback.is_fallback


def test_prompt_lists_catalog_modules() -> None:
    prompt = _build_system_prompt("t", {"crm_query_tool"}, capabilities={"crm"}, readable_modules="Quotes, acm_custom_thing")
    assert "podporované moduly pro čtení: Quotes, acm_custom_thing" in prompt
    prompt = _build_system_prompt("t", {"crm_query_tool"}, capabilities={"crm"})
    assert "podporované moduly pro čtení: Accounts, Contacts" in prompt
