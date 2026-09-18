from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.capabilities import (
    CAPABILITIES,
    TOOL_PROMPT_BLOCKS,
    TOOL_PROMPT_ORDER,
    prompt_blocks_for,
    resolve_capabilities,
    tool_names_for,
)
from app.engine.tools import build_tools


def test_crm_is_always_enabled_even_when_omitted() -> None:
    enabled, unknown = resolve_capabilities({"capabilities": []})
    assert enabled == {"crm"}
    assert unknown == []


def test_missing_capabilities_key_enables_defaults() -> None:
    enabled, _ = resolve_capabilities({})
    assert "crm" in enabled
    assert "form" in enabled  # default_on
    assert "web" not in enabled  # default_off


def test_none_context_behaves_like_empty_context() -> None:
    assert resolve_capabilities(None) == resolve_capabilities({})


def test_web_search_alias_enables_web() -> None:
    enabled, _ = resolve_capabilities({"web_search": True, "capabilities": []})
    assert enabled == {"crm", "web"}


def test_unknown_ids_are_reported_not_raised() -> None:
    enabled, unknown = resolve_capabilities({"capabilities": ["crm", "summarize", "form", "future-capability"]})
    assert enabled == {"crm", "form"}
    assert unknown == ["summarize", "future-capability"]


def test_non_string_entries_are_ignored() -> None:
    enabled, unknown = resolve_capabilities({"capabilities": ["web", 5, None, {"x": 1}]})
    assert enabled == {"crm", "web"}
    assert unknown == []


def test_tool_names_follow_enabled_capabilities() -> None:
    names = tool_names_for({"crm"})
    assert "crm_query_tool" in names
    assert "crm_action_tool" in names
    assert "web_search_tool" not in names
    assert "propose_form_fields_tool" not in names
    names = tool_names_for({"crm", "web", "form"})
    assert {"web_search_tool", "web_fetch_tool", "propose_form_fields_tool", "research_record_tool"} <= names


def test_prompt_blocks_only_for_enabled() -> None:
    assert prompt_blocks_for({"crm"}) == ""
    web_block = prompt_blocks_for({"crm", "web"})
    assert "REŽIM WEB" in web_block
    assert "propose_form_fields_tool" not in web_block
    assert "propose_form_fields_tool" in prompt_blocks_for({"crm", "form"})


def test_every_registered_tool_has_a_prompt_block_in_order() -> None:
    all_tools = set().union(*(cap.tool_names for cap in CAPABILITIES.values()))
    assert all_tools <= set(TOOL_PROMPT_BLOCKS)
    assert set(TOOL_PROMPT_ORDER) == set(TOOL_PROMPT_BLOCKS)


class _DummyCrmClient:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict) -> dict:
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict:
        return {"records": []}


def _names(tools: list) -> set[str]:
    return {str(getattr(t, "name", "")) for t in tools}


def test_build_tools_without_capabilities_keeps_legacy_full_set() -> None:
    names = _names(build_tools(
        tenant_id="t", user_id="u", input_text="ahoj", request_context=None,
        crm_client=_DummyCrmClient(), rag_service=None, action_confirmation=False,
    ))
    assert {"crm_query_tool", "crm_action_tool", "web_search_tool"} <= names


def test_build_tools_filters_by_capabilities() -> None:
    names = _names(build_tools(
        tenant_id="t", user_id="u", input_text="ahoj", request_context=None,
        crm_client=_DummyCrmClient(), rag_service=None, action_confirmation=False,
        capabilities={"crm"},
    ))
    assert "crm_query_tool" in names
    assert "web_search_tool" not in names
    assert "web_fetch_tool" not in names


def test_daily_briefing_is_default_on_and_registered():
    enabled, unknown = resolve_capabilities({})
    assert "briefing" in enabled and not unknown
    names = _names(build_tools(tenant_id="t", user_id="u", input_text="můj den", request_context=None,
                               crm_client=_DummyCrmClient(), rag_service=None, capabilities=enabled))
    assert "daily_briefing_tool" in names
    assert "daily_briefing_tool" not in tool_names_for({"crm"})
