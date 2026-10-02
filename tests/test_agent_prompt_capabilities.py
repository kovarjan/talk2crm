from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.agent import _build_system_prompt

# Gateway-native tools of the crm capability; CRM data tools come from the tenant manifest
# and reach the prompt as ToolSet blocks (see test_prompt_uses_toolset_blocks).
CRM_TOOLS = {"rag_search_tool"}


def test_prompt_lists_only_present_tools_in_order() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS, capabilities={"crm"})
    assert "1. rag_search_tool(" in prompt
    assert "web_search_tool(" not in prompt
    assert "propose_form_fields_tool(" not in prompt


def test_prompt_numbers_web_tools_after_crm_tools() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"web_search_tool", "web_fetch_tool"}, capabilities={"crm", "web"})
    assert "2. web_search_tool(" in prompt
    assert "3. web_fetch_tool(" in prompt
    assert "REŽIM WEB" in prompt


def test_form_capability_adds_block_and_tools() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"propose_form_fields_tool", "research_record_tool"}, capabilities={"crm", "form"})
    assert "2. propose_form_fields_tool(" in prompt
    assert "REŽIM FORMULÁŘ" in prompt


def test_aggregate_blocks_follow_registry_tools() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"crm_aggregate_tool", "math_tool"}, capabilities={"crm"})
    assert "2. crm_aggregate_tool(" in prompt
    assert "3. math_tool(" in prompt


def test_legacy_call_without_capabilities_still_renders() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS)
    assert "1. rag_search_tool(" in prompt
    assert "REŽIM" not in prompt


def test_braces_are_rendered_single() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"rag_search_tool"}, capabilities={"crm"})
    assert "{{" not in prompt


def test_prompt_uses_toolset_blocks_and_capability_prompt() -> None:
    blocks = ["rag_search_tool(query: str)\n   — x", "crm_query_tool(module: str)\n   — přesný dotaz"]
    prompt = _build_system_prompt("t1", set(), tool_blocks=blocks, capability_block="REŽIM DVEŘE")
    assert "1. rag_search_tool(query: str)" in prompt
    assert "2. crm_query_tool(module: str)" in prompt
    assert "REŽIM DVEŘE" in prompt


def test_prompt_says_when_crm_is_unavailable() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS, tool_blocks=["rag_search_tool(query: str)"], crm_available=False)
    assert "CRM je teď nedostupné" in prompt
