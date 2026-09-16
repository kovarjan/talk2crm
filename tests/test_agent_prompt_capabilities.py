from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.agent import _build_system_prompt

CRM_TOOLS = {"rag_search_tool", "crm_query_tool", "my_meetings_tool", "crm_action_tool", "get_company_overview"}


def test_prompt_lists_only_present_tools_in_order() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS, capabilities={"crm"})
    assert "1. rag_search_tool(" in prompt
    assert "5. get_company_overview(" in prompt
    assert "web_search_tool(" not in prompt
    assert "propose_form_fields_tool(" not in prompt


def test_prompt_numbers_web_tools_after_crm_tools() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"web_search_tool", "web_fetch_tool"}, capabilities={"crm", "web"})
    assert "6. web_search_tool(" in prompt
    assert "7. web_fetch_tool(" in prompt
    assert "REŽIM WEB" in prompt


def test_form_capability_adds_block_and_tools() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"propose_form_fields_tool", "research_record_tool"}, capabilities={"crm", "form"})
    assert "6. propose_form_fields_tool(" in prompt
    assert "REŽIM FORMULÁŘ" in prompt


def test_aggregate_blocks_follow_registry_tools() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS | {"crm_aggregate_tool", "math_tool"}, capabilities={"crm"})
    assert "6. crm_aggregate_tool(" in prompt
    assert "7. math_tool(" in prompt


def test_legacy_call_without_capabilities_still_renders() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS)
    assert "1. rag_search_tool(" in prompt
    assert "REŽIM" not in prompt


def test_braces_are_rendered_single() -> None:
    prompt = _build_system_prompt("t1", CRM_TOOLS, capabilities={"crm"})
    assert 'filters je JSON pole [{"field":"...","op":"eq","value":"..."}]' in prompt
    assert "{{" not in prompt
