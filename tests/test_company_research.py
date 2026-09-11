from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.company_research import research_company

ACCOUNT_SCHEMA = {
    "module": "Accounts",
    "record_id": None,
    "mode": "new",
    "sections": [
        {
            "group": "Základní údaje",
            "fields": [
                {"name": "name", "label": "Název", "type": "text", "required": True, "editable": True, "current_value": "ACME s.r.o."},
                {"name": "website", "label": "Web", "type": "text", "required": False, "editable": True, "current_value": None},
                {"name": "phone_office", "label": "Telefon", "type": "text", "required": False, "editable": True, "current_value": None},
                {"name": "description", "label": "Popis", "type": "textarea", "required": False, "editable": True, "current_value": None},
            ],
        }
    ],
}


class FakeCrmClient:
    mode = "coripo_public"


def _fake_tools():
    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    return [
        _Tool("web_search_tool"),
        _Tool("web_fetch_tool"),
        _Tool("crm_query_tool"),
        _Tool("get_company_overview"),
        _Tool("crm_action_tool"),
    ]


@pytest.mark.asyncio
async def test_research_company_returns_only_schema_fields_and_sources() -> None:
    fake_agent_output = {
        "output": (
            '{"message": "Doplnil jsem web, telefon a popis podle firemního webu.", '
            '"fields": {'
            '"website": "https://acme.cz", "phone_office": "+420123456789", '
            '"description": "Výrobce pneumatik se sídlem ve Zlíně.", '
            '"not_in_schema": "should be dropped"'
            "}, "
            '"sources": ["https://acme.cz", "https://acme.cz/o-nas"]}'
        ),
        "intermediate_steps": [],
    }

    with (
        patch("app.engine.company_research.build_tools", return_value=_fake_tools()) as mock_build_tools,
        patch("app.engine.company_research.run_agent", new=AsyncMock(return_value=fake_agent_output)) as mock_run_agent,
    ):
        result = await research_company(
            tenant_id="test-tenant",
            user_id="user-1",
            record_id=None,
            field_schema=ACCOUNT_SCHEMA,
            current_values={"name": "ACME s.r.o."},
            company_name="ACME s.r.o.",
            hint=None,
            crm_client=FakeCrmClient(),  # type: ignore[arg-type]
            rag_service=None,
        )

    assert result["message"] == "Doplnil jsem web, telefon a popis podle firemního webu."
    assert result["fields"] == {
        "website": "https://acme.cz",
        "phone_office": "+420123456789",
        "description": "Výrobce pneumatik se sídlem ve Zlíně.",
    }
    assert "not_in_schema" not in result["fields"]
    assert result["sources"] == ["https://acme.cz", "https://acme.cz/o-nas"]

    # crm_action_tool must never reach this agent, even though build_tools() returns it
    passed_tools = mock_run_agent.call_args.kwargs["tools"]
    assert {t.name for t in passed_tools} == {"web_search_tool", "web_fetch_tool", "crm_query_tool", "get_company_overview"}
    assert mock_build_tools.called
    # The company name must reach the agent's input text
    assert "ACME s.r.o." in mock_run_agent.call_args.kwargs["input_text"]


@pytest.mark.asyncio
async def test_research_company_defaults_message_when_nothing_found() -> None:
    fake_agent_output = {"output": '{"fields": {}}', "intermediate_steps": []}

    with (
        patch("app.engine.company_research.build_tools", return_value=_fake_tools()),
        patch("app.engine.company_research.run_agent", new=AsyncMock(return_value=fake_agent_output)),
    ):
        result = await research_company(
            tenant_id="test-tenant",
            user_id="user-1",
            record_id="acc-1",
            field_schema=ACCOUNT_SCHEMA,
            current_values={},
            company_name="Neexistující firma s.r.o.",
            hint="Zlín",
            crm_client=FakeCrmClient(),  # type: ignore[arg-type]
            rag_service=None,
        )

    assert result["fields"] == {}
    assert result["sources"] == []
    assert result["message"]
