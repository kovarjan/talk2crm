from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.extraction import extract_fields

CONTACT_SCHEMA = {
    "module": "Contacts",
    "record_id": None,
    "mode": "new",
    "sections": [
        {
            "group": "Základní údaje",
            "fields": [
                {"name": "first_name", "label": "Jméno", "type": "text", "required": False, "editable": True, "current_value": None},
                {"name": "last_name", "label": "Příjmení", "type": "text", "required": True, "editable": True, "current_value": None},
                {"name": "email1", "label": "E-mail", "type": "email", "required": False, "editable": True, "current_value": None},
                {
                    "name": "account_name", "label": "Firma", "type": "relate", "required": False,
                    "editable": True, "id_field": "account_id", "target_module": "Accounts", "current_value": None,
                },
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

    return [_Tool("rag_search_tool"), _Tool("crm_query_tool"), _Tool("get_company_overview"), _Tool("crm_action_tool")]


@pytest.mark.asyncio
async def test_extract_fields_returns_only_schema_fields_and_resolves_relate() -> None:
    fake_agent_output = {
        "output": (
            '{"message": "Vyplnil jsem jméno, příjmení a firmu.", '
            '"fields": {'
            '"first_name": "Jan", "last_name": "Novák", '
            '"account_name": {"id": "acc-42", "name": "Acme s.r.o."}, '
            '"not_in_schema": "should be dropped", '
            '"phone_mobile": "+420600000000"'
            "}}"
        ),
        "intermediate_steps": [],
    }

    with (
        patch("app.engine.extraction.build_tools", return_value=_fake_tools()) as mock_build_tools,
        patch("app.engine.extraction.run_agent", new=AsyncMock(return_value=fake_agent_output)) as mock_run_agent,
    ):
        result = await extract_fields(
            tenant_id="test-tenant",
            user_id="user-1",
            module="Contacts",
            record_id=None,
            field_schema=CONTACT_SCHEMA,
            current_values={},
            messages=[{"role": "user", "content": "Jan Novák, Acme s.r.o., prosím o nabídku."}],
            crm_client=FakeCrmClient(),  # type: ignore[arg-type]
            rag_service=None,
        )

    assert result["message"] == "Vyplnil jsem jméno, příjmení a firmu."
    assert result["fields"] == {
        "first_name": "Jan",
        "last_name": "Novák",
        "account_name": {"id": "acc-42", "name": "Acme s.r.o."},
    }
    # "phone_mobile" isn't in CONTACT_SCHEMA's fields -> must be dropped, same as "not_in_schema"
    assert "phone_mobile" not in result["fields"]

    # crm_action_tool must never be handed to this agent, even though build_tools() returns it
    passed_tools = mock_run_agent.call_args.kwargs["tools"]
    assert {t.name for t in passed_tools} == {"rag_search_tool", "crm_query_tool", "get_company_overview"}
    assert mock_build_tools.called


@pytest.mark.asyncio
async def test_extract_fields_drops_unresolved_relate_with_no_id() -> None:
    fake_agent_output = {
        "output": '{"message": "Firmu se nepodařilo najít.", "fields": {"account_name": {"name": "Neznámá firma s.r.o."}}}',
        "intermediate_steps": [],
    }

    with (
        patch("app.engine.extraction.build_tools", return_value=_fake_tools()),
        patch("app.engine.extraction.run_agent", new=AsyncMock(return_value=fake_agent_output)),
    ):
        result = await extract_fields(
            tenant_id="test-tenant",
            user_id="user-1",
            module="Contacts",
            record_id=None,
            field_schema=CONTACT_SCHEMA,
            current_values={},
            messages=[{"role": "user", "content": "firma se jmenuje nějak jako Neznámá firma"}],
            crm_client=FakeCrmClient(),  # type: ignore[arg-type]
            rag_service=None,
        )

    # No id resolved -> the harness must get {"name": ...} with no fabricated id, per spec
    assert result["fields"]["account_name"] == {"name": "Neznámá firma s.r.o."}
    assert "id" not in result["fields"]["account_name"]
