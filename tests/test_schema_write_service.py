from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.schema_write_service import SchemaDrivenWriteService, SchemaWriteError


MEETINGS_SCHEMA = {
    "module": "Meetings",
    "record_id": "meeting-1",
    "mode": "existing",
    "sections": [
        {
            "group": "Info",
            "fields": [
                {"name": "name", "type": "text", "editable": True, "current_value": None},
                {
                    "name": "assigned_user_id",
                    "type": "relate",
                    "editable": True,
                    "id_field": "assigned_user_id",
                    "target_module": "Users",
                    "current_value": None,
                },
                {
                    "name": "amount",
                    "type": "currency",
                    "editable": True,
                    "current_value": None,
                },
                {
                    "name": "parent_id",
                    "type": "polymorphic_relate",
                    "editable": True,
                    "id_field": "parent_id",
                    "type_field": "parent_type",
                    "allowed_modules": [{"value": "Accounts", "label": "Account"}],
                    "current_value": None,
                },
            ],
        },
        {
            "group": "Invitees",
            "fields": [
                {
                    "name": "invitees",
                    "type": "invitee_list",
                    "editable": True,
                    "target_modules": ["Users", "Contacts", "Leads"],
                    "current_value": {"Users": [], "Contacts": [], "Leads": []},
                },
            ],
        },
    ],
}


class FakeCoripoClient:
    def __init__(self, schema: dict[str, Any], ai_write_response: dict[str, Any]):
        self._schema = schema
        self._ai_write_response = ai_write_response
        self.last_get_ai_schema_call: dict[str, Any] | None = None
        self.last_ai_write_call: dict[str, Any] | None = None

    async def get_ai_schema(self, module, record_id=None, *, max_options=200):
        self.last_get_ai_schema_call = {
            "module": module,
            "record_id": record_id,
            "max_options": max_options,
        }
        return self._schema

    async def ai_write(self, module, data, *, record_id=None, dry_run=False):
        self.last_ai_write_call = {
            "module": module,
            "data": data,
            "record_id": record_id,
            "dry_run": dry_run,
        }
        return self._ai_write_response


def test_build_payload_folds_polymorphic_relate_and_coerces_types():
    client = FakeCoripoClient(
        MEETINGS_SCHEMA,
        {"success": True, "record_id": "meeting-1", "dry_run": False, "updated_fields": ["name"]},
    )
    service = SchemaDrivenWriteService(crm_client=client)

    result = asyncio.run(
        service.write(
            module="Meetings",
            data={
                "id": "meeting-1",
                "fields": {
                    "name": "Schůzka",
                    "assigned_user_id": "user-42",
                    "amount": 1500,
                    "parent_type": "Accounts",
                    "parent_id": "account-1",
                },
                "invitees": {
                    "Contacts": [{"id": "contact-1"}, {"id": ""}],
                    "Users": ["user-9"],
                },
            },
            dry_run=False,
        )
    )

    assert result.success is True
    assert result.record_id == "meeting-1"

    sent = client.last_ai_write_call["data"]
    assert sent["name"] == "Schůzka"
    # bare relate id gets wrapped into the standardized {id} shape
    assert sent["assigned_user_id"] == {"id": "user-42"}
    # bare currency amount gets wrapped into {amount}
    assert sent["amount"] == {"amount": 1500}
    # parent_type + parent_id fold into a single polymorphic_relate entry keyed by parent_id
    assert sent["parent_id"] == {"module": "Accounts", "id": "account-1"}
    assert "parent_type" not in sent
    # invitees: bare string ids normalized to {id}, blank ids dropped, missing module defaults to []
    assert sent["invitees"] == {
        "Users": [{"id": "user-9"}],
        "Contacts": [{"id": "contact-1"}],
        "Leads": [],
    }

    assert client.last_get_ai_schema_call == {
        "module": "Meetings",
        "record_id": "meeting-1",
        "max_options": 200,
    }
    assert client.last_ai_write_call["record_id"] == "meeting-1"
    assert client.last_ai_write_call["dry_run"] is False


def test_write_surfaces_field_errors_without_raising():
    client = FakeCoripoClient(
        MEETINGS_SCHEMA,
        {
            "success": False,
            "record_id": "meeting-1",
            "dry_run": True,
            "errors": [{"field": "amount", "code": "invalid_type", "message": "Expected {amount, currency_code?}"}],
        },
    )
    service = SchemaDrivenWriteService(crm_client=client)

    result = asyncio.run(
        service.write(module="Meetings", data={"id": "meeting-1", "fields": {}}, dry_run=True)
    )

    assert result.success is False
    assert result.errors == [{"field": "amount", "code": "invalid_type", "message": "Expected {amount, currency_code?}"}]
    assert result.dry_run is True


def test_missing_required_fields_are_warnings_not_failures():
    # Older Coripo builds put required_missing under errors; current ones under warnings.
    for response in (
        {"success": False, "record_id": None, "dry_run": True,
         "errors": [{"field": "zapis", "code": "required_missing", "message": "Field 'zapis' is required"}]},
        {"success": True, "record_id": None, "dry_run": True, "updated_fields": ["name"],
         "warnings": [{"field": "zapis", "code": "required_missing", "message": "Field 'zapis' is required"}]},
    ):
        client = FakeCoripoClient(MEETINGS_SCHEMA, response)
        result = asyncio.run(
            SchemaDrivenWriteService(crm_client=client).write(module="Meetings", data={"fields": {"name": "Schůzka"}}, dry_run=True)
        )
        assert result.success is True
        assert result.errors == []
        assert [warning["field"] for warning in result.warnings] == ["zapis"]


def test_agent_helper_fields_never_reach_ai_write():
    client = FakeCoripoClient(MEETINGS_SCHEMA, {"success": True, "record_id": None, "dry_run": True})
    asyncio.run(
        SchemaDrivenWriteService(crm_client=client).write(
            module="Meetings",
            data={"fields": {"name": "Schůzka", "contact_id": "contact-1", "contact_name": "Novák", "duration_hours": 1}},
            dry_run=True,
        )
    )
    sent = client.last_ai_write_call["data"]
    assert sent == {"name": "Schůzka"}


def test_new_record_has_no_record_id_and_omits_it_from_ai_write_call():
    client = FakeCoripoClient(
        {**MEETINGS_SCHEMA, "record_id": None, "mode": "new"},
        {"success": True, "record_id": "new-id", "dry_run": False, "updated_fields": ["name"]},
    )
    service = SchemaDrivenWriteService(crm_client=client)

    result = asyncio.run(
        service.write(module="Meetings", data={"fields": {"name": "Nová schůzka"}}, dry_run=False)
    )

    assert result.success is True
    assert result.record_id == "new-id"
    assert client.last_get_ai_schema_call["record_id"] is None
    assert client.last_ai_write_call["record_id"] is None


def test_raises_schema_write_error_when_schema_has_no_sections():
    class NoSectionsClient:
        async def get_ai_schema(self, module, record_id=None, *, max_options=200):
            return {"module": module, "sections": None}

    service = SchemaDrivenWriteService(crm_client=NoSectionsClient())

    with pytest.raises(SchemaWriteError):
        asyncio.run(service.write(module="Meetings", data={"fields": {}}, dry_run=True))
