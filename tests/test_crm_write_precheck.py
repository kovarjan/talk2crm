from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.crm_write_service import CRMWriteService

MEETINGS_SCHEMA = {
    "sections": [
        {
            "fields": [
                {"name": "name", "type": "text", "required": True},
                {"name": "description", "type": "textarea"},
                {"name": "zapis", "type": "textarea", "required": True},
                {"name": "date_start", "type": "datetime", "required": True},
                {"name": "date_end", "type": "datetime", "required": True},
                {"name": "parent_id", "type": "polymorphic_relate", "type_field": "parent_type"},
                {"name": "invitees", "type": "invitee_list"},
            ]
        }
    ]
}


class PrecheckCrmClient:
    """Mimics Coripo ai_write: unknown fields are errors, missing required are warnings."""

    mode = "coripo_public"

    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    async def get_ai_schema(self, module, record_id=None, *, max_options=200):
        return MEETINGS_SCHEMA

    async def ai_write(self, module, data, *, record_id=None, dry_run=False):
        self.writes.append(data)
        known = {field["name"] for section in MEETINGS_SCHEMA["sections"] for field in section["fields"]}
        errors = [
            {"field": name, "code": "unknown_field", "message": f"Field '{name}' is not part of this module's schema"}
            for name in data
            if name not in known
        ]
        warnings = [
            {"field": field["name"], "code": "required_missing", "message": "required"}
            for section in MEETINGS_SCHEMA["sections"]
            for field in section["fields"]
            if field.get("required") and field["name"] not in data
        ]
        return {"success": not errors, "record_id": None, "dry_run": dry_run, "errors": errors, "warnings": warnings}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}


def _service(client: PrecheckCrmClient) -> CRMWriteService:
    return CRMWriteService(
        tenant_id="ai-local",
        user_id="1",
        input_text="Naplánuj schůzku ohledně fakturace",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )


def test_unknown_fields_are_dropped_and_missing_required_does_not_block_confirmation() -> None:
    client = PrecheckCrmClient()
    result = asyncio.run(
        _service(client).execute_action(
            module="Meetings",
            action="create",
            data_json=json.dumps({"fields": {"name": "Schůzka - Fakturace", "date_start": "2026-09-15 14:00:00", "made_up_field": "x"}}),
        )
    )

    assert result["status"] == "confirmation_required"
    pending_fields = result["pending_action"]["data"]["fields"]
    assert "made_up_field" not in pending_fields
    assert any("Dropped fields" in note for note in result["adjustments"])
    # The adjustment engine derived the end time and notes, so only nothing or the
    # truly unknown remains as an advisory warning; the record is still offered.
    assert all(warning["code"] == "required_missing" for warning in result.get("missing_required", []))


def test_invalid_values_still_ask_for_a_fix_before_confirmation() -> None:
    class RejectingClient(PrecheckCrmClient):
        async def ai_write(self, module, data, *, record_id=None, dry_run=False):
            return {
                "success": False,
                "record_id": None,
                "dry_run": dry_run,
                "errors": [{"field": "date_start", "code": "invalid_type", "message": "Expected a datetime"}],
                "warnings": [],
            }

    result = asyncio.run(
        _service(RejectingClient()).execute_action(
            module="Meetings",
            action="create",
            data_json=json.dumps({"fields": {"name": "Schůzka", "date_start": "zítra"}}),
        )
    )

    assert result["status"] == "needs_more_info"
    assert [error["field"] for error in result["field_errors"]] == ["date_start"]
