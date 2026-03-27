from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import patch

from app.core.config import get_settings
from app.engine.adjustments import ModuleAdjustmentEngine
from app.engine.quick_actions import QuickActionResult, try_handle_quick_action


class DummyCrmClient:
    mode = "coripo_public"
    contact_id = "8a7af628-5d4d-4636-8967-96d331b716f6"
    account_id = "a42333d4-c035-2f73-865c-64ee30906163"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "id": "d9207f5e-0abc-4b66-96d3-bd4e6ec97200"}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        if scope == "contacts":
            q = str(query or "").lower()
            if "karl" in q or "vybih" in q:
                return {
                    "records": [
                        {
                            "id": self.contact_id,
                            "first_name": "Karel",
                            "last_name": "Vybíhal",
                            "account_id": self.account_id,
                            "account_name": "PANAS, spol. s r.o.",
                        }
                    ]
                }
            return {
                "records": [
                    {
                        "id": self.contact_id,
                        "first_name": "Jana",
                        "last_name": "Mimrová",
                        "account_id": self.account_id,
                        "account_name": "PANAS, spol. s r.o.",
                    }
                ]
            }
        if scope == "accounts":
            q = str(query or "").lower()
            if "zliner" in q:
                return {
                    "records": [
                        {
                            "id": self.account_id,
                            "name": "ZLINER s.r.o.",
                        }
                    ]
                }
            return {
                "records": [
                    {
                        "id": self.account_id,
                        "name": "PANAS, spol. s r.o.",
                    }
                ]
            }
        return {"records": []}


class FallbackLookupCrmClient:
    mode = "coripo_public"
    contact_id = "11111111-2222-3333-4444-555555555555"
    account_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((module, action, dict(data)))
        if module == "Contacts" and action == "list":
            return {
                "records": [
                    {
                        "id": self.contact_id,
                        "first_name": "Karel",
                        "last_name": "Vybíhal",
                        "account_id": self.account_id,
                        "account_name": "ZLINER s.r.o.",
                    }
                ]
            }
        if module == "Accounts" and action == "list":
            return {
                "records": [
                    {
                        "id": self.account_id,
                        "name": "ZLINER s.r.o.",
                    }
                ]
            }
        return {"status": "ok", "id": "d9207f5e-0abc-4b66-96d3-bd4e6ec97200"}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


class NoMatchCrmClient:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_meeting_datetime_parser_handles_typo_and_afternoon() -> None:
    parsed = ModuleAdjustmentEngine._derive_meeting_datetime({}, "na podnělí odpoledne")
    assert isinstance(parsed, str) and parsed

    dt = datetime.strptime(parsed, "%Y-%m-%d %H:%M:%S")
    assert dt.weekday() == 0
    assert dt.hour == 14
    assert dt.minute == 0


def test_name_variant_expansion_handles_instrumental_case() -> None:
    variants = ModuleAdjustmentEngine._expand_name_variants("Karlem Vybíhalem")
    lowered = {item.lower() for item in variants}
    assert "karlem vybíhalem" in lowered
    assert "karel vybíhal" in lowered


def test_quick_action_creates_confirmation_for_meeting_request() -> None:
    client = DummyCrmClient()
    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        result = asyncio.run(
            try_handle_quick_action(
                input_text="Vytvoř schůzku s paní Mimrovou z firmy Panas na podnělí odpoledne",
                crm_client=client,  # type: ignore[arg-type]
                user_id="28",
                action_confirmation=False,
            )
        )

    assert isinstance(result, QuickActionResult)
    assert not result.should_fallback
    assert result.data.get("status") == "confirmation_required"
    assert "Potvrďte prosím provedení" in str(result.data.get("message_to_user") or "")
    assert "kontakt navázán" in str(result.data.get("message_to_user") or "").lower()

    pending = result.data.get("pending_action") or {}
    assert pending.get("module") == "Meetings"
    assert pending.get("action") == "create"

    data = pending.get("data") or {}
    fields = data.get("fields") or {}
    assert fields.get("parent_type") == "Contacts"
    assert fields.get("parent_id") == client.contact_id
    assert fields.get("contact_name") == "Mimrová"

    invitees = data.get("invitees") or {}
    contacts = invitees.get("Contacts") or []
    assert any(str(item.get("id") or "").strip() == client.contact_id for item in contacts)

    date_start = str(fields.get("date_start") or "")
    assert date_start
    dt = datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S")
    assert dt.weekday() == 0
    assert dt.hour == 14


def test_quick_action_resolves_contact_from_non_title_phrase() -> None:
    client = DummyCrmClient()
    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        result = asyncio.run(
            try_handle_quick_action(
                input_text="naplánuj schůzku s Karlem vybíhalem na středu ráno",
                crm_client=client,  # type: ignore[arg-type]
                user_id="28",
                action_confirmation=False,
            )
        )

    assert isinstance(result, QuickActionResult)
    assert not result.should_fallback
    assert result.data.get("status") == "confirmation_required"

    pending = result.data.get("pending_action") or {}
    data = pending.get("data") or {}
    fields = data.get("fields") or {}
    assert "karl" in str(result.data.get("message_to_user") or "").lower()

    # For synthetic dummy dataset, fuzzy resolution picks the available contact.
    assert fields.get("parent_type") == "Contacts"
    assert fields.get("parent_id") == client.contact_id
    assert str(fields.get("contact_name") or "").strip()
    assert "kontakt navázán" in str(result.data.get("message_to_user") or "").lower()


def test_quick_action_resolves_contact_when_phrase_contains_company_clause() -> None:
    client = DummyCrmClient()
    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        result = asyncio.run(
            try_handle_quick_action(
                input_text="naplánuj schůzku s Karlem vybíhalem z firmy zliner na středu ráno",
                crm_client=client,  # type: ignore[arg-type]
                user_id="28",
                action_confirmation=False,
            )
        )

    assert isinstance(result, QuickActionResult)
    assert not result.should_fallback
    assert result.data.get("status") == "confirmation_required"
    assert "karlem vybíhalem" in str(result.data.get("message_to_user") or "").lower()
    assert "kontakt navázán" in str(result.data.get("message_to_user") or "").lower()

    pending = result.data.get("pending_action") or {}
    assert isinstance(pending.get("adjustments"), list)

    data = pending.get("data") or {}
    fields = data.get("fields") or {}
    assert fields.get("parent_type") == "Contacts"
    assert fields.get("parent_id") == client.contact_id


def test_quick_action_uses_fallback_list_lookup_when_generic_search_returns_empty() -> None:
    client = FallbackLookupCrmClient()
    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        result = asyncio.run(
            try_handle_quick_action(
                input_text="naplánuj schůzku s Karlem vybíhalem z firmy zliner na středu ráno",
                crm_client=client,  # type: ignore[arg-type]
                user_id="28",
                action_confirmation=False,
            )
        )

    assert isinstance(result, QuickActionResult)
    assert not result.should_fallback
    assert result.data.get("status") == "confirmation_required"
    assert any(module == "Contacts" and action == "list" for module, action, _ in client.calls)

    pending = result.data.get("pending_action") or {}
    data = pending.get("data") or {}
    fields = data.get("fields") or {}
    assert fields.get("parent_type") == "Contacts"
    assert fields.get("parent_id") == client.contact_id


def test_quick_action_hard_stops_when_contact_cannot_be_resolved() -> None:
    client = NoMatchCrmClient()
    with patch.object(get_settings(), "quick_action_min_confidence", 0.0):
        result = asyncio.run(
            try_handle_quick_action(
                input_text="naplánuj schůzku s Karlem vybíhalem z firmy zliner na středu ráno",
                crm_client=client,  # type: ignore[arg-type]
                user_id="28",
                action_confirmation=False,
            )
        )

    # With fallback-on-no-candidates enabled (default), the quick action signals
    # fallback so the agent can handle unresolved entities more intelligently.
    assert isinstance(result, QuickActionResult)
    assert result.should_fallback
    assert result.data.get("status") == "resolution_required"
    assert "nepodařilo se mi spolehlivě dohledat kontakt" in str(result.data.get("message_to_user") or "").lower()
    pending = result.data.get("pending_action") or {}
    assert pending.get("module") == "Meetings"
    assert pending.get("action") == "create"


def test_adjust_meeting_normalizes_card_style_contact_id_into_invitees() -> None:
    client = DummyCrmClient()
    engine = ModuleAdjustmentEngine(
        tenant_id="ai-local",
        user_id="28",
        crm_client=client,  # type: ignore[arg-type]
        input_text="Naplánuj schůzku s Liborem Adamcem",
        request_context={},
    )

    raw_contact_id = f"Contacts-{client.contact_id}"
    adjusted = asyncio.run(
        engine.apply(
            module="Meetings",
            action="create",
            data={
                "fields": {
                    "contact_id": raw_contact_id,
                    "date_start": "2026-04-06 09:00:00",
                    "description": "Adaptace napojení API Helios",
                }
            },
        )
    )

    fields = (adjusted.data or {}).get("fields") or {}
    assert fields.get("contact_id") == client.contact_id
    assert fields.get("parent_type") == "Contacts"
    assert fields.get("parent_id") == client.contact_id

    invitees = (adjusted.data or {}).get("invitees") or {}
    contacts = invitees.get("Contacts") or []
    assert any(str(item.get("id") or "").strip() == client.contact_id for item in contacts)
