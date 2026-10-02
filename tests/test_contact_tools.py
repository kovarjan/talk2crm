from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.engine.contact_tools import NAME, PrepareContactsArgs, build_contact_tool, contact_form_misroute
from app.engine.form_tools import build_form_tools
from app.presentation.agent_result import normalize_agent_result_for_ui
from app.tools.toolset import ToolSet, build_turn_toolset

ACCOUNT = "d57d57de-6be7-3215-9b2f-61388ffdd307"
PERSON = {"first_name": "Josef", "last_name": "Štádler", "email": "josef.stadler@example.cz",
          "phone": "+420 777 175 189", "department": "Spedice Plzeň", "source_url": "https://example.cz/kontakty"}


class FakeToolSet:
    def __init__(self, *, existing=None, drop=None, fail_lookup=False):
        self.ctx = SimpleNamespace(request_context={"module": "Accounts", "record": ACCOUNT, "chat_id": "chat-1"})
        self.calls = []
        self.existing = existing
        self.drop = drop
        self.fail_lookup = fail_lookup

    async def invoke(self, name, args):
        self.calls.append((name, args))
        if name == "crm_query_tool":
            if args["module"] == "Accounts":
                return json.dumps({"status": "ok", "records": [{"id": ACCOUNT, "name": "Skylog"}]})
            if self.fail_lookup:
                return json.dumps({"status": "crm_unavailable"})
            email = next((f["value"] for f in args["filters"] if f["field"] == "email1"), None)
            rows = [self.existing] if self.existing and email == PERSON["email"] else []
            return json.dumps({"status": "ok", "records": rows})
        assert name == "crm_action_tool"
        assert args["module"] == "Contacts" and args["action"] == "create"
        confirmed = copy.deepcopy(args)
        if self.drop:
            confirmed["data_json"]["fields"].pop(self.drop, None)
        return json.dumps({"status": "confirmation_required", "pending_action": {
            "tool": name, "arguments": confirmed, "confirmation_token": "token-for-person", "expires_at": "2030-01-01",
        }})


@pytest.mark.asyncio
async def test_multiple_people_have_independent_confirmations_and_the_exact_company_relation():
    toolset = FakeToolSet()
    second = {**PERSON, "first_name": "Nikola", "last_name": "Kosíková", "email": "nikola@example.cz"}
    result = json.loads(await build_contact_tool(toolset).ainvoke({"contacts": [PERSON, second]}))
    assert result["status"] == "contact_proposals"
    assert len(result["cards"]) == 2
    for card in result["cards"]:
        command = card["command"]
        assert command["tool"] == "crm_action_tool"
        assert command["module"] == "Contacts"
        assert command["arguments"]["data_json"]["fields"]["account_id"] == ACCOUNT
        assert "description" not in command["arguments"]["data_json"]["fields"]
        assert command["confirmation_token"] and command["chat_id"] == "chat-1"
        assert PERSON["source_url"] in card["text"]
    normalized = normalize_agent_result_for_ui({"output": result["message_to_user"], "intermediate_steps": [{"tool": NAME, "observation": result}]})
    assert normalized["cards"] == result["cards"]


@pytest.mark.asyncio
async def test_duplicate_email_offers_existing_record_without_creating_or_reassigning_it():
    toolset = FakeToolSet(existing={"id": "existing-id", "name": "Josef Štádler", "account_id": "another-company"})
    result = json.loads(await build_contact_tool(toolset).ainvoke({"contacts": [PERSON]}))
    assert not any(name == "crm_action_tool" for name, _ in toolset.calls)
    assert "command" not in result["cards"][0]
    assert result["cards"][0]["actions"][0]["url"] == "/#detail/Contacts/existing-id"
    assert result["skipped"]


@pytest.mark.asyncio
@pytest.mark.parametrize("drop", ["account_id", "email1", "phone_work", "last_name"])
async def test_refuses_a_preview_that_silently_drops_contact_data_or_company_link(drop):
    result = json.loads(await build_contact_tool(FakeToolSet(drop=drop)).ainvoke({"contacts": [PERSON]}))
    assert result["cards"] == []
    assert len(result["skipped"]) == 1


@pytest.mark.asyncio
async def test_repeated_candidates_missing_evidence_and_lookup_failures_are_skipped():
    toolset = FakeToolSet()
    result = json.loads(await build_contact_tool(toolset).ainvoke({"contacts": [PERSON, PERSON,
        {**PERSON, "first_name": "X", "email": None, "phone": "725 891…"}]}))
    assert len(result["cards"]) == 1 and len(result["skipped"]) == 2
    failed = json.loads(await build_contact_tool(FakeToolSet(fail_lookup=True)).ainvoke({"contacts": [PERSON]}))
    assert failed["cards"] == []
    with pytest.raises(ValidationError):
        PrepareContactsArgs.model_validate({"contacts": [{"email": "obchod@example.cz", "source_url": PERSON["source_url"]}]})


@pytest.mark.asyncio
async def test_requires_a_saved_company():
    toolset = FakeToolSet()
    toolset.ctx.request_context["record"] = ""
    result = json.loads(await build_contact_tool(toolset).ainvoke({"contacts": [PERSON]}))
    assert result["status"] == "account_required" and toolset.calls == []


@pytest.mark.asyncio
async def test_the_failed_conversation_cannot_turn_a_contact_list_into_company_description():
    tools = build_form_tools(tenant_id="t", user_id="u", request_context={}, crm_client=None, rag_service=None, input_text="ano")
    form = next(tool for tool in tools if tool.name == "propose_form_fields_tool")
    result = json.loads(await form.ainvoke({"module": "Accounts", "record_id": ACCOUNT,
        "instructions": "Vlož do description veřejné kontakty: Josef josef@example.cz, Nikola nikola@example.cz"}))
    assert result["status"] == "related_contacts_required"
    assert "form_patch" not in result
    research = next(tool for tool in tools if tool.name == "research_record_tool")
    result = json.loads(await research.ainvoke({"module": "Accounts", "record_id": ACCOUNT, "subject": "Skylog", "hint": "Dohledej další kontakty k této firmě"}))
    assert result["status"] == "related_contacts_required"
    assert not contact_form_misroute("Dohledej kontakty", user_input="Vlož seznam kontaktů do description firmy")


@pytest.mark.asyncio
async def test_turn_registers_contact_workflow_and_blocks_the_company_update_fallback(monkeypatch):
    target = FakeToolSet()
    target.crm_specs = {"crm_query_tool": object(), "crm_action_tool": object()}
    target.native_tools, target.interceptors = {}, []
    monkeypatch.setattr(ToolSet, "build", AsyncMock(return_value=target))
    toolset = await build_turn_toolset(tenant_id="t", user_id="u", input_text="ano", request_context={},
                                      crm_client=None, rag_service=None)
    assert NAME in toolset.native_tools
    intercepted = await toolset.interceptors[0](SimpleNamespace(name="crm_action_tool"), {
        "module": "Accounts", "action": "update", "data_json": {"fields": {
            "description": "Veřejné kontakty: Josef josef@example.cz, Nikola nikola@example.cz"}}})
    assert json.loads(intercepted)["status"] == "related_contacts_required"
    assert await toolset.interceptors[0](SimpleNamespace(name="crm_action_tool"), {
        "module": "Contacts", "action": "create", "data_json": {"fields": {"first_name": "Josef"}}}) is None
