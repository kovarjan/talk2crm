from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx

from app.utils.text import normalize_text
from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.contracts import NormalizedCommand, build_pending_action_envelope
from app.engine.adjustments import ModuleAdjustmentEngine
from app.engine.rag import TenantRAGService
from app.presentation.cards import parse_datetime
from app.services.crm_client import CoripoClient
from app.services.schema_write_service import SchemaDrivenWriteService


logger = get_logger(__name__)


class CRMWriteService:
    def __init__(
        self,
        *,
        tenant_id: str,
        user_id: str,
        input_text: str,
        request_context: dict[str, Any] | None,
        crm_client: CoripoClient,
        rag_service: TenantRAGService | None,
        action_confirmation: bool,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.input_text = input_text
        self.request_context = request_context or {}
        self.crm_client = crm_client
        self.rag_service = rag_service
        self.action_confirmation = action_confirmation
        self.adjustment_engine = ModuleAdjustmentEngine(
            tenant_id=tenant_id,
            user_id=user_id,
            crm_client=crm_client,
            input_text=input_text,
            request_context=request_context,
        )

    @classmethod
    def infer_module_from_intent(cls, *, input_text: str, requested_module: str) -> str:
        normalized = normalize_text(input_text)
        module_norm = (requested_module or "").strip().lower()
        module_aliases = {
            "meeting": "Meetings",
            "meetings": "Meetings",
            "schuzka": "Meetings",
            "schuzky": "Meetings",
            "call": "Calls",
            "calls": "Calls",
            "hovor": "Calls",
            "task": "Tasks",
            "tasks": "Tasks",
            "ukol": "Tasks",
            "note": "Notes",
            "notes": "Notes",
            "poznamka": "Notes",
            "poznamky": "Notes",
        }

        call_tokens = ("hovor", "telefonat", "zavolej", "volat", "call")
        task_tokens = ("ukol", "task", "todo", "pripomen", "follow up")
        note_tokens = ("poznamk", "zapis", "zapisek", "note")
        meeting_tokens = ("schuzk", "meeting")

        has_call = any(token in normalized for token in call_tokens)
        has_task = any(token in normalized for token in task_tokens)
        has_note = any(token in normalized for token in note_tokens)
        has_meeting = any(token in normalized for token in meeting_tokens)

        explicit_module = module_aliases.get(module_norm)
        if explicit_module:
            if explicit_module == "Meetings" and has_call and not has_meeting:
                return "Calls"
            if explicit_module == "Calls" and has_meeting and not has_call:
                return "Meetings"
            return explicit_module
        if module_norm:
            return requested_module

        if has_task:
            return "Tasks"
        if has_call:
            return "Calls"
        if has_meeting:
            return "Meetings"
        if has_note:
            return "Notes"

        return "Meetings"

    @staticmethod
    def _candidate_title(record: dict[str, Any]) -> str:
        first = str(record.get("first_name") or "").strip()
        last = str(record.get("last_name") or "").strip()
        full_name = f"{first} {last}".strip()
        return full_name or str(record.get("name") or "(bez názvu)").strip()

    def _resolution_payload(
        self,
        *,
        module: str,
        action: str,
        data: dict[str, Any],
        message: str,
        reason: str,
        ambiguities: list[dict[str, Any]] | None = None,
        adjustments: list[str] | None = None,
    ) -> dict[str, Any]:
        pending_action = build_pending_action_envelope(
            module=module,
            action=action,
            data=data,
            adjustments=list(adjustments or []),
            ambiguities=list(ambiguities or []),
            requires_confirmation=True,
        )
        return {
            "status": "resolution_required",
            "reason": reason,
            "pending_action": pending_action,
            "resolution": {
                "status": "unresolved",
                "confidence": 0.0,
                "candidates": list(ambiguities or []),
            },
            "output": json.dumps(
                {
                    "action": action,
                    "module": module,
                    "data_json": json.dumps(data, ensure_ascii=False),
                    "message_to_user": message,
                },
                ensure_ascii=False,
            ),
            "message_to_user": message,
        }

    async def _resolve_contact_in_company(
        self,
        *,
        contact_name: str,
        account_id: str,
        account_name: str,
    ) -> dict[str, Any] | None:
        records: list[dict[str, Any]] = []
        try:
            payload = await self.crm_client.generic_search(query=contact_name, scope="contacts")
            records = self.adjustment_engine._extract_records(payload)
        except Exception:
            records = []

        if not records:
            try:
                payload = await self.crm_client.execute_module_action(
                    module="Contacts",
                    action="list",
                    data={
                        "query": contact_name,
                        "q": contact_name,
                        "max_results": 80,
                    },
                )
                records = self.adjustment_engine._extract_records(payload)
            except Exception:
                records = []

        if not records:
            return None

        target_account_norm = normalize_text(account_name)
        scoped: list[dict[str, Any]] = []
        for row in records:
            row_account_id = str(
                row.get("account_id") or row.get("accountid") or row.get("accountId") or ""
            ).strip()
            row_account_name = str(
                row.get("account_name") or row.get("accountName") or row.get("company_name") or ""
            ).strip()
            if account_id and row_account_id == account_id:
                scoped.append(row)
                continue
            if target_account_norm:
                row_norm = normalize_text(row_account_name)
                if row_norm and (target_account_norm in row_norm or row_norm in target_account_norm):
                    scoped.append(row)

        if not scoped:
            return None

        best: tuple[int, dict[str, Any]] | None = None
        for row in scoped:
            score = self.adjustment_engine._score_match(contact_name, row)
            if best is None or score > best[0]:
                best = (score, row)
        return best[1] if best is not None else None

    async def _execute_crm_action(self, *, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        normalized_action = (action or "").strip().lower()

        if normalized_action in {"create", "update", "patch"} and get_settings().ai_write_enabled:
            schema_result = await self._try_schema_driven_write(module=module, data=data)
            if schema_result is not None:
                result = schema_result
                if normalized_action in {"create", "update", "patch", "delete"} and self.rag_service is not None:
                    try:
                        ingested = await self.rag_service.ingest_from_crm(
                            tenant_id=self.tenant_id,
                            crm_client=self.crm_client,
                            module=module,
                        )
                        result["_rag_ingested"] = ingested
                    except Exception:
                        result["_rag_ingest_error"] = "failed"
                return result

        result = await self.crm_client.execute_module_action(module=module, action=action, data=data)
        if normalized_action in {"create", "update", "patch", "delete"} and self.rag_service is not None:
            try:
                ingested = await self.rag_service.ingest_from_crm(
                    tenant_id=self.tenant_id,
                    crm_client=self.crm_client,
                    module=module,
                )
                if isinstance(result, dict):
                    result["_rag_ingested"] = ingested
            except Exception:
                if isinstance(result, dict):
                    result["_rag_ingest_error"] = "failed"
        return result if isinstance(result, dict) else {"result": result}

    async def _try_schema_driven_write(self, *, module: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Attempts the write via Coripo's live ai_schema/ai_write.

        Returns None (never raises) on ANY failure in the new path - unavailable
        endpoints (older Coripo deployment), an unrecognized module, a CRM client
        double in tests that doesn't implement get_ai_schema/ai_write, transport
        errors, anything - so the caller always falls back to the legacy
        set/{module} path rather than breaking a request. Worst case this behaves
        exactly like ai_write_enabled=False; it should never behave worse than that.
        """
        try:
            write_service = SchemaDrivenWriteService(crm_client=self.crm_client)
            result = await write_service.write(module=module, data=data, dry_run=False)
        except Exception as exc:  # noqa: BLE001 - intentional broad fallback, see docstring
            logger.info(
                "schema_driven_write_fallback module=%s error=%s: %s",
                module,
                type(exc).__name__,
                exc,
            )
            return None

        if result.success:
            response: dict[str, Any] = {"status": "ok"}
            if result.record_id:
                response["id"] = result.record_id
            return response

        return {
            "status": "validation_error",
            "id": result.record_id,
            "errors": result.errors,
            "message": "CRM rejected one or more fields; see errors for the exact field/code/message.",
        }

    async def _try_precheck_write(
        self, *, module: str, data: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Dry-run validates a pending create/update against the live CRM schema before
        the confirmation prompt is shown, so the agent can fix invalid fields instead of
        only discovering the gap after the user clicks confirm.

        Returns (errors, warnings): errors block the confirmation (unknown/invalid
        fields), warnings (missing required fields) are advisory and never block a
        record the user asked for. Returns None if the schema-driven path isn't
        available for this module - same defensive fallback as
        _try_schema_driven_write, so unsupported modules just skip pre-validation.
        """
        try:
            write_service = SchemaDrivenWriteService(crm_client=self.crm_client)
            result = await write_service.write(module=module, data=data, dry_run=True)
        except Exception as exc:  # noqa: BLE001 - intentional broad fallback, see docstring
            logger.info(
                "schema_precheck_fallback module=%s error=%s: %s",
                module,
                type(exc).__name__,
                exc,
            )
            return None

        return result.errors, result.warnings

    @staticmethod
    def _strip_unknown_fields(data: dict[str, Any], errors: list[dict[str, Any]]) -> list[str]:
        """Drops fields the CRM reported as unknown_field so a typo or a helper key the
        LLM invented never blocks the whole record. Returns the dropped names."""
        fields = data.get("fields") if isinstance(data.get("fields"), dict) else None
        if fields is None:
            return []
        dropped: list[str] = []
        for error in errors:
            name = str(error.get("field") or "")
            if error.get("code") == "unknown_field" and name in fields:
                fields.pop(name, None)
                dropped.append(name)
        return dropped

    async def execute_action(self, *, module: str, action: str, data_json: str = "{}") -> dict[str, Any]:
        try:
            data: dict[str, Any] = json.loads(data_json) if data_json else {}
            if not isinstance(data, dict):
                raise ValueError("data_json must decode to an object")
        except Exception as exc:
            return {"error": f"Invalid data_json: {exc}"}

        normalized_action = (action or "").strip().lower()
        mutating_actions = {"create", "update", "patch", "delete"}
        effective_module = module
        if normalized_action in {"create", "update", "patch"}:
            effective_module = self.infer_module_from_intent(input_text=self.input_text, requested_module=module)

        adjustment = await self.adjustment_engine.apply(
            module=effective_module,
            action=normalized_action,
            data=data,
        )
        data = adjustment.data

        # Ensure any contact resolved into fields is also present in invitees.
        # The adjustment engine only does this for coripo_public mode; we guard
        # it here unconditionally so the pending action always carries the full
        # invitee list regardless of CRM mode.
        if effective_module.lower() in {"meetings", "meeting", "calls", "call"}:
            _fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
            _contact_id = str(_fields.get("contact_id") or "").strip()
            if _contact_id:
                _invitees = data.get("invitees")
                if not isinstance(_invitees, dict):
                    _invitees = {"Users": [], "Contacts": [], "Leads": []}
                    data["invitees"] = _invitees
                _invitees.setdefault("Users", [])
                _invitees.setdefault("Contacts", [])
                _invitees.setdefault("Leads", [])
                if not any(str(inv.get("id") or "") == _contact_id for inv in _invitees["Contacts"]):
                    _invitees["Contacts"].append({"id": _contact_id})

        adjustment_notes = list(adjustment.notes)
        missing_required: list[dict[str, Any]] = []
        if (
            normalized_action in {"create", "update", "patch"}
            and not self.action_confirmation
            and get_settings().ai_write_enabled
        ):
            precheck = await self._try_precheck_write(module=effective_module, data=data)
            if precheck is not None:
                errors, warnings = precheck
                dropped = self._strip_unknown_fields(data, errors)
                if dropped:
                    adjustment_notes.append(f"Dropped fields not on the {effective_module} form: {', '.join(dropped)}")
                    precheck = await self._try_precheck_write(module=effective_module, data=data)
                    if precheck is not None:
                        errors, warnings = precheck
                missing_required = warnings
                if errors:
                    return {
                        "status": "needs_more_info",
                        "message": (
                            "The CRM rejected some fields before confirmation. Fix data_json yourself "
                            "(correct the value, or derive/remove the field) and retry crm_action_tool with "
                            "the same module/action; ask the user only for information that cannot be "
                            "derived. Do not present a confirmation prompt yet."
                        ),
                        "module": effective_module,
                        "action": normalized_action,
                        "field_errors": errors,
                        "missing_required": warnings,
                    }

        if normalized_action in mutating_actions and not self.action_confirmation:
            pending_action = build_pending_action_envelope(
                module=module,
                requested_module=module,
                effective_module=effective_module,
                action=normalized_action,
                data=data,
                adjustments=adjustment_notes,
                ambiguities=[],
                requires_confirmation=True,
            )
            response: dict[str, Any] = {
                "status": "confirmation_required",
                "message": "Akce mění CRM data a vyžaduje potvrzení uživatele.",
                "pending_action": pending_action,
                "adjustments": adjustment_notes,
            }
            if missing_required:
                # Advisory only: the form marks these as required, but the record is
                # created without them if the assistant could not derive a value.
                response["missing_required"] = missing_required
            return response

        data.setdefault("requested_by_user_id", self.user_id)
        try:
            result = await self._execute_crm_action(module=effective_module, action=action, data=data)
            if adjustment_notes and isinstance(result, dict):
                result["_adjustments"] = adjustment_notes
            return result
        except httpx.HTTPStatusError as exc:
            body_preview = ""
            try:
                body_preview = exc.response.text[:1000]
            except Exception:
                body_preview = ""
            return {
                "status": "crm_http_error",
                "http_status": exc.response.status_code,
                "url": str(exc.request.url),
                "message": "CRM rejected the action. Confirm field mapping/required values for this module.",
                "response_body": body_preview,
                "failed_action": {
                    "module": module,
                    "effective_module": effective_module,
                    "action": normalized_action,
                    "data": data,
                },
            }
        except Exception as exc:
            return {
                "status": "crm_action_error",
                "message": str(exc),
                "failed_action": {
                    "module": module,
                    "effective_module": effective_module,
                    "action": normalized_action,
                    "data": data,
                },
            }

    async def execute_quick_command(self, command: NormalizedCommand) -> dict[str, Any] | None:
        if command.intent == "create_contact":
            data = dict(command.slots)
            company_hint = str(data.pop("company_name_hint", "") or "").strip()
            if company_hint:
                account_search = await self.crm_client.generic_search(query=company_hint, scope="accounts")
                account_records = self.adjustment_engine._extract_records(account_search)
                if account_records:
                    account = account_records[0]
                    account_id = str(account.get("id") or "").strip()
                    account_name = str(account.get("name") or company_hint).strip()
                    if account_id:
                        data["account_id"] = account_id
                    if account_name:
                        data["account_name"] = account_name

            full_name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
            confirmation_text = f"Připraveno: vytvořit kontakt {full_name}."
            if data.get("phone_mobile"):
                confirmation_text += f" Tel: {data['phone_mobile']}."
            if data.get("email1"):
                confirmation_text += f" Email: {data['email1']}."
            if data.get("account_name"):
                confirmation_text += f" Společnost: {data['account_name']}."
            confirmation_text += " Potvrďte prosím provedení."

            pending_action = build_pending_action_envelope(
                module="Contacts",
                action="create",
                data=data,
                requires_confirmation=True,
            )

            if not self.action_confirmation:
                return {
                    "status": "confirmation_required",
                    "pending_action": pending_action,
                    "output": json.dumps(
                        {
                            "action": "create",
                            "module": "Contacts",
                            "data_json": json.dumps(data, ensure_ascii=False),
                            "message_to_user": confirmation_text,
                        },
                        ensure_ascii=False,
                    ),
                    "message_to_user": confirmation_text,
                }

            result = await self._execute_crm_action(module="Contacts", action="create", data=data)
            created_id = str(result.get("id") or result.get("record_id") or "").strip()
            message = (
                f"Hotovo. Kontakt {full_name} byl vytvořen (ID: {created_id})."
                if created_id
                else f"Hotovo. Kontakt {full_name} byl vytvořen."
            )
            return {
                "status": "ok",
                "output": json.dumps(result, ensure_ascii=False),
                "message_to_user": message,
            }

        if command.intent == "create_meeting":
            write_v2_enabled = get_settings().write_service_v2_active
            create_meeting = dict(command.slots)
            raw_fields = create_meeting.get("fields")
            raw_fields_obj = raw_fields if isinstance(raw_fields, dict) else {}
            raw_contact_name = str(raw_fields_obj.get("contact_name") or "").strip()
            raw_account_name = str(raw_fields_obj.get("account_name") or "").strip()

            pre_resolution_ambiguities: list[dict[str, Any]] = []
            if write_v2_enabled and raw_account_name:
                account_resolution = await self.adjustment_engine.entity_resolver.resolve_record_by_name(
                    scope="accounts",
                    name=raw_account_name,
                    for_mutation=True,
                )
                if account_resolution.status == "resolved" and account_resolution.selected:
                    account_row = account_resolution.selected
                    fields = create_meeting.setdefault("fields", {})
                    if isinstance(fields, dict):
                        account_id = str(account_row.get("id") or "").strip()
                        account_name = str(account_row.get("name") or raw_account_name).strip()
                        if account_id:
                            fields.setdefault("account_id", account_id)
                        if account_name:
                            fields.setdefault("account_name", account_name)
                elif account_resolution.candidates:
                    pre_resolution_ambiguities.append(
                        {
                            "type": "account",
                            "query": raw_account_name,
                            "candidates": [
                                {
                                    "id": str(row.get("id") or "").strip(),
                                    "name": self._candidate_title(row),
                                }
                                for row in account_resolution.candidates[:5]
                            ],
                            "reason": account_resolution.reason or account_resolution.status,
                            "confidence": float(account_resolution.confidence or 0.0),
                        }
                    )

            if write_v2_enabled and raw_contact_name:
                fields = create_meeting.setdefault("fields", {})
                if not isinstance(fields, dict):
                    fields = {}
                    create_meeting["fields"] = fields
                scoped_account_id = str(fields.get("account_id") or "").strip()
                scoped_account_name = str(fields.get("account_name") or raw_account_name).strip()
                scoped_contact = None
                if scoped_account_id:
                    scoped_contact = await self._resolve_contact_in_company(
                        contact_name=raw_contact_name,
                        account_id=scoped_account_id,
                        account_name=scoped_account_name,
                    )
                if scoped_contact:
                    contact_id = str(scoped_contact.get("id") or "").strip()
                    if contact_id:
                        fields.setdefault("contact_id", contact_id)
                        fields.setdefault("contact_name", self._candidate_title(scoped_contact))
                else:
                    contact_resolution = await self.adjustment_engine.entity_resolver.resolve_record_by_name(
                        scope="contacts",
                        name=raw_contact_name,
                        for_mutation=True,
                    )
                    if contact_resolution.status == "resolved" and contact_resolution.selected:
                        contact_row = contact_resolution.selected
                        contact_id = str(contact_row.get("id") or "").strip()
                        if contact_id:
                            fields.setdefault("contact_id", contact_id)
                            fields.setdefault("contact_name", self._candidate_title(contact_row))
                    elif contact_resolution.candidates:
                        pre_resolution_ambiguities.append(
                            {
                                "type": "contact",
                                "query": raw_contact_name,
                                "candidates": [
                                    {
                                        "id": str(row.get("id") or "").strip(),
                                        "name": self._candidate_title(row),
                                        "account_name": str(row.get("account_name") or "").strip(),
                                    }
                                    for row in contact_resolution.candidates[:5]
                                ],
                                "reason": contact_resolution.reason or contact_resolution.status,
                                "confidence": float(contact_resolution.confidence or 0.0),
                            }
                        )

            adjusted = await self.adjustment_engine.apply(
                module="Meetings",
                action="create",
                data=create_meeting,
            )
            prepared_data = adjusted.data if isinstance(adjusted.data, dict) else create_meeting
            adjustment_notes = adjusted.notes if isinstance(adjusted.notes, list) else []
            fields_raw = prepared_data.get("fields")
            fields = fields_raw if isinstance(fields_raw, dict) else {}

            date_start = str(fields.get("date_start") or "").strip()
            if not date_start:
                message = (
                    "Pro vytvoření schůzky potřebuji i termín. "
                    "Napište prosím den a čas (např. v pondělí 14:00)."
                )
                return {
                    "status": "needs_input",
                    "output": json.dumps({"status": "needs_input", "message_to_user": message}, ensure_ascii=False),
                    "message_to_user": message,
                }

            contact_name = str(raw_contact_name or fields.get("contact_name") or "").strip()
            account_name = str(fields.get("account_name") or raw_account_name).strip()
            parent_type = str(fields.get("parent_type") or "").strip()
            parent_id = str(fields.get("parent_id") or "").strip()
            # Meetings link the person through invitees (Coripo's parent only takes companies).
            invitee_contacts = (prepared_data.get("invitees") or {}).get("Contacts") if isinstance(prepared_data.get("invitees"), dict) else None
            has_contact_link = bool(str(fields.get("contact_id") or "").strip()) or bool(invitee_contacts)
            has_account_link = parent_type == "Accounts" and bool(parent_id)

            if raw_contact_name and not has_contact_link:
                if not write_v2_enabled:
                    return None
                message = (
                    f"Nepodařilo se mi spolehlivě dohledat kontakt '{raw_contact_name}' pro vytvoření schůzky. "
                    "Doplňte prosím přesnější jméno nebo potvrďte konkrétního kandidáta."
                )
                return self._resolution_payload(
                    module="Meetings",
                    action="create",
                    data=prepared_data,
                    message=message,
                    reason="contact_unresolved",
                    ambiguities=pre_resolution_ambiguities,
                    adjustments=adjustment_notes,
                )
            if raw_account_name and not (has_contact_link or has_account_link):
                if not write_v2_enabled:
                    return None
                message = (
                    f"Nepodařilo se mi spolehlivě dohledat firmu '{raw_account_name}' pro vytvoření schůzky. "
                    "Upřesněte prosím název firmy nebo vyberte kandidáta."
                )
                return self._resolution_payload(
                    module="Meetings",
                    action="create",
                    data=prepared_data,
                    message=message,
                    reason="account_unresolved",
                    ambiguities=pre_resolution_ambiguities,
                    adjustments=adjustment_notes,
                )

            start_dt = parse_datetime(date_start)
            when_text = start_dt.strftime("%d.%m.%Y %H:%M") if isinstance(start_dt, datetime) else date_start

            confirmation_text = "Připraveno: vytvořit schůzku"
            if contact_name:
                confirmation_text += f" s {contact_name}"
            if account_name:
                confirmation_text += f" ({account_name})"
            if has_contact_link:
                confirmation_text += " [kontakt navázán]"
            elif raw_contact_name:
                confirmation_text += " [kontakt se nepodařilo automaticky dohledat]"
            if has_account_link:
                confirmation_text += " [firma navázána]"
            confirmation_text += f" na {when_text}. Potvrďte prosím provedení."

            pending_action = build_pending_action_envelope(
                module="Meetings",
                action="create",
                data=prepared_data,
                adjustments=adjustment_notes,
                requires_confirmation=True,
            )

            if not self.action_confirmation:
                return {
                    "status": "confirmation_required",
                    "pending_action": pending_action,
                    "output": json.dumps(
                        {
                            "action": "create",
                            "module": "Meetings",
                            "data_json": json.dumps(prepared_data, ensure_ascii=False),
                            "adjustments": adjustment_notes,
                            "message_to_user": confirmation_text,
                        },
                        ensure_ascii=False,
                    ),
                    "message_to_user": confirmation_text,
                }

            result = await self._execute_crm_action(module="Meetings", action="create", data=prepared_data)
            created_id = str(result.get("id") or result.get("record_id") or "").strip()
            message = (
                f"Hotovo. Schůzka byla vytvořena (ID: {created_id})." if created_id else "Hotovo. Schůzka byla vytvořena."
            )
            return {
                "status": "ok",
                "output": json.dumps(result, ensure_ascii=False),
                "message_to_user": message,
            }

        return None
