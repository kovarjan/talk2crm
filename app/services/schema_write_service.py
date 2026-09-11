# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Builds CRM writes from Coripo's live ai_schema instead of a blind field dict.

Coripo's GET /public/ai_schema/{module} always reflects the module's current
Studio layout + vardefs (third/custom/intern layers) merged with a record's
current values; POST /public/ai_write/{module} validates and applies a write
against that same live schema. This service is the bridge between talk2api2's
existing flat `{id?, fields: {...}, invitees?: {...}}` write shape (produced by
ModuleAdjustmentEngine) and ai_write's standardized per-type value shapes
(`{id}` for relate, `{module, id}` for polymorphic_relate, `{amount}` for
currency, etc.) - so the field-type knowledge that used to live in scattered
per-module Python lives in the CRM's own schema instead.

No caching: every call fetches the schema fresh, per Coripo's design ("no
talk2crm-owned schema store"). A dry_run is always issued before a real write,
and the real write re-validates independently - dry_run and commit are two
separate calls against live data.
"""

from __future__ import annotations

from typing import Any

from app.services.crm_client import CoripoClient


class SchemaWriteError(Exception):
    """Raised when the module has no usable ai_schema (e.g. unsupported module)."""


class SchemaDrivenWriteResult:
    __slots__ = ("success", "record_id", "dry_run", "errors", "warnings", "updated_fields", "raw")

    def __init__(
        self,
        *,
        success: bool,
        record_id: str | None,
        dry_run: bool,
        errors: list[dict[str, Any]],
        updated_fields: list[str],
        raw: dict[str, Any],
        warnings: list[dict[str, Any]] | None = None,
    ):
        self.success = success
        self.record_id = record_id
        self.dry_run = dry_run
        self.errors = errors
        self.warnings = warnings or []
        self.updated_fields = updated_fields
        self.raw = raw

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "record_id": self.record_id,
            "dry_run": self.dry_run,
            "errors": self.errors,
            "warnings": self.warnings,
            "updated_fields": self.updated_fields,
        }


class SchemaDrivenWriteService:
    """Translates a flat write `data` dict into an ai_write call for `module`."""

    # Keys the adjustment engine / LLM use that are not fields on any Coripo form.
    HELPER_FIELDS = frozenset({
        "contact_id",
        "contact_name",
        "related_contact_id",
        "related_contact_name",
        "invite_contact_id",
        "invite_contact_name",
        "participant_name",
        "account_name",
        "related_account_name",
        "company",
        "company_name",
        "duration_hours",
        "duration_minutes",
        "duration",
    })

    def __init__(self, *, crm_client: CoripoClient):
        self.crm_client = crm_client

    async def write(
        self,
        *,
        module: str,
        data: dict[str, Any],
        dry_run: bool,
    ) -> SchemaDrivenWriteResult:
        record_id = str(data.get("id") or "").strip() or None
        schema = await self.crm_client.get_ai_schema(module, record_id)

        sections = schema.get("sections")
        if not isinstance(sections, list):
            raise SchemaWriteError(f"ai_schema returned no sections for module '{module}'")

        field_index = self._index_fields(sections)
        payload = self._build_payload(field_index, data)

        raw = await self.crm_client.ai_write(module, payload, record_id=record_id, dry_run=dry_run)
        return self._normalize_result(raw, dry_run=dry_run, fallback_record_id=record_id)

    @staticmethod
    def _index_fields(sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for section in sections:
            for field in section.get("fields") or []:
                name = field.get("name")
                if isinstance(name, str) and name:
                    index[name] = field
        return index

    def _build_payload(self, field_index: dict[str, dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
        fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
        payload: dict[str, Any] = {}

        # type_field names (e.g. "parent_type") that a polymorphic_relate field folds
        # into its combined entry - never sent standalone even if present in `fields`.
        folded_type_fields = {
            field["type_field"]
            for field in field_index.values()
            if field.get("type") == "polymorphic_relate" and field.get("type_field")
        }

        for name, raw_value in fields.items():
            if name in folded_type_fields:
                continue
            field_schema = field_index.get(name)
            if field_schema is None:
                if name in self.HELPER_FIELDS:
                    # Agent-side helpers (contact_id/contact_name live in invitees on
                    # Coripo, duration_* are derived into date_end) - never a real
                    # field on the form, so they must not surface as unknown_field.
                    continue
                # Not part of this module's schema (or already excluded, e.g. a
                # non-writable display companion) - let ai_write report unknown_field
                # rather than silently dropping it, so the caller can see the mismatch.
                payload[name] = raw_value
                continue

            field_type = field_schema.get("type")
            if field_type == "polymorphic_relate":
                type_field = field_schema.get("type_field")
                module_value = fields.get(type_field) if type_field else None
                if module_value and raw_value:
                    payload[name] = {"module": module_value, "id": raw_value}
                continue

            payload[name] = self._coerce_value(field_type, raw_value)

        invitees = data.get("invitees")
        if isinstance(invitees, dict) and "invitees" in field_index:
            payload["invitees"] = self._normalize_invitees(invitees)

        return payload

    @staticmethod
    def _coerce_value(field_type: str | None, raw_value: Any) -> Any:
        if raw_value is None:
            return raw_value

        if field_type == "relate":
            if isinstance(raw_value, dict):
                return raw_value
            return {"id": str(raw_value)}

        if field_type == "currency":
            if isinstance(raw_value, dict):
                return raw_value
            return {"amount": raw_value}

        if field_type == "multi_enum" and not isinstance(raw_value, list):
            return [raw_value]

        return raw_value

    @staticmethod
    def _normalize_invitees(invitees: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        # ai_write does the real existence check per id server-side (it's the source of
        # truth); this only shapes the payload. No id-format filtering here - some Coripo
        # installs use legacy non-UUID Users ids (e.g. plain integers) alongside UUIDs for
        # Contacts/Leads, so any shape-based prefilter would silently drop valid invitees.
        normalized: dict[str, list[dict[str, str]]] = {}
        for module in ("Users", "Contacts", "Leads"):
            rows = invitees.get(module)
            out: list[dict[str, str]] = []
            if isinstance(rows, list):
                for row in rows:
                    row_id = str(row.get("id") if isinstance(row, dict) else row or "").strip()
                    if row_id:
                        out.append({"id": row_id})
            normalized[module] = out
        return normalized

    @staticmethod
    def _normalize_result(raw: dict[str, Any], *, dry_run: bool, fallback_record_id: str | None) -> SchemaDrivenWriteResult:
        errors = raw.get("errors")
        warnings = raw.get("warnings")
        errors = errors if isinstance(errors, list) else []
        warnings = warnings if isinstance(warnings, list) else []
        # Older Coripo builds report missing required fields as errors; they are
        # advisory here (the assistant fills what it can derive, the form flags the
        # rest), so fold them into warnings regardless of where the CRM put them.
        soft = [error for error in errors if error.get("code") == "required_missing"]
        hard = [error for error in errors if error.get("code") != "required_missing"]
        return SchemaDrivenWriteResult(
            success=bool(raw.get("success")) or (not hard and bool(errors)),
            record_id=raw.get("record_id") or fallback_record_id,
            dry_run=bool(raw.get("dry_run", dry_run)),
            errors=hard,
            warnings=warnings + soft,
            updated_fields=raw.get("updated_fields") or [],
            raw=raw,
        )
