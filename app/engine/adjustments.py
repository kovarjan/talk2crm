from __future__ import annotations

import copy
import re
import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.core.logging import get_logger
from app.domain.entity_resolver import EntityResolver
from app.domain.temporal_resolver import TemporalResolver
from app.services.crm_client import SugarClient


logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_CRM_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_PLACEHOLDER_ID_RE = re.compile(r"^[A-Z_]+_ID$")
_TIME_RE = re.compile(r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b")


@dataclass
class AdjustmentResult:
    data: dict[str, Any]
    notes: list[str] = field(default_factory=list)


class ModuleAdjustmentEngine:
    def __init__(
        self,
        *,
        tenant_id: str,
        user_id: str,
        crm_client: SugarClient,
        input_text: str,
        request_context: dict[str, Any] | None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.crm_client = crm_client
        self.input_text = input_text or ""
        self.request_context = request_context or {}
        self.entity_resolver = EntityResolver.from_settings(
            tenant_id=tenant_id,
            crm_client=crm_client,
        )
        self.temporal_resolver = TemporalResolver.from_settings()

    async def apply(
        self,
        *,
        module: str,
        action: str,
        data: dict[str, Any],
    ) -> AdjustmentResult:
        module_norm = (module or "").strip().lower()
        if module_norm in {"meeting", "meetings"}:
            return await self._adjust_meetings(action=action, data=data)
        if module_norm in {"call", "calls", "task", "tasks", "note", "notes"}:
            return await self._adjust_activity_parent(module=module_norm, action=action, data=data)
        return AdjustmentResult(data=copy.deepcopy(data), notes=[])

    async def _adjust_activity_parent(
        self,
        *,
        module: str,
        action: str,
        data: dict[str, Any],
    ) -> AdjustmentResult:
        normalized_action = (action or "").strip().lower()
        if normalized_action not in {"create", "update", "patch"}:
            return AdjustmentResult(data=copy.deepcopy(data), notes=[])

        adjusted = copy.deepcopy(data)
        notes: list[str] = []
        if self.crm_client.mode != "coripo_public":
            return AdjustmentResult(data=adjusted, notes=notes)

        fields = self._collect_coripo_fields(adjusted)
        contact_name = self._first_nonempty(
            fields,
            ["contact_name", "related_contact_name", "invite_contact_name", "participant_name"],
        )
        account_name = self._first_nonempty(
            fields,
            ["account_name", "related_account_name", "company", "company_name"],
        )
        self._drop_helper_fields(fields)

        contact_id = self._first_nonempty(fields, ["contact_id", "related_contact_id", "invite_contact_id"])
        account_id = self._first_nonempty(fields, ["account_id", "related_account_id", "company_id"])
        if contact_id and not self._is_valid_crm_id(contact_id):
            contact_id = None
        if account_id and not self._is_valid_crm_id(account_id):
            account_id = None

        if not account_id and account_name:
            account_candidate = await self._resolve_record_by_name(scope="accounts", name=account_name)
            if account_candidate:
                resolved = str(account_candidate.get("id") or "").strip()
                if self._is_valid_crm_id(resolved):
                    account_id = resolved
        if not contact_id and contact_name:
            contact_candidate = await self._resolve_contact_by_name_in_account(
                contact_name=contact_name,
                account_id=account_id,
                account_name=account_name,
            )
            if contact_candidate:
                resolved = str(contact_candidate.get("id") or "").strip()
                if self._is_valid_crm_id(resolved):
                    contact_id = resolved

        # Always prefer person relation for "Týká se" when available.
        if contact_id:
            fields["parent_type"] = "Contacts"
            fields["parent_id"] = contact_id
        elif account_id:
            fields["parent_type"] = "Accounts"
            fields["parent_id"] = account_id

        topic = self._derive_topic(fields, self.input_text)
        dt_value = self._derive_meeting_datetime(fields, self.input_text)
        if module in {"call", "calls"}:
            if topic and not str(fields.get("name") or "").strip():
                fields["name"] = f"Hovor - {topic}"
            if dt_value and not str(fields.get("date_start") or "").strip():
                fields["date_start"] = dt_value
            if "duration_minutes" not in fields and "duration_hours" not in fields:
                fields["duration_hours"] = 0
                fields["duration_minutes"] = 30
        elif module in {"task", "tasks"}:
            if topic and not str(fields.get("name") or "").strip():
                fields["name"] = f"Úkol - {topic}"
            if dt_value and not str(fields.get("date_due") or "").strip():
                fields["date_due"] = dt_value
        elif module in {"note", "notes"}:
            if topic and not str(fields.get("name") or "").strip():
                fields["name"] = f"Poznámka - {topic}"
            if topic and not str(fields.get("description") or "").strip():
                fields["description"] = topic

        adjusted["fields"] = fields
        return AdjustmentResult(data=adjusted, notes=notes)

    async def _adjust_meetings(self, *, action: str, data: dict[str, Any]) -> AdjustmentResult:
        normalized_action = (action or "").strip().lower()
        if normalized_action not in {"create", "update", "patch"}:
            return AdjustmentResult(data=copy.deepcopy(data), notes=[])

        adjusted = copy.deepcopy(data)
        notes: list[str] = []

        if self.crm_client.mode != "coripo_public":
            return AdjustmentResult(data=adjusted, notes=notes)

        fields = self._collect_coripo_fields(adjusted)
        contact_name = self._first_nonempty(
            fields,
            [
                "contact_name",
                "related_contact_name",
                "invite_contact_name",
                "participant_name",
            ],
        )
        account_name = self._first_nonempty(
            fields,
            [
                "account_name",
                "related_account_name",
                "company",
                "company_name",
            ],
        )
        self._drop_helper_fields(fields)

        contact_id = self._first_nonempty(
            fields,
            [
                "contact_id",
                "related_contact_id",
                "invite_contact_id",
            ],
        )
        account_id = self._first_nonempty(
            fields,
            [
                "account_id",
                "related_account_id",
                "company_id",
            ],
        )

        if contact_id and not self._is_valid_crm_id(contact_id):
            fields.pop("contact_id", None)
            fields.pop("related_contact_id", None)
            fields.pop("invite_contact_id", None)
            contact_id = None
        if account_id and not self._is_valid_crm_id(account_id):
            fields.pop("account_id", None)
            fields.pop("related_account_id", None)
            fields.pop("company_id", None)
            account_id = None

        ctx_entities = self.request_context.get("entities", {})
        if isinstance(ctx_entities, dict):
            contact_name = contact_name or self._first_nonempty(
                ctx_entities,
                ["contact", "contact_name", "person", "person_name"],
            )
            account_name = account_name or self._first_nonempty(
                ctx_entities,
                ["account", "account_name", "company", "company_name"],
            )

        contact_candidate: dict[str, Any] | None = None
        account_candidate: dict[str, Any] | None = None

        if not account_id and account_name:
            account_candidate = await self._resolve_record_by_name(
                scope="accounts",
                name=account_name,
            )
            if account_candidate:
                account_id = str(account_candidate.get("id") or "").strip()
                if account_id:
                    notes.append(f"Resolved account '{account_name}' -> {account_id}")
            else:
                notes.append(f"Account '{account_name}' was not resolved in CRM search")

        if not contact_id and contact_name:
            contact_candidate = await self._resolve_contact_by_name_in_account(
                contact_name=contact_name,
                account_id=account_id,
                account_name=account_name,
            )
            if contact_candidate:
                contact_id = str(contact_candidate.get("id") or "").strip()
                if contact_id:
                    notes.append(f"Resolved contact '{contact_name}' -> {contact_id}")
            else:
                notes.append(f"Contact '{contact_name}' was not resolved in CRM search")

        if contact_candidate and not account_id:
            linked_account_id = self._first_nonempty(
                contact_candidate,
                ["account_id", "accountid", "accountId"],
            )
            linked_account_name = self._first_nonempty(
                contact_candidate,
                ["account_name", "accountName", "company_name"],
            )
            if linked_account_id:
                account_id = linked_account_id
                notes.append("Derived account from resolved contact")
            elif linked_account_name:
                account_candidate = await self._resolve_record_by_name(
                    scope="accounts",
                    name=linked_account_name,
                )
                if account_candidate:
                    account_id = str(account_candidate.get("id") or "").strip()
                    if account_id:
                        notes.append(
                            f"Resolved account '{linked_account_name}' from contact relation"
                        )

        if contact_id and not self._is_valid_crm_id(contact_id):
            notes.append(f"Ignoring non-CRM contact id '{contact_id}'")
            contact_id = None
        if account_id and not self._is_valid_crm_id(account_id):
            notes.append(f"Ignoring non-CRM account id '{account_id}'")
            account_id = None

        # Always prefer person relation for "Týká se" when available.
        if contact_id:
            fields["parent_type"] = "Contacts"
            fields["parent_id"] = contact_id
        elif account_id:
            fields["parent_type"] = "Accounts"
            fields["parent_id"] = account_id

        topic = self._derive_topic(fields, self.input_text)
        contact_display_name = self._record_label(contact_candidate or {}) or contact_name
        surname = self._extract_last_name(contact_display_name)
        surname = self._normalize_czech_last_name(surname)
        if surname:
            fields["contact_name"] = surname
        if topic:
            if surname:
                fields["name"] = f"Schůzka {surname} - {topic}"
            else:
                fields["name"] = f"Schůzka - {topic}"
            fields["description"] = topic
            notes.append("Normalized meeting name/description policy")

        if (
            "duration_minutes" not in fields
            and "duration_hours" not in fields
            and "duration" not in fields
        ):
            fields["duration_minutes"] = 0
            fields["duration_hours"] = 1
            notes.append("Applied default meeting duration 60 minutes")

        dt_value = self._derive_meeting_datetime(fields, self.input_text)
        if dt_value:
            fields["date_start"] = dt_value
            notes.append(f"Normalized meeting datetime -> {dt_value}")

        invitees = self._normalize_invitees(adjusted.get("invitees"))
        if self.user_id and _UUID_RE.match(str(self.user_id).strip()):
            self._ensure_invitee(invitees, "Users", self.user_id)
            notes.append("Ensured creator is included in invitees")
        elif self.user_id:
            notes.append("Creator invitee ID is non-UUID; relying on backend auto-invite")
        if contact_id:
            self._ensure_invitee(invitees, "Contacts", contact_id)
            notes.append("Ensured related contact is included in invitees")

        adjusted["fields"] = fields
        adjusted["invitees"] = invitees
        adjusted.setdefault("inviteesBackup", copy.deepcopy(invitees))
        return AdjustmentResult(data=adjusted, notes=notes)

    def _collect_coripo_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        base_fields: dict[str, Any] = {}
        if isinstance(data.get("fields"), dict):
            base_fields.update(data["fields"])

        reserved = {
            "id",
            "fields",
            "relationships",
            "customData",
            "files",
            "invitees",
            "inviteesBackup",
            "requested_by_user_id",
            "confirm_action",
        }
        for key, value in data.items():
            if key in reserved or value is None:
                continue
            if key not in base_fields:
                base_fields[key] = value
        return base_fields

    @staticmethod
    def _drop_helper_fields(fields: dict[str, Any]) -> None:
        helper_keys = {
            "contact_name",
            "related_contact_name",
            "invite_contact_name",
            "participant_name",
            "account_name",
            "related_account_name",
            "company_name",
            "company",
        }
        for key in helper_keys:
            fields.pop(key, None)

    @staticmethod
    def _first_nonempty(data: dict[str, Any], keys: list[str]) -> str | None:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = value.strip().lower()
        unaccented = "".join(
            c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()

    @staticmethod
    def _extract_last_name(full_name: str | None) -> str | None:
        text = str(full_name or "").strip()
        if not text:
            return None
        parts = [p for p in re.split(r"\s+", text) if p]
        if not parts:
            return None
        return parts[-1]

    @staticmethod
    def _normalize_czech_last_name(last_name: str | None) -> str | None:
        value = str(last_name or "").strip()
        if not value:
            return None
        lowered = ModuleAdjustmentEngine._normalize_text(value)
        if lowered.endswith("ovou"):
            return value[:-4] + "ová"
        if lowered.endswith("ou"):
            return value[:-2] + "á"
        return value

    @staticmethod
    def _is_valid_crm_id(value: str | None) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if _PLACEHOLDER_ID_RE.match(text):
            return False
        return bool(_CRM_ID_RE.match(text))

    @classmethod
    def _derive_topic(cls, fields: dict[str, Any], input_text: str) -> str:
        for key in ("description", "note", "subject"):
            val = str(fields.get(key) or "").strip()
            if val:
                return val

        current_name = str(fields.get("name") or "").strip()
        if current_name and not current_name.lower().startswith("sch"):
            return current_name
        if " - " in current_name:
            return current_name.split(" - ", 1)[1].strip()
        return ""

    @staticmethod
    def _weekday_from_text(text: str) -> int | None:
        return TemporalResolver.from_settings()._weekday_from_text(text)

    @classmethod
    def _derive_meeting_datetime(cls, fields: dict[str, Any], input_text: str) -> str | None:
        resolution = TemporalResolver.from_settings().resolve_meeting_datetime(
            fields=fields,
            input_text=input_text,
        )
        return resolution.datetime_value if resolution.status == "resolved" else None

    @classmethod
    def _expand_name_variants(cls, name: str) -> list[str]:
        return EntityResolver.expand_name_variants(name)

    @classmethod
    def _extract_records(cls, payload: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                direct_records = value.get("records")
                if isinstance(direct_records, list):
                    for item in direct_records:
                        if isinstance(item, dict):
                            records.append(item)

                result_obj = value.get("result")
                if isinstance(result_obj, dict):
                    collect(result_obj)

                data_obj = value.get("data")
                if isinstance(data_obj, dict):
                    collect(data_obj)

                message_obj = value.get("message")
                if isinstance(message_obj, dict):
                    collect(message_obj)

                for key in ("contacts", "accounts", "meetings", "crm"):
                    nested = value.get(key)
                    if nested is not None:
                        collect(nested)
                return

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        records.append(item)

        collect(payload)
        unique: dict[str, dict[str, Any]] = {}
        for row in records:
            row_id = str(row.get("id") or row.get("record_id") or "").strip()
            if not cls._is_valid_crm_id(row_id):
                continue
            if row_id not in unique:
                unique[row_id] = row
        return list(unique.values())

    @staticmethod
    def _record_label(record: dict[str, Any]) -> str:
        if record.get("name"):
            return str(record["name"])
        first = str(record.get("first_name") or "").strip()
        last = str(record.get("last_name") or "").strip()
        if first or last:
            return f"{first} {last}".strip()
        return ""

    @classmethod
    def _score_match(cls, search_name: str, record: dict[str, Any]) -> int:
        candidate = cls._record_label(record)
        if not candidate:
            return 0
        n_search = cls._normalize_text(search_name)
        n_candidate = cls._normalize_text(candidate)
        if not n_search or not n_candidate:
            return 0
        if n_search == n_candidate:
            return 100
        if n_search in n_candidate or n_candidate in n_search:
            return 80
        s_tokens = set(n_search.split())
        c_tokens = set(n_candidate.split())
        if not s_tokens:
            return 0
        overlap = len(s_tokens & c_tokens) / len(s_tokens)
        score = int(overlap * 60)

        # Handle Czech inflection variants (e.g. "Karlem Vybíhalem" vs "Karel Vybíhal").
        seq_ratio = SequenceMatcher(None, n_search, n_candidate).ratio()
        if seq_ratio >= 0.72:
            score = max(score, int(seq_ratio * 85))
        return score

    @staticmethod
    def _lookup_filter_for_scope(scope: str, query: str) -> dict[str, Any]:
        value = str(query or "").strip()
        if not value:
            return {"operator": "and", "operands": []}

        if scope == "contacts":
            return {
                "operator": "and",
                "operands": [
                    {
                        "operator": "or",
                        "operands": [
                            {"field": "name", "type": "cont", "value": value},
                            {"field": "first_name", "type": "cont", "value": value},
                            {"field": "last_name", "type": "cont", "value": value},
                        ],
                    }
                ],
            }
        return {
            "operator": "and",
            "operands": [
                {
                    "operator": "or",
                    "operands": [
                        {"field": "name", "type": "cont", "value": value},
                    ],
                }
            ],
        }

    async def _resolve_record_by_name(self, *, scope: str, name: str) -> dict[str, Any] | None:
        resolution = await self.entity_resolver.resolve_record_by_name(
            scope=scope,
            name=name,
            for_mutation=True,
        )
        if resolution.status == "resolved":
            return resolution.selected
        return None

    async def _resolve_contact_by_name_in_account(
        self,
        *,
        contact_name: str,
        account_id: str | None,
        account_name: str | None,
    ) -> dict[str, Any] | None:
        if not contact_name:
            return None

        resolved_contact = await self._resolve_record_by_name(scope="contacts", name=contact_name)
        if not resolved_contact:
            return None

        target_account_id = str(account_id or "").strip()
        target_account_norm = self._normalize_text(str(account_name or ""))
        if not target_account_id and not target_account_norm:
            return resolved_contact

        resolved_account_id = self._first_nonempty(
            resolved_contact,
            ["account_id", "accountid", "accountId"],
        ) or ""
        resolved_account_norm = self._normalize_text(
            self._first_nonempty(
                resolved_contact,
                ["account_name", "accountName", "company_name"],
            )
            or ""
        )

        if target_account_id and resolved_account_id and target_account_id == resolved_account_id:
            return resolved_contact
        if target_account_norm and resolved_account_norm and (
            target_account_norm in resolved_account_norm or resolved_account_norm in target_account_norm
        ):
            return resolved_contact
        return None

    @staticmethod
    def _normalize_invitees(invitees: Any) -> dict[str, list[dict[str, str]]]:
        normalized: dict[str, list[dict[str, str]]] = {
            "Users": [],
            "Contacts": [],
            "Leads": [],
        }
        if not isinstance(invitees, dict):
            return normalized

        for module in ("Users", "Contacts", "Leads"):
            raw = invitees.get(module)
            if not isinstance(raw, list):
                continue
            out: list[dict[str, str]] = []
            for row in raw:
                if isinstance(row, dict):
                    row_id = str(row.get("id") or "").strip()
                else:
                    row_id = str(row).strip()
                if not row_id:
                    continue
                if module in {"Contacts", "Leads"} and not ModuleAdjustmentEngine._is_valid_crm_id(
                    row_id
                ):
                    continue
                out.append({"id": row_id})
            normalized[module] = out
        return normalized

    @staticmethod
    def _ensure_invitee(invitees: dict[str, list[dict[str, str]]], module: str, invitee_id: str) -> None:
        value = str(invitee_id).strip()
        if not value:
            return
        if module in {"Contacts", "Leads"} and not ModuleAdjustmentEngine._is_valid_crm_id(value):
            return
        rows = invitees.setdefault(module, [])
        if any(str(row.get("id") or "").strip() == value for row in rows):
            return
        rows.append({"id": value})
