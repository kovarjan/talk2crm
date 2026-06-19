from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.engine.rag import TenantRAGService
from app.services.crm_client import CoripoClient


logger = get_logger(__name__)


SafeTextFn = Callable[[Any], str]
ExtractRecordsFn = Callable[[Any], list[dict[str, Any]]]
NormalizeTextFn = Callable[[str], str]


@dataclass
class CRMReadHelpers:
    safe_text: SafeTextFn
    normalize_text: NormalizeTextFn
    parse_datetime: Callable[[Any], Any]
    extract_records: ExtractRecordsFn
    extract_date_range_from_text: Callable[[str], tuple[Any, Any] | None]
    infer_data_query_type: Callable[[str, str], str]
    next_week_range: Callable[[], tuple[Any, Any]]
    build_meetings_date_filter: Callable[[Any, Any], dict[str, Any]]
    sort_records_by_datetime: Callable[[list[dict[str, Any]], str], list[dict[str, Any]]]
    dedupe_and_limit: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]
    meeting_cards: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]
    filter_valid_contact_records: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    contact_cards: Callable[[list[dict[str, Any]], int, str], list[dict[str, Any]]]
    contact_cards_force_table: Callable[[list[dict[str, Any]], int, str], list[dict[str, Any]]]
    generic_cards: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]
    extract_person_name: Callable[[str, dict[str, Any] | None], str]
    recent_contacts_from_context: Callable[[dict[str, Any] | None], list[dict[str, Any]]]
    best_recent_contact_match: Callable[[str, list[dict[str, Any]]], dict[str, Any] | None]
    extract_rag_records: Callable[[list[dict[str, Any]], str | None], list[dict[str, Any]]]
    merge_records_by_id: Callable[..., list[dict[str, Any]]]
    best_record_match: Callable[[list[dict[str, Any]], str], dict[str, Any] | None]
    extract_company_name: Callable[[str, dict[str, Any] | None], str]
    extract_account_ids_from_query: Callable[[Any, dict[str, Any] | None], list[str]]
    resolve_accounts_from_qdrant: Callable[[str], Awaitable[list[dict[str, Any]]]]
    select_account_candidates: Callable[[str, list[dict[str, Any]], int, int], list[dict[str, Any]]]
    query_requires_filled_email: Callable[[str], bool]
    build_contacts_filter_by_account_ids: Callable[[list[str], bool], dict[str, Any]]
    filter_records_by_company: Callable[[list[dict[str, Any]], str, bool], list[dict[str, Any]]]


class CRMReadService:
    def __init__(
        self,
        *,
        tenant_id: str,
        user_id: str,
        input_text: str,
        request_context: dict[str, Any] | None,
        crm_client: CoripoClient,
        rag_service: TenantRAGService | None,
        helpers: CRMReadHelpers,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.input_text = input_text
        self.request_context = request_context or {}
        self.crm_client = crm_client
        self.rag_service = rag_service
        self.h = helpers

    @staticmethod
    def _module_aliases(scope: str) -> set[str]:
        key = (scope or "").strip().lower()
        aliases = {
            "accounts": {"accounts", "account"},
            "contacts": {"contacts", "contact"},
            "meetings": {"meetings", "meeting"},
        }
        return aliases.get(key, {key} if key else set())

    def _record_module_name(self, record: dict[str, Any]) -> str:
        for key in ("_module_hint", "_module", "module", "record_module", "tag"):
            value = self.h.safe_text(record.get(key)).lower()
            if value:
                return value
        record_id = self.h.safe_text(record.get("id"))
        if "-" in record_id and not re_match_uuid(record_id):
            prefix = record_id.split("-", 1)[0].strip().lower()
            if prefix in {"accounts", "account", "contacts", "contact", "meetings", "meeting", "calls", "tasks", "notes"}:
                return prefix
        return ""

    def _enforce_scope_purity(self, records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
        aliases = self._module_aliases(scope)
        if not aliases:
            return records
        filtered: list[dict[str, Any]] = []
        for row in records:
            module_name = self._record_module_name(row)
            if module_name and module_name not in aliases:
                continue
            filtered.append(row)
        return filtered

    async def execute_structured_list(self, module: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload or {})
        return await self.crm_client.execute_module_action(module=module, action="list", data=data)

    async def run_search(self, *, query: Any, scope: str = "all", limit: int = 20) -> dict[str, Any]:
        normalized_scope = self.h.safe_text(scope).lower() or "all"
        requested = int(limit or 20)
        safe_limit = max(1, min(requested, 15))
        if requested > 15:
            logger.debug(
                "crm_read limit clamped: requested=%d → %d (hard cap)",
                requested, safe_limit,
            )
        module_map = {
            "contacts": "Contacts",
            "accounts": "Accounts",
            "meetings": "Meetings",
            "opportunities": "Opportunities",
            "opportunites": "Opportunities",
            "quotes": "Quotes",
            "acm_invoices": "acm_invoices",
        }

        if normalized_scope in module_map:
            payload: dict[str, Any]
            if isinstance(query, dict):
                payload = dict(query)
            else:
                query_text = self.h.safe_text(query)
                payload = {"query": query_text, "q": query_text}

            payload.setdefault("limit", max(50, safe_limit * 4))
            payload.setdefault("offset", 0)
            payload.setdefault("include_field_names", False)

            if normalized_scope == "meetings":
                payload.setdefault(
                    "columns",
                    [
                        {"field": "date_start", "module": "Meetings"},
                        {"field": "name", "module": "Meetings"},
                        {"field": "location", "module": "Meetings"},
                        {"field": "status", "module": "Meetings"},
                    ],
                )
                payload.setdefault(
                    "response_fields",
                    [
                        "id",
                        "name",
                        "date_start",
                        "date_entered",
                        "location",
                        "status",
                        "assigned_user_name",
                    ],
                )
            elif normalized_scope == "contacts":
                payload.setdefault(
                    "response_fields",
                    [
                        "id",
                        "name",
                        "first_name",
                        "last_name",
                        "account_name",
                        "email",
                        "email1",
                        "phone_mobile",
                        "phone_work",
                    ],
                )
            elif normalized_scope == "accounts":
                payload.setdefault("response_fields", ["id", "name", "account_name", "billing_address_city"])

            result = await self.execute_structured_list(module_map[normalized_scope], payload)
        else:
            result = await self.crm_client.generic_search(query=self.h.safe_text(query), scope=normalized_scope)

        if normalized_scope in module_map and isinstance(result, dict):
            records = self._enforce_scope_purity(self.h.extract_records(result), normalized_scope)
            if normalized_scope == "meetings":
                ordered = self.h.sort_records_by_datetime(records, "date_start")
                selected = self.h.dedupe_and_limit(ordered, safe_limit)
                total_count = len(ordered)
                cards = self.h.meeting_cards(selected, total_count)
                message = f"Našla jsem {total_count} schůzek." if total_count else "Nenašla jsem žádné schůzky."
                return {
                    "status": "ok",
                    "query_type": "crm_search_meetings",
                    "module": "Meetings",
                    "total_count": total_count,
                    "cards": cards,
                    "message_to_user": message,
                }
            if normalized_scope == "contacts":
                selected = self.h.dedupe_and_limit(self.h.filter_valid_contact_records(records), safe_limit)
                total_count = len(selected)
                cards = self.h.contact_cards(selected, total_count, f"Kontakty ({total_count})")
                return {
                    "status": "ok",
                    "query_type": "crm_search_contacts",
                    "module": "Contacts",
                    "total_count": total_count,
                    "cards": cards,
                }
            merged = self.h.dedupe_and_limit(records, safe_limit)
            total_count = len(records)
            cards = self.h.generic_cards(merged, total_count)
            return {
                "status": "ok",
                "query_type": "crm_search_generic",
                "module": module_map[normalized_scope],
                "total_count": total_count,
                "cards": cards,
            }

        return result if isinstance(result, dict) else {"result": result}

    async def run_data(
        self,
        *,
        query: Any,
        query_type: str = "auto",
        scope: Any | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        requested = int(limit or 20)
        safe_limit = max(1, min(requested, 15))
        if requested > 15:
            logger.debug(
                "crm_read limit clamped: requested=%d → %d (hard cap)",
                requested, safe_limit,
            )
        user_query = self.h.safe_text(self.input_text)
        effective_query = self.h.safe_text(query) or user_query
        normalized_scope = self.h.safe_text(scope).lower()
        scope_module_map = {
            "contacts": "Contacts",
            "accounts": "Accounts",
            "meetings": "Meetings",
            "opportunities": "Opportunities",
            "opportunites": "Opportunities",
            "quotes": "Quotes",
            "acm_invoices": "acm_invoices",
        }
        requested_query_type = self.h.safe_text(query_type).lower()
        read_v2_enabled = get_settings().read_service_v2_active

        # Scoped `search` must stay deterministic and module-pure.
        if read_v2_enabled and requested_query_type == "search" and normalized_scope in scope_module_map:
            scoped_result = await self.run_search(
                query=query,
                scope=normalized_scope,
                limit=safe_limit,
            )
            if isinstance(scoped_result, dict):
                scoped_result["query_type"] = f"scoped_search_{normalized_scope}"
            return scoped_result

        resolved_type = self.h.infer_data_query_type(user_query or effective_query, query_type)
        explicit_range = self.h.extract_date_range_from_text(effective_query)
        normalized_user_query = self.h.normalize_text(user_query)
        user_mentions_next_week = any(token in normalized_user_query for token in ("dalsi tyden", "pristi tyden", "next week"))
        if explicit_range is not None and resolved_type in {"meetings_next_week", "meetings_range"} and not user_mentions_next_week:
            resolved_type = "meetings_range"

        try:
            if resolved_type in {"meetings_next_week", "meetings_range"}:
                if resolved_type == "meetings_next_week":
                    range_start, range_end = self.h.next_week_range()
                elif explicit_range:
                    range_start, range_end = explicit_range
                else:
                    range_start, range_end = self.h.next_week_range()

                meeting_filter = self.h.build_meetings_date_filter(range_start, range_end)
                result = await self.crm_client.execute_module_action(
                    module="Meetings",
                    action="list",
                    data={
                        "limit": 500,
                        "offset": 0,
                        "filter": meeting_filter,
                        "columns": [
                            {"field": "date_start", "module": "Meetings"},
                            {"field": "name", "module": "Meetings"},
                            {"field": "location", "module": "Meetings"},
                            {"field": "status", "module": "Meetings"},
                            {"field": "assigned_user_name", "module": "Meetings"},
                        ],
                        "order": [{"field": "date_start", "sort": "ASC", "module": "Meetings"}],
                        "include_field_names": False,
                        "response_fields": [
                            "id",
                            "name",
                            "date_start",
                            "date_entered",
                            "location",
                            "status",
                            "assigned_user_id",
                            "assigned_user_name",
                            "users_id_c",
                        ],
                    },
                )
                all_records = self.h.extract_records(result)

                selected: list[dict[str, Any]] = []
                for row in all_records:
                    start_value = row.get("date_start") or row.get("date_entered")
                    dt = self.h.parse_datetime(start_value)
                    if dt is None or not (range_start <= dt < range_end):
                        continue
                    assigned_user = self.h.safe_text(row.get("assigned_user_id") or row.get("users_id_c"))
                    if (
                        re_match_uuid(self.h.safe_text(self.user_id))
                        and assigned_user
                        and assigned_user != self.h.safe_text(self.user_id)
                    ):
                        continue
                    selected.append(row)

                selected = self.h.sort_records_by_datetime(selected, "date_start")
                limited = self.h.dedupe_and_limit(selected, safe_limit)
                total_count = len(selected)
                cards = self.h.meeting_cards(limited, total_count)

                if total_count == 0:
                    message = (
                        "Na příští týden nemáte žádné schůzky. "
                        f"Kontrolované období: {range_start.strftime('%d.%m.%Y')} - {range_end.strftime('%d.%m.%Y')}."
                    )
                else:
                    message = (
                        f"Na příští týden jsem našla {total_count} schůzek. "
                        f"Období: {range_start.strftime('%d.%m.%Y')} - {range_end.strftime('%d.%m.%Y')}."
                    )

                return {
                    "status": "ok",
                    "query_type": resolved_type,
                    "module": "Meetings",
                    "total_count": total_count,
                    "cards": cards,
                    "message_to_user": message,
                }

            if resolved_type == "contact_by_name":
                explicit_query = self.h.safe_text(query)
                person_source = explicit_query or user_query or effective_query
                person_name = self.h.extract_person_name(person_source, self.request_context)

                recent_contacts = self.h.recent_contacts_from_context(self.request_context)
                recent_match = self.h.best_recent_contact_match(person_name or person_source, recent_contacts)
                if recent_match is not None:
                    selected = [recent_match]
                    total_count = 1
                    cards = self.h.contact_cards(selected, total_count, f"Kontakty ({total_count})")
                    return {
                        "status": "ok",
                        "query_type": resolved_type,
                        "module": "Contacts",
                        "total_count": total_count,
                        "cards": cards,
                        "message_to_user": (
                            "Navazuji na poslední kontext. "
                            f"Kontakt '{self.h.safe_text(recent_match.get('name'))}' jsem našla."
                        ),
                        "source": "recent_contacts_context",
                    }

                rag_candidates: list[dict[str, Any]] = []
                if self.rag_service is not None:
                    try:
                        rag_hits = await self.rag_service.search_entities(
                            tenant_id=self.tenant_id,
                            query=person_name,
                            entity_type="contact",
                            limit=max(20, safe_limit * 3),
                        )
                        rag_candidates = self.h.filter_valid_contact_records(self.h.extract_rag_records(rag_hits, "contacts"))
                    except Exception:
                        rag_candidates = []

                list_result = await self.crm_client.execute_module_action(
                    module="Contacts",
                    action="list",
                    data={"query": person_name, "q": person_name, "max_results": max(50, safe_limit)},
                )
                search_result = await self.crm_client.generic_search(query=person_name, scope="contacts")
                all_records = self.h.merge_records_by_id(
                    rag_candidates,
                    self.h.filter_valid_contact_records(self.h.extract_records(list_result)),
                    self.h.filter_valid_contact_records(self.h.extract_records(search_result)),
                )
                all_records = self._enforce_scope_purity(all_records, "contacts")
                match = self.h.best_record_match(all_records, person_name)
                selected = [match] if match else self.h.dedupe_and_limit(all_records, safe_limit)
                total_count = len(selected)
                cards = self.h.contact_cards(selected, total_count, f"Kontakty ({total_count})")

                if total_count == 0:
                    message = f"Kontakt '{person_name}' jsem nenašla."
                elif match:
                    message = f"Kontakt '{self.h.safe_text(match.get('name'))}' jsem našla."
                else:
                    message = f"Našla jsem {total_count} kontaktů k dotazu '{person_name}'."

                return {
                    "status": "ok",
                    "query_type": resolved_type,
                    "module": "Contacts",
                    "total_count": total_count,
                    "cards": cards,
                    "message_to_user": message,
                }

            if resolved_type == "contacts_by_company":
                company_name = self.h.extract_company_name(user_query or effective_query, self.request_context)
                direct_account_ids = self.h.extract_account_ids_from_query(query, self.request_context)
                if not direct_account_ids and scope is not None:
                    direct_account_ids = self.h.extract_account_ids_from_query(scope, self.request_context)

                resolved_accounts: list[dict[str, Any]] = []
                if direct_account_ids:
                    account_ids = direct_account_ids
                    try:
                        id_filter = {
                            "operator": "and",
                            "operands": [
                                {
                                    "operator": "or",
                                    "operands": [{"field": "id", "type": "eq", "value": row_id} for row_id in account_ids],
                                }
                            ],
                        }
                        accounts_by_id = await self.crm_client.execute_module_action(
                            module="Accounts",
                            action="list",
                            data={
                                "limit": max(30, len(account_ids) * 2),
                                "offset": 0,
                                "filter": id_filter,
                                "include_field_names": False,
                                "response_fields": ["id", "name", "account_name", "billing_address_city"],
                            },
                        )
                        resolved_accounts = self.h.extract_records(accounts_by_id)
                    except Exception:
                        resolved_accounts = [{"id": row_id} for row_id in account_ids]
                else:
                    resolved_accounts = await self.h.resolve_accounts_from_qdrant(company_name)
                    if not resolved_accounts:
                        accounts_list_result = await self.crm_client.execute_module_action(
                            module="Accounts",
                            action="list",
                            data={
                                "query": company_name,
                                "q": company_name,
                                "max_results": 60,
                                "include_field_names": False,
                                "response_fields": ["id", "name", "account_name", "billing_address_city"],
                            },
                        )
                        crm_account_candidates = self.h.extract_records(accounts_list_result)
                        resolved_accounts = self.h.select_account_candidates(company_name, crm_account_candidates, 5, 65)

                    account_ids = [
                        self.h.safe_text(row.get("id"))
                        for row in resolved_accounts
                        if re_match_uuid(self.h.safe_text(row.get("id")))
                    ]

                require_email = self.h.query_requires_filled_email(user_query or effective_query)
                relation_contacts: list[dict[str, Any]] = []
                if account_ids:
                    relation_filter = self.h.build_contacts_filter_by_account_ids(account_ids, require_email)
                    contacts_relation_result = await self.crm_client.execute_module_action(
                        module="Contacts",
                        action="list",
                        data={
                            "limit": 200,
                            "offset": 0,
                            "filter": relation_filter,
                            "columns": [
                                {"field": "name", "module": "Contacts"},
                                {"field": "title", "module": "Contacts"},
                                {"field": "account_name", "module": "Contacts"},
                                {"field": "phone_mobile", "module": "Contacts"},
                                {"field": "phone_work", "module": "Contacts"},
                                {"field": "email", "module": "Contacts"},
                                {"field": "assigned_user_name", "module": "Contacts"},
                            ],
                            "order": [],
                            "groupBy": [],
                            "function": {},
                            "alterName": {},
                            "groupByDate": [],
                            "savedSearch": True,
                            "include_field_names": False,
                            "response_fields": [
                                "id",
                                "name",
                                "first_name",
                                "last_name",
                                "account_name",
                                "email",
                                "email1",
                                "phone_mobile",
                                "phone_work",
                            ],
                        },
                    )
                    relation_contacts = self.h.filter_valid_contact_records(self.h.extract_records(contacts_relation_result))

                contacts: list[dict[str, Any]]
                used_relation_filter = bool(account_ids)
                if account_ids and relation_contacts:
                    contacts = relation_contacts
                else:
                    contacts_list_result = await self.crm_client.execute_module_action(
                        module="Contacts",
                        action="list",
                        data={
                            "query": company_name,
                            "q": company_name,
                            "max_results": 200,
                            "include_field_names": False,
                            "response_fields": [
                                "id",
                                "name",
                                "first_name",
                                "last_name",
                                "account_name",
                                "email",
                                "email1",
                                "phone_mobile",
                                "phone_work",
                            ],
                        },
                    )
                    contacts_search_result = await self.crm_client.generic_search(query=company_name, scope="contacts")
                    contacts = self.h.merge_records_by_id(
                        self.h.filter_valid_contact_records(self.h.extract_records(contacts_list_result)),
                        self.h.filter_valid_contact_records(self.h.extract_records(contacts_search_result)),
                    )

                contacts = self._enforce_scope_purity(contacts, "contacts")
                filtered_contacts = self.h.filter_records_by_company(
                    contacts,
                    company_name,
                    not used_relation_filter,
                )
                display_limit = min(safe_limit, 10)
                selected = self.h.dedupe_and_limit(filtered_contacts, display_limit)
                total_count = len(filtered_contacts)
                cards = self.h.contact_cards_force_table(selected, total_count, f"Kontakty firmy ({total_count})")

                if total_count == 0:
                    message = f"Kontakty k firmě '{company_name}' jsem nenašla."
                else:
                    if account_ids:
                        account_labels = ", ".join(
                            self.h.safe_text(row.get("name")) or self.h.safe_text(row.get("account_name"))
                            for row in resolved_accounts[:3]
                            if self.h.safe_text(row.get("name")) or self.h.safe_text(row.get("account_name"))
                        )
                        if account_labels:
                            message = (
                                f"K firmě '{company_name}' jsem našla {total_count} kontaktů. "
                                f"Filtrovala jsem přes účet: {account_labels}."
                            )
                        else:
                            message = f"K firmě '{company_name}' jsem našla {total_count} kontaktů."
                    else:
                        message = (
                            f"K firmě '{company_name}' jsem našla {total_count} kontaktů. "
                            "Účet nebyl nalezen v RAG indexu, použila jsem textové CRM filtrování."
                        )

                return {
                    "status": "ok",
                    "query_type": resolved_type,
                    "module": "Contacts",
                    "total_count": total_count,
                    "resolved_account_ids": account_ids,
                    "cards": cards,
                    "message_to_user": message,
                }

            if read_v2_enabled and normalized_scope in scope_module_map:
                scoped_result = await self.run_search(
                    query=effective_query,
                    scope=normalized_scope,
                    limit=safe_limit,
                )
                if isinstance(scoped_result, dict):
                    scoped_result["query_type"] = f"scoped_search_{normalized_scope}"
                return scoped_result

            aggregate = await self.crm_client.generic_search(query=effective_query, scope="all")
            merged: list[dict[str, Any]] = []
            if self.rag_service is not None:
                try:
                    rag_hits = await self.rag_service.search(
                        tenant_id=self.tenant_id,
                        query=effective_query,
                        limit=max(30, safe_limit * 4),
                    )
                    rag_rows = self.h.extract_rag_records(rag_hits, None)
                    for row in rag_rows:
                        module_hint = self.h.safe_text(row.get("_module_hint")).capitalize()
                        if module_hint:
                            row["_module_hint"] = module_hint
                        merged.append(row)
                except Exception:
                    pass
            if isinstance(aggregate, dict):
                module_map = {
                    "contacts": "Contacts",
                    "accounts": "Accounts",
                    "meetings": "Meetings",
                    "opportunities": "Opportunities",
                    "opportunites": "Opportunities",
                    "quotes": "Quotes",
                    "acm_invoices": "acm_invoices",
                }
                for module_key, payload in aggregate.items():
                    module_name = module_map.get(str(module_key).lower(), str(module_key))
                    for row in self.h.extract_records(payload):
                        row["_module_hint"] = module_name
                        merged.append(row)
            merged = self.h.dedupe_and_limit(merged, safe_limit)
            total_count = len(merged)
            cards = self.h.generic_cards(merged, total_count)
            message = (
                f"Našla jsem {total_count} záznamů pro dotaz '{effective_query}'."
                if total_count
                else f"K dotazu '{effective_query}' jsem nic nenašla."
            )
            return {
                "status": "ok",
                "query_type": "generic_search",
                "module": "CRM",
                "total_count": total_count,
                "cards": cards,
                "message_to_user": message,
            }
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            return {
                "status": "crm_http_error",
                "query_type": resolved_type,
                "http_status": status_code,
                "message_to_user": "CRM data se nepodařilo načíst. Zkuste to prosím znovu.",
                "cards": [],
            }
        except Exception as exc:
            return {
                "status": "crm_data_error",
                "query_type": resolved_type,
                "message_to_user": "Nastala chyba při načítání CRM dat.",
                "error": str(exc),
                "cards": [],
            }



def re_match_uuid(value: str) -> bool:
    import re

    return bool(
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            (value or "").strip(),
            re.IGNORECASE,
        )
    )
