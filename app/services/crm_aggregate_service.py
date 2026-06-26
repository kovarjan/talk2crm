from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.core.logging import get_logger
from app.domain.aggregate_contracts import (
    AggregateFilters,
    AggregateMetric,
    AggregateOperation,
    AggregateRequest,
    AggregateResult,
)
from app.engine.filter_builder import FilterSpec
from app.services.crm_read_service import CRMReadService


logger = get_logger(__name__)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

MODULE_MAP: dict[str, tuple[str, ...]] = {
    "invoices": ("acm_invoices", "AOS_Invoices"),
    "invoice_items": ("AOS_Invoices_Quotes",),
    "quotes": ("Quotes",),
    "quote_items": ("AOS_Products_Quotes",),
    "opportunities": ("Opportunities",),
}

METRIC_FIELD_CANDIDATES: dict[str, list[str]] = {
    "amount": [
        "amount",
        "cena_s_dph_celkem",
        "grand_total",
        "_raw_amount_usdollar",
        "amount_usdollar",
        "cena_bez_dph_celkem",
    ],
    "amount_total": [
        "amount_total",
        "cena_s_dph_celkem",
        "grand_total",
        "total_amount",
        "amount",
    ],
    "amount_without_vat": [
        "amount_without_vat",
        "cena_bez_dph_celkem",
        "subtotal_amount",
        "amount",
    ],
    "vat_amount": ["vat_amount", "cena_dph_celkem", "tax_amount"],
    "quantity": ["quantity"],
    "unit_price": ["unit_price", "price"],
    "discount": ["discount", "discount_amount"],
    "margin": ["margin", "gross_margin"],
}

STATUS_ALIASES: dict[str, dict[str, str]] = {
    "invoices": {
        "uhrazeno": "uhrazeno",
        "zaplaceno": "uhrazeno",
        "paid": "uhrazeno",
        "neuhrazeno": "neuhrazeno",
        "unpaid": "neuhrazeno",
        "po splatnosti": "po splatnosti",
        "overdue": "po splatnosti",
        "storno": "storno",
        "cancelled": "storno",
        "canceled": "storno",
    },
    "quotes": {},
    "quote_items": {},
    "opportunities": {
        "closed won": "closed won",
        "won": "closed won",
        "closed lost": "closed lost",
        "lost": "closed lost",
    },
}

DATE_FIELD_MAP: dict[str, dict[str, str]] = {
    "acm_invoices": {
        "date_issued": "datum_vystaveni",
        "date_paid": "datum_uhrazeni",
        "due_date": "datum_splatnosti",
        "date_entered": "date_entered",
        "date_modified": "date_modified",
    },
    "Quotes": {
        "date_issued": "date_quote_expected_closed",
        "date_paid": "date_quote_expected_closed",
        "due_date": "date_quote_expected_closed",
        "date_entered": "date_entered",
        "date_modified": "date_modified",
    },
    "Opportunities": {
        "date_issued": "date_closed",
        "date_paid": "date_closed",
        "due_date": "date_closed",
        "date_entered": "date_entered",
        "date_modified": "date_modified",
    },
    "AOS_Invoices_Quotes": {
        "date_issued": "date_entered",
        "date_paid": "date_entered",
        "due_date": "date_entered",
        "date_entered": "date_entered",
        "date_modified": "date_modified",
    },
    "AOS_Products_Quotes": {
        "date_issued": "date_entered",
        "date_paid": "date_entered",
        "due_date": "date_entered",
        "date_entered": "date_entered",
        "date_modified": "date_modified",
    },
}

CANCELLED_STATUSES = {"cancelled", "storno", "void", "draft_cancelled"}
TURNOVER_QUERY_MARKERS = ("obrat", "trzby", "tržby", "revenue", "turnover")


class CRMAggregateService:
    def __init__(self, crm_read_service: CRMReadService) -> None:
        self._crm = crm_read_service

    async def aggregate(self, request: AggregateRequest) -> AggregateResult:
        crm_modules = MODULE_MAP.get(request.module)
        if not crm_modules:
            return AggregateResult(
                success=False,
                operation=request.operation,
                metric=request.metric,
                module=request.module,
                record_count=0,
                error=f"Unknown module: {request.module}",
            )

        fetch_notes: list[str] = []
        records: list[dict[str, object]] = []
        actual_date_field = "date_entered"
        try:
            for crm_module in crm_modules:
                records, actual_date_field, module_notes = await self._fetch_records(crm_module, request.filters)
                fetch_notes.extend(module_notes)
                if records:
                    break
            if not records and request.module == "invoices" and request.filters.account_id:
                records, overview_notes = await self._fetch_invoice_records_from_overview(
                    account_id=request.filters.account_id
                )
                fetch_notes.extend(overview_notes)
        except Exception as exc:
            logger.exception("crm aggregate fetch failed for %s", request.module)
            return AggregateResult(
                success=False,
                operation=request.operation,
                metric=request.metric,
                module=request.module,
                record_count=0,
                error=str(exc),
            )

        filtered_records, status_notes = self._apply_status_filter(records, request)
        fetch_notes.extend(status_notes)
        values, skipped, field_used = self._extract_values(filtered_records, request.metric)
        assumptions = self._build_assumptions(
            request=request,
            field_used=field_used,
            skipped=skipped,
            total_filtered=len(filtered_records),
            actual_date_field=actual_date_field,
            fetch_notes=fetch_notes,
        )

        if request.operation == "count":
            result_value = Decimal(len(filtered_records)) if filtered_records else None
            record_count = len(filtered_records)
        else:
            result_value = self._compute(values, request.operation)
            record_count = len(values)

        return AggregateResult(
            success=True,
            operation=request.operation,
            metric=request.metric,
            module=request.module,
            value=result_value,
            record_count=record_count,
            currency="CZK",
            assumptions=assumptions,
        )

    async def _fetch_records(
        self,
        crm_module: str,
        filters: AggregateFilters,
    ) -> tuple[list[dict[str, object]], str, list[str]]:
        specs: list[FilterSpec] = []
        if filters.account_id:
            specs.append(FilterSpec(field="account_id", op="eq", value=filters.account_id))
        if filters.assigned_user_id:
            specs.append(FilterSpec(field="assigned_user_id", op="eq", value=filters.assigned_user_id))

        actual_date_field = self._resolve_date_field(crm_module, filters.date_field)
        date_from, date_to, notes = self._sanitize_date_filters(filters.date_from, filters.date_to)
        records = await self._crm.list_module(
            crm_module,
            filters=specs,
            date_from=date_from,
            date_to=date_to,
            date_field=actual_date_field,
            limit=10000,
            fetch_all=True,
            include_field_names=False,
        )
        return records, actual_date_field, notes

    async def _fetch_invoice_records_from_overview(
        self,
        *,
        account_id: str,
    ) -> tuple[list[dict[str, object]], list[str]]:
        raw = await self._crm.crm_client.execute_module_action(
            module="Accounts",
            action="company_overview",
            data={"id": account_id},
        )
        if not isinstance(raw, dict):
            return [], []

        related_records = raw.get("related_records")
        if not isinstance(related_records, dict):
            return [], []

        for key in ("AOS_Invoices", "acm_invoices"):
            rows = related_records.get(key)
            if isinstance(rows, list):
                normalized = self._crm._normalize_records([row for row in rows if isinstance(row, dict)])
                if normalized:
                    return normalized, [f"zdroj: company_overview/{key}"]
        return [], []

    @staticmethod
    def _sanitize_date_filters(
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[str | None, str | None, list[str]]:
        notes: list[str] = []
        normalized_from = date_from.strip() if isinstance(date_from, str) else None
        normalized_to = date_to.strip() if isinstance(date_to, str) else None

        if normalized_from and not _ISO_DATE_RE.match(normalized_from):
            notes.append(f"ignorován neplatný date_from: {normalized_from}")
            normalized_from = None
        if normalized_to and not _ISO_DATE_RE.match(normalized_to):
            notes.append(f"ignorován neplatný date_to: {normalized_to}")
            normalized_to = None

        return normalized_from, normalized_to, notes

    @staticmethod
    def _resolve_date_field(crm_module: str, date_field: str) -> str:
        module_map = DATE_FIELD_MAP.get(crm_module, {})
        return module_map.get(date_field, module_map.get("date_issued", "date_entered"))

    def _apply_status_filter(
        self,
        records: list[dict[str, object]],
        request: AggregateRequest,
    ) -> tuple[list[dict[str, object]], list[str]]:
        filters = request.filters
        status_aliases = STATUS_ALIASES.get(request.module, {})
        requested_statuses = [str(item).strip().lower() for item in (filters.statuses or []) if str(item).strip()]
        normalized_requested = [
            status_aliases.get(item, item)
            for item in requested_statuses
        ]
        allowed = {item for item in normalized_requested if item in status_aliases.values() or not status_aliases}
        notes: list[str] = []
        if requested_statuses and status_aliases and not allowed:
            notes.append("ignorován neplatný status filtr pro tento modul")
        implied_statuses, implied_notes = self._implied_status_filter(request, allowed)
        notes.extend(implied_notes)
        if implied_statuses:
            allowed = implied_statuses

        result: list[dict[str, object]] = []
        for row in records:
            status = str(
                row.get("status")
                or row.get("invoice_status")
                or row.get("stav_uhrazeni")
                or ""
            ).strip().lower()
            if status_aliases:
                status = status_aliases.get(status, status)
            if filters.exclude_cancelled and status in CANCELLED_STATUSES:
                continue
            if allowed and status and status not in allowed:
                continue
            result.append(row)
        return result, notes

    def _implied_status_filter(
        self,
        request: AggregateRequest,
        allowed: set[str],
    ) -> tuple[set[str], list[str]]:
        if request.module != "invoices":
            return set(), []

        user_text = str(getattr(self._crm, "input_text", "") or "").strip().lower()
        if not user_text:
            return set(), []

        if not any(marker in user_text for marker in TURNOVER_QUERY_MARKERS):
            return set(), []

        if allowed and allowed != {"uhrazeno"}:
            return allowed, []

        return {"uhrazeno"}, ["výklad obratu: započteny pouze uhrazené faktury"]

    def _extract_values(
        self,
        records: list[dict[str, object]],
        metric: AggregateMetric,
    ) -> tuple[list[Decimal], int, str]:
        candidates = METRIC_FIELD_CANDIDATES.get(metric, [metric])
        values: list[Decimal] = []
        skipped = 0
        field_used = candidates[0]

        for record in records:
            raw = None
            selected_field = field_used
            for candidate in candidates:
                if candidate in record and record[candidate] is not None:
                    raw = record[candidate]
                    selected_field = candidate
                    break
            if raw is None:
                skipped += 1
                continue

            decimal_value = self._parse_decimal(raw)
            if decimal_value is None:
                logger.warning(
                    "crm aggregate cannot parse value %r for field %s",
                    raw,
                    selected_field,
                )
                skipped += 1
                continue

            field_used = selected_field
            values.append(decimal_value)

        return values, skipped, field_used

    @staticmethod
    def _parse_decimal(raw: object) -> Decimal | None:
        text = str(raw).strip().replace("\xa0", "").replace(" ", "")
        if not text:
            return None
        if "," in text and "." in text:
            text = text.replace(",", "")
        elif "," in text:
            text = text.replace(",", ".")
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    @staticmethod
    def _compute(values: list[Decimal], operation: AggregateOperation) -> Decimal | None:
        if not values:
            return None
        if operation == "sum":
            return sum(values, Decimal("0"))
        if operation == "avg":
            return sum(values, Decimal("0")) / Decimal(len(values))
        if operation == "min":
            return min(values)
        if operation == "max":
            return max(values)
        if operation == "count":
            return Decimal(len(values))
        raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def _build_assumptions(
        *,
        request: AggregateRequest,
        field_used: str,
        skipped: int,
        total_filtered: int,
        actual_date_field: str,
        fetch_notes: list[str],
    ) -> list[str]:
        assumptions = [f"pole: {field_used}", f"datum: {actual_date_field}"]
        if request.filters.exclude_cancelled:
            assumptions.append("stornované záznamy vynechány")
        if request.filters.statuses:
            assumptions.append("stavy: " + ", ".join(request.filters.statuses))
        assumptions.extend(dict.fromkeys(fetch_notes))
        if request.operation == "count":
            assumptions.append("count počítá záznamy po filtrování bez ohledu na metriku")
        elif skipped > 0:
            assumptions.append(
                f"{skipped} z {total_filtered} záznamů vynecháno (chybějící nebo nečíselná hodnota)"
            )
        return assumptions
