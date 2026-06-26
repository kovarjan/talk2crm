from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain.aggregate_contracts import AggregateFilters, AggregateRequest, AggregateResult
from app.engine.tools import build_tools
from app.services.crm_aggregate_service import CRMAggregateService
from app.services.crm_read_service import CRMReadHelpers, CRMReadService


class DummyCrmClient:
    def __init__(self, pages: list[dict] | None = None, company_overview: dict | None = None) -> None:
        self.pages = pages or [{"records": [], "source_record_count": 0}]
        self.company_overview = company_overview or {}
        self.calls: list[tuple[str, str, dict]] = []

    async def execute_module_action(self, module: str, action: str, data: dict) -> dict:
        self.calls.append((module, action, data))
        if module == "Accounts" and action == "company_overview":
            return self.company_overview
        offset = int(data.get("offset", 0) or 0)
        page_index = 0 if offset == 0 else 1
        return self.pages[min(page_index, len(self.pages) - 1)]

    async def generic_search(self, query: str, scope: str = "all") -> dict:
        return {"records": []}


def _noop_helpers() -> CRMReadHelpers:
    return CRMReadHelpers(
        safe_text=lambda v: "" if v is None else str(v).strip(),
        normalize_text=lambda s: str(s or "").strip().lower(),
        parse_datetime=lambda v: None,
        extract_records=lambda v: list(v.get("records", [])) if isinstance(v, dict) else [],
        extract_date_range_from_text=lambda text: None,
        infer_data_query_type=lambda query, requested: "generic_search",
        next_week_range=lambda: (None, None),
        build_meetings_date_filter=lambda start, end: {},
        sort_records_by_datetime=lambda records, field: records,
        dedupe_and_limit=lambda records, limit: records[:limit],
        meeting_cards=lambda records, total: [],
        filter_valid_contact_records=lambda records: records,
        contact_cards=lambda records, total, title: [],
        contact_cards_force_table=lambda records, total, title: [],
        generic_cards=lambda records, total: [],
        extract_person_name=lambda query, context: query,
        recent_contacts_from_context=lambda context: [],
        best_recent_contact_match=lambda query, candidates: None,
        extract_rag_records=lambda results, module_hint: [],
        merge_records_by_id=lambda *sets: [],
        best_record_match=lambda records, search: None,
        extract_company_name=lambda query, context: query,
        extract_account_ids_from_query=lambda query, context: [],
        resolve_accounts_from_qdrant=lambda company_name: [],
        select_account_candidates=lambda company_name, records, limit, min_score: [],
        query_requires_filled_email=lambda text: False,
        build_contacts_filter_by_account_ids=lambda ids, require_email: {},
        filter_records_by_company=lambda records, company_name, fallback: records,
    )


def _make_read_service(client: DummyCrmClient | None = None) -> CRMReadService:
    return CRMReadService(
        tenant_id="ai-local",
        user_id="1",
        input_text="",
        request_context={},
        crm_client=client or DummyCrmClient(),  # type: ignore[arg-type]
        rag_service=None,
        helpers=_noop_helpers(),
    )


def _make_tools(*, crm_client=None, aggregate_tools_enabled: bool = True, input_text: str = "součet faktur"):
    client = crm_client or DummyCrmClient()
    with patch("app.engine.tools.get_settings") as mock_settings:
        mock_settings.return_value.crm_mode = "on"
        mock_settings.return_value.tool_call_logging = False
        mock_settings.return_value.aggregate_tools_enabled = aggregate_tools_enabled
        return build_tools(
            tenant_id="test-tenant",
            user_id="user1",
            input_text=input_text,
            request_context=None,
            crm_client=client,
            rag_service=None,
            action_confirmation=False,
        )


def _get_tool(tools: list, name: str):
    return next(tool for tool in tools if getattr(tool, "name", "") == name)


class TestNormalizeAmountField:
    def test_renames_amount_usdollar_to_amount(self) -> None:
        record = {"id": "1", "amount_usdollar": "15000.00"}
        result = CRMReadService._normalize_amount_field(record)
        assert result["amount"] == "15000.00"
        assert result["_raw_amount_usdollar"] == "15000.00"
        assert "amount_usdollar" not in result

    def test_does_not_overwrite_existing_amount(self) -> None:
        record = {"amount": "20000.00", "amount_usdollar": "15000.00"}
        result = CRMReadService._normalize_amount_field(record)
        assert result["amount"] == "20000.00"

    def test_adds_currency_hint(self) -> None:
        result = CRMReadService._normalize_amount_field({"amount_usdollar": "5000"})
        assert result["amount_currency"] == "CZK"

    def test_does_not_mutate_original(self) -> None:
        original = {"amount_usdollar": "999"}
        CRMReadService._normalize_amount_field(original)
        assert "amount_usdollar" in original


def test_list_module_fetches_all_pages_and_normalizes_amounts() -> None:
    first_page = {
        "records": [{"id": str(index), "amount_usdollar": "100.00"} for index in range(500)],
        "source_record_count": 500,
    }
    second_page = {
        "records": [{"id": "500", "amount_usdollar": "250.00"}, {"id": "501", "amount_usdollar": "400.00"}],
        "source_record_count": 2,
    }
    client = DummyCrmClient(pages=[first_page, second_page])
    service = _make_read_service(client)

    records = asyncio.run(service.list_module("acm_invoices", limit=502, fetch_all=True))

    assert len(records) == 502
    assert len(client.calls) == 2
    assert records[0]["amount"] == "100.00"
    assert records[-1]["_raw_amount_usdollar"] == "400.00"


class TestAggregateContracts:
    def test_valid_request(self) -> None:
        req = AggregateRequest(
            module="invoices",
            operation="sum",
            filters=AggregateFilters(date_from="2026-01-01", date_to="2026-03-31"),
        )
        assert req.filters.exclude_cancelled is True

    def test_invalid_module_rejected(self) -> None:
        with pytest.raises(Exception):
            AggregateRequest(module="bad-module", operation="sum")  # type: ignore[arg-type]

    def test_result_to_agent_string_sum(self) -> None:
        result = AggregateResult(
            success=True,
            operation="sum",
            metric="amount",
            module="invoices",
            value=Decimal("184500.00"),
            record_count=12,
            currency="CZK",
        )
        text = result.to_agent_string()
        assert "184" in text
        assert "CZK" in text


SAMPLE_INVOICES = [
    {"id": "1", "amount": "10000.00", "status": "paid"},
    {"id": "2", "amount": "20000.00", "status": "issued"},
    {"id": "3", "amount": "5000.00", "status": "cancelled"},
    {"id": "4", "amount": "15000.00", "status": "paid"},
    {"id": "5", "status": "paid"},
]


def _make_stub_crm(records: list[dict]):
    stub = MagicMock()
    stub.list_module = AsyncMock(return_value=records)
    return stub


@pytest.mark.asyncio
async def test_sum_excludes_cancelled_by_default() -> None:
    svc = CRMAggregateService(_make_stub_crm(SAMPLE_INVOICES))
    result = await svc.aggregate(AggregateRequest(module="invoices", operation="sum", metric="amount"))
    assert result.success is True
    assert result.value == Decimal("45000.00")
    assert result.record_count == 3


@pytest.mark.asyncio
async def test_avg_uses_decimal_math() -> None:
    svc = CRMAggregateService(_make_stub_crm(SAMPLE_INVOICES))
    result = await svc.aggregate(AggregateRequest(module="invoices", operation="avg", metric="amount"))
    assert result.value == Decimal("45000.00") / Decimal("3")


@pytest.mark.asyncio
async def test_count_counts_filtered_records_even_without_metric_value() -> None:
    svc = CRMAggregateService(_make_stub_crm(SAMPLE_INVOICES))
    result = await svc.aggregate(AggregateRequest(module="invoices", operation="count", metric="amount"))
    assert result.record_count == 4
    assert result.value == Decimal("4")
    assert any("count počítá" in item for item in result.assumptions)


@pytest.mark.asyncio
async def test_status_filter_is_applied() -> None:
    svc = CRMAggregateService(_make_stub_crm(SAMPLE_INVOICES))
    result = await svc.aggregate(
        AggregateRequest(
            module="invoices",
            operation="sum",
            metric="amount",
            filters=AggregateFilters(statuses=["paid"]),
        )
    )
    assert result.value == Decimal("25000.00")
    assert result.record_count == 2


@pytest.mark.asyncio
async def test_invoice_sum_uses_czech_total_fields() -> None:
    invoice_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "stav_uhrazeni": "neuhrazeno",
            "datum_vystaveni": "2026-02-06",
            "cena_bez_dph_celkem": "9600.000000",
            "cena_dph_celkem": "2016.000000",
            "cena_s_dph_celkem": "11616.000000",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "stav_uhrazeni": "neuhrazeno",
            "datum_vystaveni": "2026-02-06",
            "cena_bez_dph_celkem": "43350.000000",
            "cena_dph_celkem": "9103.500000",
            "cena_s_dph_celkem": "52453.500000",
        },
    ]
    svc = CRMAggregateService(_make_stub_crm(invoice_rows))

    result = await svc.aggregate(
        AggregateRequest(module="invoices", operation="sum", metric="amount")
    )

    assert result.success is True
    assert result.value == Decimal("64069.500000")
    assert result.record_count == 2
    assert any("pole: cena_s_dph_celkem" == item for item in result.assumptions)


@pytest.mark.asyncio
async def test_invoice_status_filter_reads_stav_uhrazeni() -> None:
    invoice_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "stav_uhrazeni": "uhrazeno",
            "cena_s_dph_celkem": "1000.00",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "stav_uhrazeni": "neuhrazeno",
            "cena_s_dph_celkem": "2000.00",
        },
    ]
    svc = CRMAggregateService(_make_stub_crm(invoice_rows))

    result = await svc.aggregate(
        AggregateRequest(
            module="invoices",
            operation="sum",
            metric="amount",
            filters=AggregateFilters(statuses=["uhrazeno"]),
        )
    )

    assert result.value == Decimal("1000.00")
    assert result.record_count == 1


@pytest.mark.asyncio
async def test_invoice_turnover_defaults_to_paid_only() -> None:
    invoice_rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "stav_uhrazeni": "uhrazeno",
            "cena_s_dph_celkem": "813531.40",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "stav_uhrazeni": "neuhrazeno",
            "cena_s_dph_celkem": "64069.50",
        },
    ]
    service = CRMReadService(
        tenant_id="ai-local",
        user_id="1",
        input_text="sečti všechny faktury této firmy kolik je její celkový obrat?",
        request_context={},
        crm_client=DummyCrmClient(),  # type: ignore[arg-type]
        rag_service=None,
        helpers=_noop_helpers(),
    )
    aggregate = CRMAggregateService(service)
    aggregate._fetch_records = AsyncMock(return_value=(invoice_rows, "datum_vystaveni", []))  # type: ignore[method-assign]

    result = await aggregate.aggregate(
        AggregateRequest(
            module="invoices",
            operation="sum",
            metric="amount",
        )
    )

    assert result.value == Decimal("813531.40")
    assert result.record_count == 1
    assert any("uhrazené faktury" in item for item in result.assumptions)


@pytest.mark.asyncio
async def test_crm_fetch_error_returns_failure() -> None:
    stub = MagicMock()
    stub.list_module = AsyncMock(side_effect=ConnectionError("CRM down"))
    result = await CRMAggregateService(stub).aggregate(
        AggregateRequest(module="invoices", operation="sum", metric="amount")
    )
    assert result.success is False
    assert "CRM down" in str(result.error)


@pytest.mark.asyncio
async def test_invoice_aggregate_falls_back_to_company_overview_records() -> None:
    client = DummyCrmClient(
        pages=[{"records": [], "source_record_count": 0}],
        company_overview={
            "module": "Accounts",
            "related_records": {
                "AOS_Invoices": [
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "grand_total": "12000.00",
                        "status": "Sent",
                    },
                    {
                        "id": "22222222-2222-2222-2222-222222222222",
                        "grand_total": "8500.00",
                        "status": "Paid",
                    },
                ]
            },
        },
    )
    service = _make_read_service(client)

    result = await CRMAggregateService(service).aggregate(
        AggregateRequest(
            module="invoices",
            operation="sum",
            metric="amount",
            filters=AggregateFilters(account_id="acc-1"),
        )
    )

    assert result.success is True
    assert result.value == Decimal("20500.00")
    assert result.record_count == 2
    assert any("company_overview/AOS_Invoices" in item for item in result.assumptions)


@pytest.mark.asyncio
async def test_invalid_date_placeholders_are_ignored() -> None:
    stub = MagicMock()
    stub.list_module = AsyncMock(
        return_value=[
            {"id": "1", "amount": "1000.00", "status": "paid"},
            {"id": "2", "amount": "250.00", "status": "paid"},
        ]
    )

    result = await CRMAggregateService(stub).aggregate(
        AggregateRequest(
            module="invoices",
            operation="sum",
            metric="amount",
            filters=AggregateFilters(
                date_from="tentokrat_start",
                date_to="tentokrat_end",
            ),
        )
    )

    assert result.success is True
    assert result.value == Decimal("1250.00")
    assert any("ignorován neplatný date_from" in item for item in result.assumptions)
    assert any("ignorován neplatný date_to" in item for item in result.assumptions)


def test_build_tools_includes_aggregate_tools_when_enabled() -> None:
    names = {getattr(tool, "name", "") for tool in _make_tools(aggregate_tools_enabled=True)}
    assert "crm_aggregate_tool" in names
    assert "math_tool" in names


def test_build_tools_omits_aggregate_tools_when_disabled() -> None:
    names = {getattr(tool, "name", "") for tool in _make_tools(aggregate_tools_enabled=False)}
    assert "crm_aggregate_tool" not in names
    assert "math_tool" not in names


def test_build_tools_omits_aggregate_tools_for_non_reporting_prompt() -> None:
    names = {getattr(tool, "name", "") for tool in _make_tools(input_text="najdi firmu eleman")}
    assert "crm_aggregate_tool" not in names
    assert "math_tool" not in names


def test_math_tool_sum() -> None:
    tool = _get_tool(_make_tools(), "math_tool")
    result = tool.invoke({"numbers": [100, 200, 300], "operation": "sum"})
    assert "600" in result


def test_math_tool_unknown_operation() -> None:
    tool = _get_tool(_make_tools(), "math_tool")
    result = tool.invoke({"numbers": [1, 2], "operation": "magic"})
    assert "Neznámá" in result


def test_crm_aggregate_tool_returns_czech_summary() -> None:
    client = DummyCrmClient(
        pages=[
            {
                "records": [
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "amount_usdollar": "100.00",
                        "status": "paid",
                    },
                    {
                        "id": "22222222-2222-2222-2222-222222222222",
                        "amount_usdollar": "250.00",
                        "status": "paid",
                    },
                ],
                "source_record_count": 2,
            }
        ]
    )
    tool = _get_tool(_make_tools(crm_client=client), "crm_aggregate_tool")

    result = asyncio.run(tool.ainvoke({"module": "invoices", "operation": "sum"}))

    assert "Součet" in result
    assert "350,00" in result
