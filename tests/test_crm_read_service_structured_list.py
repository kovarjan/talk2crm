from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.crm_read_service import CRMReadHelpers, CRMReadService


class DummyCrmClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((module, action, data))
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}



def _noop_helpers() -> CRMReadHelpers:
    return CRMReadHelpers(
        safe_text=lambda v: "" if v is None else str(v).strip(),
        normalize_text=lambda s: str(s or "").strip().lower(),
        parse_datetime=lambda v: None,
        extract_records=lambda v: [],
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



def test_execute_structured_list_passthrough_payload() -> None:
    client = DummyCrmClient()
    service = CRMReadService(
        tenant_id="ai-local",
        user_id="28",
        input_text="",
        request_context={},
        crm_client=client,  # type: ignore[arg-type]
        rag_service=None,
        helpers=_noop_helpers(),
    )

    payload = {
        "filter": {"operator": "and", "operands": [{"field": "name", "type": "cont", "value": "Igor"}]},
        "columns": [{"field": "name", "module": "Contacts"}],
        "order": [{"field": "date_modified", "sort": "DESC", "module": "Contacts"}],
        "response_fields": ["id", "name", "email1"],
    }

    asyncio.run(service.execute_structured_list("Contacts", payload))

    assert client.calls
    module, action, data = client.calls[0]
    assert module == "Contacts"
    assert action == "list"
    assert data == payload
