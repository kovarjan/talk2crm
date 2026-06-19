# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Account/company candidate resolution against RAG and contact records."""

from __future__ import annotations

from typing import Any

from app.engine.rag import TenantRAGService
from app.engine.record_extract import (
    UUID_RE,
    extract_rag_records,
    record_primary_name,
    score_text_match,
)
from app.utils.text import safe_text


def score_account_candidate(company_name: str, record: dict[str, Any]) -> int:
    names = [
        safe_text(record.get("name")),
        safe_text(record.get("account_name")),
        safe_text(record.get("company")),
    ]
    base = max((score_text_match(company_name, name) for name in names), default=0)
    rag_boost = min(10, int(float(record.get("_rag_score", 0.0)) * 10))
    return base + rag_boost

def select_account_candidates(
    *,
    company_name: str,
    records: list[dict[str, Any]],
    limit: int = 5,
    min_name_score: int = 70,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in records:
        row_id = safe_text(row.get("id"))
        if not UUID_RE.match(row_id):
            continue
        names = [
            safe_text(row.get("name")),
            safe_text(row.get("account_name")),
            safe_text(row.get("company")),
        ]
        base_score = max((score_text_match(company_name, name) for name in names), default=0)
        if base_score < min_name_score:
            continue
        score = score_account_candidate(company_name, row)
        enriched = dict(row)
        enriched["_name_score"] = base_score
        scored.append((score, enriched))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []
    best_score = scored[0][0]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, row in scored:
        if score < max(min_name_score, best_score - 8):
            continue
        row_id = safe_text(row.get("id"))
        if row_id in seen:
            continue
        seen.add(row_id)
        selected.append(row)
        if len(selected) >= max(1, limit):
            break
    return selected

def contact_row_account_ref(row: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(row, dict):
        return "", ""
    account_id = safe_text(
        row.get("account_id")
        or row.get("accounts|id")
        or row.get("account")
        or row.get("accountid")
    )
    account_name = safe_text(
        row.get("account_name")
        or row.get("company")
        or row.get("accounts|name")
        or row.get("name")
    )
    return account_id, account_name

def account_names_from_contact_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    names_by_id: dict[str, str] = {}
    best_score: dict[str, float] = {}
    for row in rows:
        account_id, account_name = contact_row_account_ref(row)
        if not account_id or not account_name:
            continue
        score = float(row.get("_rag_score", 0.0))
        prev_score = best_score.get(account_id, -1.0)
        prev_name = names_by_id.get(account_id, "")
        if score > prev_score or (score == prev_score and len(account_name) > len(prev_name)):
            names_by_id[account_id] = account_name
            best_score[account_id] = score
    return names_by_id

def enrich_account_rows_with_contact_names(
    *,
    account_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not account_rows:
        return []
    names_by_id = account_names_from_contact_rows(contact_rows)
    if not names_by_id:
        return account_rows
    enriched: list[dict[str, Any]] = []
    for row in account_rows:
        row_id = safe_text(row.get("id"))
        name = safe_text(row.get("name") or row.get("account_name") or row.get("company"))
        if row_id and not name and row_id in names_by_id:
            updated = dict(row)
            updated["name"] = names_by_id[row_id]
            enriched.append(updated)
            continue
        enriched.append(row)
    return enriched

async def lookup_account_names_in_contacts_by_ids(
    *,
    rag_service: TenantRAGService,
    tenant_id: str,
    account_ids: list[str],
) -> dict[str, str]:
    wanted = {row_id for row_id in account_ids if UUID_RE.match(safe_text(row_id))}
    if not wanted:
        return {}

    names_by_id: dict[str, str] = {}
    async for payload in rag_service.iter_payloads(
        tenant_id=tenant_id,
        module="Contacts",
        page_size=768,
        max_scanned=160000,
    ):
        record = payload.get("record")
        if not isinstance(record, dict):
            continue
        account_id, account_name = contact_row_account_ref(record)
        if not account_id or not account_name:
            continue
        if account_id not in wanted:
            continue
        prev = names_by_id.get(account_id, "")
        if len(account_name) > len(prev):
            names_by_id[account_id] = account_name
        if wanted.issubset(names_by_id.keys()):
            break

    return names_by_id

async def resolve_accounts_from_qdrant(
    *,
    rag_service: TenantRAGService | None,
    tenant_id: str,
    company_name: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if rag_service is None:
        return []
    query = safe_text(company_name)
    if not query:
        return []

    results = await rag_service.search_entities(
        tenant_id=tenant_id,
        query=query,
        entity_type="account",
        limit=50,
    )
    account_rows = extract_rag_records(results, module_hint="accounts")
    contact_rows = extract_rag_records(results, module_hint="contacts")
    account_rows = enrich_account_rows_with_contact_names(
        account_rows=account_rows,
        contact_rows=contact_rows,
    )
    missing_ids = [
        safe_text(row.get("id"))
        for row in account_rows
        if UUID_RE.match(safe_text(row.get("id"))) and not record_primary_name(row)
    ]
    if missing_ids:
        id_lookup = await lookup_account_names_in_contacts_by_ids(
            rag_service=rag_service,
            tenant_id=tenant_id,
            account_ids=missing_ids,
        )
        if id_lookup:
            updated_rows: list[dict[str, Any]] = []
            for row in account_rows:
                row_id = safe_text(row.get("id"))
                if row_id and row_id in id_lookup and not record_primary_name(row):
                    row = dict(row)
                    row["name"] = id_lookup[row_id]
                updated_rows.append(row)
            account_rows = updated_rows

    # Fallback: derive account candidates from contact records when Accounts are not ingested.
    if not account_rows:
        for row in contact_rows:
            account_id, account_name = contact_row_account_ref(row)
            if not account_id or not account_name:
                continue
            # Do not promote unrelated contact companies into account candidates.
            if score_text_match(query, account_name) < 70:
                continue
            account_rows.append(
                {
                    "id": account_id,
                    "name": account_name,
                    "_module_hint": "accounts",
                    "_rag_score": float(row.get("_rag_score", 0.0)),
                }
            )

    return select_account_candidates(
        company_name=query,
        records=account_rows,
        limit=limit,
        min_name_score=70,
    )

def build_contacts_filter_by_account_ids(
    *,
    account_ids: list[str],
    require_email: bool = False,
) -> dict[str, Any]:
    account_operands = [{"field": "id", "type": "eq", "value": row_id} for row_id in account_ids if row_id]
    relation_operator = "and" if len(account_operands) <= 1 else "or"
    relation_filter: dict[str, Any] = {
        "operator": relation_operator,
        "operands": account_operands,
    }
    filter_operands: list[dict[str, Any]] = [
        {
            "operator": "and",
            "operands": [
                {
                    "module": "Accounts",
                    "type": "relate",
                    "name": "account_name",
                    "relationship": ["accounts"],
                    "filter": relation_filter,
                }
            ],
        }
    ]
    if require_email:
        filter_operands.append({"field": "email", "type": "nnull", "value": None})
    return {"operator": "and", "operands": filter_operands}
