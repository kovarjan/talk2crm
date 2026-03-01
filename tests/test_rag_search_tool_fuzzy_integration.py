from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.rag import TenantRAGService
from app.engine.tools import build_tools
from app.services.crm_client import SugarClient


def _norm(value: str) -> str:
    import re
    import unicodedata

    lowered = (value or "").strip().lower()
    unaccented = "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()


def _build_rag_tool():
    tenant_id = os.getenv("RAG_TEST_TENANT_ID", "ai-local").strip()
    user_id = (
        os.getenv("CORIPO_TEST_USER_ID")
        or os.getenv("RAG_TEST_USER_ID")
        or os.getenv("CRM_TEST_USER_ID")
        or "28"
    ).strip()
    user_name = (
        os.getenv("CORIPO_TEST_USER_NAME")
        or os.getenv("RAG_TEST_USER_NAME")
        or os.getenv("CRM_TEST_USER_NAME")
        or ""
    ).strip()
    if os.getenv("CORIPO_TEST_BASE_URL"):
        crm_base_url = str(os.getenv("CORIPO_TEST_BASE_URL")).strip()
    else:
        in_container = Path("/.dockerenv").exists()
        crm_base_url = "http://host.docker.internal:2000/public" if in_container else "http://localhost:2000/public"
    crm_token = (
        os.getenv("CORIPO_TEST_TOKEN")
        or os.getenv("CRM_TEST_TOKEN")
        or os.getenv("CRM_TOKEN")
        or ""
    ).strip()

    if not crm_token:
        pytest.skip("Set CORIPO_TEST_TOKEN (or CRM_TEST_TOKEN/CRM_TOKEN) to run real RAG integration tests.")
    if user_id and "-" not in user_id and not user_name:
        pytest.skip(
            "CORIPO_TEST_USER_ID appears non-UUID; set CORIPO_TEST_USER_NAME "
            "(for example 'jkovar') for HMAC fallback auth."
        )

    rag_service = TenantRAGService()
    crm_client = SugarClient(
        base_url=crm_base_url,
        token=crm_token,
        user_id=user_id,
        user_name=user_name,
    )
    tools = build_tools(
        tenant_id=tenant_id,
        user_id=user_id,
        input_text="integration fuzzy rag search",
        request_context={},
        crm_client=crm_client,
        rag_service=rag_service,
        action_confirmation=False,
    )
    rag_tool = next((tool for tool in tools if getattr(tool, "name", "") == "rag_search_tool"), None)
    assert rag_tool is not None, "rag_search_tool not found in build_tools output"
    return rag_tool


def _invoke_rag_tool(rag_tool, *, query: str, module: str, limit: int = 20) -> list[dict[str, Any]]:
    try:
        raw = asyncio.run(rag_tool.ainvoke({"query": query, "module": module, "limit": limit}))
    except ValueError as exc:
        pytest.skip(f"Auth not configured for checksid/HMAC flow: {exc}")
    except httpx.ConnectError as exc:
        pytest.skip(f"CRM endpoint not reachable for integration run: {exc}")
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            pytest.skip(
                "Coripo auth rejected credentials (401). Verify token/keyId/user_id/user_name. "
                f"Body: {exc.response.text[:400]}"
            )
        raise
    data = json.loads(raw)
    assert isinstance(data, list), f"rag_search_tool returned unexpected payload: {data}"
    return data


def _extract_hits(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _name_from_record(record: dict[str, Any]) -> str:
        name = str(record.get("name") or record.get("NAME") or "").strip()
        if name:
            return name
        acc = str(record.get("account_name") or record.get("ACCOUNT_NAME") or record.get("company") or "").strip()
        if acc:
            return acc
        first = str(record.get("first_name") or record.get("FIRST_NAME") or "").strip()
        last = str(record.get("last_name") or record.get("LAST_NAME") or "").strip()
        full = f"{first} {last}".strip()
        return full

    hits: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        record = payload.get("record")
        if not isinstance(record, dict):
            continue
        name = _name_from_record(record)
        hits.append(
            {
                "score": float(item.get("score") or 0.0),
                "module": str(payload.get("module") or ""),
                "id": str(record.get("id") or ""),
                "name": name,
            }
        )
    return hits


def _print_hits(query: str, module: str, hits: list[dict[str, Any]]) -> None:
    print("\n=== RAG FUZZY TEST ===")
    print(f"query={query!r} module={module!r} total_hits={len(hits)}")
    for idx, hit in enumerate(hits[:12], start=1):
        print(
            f"{idx:02d}. score={hit['score']:.4f} module={hit['module']} "
            f"id={hit['id']} name={hit['name']}"
        )


@pytest.mark.parametrize(
    "query,module,expected_name",
    [
        ("Zliner", "Accounts", "ZLINER s.r.o."),
        ("panas", "Accounts", "PANAS, spol. s r.o."),
        ("az pokorny", "Accounts", "AZ - Pokorny Trade, s.r.o."),
    ],
)
def test_rag_search_tool_accounts_fuzzy_real_data(query: str, module: str, expected_name: str) -> None:
    rag_tool = _build_rag_tool()
    results = _invoke_rag_tool(rag_tool, query=query, module=module, limit=25)
    hits = _extract_hits(results)
    _print_hits(query, module, hits)

    expected_norm = _norm(expected_name)
    matching = [
        hit
        for hit in hits
        if _norm(hit["name"])
        and (expected_norm in _norm(hit["name"]) or _norm(hit["name"]) in expected_norm)
    ]
    assert matching, (
        f"Expected fuzzy account match '{expected_name}' for query '{query}', "
        f"but not found in top hits. Got names: {[hit['name'] for hit in hits[:12]]}"
    )


@pytest.mark.parametrize(
    "query,module,expected_id,expected_name",
    [
        ("karelm vodickou", "Contacts", "1f39facd-bc89-da5c-1f89-67d2bf0ac5dd", "Karel Vodička"),
        ("David Kostka", "Contacts", "bbf4032e-7fee-8b8c-869e-5f1543450ae6", "David Kostka"),
    ],
)
def test_rag_search_tool_contacts_fuzzy_real_data(
    query: str,
    module: str,
    expected_id: str,
    expected_name: str,
) -> None:
    rag_tool = _build_rag_tool()
    results = _invoke_rag_tool(rag_tool, query=query, module=module, limit=30)
    hits = _extract_hits(results)
    _print_hits(query, module, hits)

    by_id = [hit for hit in hits if hit["id"] == expected_id]
    assert by_id, (
        f"Expected contact id '{expected_id}' for query '{query}', "
        f"but not present in hits. Got ids: {[hit['id'] for hit in hits[:15]]}"
    )

    expected_name_norm = _norm(expected_name)
    assert any(expected_name_norm in _norm(hit["name"]) for hit in by_id), (
        f"Expected contact name similar to '{expected_name}' for id '{expected_id}', "
        f"but got: {[hit['name'] for hit in by_id]}"
    )
