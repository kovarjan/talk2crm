from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from app.engine import web_search
from app.engine.tools import build_tools
from app.engine.web_search import normalize_searxng_results


class DummyCrmClient:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_normalize_dedupes_by_url_and_trims_fields() -> None:
    payload = {
        "results": [
            {"title": " ACME s.r.o. ", "url": "https://acme.cz", "content": "Výrobce  pneumatik\n Zlín"},
            {"title": "duplicate", "url": "https://acme.cz", "content": "x"},
            {"title": "Other", "url": "https://other.cz", "content": "y" * 1000},
            "not-a-dict",
        ]
    }
    rows = normalize_searxng_results(payload, 5)
    assert [row["url"] for row in rows] == ["https://acme.cz", "https://other.cz"]
    assert rows[0]["title"] == "ACME s.r.o."
    assert rows[0]["content"] == "Výrobce pneumatik Zlín"
    assert len(rows[1]["content"]) == 600


def test_web_search_tool_calls_searxng_json_api(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"results": [{"title": "ACME", "url": "https://acme.cz", "content": "info"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(web_search, "get_shared_http_client", lambda: client)

    tools = build_tools(
        tenant_id="ai-local",
        user_id="1",
        input_text="najdi na webu firmu ACME",
        request_context={"web_search": True},
        crm_client=DummyCrmClient(),  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    tool = next(item for item in tools if getattr(item, "name", "") == "web_search_tool")
    raw = asyncio.run(tool.ainvoke({"query": "ACME s.r.o.", "max_results": 3}))
    payload = json.loads(raw)

    assert payload["status"] == "ok"
    assert payload["results"] == [{"title": "ACME", "url": "https://acme.cz", "content": "info"}]
    assert "format=json" in captured["url"]
    assert "q=ACME" in captured["url"]


def test_web_search_tool_reports_unavailable_instance(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("searxng down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(web_search, "get_shared_http_client", lambda: client)

    tools = build_tools(
        tenant_id="ai-local",
        user_id="1",
        input_text="najdi na webu firmu ACME",
        request_context={},
        crm_client=DummyCrmClient(),  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    tool = next(item for item in tools if getattr(item, "name", "") == "web_search_tool")
    payload = json.loads(asyncio.run(tool.ainvoke({"query": "ACME"})))
    assert payload["status"] == "web_search_unavailable"
    assert payload["results"] == []
