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
import pytest

from app.engine import web_fetch
from app.engine.tools import build_tools
from app.engine.web_fetch import WebFetchError, _TextExtractor, _validate_url


class DummyCrmClient:
    mode = "coripo_public"

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"records": []}

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        return {"records": []}


def test_validate_url_rejects_private_and_non_http_targets() -> None:
    with pytest.raises(WebFetchError):
        _validate_url("http://127.0.0.1/admin")
    with pytest.raises(WebFetchError):
        _validate_url("http://localhost:8080/")
    with pytest.raises(WebFetchError):
        _validate_url("ftp://example.com/file")


def test_text_extractor_strips_scripts_and_collapses_whitespace() -> None:
    html = """
    <html><head><title> ACME  s.r.o. </title><style>.x{color:red}</style></head>
    <body>
      <nav>menu</nav>
      <script>alert(1)</script>
      <h1>O nás</h1>
      <p>Vyrábíme   pneumatiky   od roku 1995.</p>
      <div>Kontakt: info@acme.cz</div>
    </body></html>
    """
    parser = _TextExtractor()
    parser.feed(html)
    text = parser.text()
    assert parser.title == "ACME s.r.o."
    assert "alert(1)" not in text
    assert "menu" not in text
    assert "O nás" in text
    assert "Vyrábíme pneumatiky od roku 1995." in text


def test_web_fetch_tool_extracts_page_text(monkeypatch) -> None:
    monkeypatch.setattr(web_fetch, "_is_public_host", lambda host: True)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            html="<html><head><title>ACME</title></head><body><p>Výrobce pneumatik.</p></body></html>",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(web_fetch, "get_shared_http_client", lambda: client)

    tools = build_tools(
        tenant_id="ai-local",
        user_id="1",
        input_text="precti stranku acme.cz",
        request_context={},
        crm_client=DummyCrmClient(),  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    tool = next(item for item in tools if getattr(item, "name", "") == "web_fetch_tool")
    payload = json.loads(asyncio.run(tool.ainvoke({"url": "https://acme.cz/o-nas"})))

    assert payload["status"] == "ok"
    assert payload["title"] == "ACME"
    assert "Výrobce pneumatik." in payload["text"]


def test_web_fetch_tool_rejects_private_url() -> None:
    tools = build_tools(
        tenant_id="ai-local",
        user_id="1",
        input_text="precti localhost",
        request_context={},
        crm_client=DummyCrmClient(),  # type: ignore[arg-type]
        rag_service=None,
        action_confirmation=False,
    )
    tool = next(item for item in tools if getattr(item, "name", "") == "web_fetch_tool")
    payload = json.loads(asyncio.run(tool.ainvoke({"url": "http://127.0.0.1/secret"})))
    assert payload["status"] == "web_fetch_unavailable"
