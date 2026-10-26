"""The gateway signs CRM requests in the strict format the PHP HmacVerifier expects."""
import asyncio
import base64
import hashlib
import hmac
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.services import crm_client
from app.services.crm_client import CoripoClient


def _client() -> CoripoClient:
    client = CoripoClient(base_url="http://crm.test/public", token="11111111-2222-3333-4444-555555555555")
    client.coripo_hmac_secret = "s3cret"
    client.coripo_hmac_key_id = "kid"
    client.user_id = "user-1"
    client._coripo_session_id = "sid"
    return client


def test_signature_binds_method_path_query_and_body(monkeypatch) -> None:
    sent: dict[str, Any] = {}

    class FakeHttp:
        async def request(self, *, method, url, headers, content=None, timeout=None, **kw):
            sent.update(method=method, url=str(url), headers=headers, content=content or b"")
            return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, str(url)))

    monkeypatch.setattr(crm_client, "get_shared_http_client", lambda: FakeHttp())
    asyncio.run(
        _client()._coripo_request_response(
            "POST", "rest/ai/tools/call", params={"a": "x y", "b": "1"}, json_body={"k": "v"}
        )
    )

    parts = urlsplit(sent["url"])
    h = sent["headers"]
    base = (
        f"{h['X-Timestamp']}|{h['X-Nonce']}|user-1|POST|{parts.path}|{parts.query}|"
    ).encode() + sent["content"]
    expected = base64.b64encode(hmac.new(b"s3cret", base, hashlib.sha256).digest()).decode()
    assert parts.query == "a=x+y&b=1"
    assert h["Authorization"] == f"HMAC keyId=kid, signature={expected}"


def test_signature_changes_with_path_and_query() -> None:
    c = _client()
    a = c._build_coripo_hmac_headers(b"", method="GET", path="/a", query="")
    b = c._build_coripo_hmac_headers(b"", method="GET", path="/b", query="")
    assert a["Authorization"] != b["Authorization"]
