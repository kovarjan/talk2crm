from __future__ import annotations

import httpx
import pytest

from _tool_fakes import FakeTransport, error_result, http_error, ok_result
from app.tools.contracts import CallContext, Manifest, ToolAnnotations, ToolSpec
from app.tools.crm_provider import MODE_PREVIEW, CrmToolProvider
from app.tools.manifest_cache import ManifestCache
from app.tools.resilience import TenantGuards

CTX = CallContext(tenant_id="t1", user_id="u1")
READ = ToolSpec("crm_query_tool", "q", {"type": "object"}, ToolAnnotations(read_only=True, timeout_seconds=15), "crm", "1.0.0")
WRITE = ToolSpec("crm_action_tool", "a", {"type": "object"}, ToolAnnotations(read_only=False, requires_confirmation=True, timeout_seconds=30), "crm", "1.0.0")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("app.tools.crm_provider.asyncio.sleep", instant)


def _provider(transport: FakeTransport, guards: TenantGuards | None = None, cache: ManifestCache | None = None) -> CrmToolProvider:
    return CrmToolProvider(tenant_id="t1", transport=transport, cache=cache or ManifestCache(), guards=guards or TenantGuards())


@pytest.mark.asyncio
async def test_call_passes_mode_timeout_and_request_id() -> None:
    transport = FakeTransport(responses=[ok_result({"status": "ok", "total": 1}, meta={"request_id": "x"})])
    result = await _provider(transport).call(READ, {"module": "Contacts"}, CTX)
    call = transport.calls[0]
    assert call["name"] == "crm_query_tool" and call["mode"] == "execute"
    assert call["timeout"].read == 15 and call["timeout"].connect == 3
    assert call["request_id"]
    assert result.structured["total"] == 1


@pytest.mark.asyncio
async def test_tool_errors_are_results_and_do_not_trip_the_breaker() -> None:
    guards = TenantGuards(failure_threshold=2)
    transport = FakeTransport(responses=[error_result("invalid_arguments", "bad")] * 3)
    provider = _provider(transport, guards)
    for _ in range(3):
        result = await provider.call(READ, {}, CTX)
        assert result.error_code == "invalid_arguments"
    assert guards.state("t1")["breaker"] == "closed"


@pytest.mark.asyncio
async def test_read_only_tool_retries_once_on_503() -> None:
    transport = FakeTransport(responses=[http_error(503), ok_result({"status": "ok"})])
    result = await _provider(transport).call(READ, {}, CTX)
    assert not result.is_error
    assert len(transport.calls) == 2
    assert transport.calls[0]["request_id"] == transport.calls[1]["request_id"]


@pytest.mark.asyncio
async def test_write_is_never_retried_and_timeout_warns_it_may_have_applied() -> None:
    transport = FakeTransport(responses=[httpx.ReadTimeout("slow"), ok_result({})])
    result = await _provider(transport).call(WRITE, {}, CTX, mode=MODE_PREVIEW)
    assert len(transport.calls) == 1
    assert result.error_code == "unavailable"
    assert "mohla provést" in result.text


@pytest.mark.asyncio
async def test_breaker_opens_after_repeated_transport_failures_and_fails_fast() -> None:
    guards = TenantGuards(failure_threshold=3, reset_seconds=30)
    transport = FakeTransport(responses=[httpx.ConnectError("down")] * 6)
    provider = _provider(transport, guards)
    for _ in range(3):
        assert (await provider.call(WRITE, {}, CTX)).error_code == "unavailable"
    assert guards.state("t1")["breaker"] == "open"
    calls = len(transport.calls)
    result = await provider.call(READ, {}, CTX)
    assert result.error_code == "unavailable"
    assert len(transport.calls) == calls  # failed fast, CRM not called


@pytest.mark.asyncio
async def test_breaker_is_per_tenant() -> None:
    guards = TenantGuards(failure_threshold=1)
    down = FakeTransport(responses=[httpx.ConnectError("down")])
    await _provider(down, guards).call(WRITE, {}, CTX)
    other = CrmToolProvider(tenant_id="t2", transport=FakeTransport(), cache=ManifestCache(), guards=guards)
    assert not (await other.call(READ, {}, CallContext("t2", "u"))).is_error


@pytest.mark.asyncio
async def test_404_invalidates_the_manifest() -> None:
    cache = ManifestCache()
    transport = FakeTransport(responses=[http_error(404)])
    provider = _provider(transport, cache=cache)
    await provider.manifest()
    result = await provider.call(READ, {}, CTX)
    assert result.error_code == "not_found"
    await provider.manifest()
    assert len(transport.manifest_calls) == 2


@pytest.mark.asyncio
async def test_half_open_breaker_recovers_after_success() -> None:
    guards = TenantGuards(failure_threshold=1, reset_seconds=0)
    transport = FakeTransport(responses=[httpx.ConnectError("down"), ok_result({"status": "ok"})])
    provider = _provider(transport, guards)
    await provider.call(WRITE, {}, CTX)
    assert not (await provider.call(WRITE, {}, CTX)).is_error
    assert guards.state("t1")["breaker"] == "closed"


def test_manifest_parse_reads_annotations() -> None:
    from _tool_fakes import sample_manifest

    manifest = Manifest.parse(sample_manifest())
    action = next(t for t in manifest.tools if t.name == "crm_action_tool")
    assert action.annotations.needs_confirmation and not action.annotations.read_only
    query = next(t for t in manifest.tools if t.name == "crm_query_tool")
    assert query.annotations.read_only and query.annotations.capability == "crm"
