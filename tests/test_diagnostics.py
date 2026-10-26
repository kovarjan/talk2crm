import asyncio

from app.services import diagnostics


def test_collect_diagnostics_isolates_failures_and_hides_secrets(monkeypatch) -> None:
    async def ok_db(db):
        return {"migration": "0013"}

    async def broken_qdrant(tenant_id):
        raise ConnectionError("http://user:pw@qdrant:6333 refused")

    async def ok_llm(base_url, api_key, model):
        return {"host": "llm", "model": model, "model_available": True}

    monkeypatch.setattr(diagnostics, "_probe_database", ok_db)
    monkeypatch.setattr(diagnostics, "_probe_qdrant", broken_qdrant)
    monkeypatch.setattr(diagnostics, "_probe_openai_compatible", ok_llm)
    monkeypatch.setattr(diagnostics, "_crm_state", lambda tenant_id: {"manifest": {"cached": False}, "circuit": {}})

    data = asyncio.run(diagnostics.collect_diagnostics(tenant_id="t1", user_id="u1", db=None))

    assert data["status"] == "degraded"
    assert data["database"] == {"ok": True, "latency_ms": data["database"]["latency_ms"], "migration": "0013"}
    assert data["qdrant"]["ok"] is False and data["qdrant"]["error"] == "ConnectionError"
    assert "pw" not in str(data)  # exception text (may carry credentials) is never returned
    assert data["app"]["version"] and data["security"].keys() >= {"require_hmac", "bind_identity"}


def test_probe_timeout_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "_PROBE_TIMEOUT", 0.05)

    async def slow():
        await asyncio.sleep(1)

    result = asyncio.run(diagnostics._timed(slow()))
    assert result["ok"] is False and "timeout" in result["error"]
