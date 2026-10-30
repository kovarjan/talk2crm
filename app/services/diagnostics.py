"""Gateway self-diagnostics for the hidden `/ping` command.

Every probe is timed and isolated: a failing component is reported as
``ok: false`` with a short error class, never raised, and no secret, token or
credential-bearing URL is returned (only booleans and host names).
"""
from __future__ import annotations

import asyncio
import platform
import sys
import time
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import text

from app.core.config import get_settings
from app.services.tenant_llm import resolve_llm_config
from app.core.http import get_shared_http_client
from app.core.logging import get_logger
from app.engine.rag import TenantRAGService, get_rag_service

logger = get_logger(__name__)

_STARTED_AT = time.time()
_PROBE_TIMEOUT = 5.0
# Modules whose synced record counts are worth showing when testing the RAG sync.
_COUNTED_MODULES = ["Accounts", "Contacts", "Leads", "Opportunities", "Meetings", "Calls", "Tasks", "Notes"]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


def _host(url: str | None) -> str:
    parts = urlsplit(url or "")
    return parts.netloc.rsplit("@", 1)[-1] or (url or "")


async def _timed(coro) -> dict[str, Any]:
    """Run a probe; always return a dict with ok/latency_ms (and error on failure)."""
    started = time.perf_counter()
    try:
        data = await asyncio.wait_for(coro, timeout=_PROBE_TIMEOUT)
        return {"ok": True, "latency_ms": _ms(started), **(data or {})}
    except asyncio.TimeoutError:
        return {"ok": False, "latency_ms": _ms(started), "error": f"timeout after {_PROBE_TIMEOUT:g}s"}
    except Exception as exc:  # diagnostics must never raise
        logger.warning("diagnostics probe failed: %s", type(exc).__name__)
        return {"ok": False, "latency_ms": _ms(started), "error": type(exc).__name__}


async def _probe_database(db) -> dict[str, Any]:
    await db.execute(text("SELECT 1"))
    revision = None
    try:
        revision = (await db.execute(text("SELECT version_num FROM alembic_version"))).scalar()
    except Exception:
        await db.rollback()
    return {"dialect": db.bind.dialect.name if db.bind is not None else None, "migration": revision}


async def _probe_qdrant(tenant_id: str) -> dict[str, Any]:
    settings = get_settings()
    rag = get_rag_service()
    if rag is None:
        raise RuntimeError("RAG service unavailable")
    collection = await rag._tenant_collection(tenant_id)
    client = await rag._get_client()
    info = await client.get_collection(collection)
    total = int((await client.count(collection_name=collection, exact=True)).count)

    async def count_module(module: str) -> tuple[str, int]:
        result = await client.count(
            collection_name=collection,
            count_filter=TenantRAGService._module_filter([module]),
            exact=True,
        )
        return module, int(result.count)

    per_module = dict(await asyncio.gather(*(count_module(m) for m in _COUNTED_MODULES)))
    local = getattr(client, "_client", None).__class__.__name__ == "AsyncQdrantLocal"
    return {
        "host": "local-disk" if local else _host(settings.qdrant_url),
        "collection": collection,
        "status": str(getattr(info.status, "value", info.status)),
        "vector_count": total,
        "points_count": getattr(info, "points_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "modules": {m: c for m, c in per_module.items() if c},
        "embedder": type(rag.embedder).__name__,
        "embedding_model": settings.rag_embedding_model,
        "embedding_size": settings.rag_embedding_size,
    }


async def _probe_openai_compatible(base_url: str, api_key: str | None, model: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = await get_shared_http_client().get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=_PROBE_TIMEOUT)
    response.raise_for_status()
    ids = [str(m.get("id")) for m in (response.json().get("data") or []) if isinstance(m, dict)]
    return {"host": _host(base_url), "model": model, "model_available": model in ids if ids else None}


async def _probe_tenant_llm(tenant_id: str) -> dict[str, Any]:
    # The tenant's own endpoint/model when configured, else the system default.
    config = await resolve_llm_config(tenant_id)
    result = await _probe_openai_compatible(config.base_url, config.api_key, config.model)
    return {**result, "tenant_endpoint": config.custom_endpoint}


def _crm_state(tenant_id: str) -> dict[str, Any]:
    from app.tools.toolset import get_manifest_cache, get_tenant_guards

    return {"manifest": get_manifest_cache().state(tenant_id), "circuit": get_tenant_guards().state(tenant_id)}


def _security_flags() -> dict[str, Any]:
    settings = get_settings()
    return {
        "require_hmac": settings.require_hmac,
        "bind_identity": settings.hmac_bind_identity,
        "nonce_store": (settings.hmac_nonce_store or "auto"),
    }


async def collect_diagnostics(*, tenant_id: str, user_id: str, db) -> dict[str, Any]:
    settings = get_settings()
    started = time.perf_counter()
    database, qdrant, llm, embeddings = await asyncio.gather(
        _timed(_probe_database(db)),
        _timed(_probe_qdrant(tenant_id)),
        _timed(_probe_tenant_llm(tenant_id)),
        _timed(_probe_openai_compatible(settings.rag_embedding_base_url, settings.rag_embedding_api_key, settings.rag_embedding_model)),
    )
    components = {"database": database, "qdrant": qdrant, "llm": llm, "embeddings": embeddings}
    return {
        "status": "ok" if all(c["ok"] for c in components.values()) else "degraded",
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "uptime_seconds": int(time.time() - _STARTED_AT),
        },
        "tenant_id": tenant_id,
        "user_id": user_id,
        **components,
        "crm": _crm_state(tenant_id),
        "security": _security_flags(),
        "probe_total_ms": _ms(started),
    }
