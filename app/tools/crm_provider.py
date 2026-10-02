# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Tools published by the tenant's Coripo (coripo-tools/1), called over its REST API.

Call path: circuit breaker → per-tenant semaphore → timeout from the tool's annotations
→ POST ai/tools/call. Read-only tools get one jittered retry on connection-level
failures; writes are never retried (a timeout after sending may have written already).
Tool-level errors (isError) are normal results and do not count against the breaker.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Protocol

import httpx

from app.core.logging import get_logger
from app.tools.contracts import CallContext, Manifest, ToolCallResult, ToolSpec
from app.tools.manifest_cache import ManifestCache
from app.tools.resilience import CircuitOpenError, TenantGuards

logger = get_logger(__name__)

MODE_EXECUTE = "execute"
MODE_PREVIEW = "preview"
CONNECT_TIMEOUT_SECONDS = 3.0
MAX_TIMEOUT_SECONDS = 60.0
_RETRYABLE_STATUS = {502, 503, 504}


class ToolTransport(Protocol):
    async def fetch_manifest(self, etag: str | None) -> tuple[bool, dict[str, Any] | None]: ...

    async def call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        mode: str,
        confirmation_token: str | None,
        request_id: str,
        timeout: httpx.Timeout,
    ) -> dict[str, Any]: ...


class CoripoToolTransport:
    """ToolTransport over the tenant's CoripoClient (HMAC/SID auth, shared HTTP pool)."""

    def __init__(self, crm_client: Any) -> None:
        self.crm_client = crm_client

    async def fetch_manifest(self, etag: str | None) -> tuple[bool, dict[str, Any] | None]:
        return await self.crm_client.get_tool_manifest(etag=etag, timeout=httpx.Timeout(15.0, connect=CONNECT_TIMEOUT_SECONDS))

    async def call(self, **kwargs: Any) -> dict[str, Any]:
        return await self.crm_client.call_tool(**kwargs)


class CrmToolProvider:
    def __init__(self, *, tenant_id: str, transport: ToolTransport, cache: ManifestCache, guards: TenantGuards) -> None:
        self.tenant_id = tenant_id
        self.transport = transport
        self.cache = cache
        self.guards = guards

    async def manifest(self) -> Manifest | None:
        return await self.cache.get(self.tenant_id, self.transport.fetch_manifest)

    async def call(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        ctx: CallContext,
        *,
        mode: str = MODE_EXECUTE,
        confirmation_token: str | None = None,
    ) -> ToolCallResult:
        request_id = CallContext.new_request_id()
        read_timeout = min(max(spec.annotations.timeout_seconds, 1.0), MAX_TIMEOUT_SECONDS)
        timeout = httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT_SECONDS)
        guard = self.guards.get(self.tenant_id)
        attempts = 2 if spec.annotations.read_only else 1
        started = time.perf_counter()
        status = "ok"
        result: ToolCallResult
        try:
            guard.breaker.before_call()
        except CircuitOpenError:
            result = ToolCallResult.error("unavailable", "CRM je dočasně nedostupné, zkuste to prosím za chvíli.")
            self._log(spec, mode, request_id, "circuit_open", started)
            return result

        try:
            async with guard.semaphore:
                for attempt in range(1, attempts + 1):
                    try:
                        raw = await self.transport.call(
                            name=spec.name, arguments=args, mode=mode, confirmation_token=confirmation_token,
                            request_id=request_id, timeout=timeout,
                        )
                        guard.breaker.record_success()
                        result = ToolCallResult.from_wire(raw)
                        status = result.error_code or ("confirmation_required" if result.confirmation else "ok")
                        break
                    except httpx.HTTPStatusError as exc:
                        code = exc.response.status_code
                        if code in _RETRYABLE_STATUS and attempt < attempts:
                            await asyncio.sleep(random.uniform(0.2, 0.6))
                            continue
                        result, status = self._http_error(spec, exc, guard)
                        break
                    except (httpx.TransportError, asyncio.TimeoutError) as exc:
                        if attempt < attempts and not isinstance(exc, httpx.ReadTimeout):
                            await asyncio.sleep(random.uniform(0.2, 0.6))
                            continue
                        guard.breaker.record_failure()
                        status = type(exc).__name__
                        result = self._transport_error(spec, exc)
                        break
        except asyncio.CancelledError:
            guard.breaker.release_trial()
            raise
        except Exception as exc:  # noqa: BLE001 - malformed response etc.; the turn goes on
            guard.breaker.release_trial()
            logger.exception("crm_tool.call_failed tenant=%s tool=%s request_id=%s", self.tenant_id, spec.name, request_id)
            status = "client_error"
            result = ToolCallResult.error("internal", f"Volání nástroje {spec.name} selhalo ({type(exc).__name__}).")

        result.meta.setdefault("request_id", request_id)
        self._log(spec, mode, request_id, status, started)
        return result

    def _http_error(self, spec: ToolSpec, exc: httpx.HTTPStatusError, guard: Any) -> tuple[ToolCallResult, str]:
        code = exc.response.status_code
        if code == 404:
            # Tool vanished from this CRM (deploy/rollback): refetch the manifest next turn.
            guard.breaker.record_success()
            self.cache.invalidate(self.tenant_id)
            return ToolCallResult.error("not_found", f"Nástroj {spec.name} v tomto CRM už není k dispozici."), "http_404"
        if code in {401, 403}:
            guard.breaker.record_success()
            return ToolCallResult.error("access_denied", "CRM odmítlo přístup k nástroji."), f"http_{code}"
        if code >= 500:
            guard.breaker.record_failure()
        else:
            guard.breaker.record_success()
        return ToolCallResult.error("unavailable", f"CRM vrátilo chybu {code}."), f"http_{code}"

    @staticmethod
    def _transport_error(spec: ToolSpec, exc: Exception) -> ToolCallResult:
        if not spec.annotations.read_only and isinstance(exc, (httpx.ReadTimeout, asyncio.TimeoutError)):
            return ToolCallResult.error(
                "unavailable",
                "CRM neodpovědělo včas. Změna se mohla provést — ověřte prosím stav v CRM, než ji zopakujete.",
            )
        return ToolCallResult.error("unavailable", "CRM je dočasně nedostupné, zkuste to prosím za chvíli.")

    def _log(self, spec: ToolSpec, mode: str, request_id: str, status: str, started: float) -> None:
        logger.info(
            "crm_tool.call tenant=%s tool=%s version=%s mode=%s request_id=%s status=%s duration_ms=%.0f",
            self.tenant_id, spec.name, spec.version, mode, request_id, status, (time.perf_counter() - started) * 1000,
        )
