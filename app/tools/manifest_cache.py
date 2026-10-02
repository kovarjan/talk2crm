# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Per-tenant cache of CRM tool manifests.

- fresh for ``ttl_seconds``; then revalidated with the manifest hash (ETag → cheap 304);
- one fetch per tenant at a time (single-flight), concurrent turns wait for it;
- a failed refresh keeps serving the last good manifest (flagged ``stale``) for up to
  ``stale_max_seconds`` and retries after ``retry_seconds`` — never on every turn;
- the last good manifest is also persisted (SnapshotStore) so a freshly started worker
  can serve tools while the CRM is down.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from app.core.logging import get_logger
from app.tools.contracts import Manifest

logger = get_logger(__name__)

# (etag or None) -> (not_modified, raw manifest or None)
FetchFn = Callable[[str | None], Awaitable[tuple[bool, dict[str, Any] | None]]]


class SnapshotStore(Protocol):
    async def load(self, tenant_id: str) -> tuple[dict[str, Any], float] | None:
        """(raw manifest, fetched_at unix time) or None."""

    async def save(self, tenant_id: str, raw: dict[str, Any]) -> None: ...


@dataclass
class _Entry:
    raw: dict[str, Any]
    manifest: Manifest
    fetched_at: float  # unix time of the last successful fetch/revalidation (staleness)
    fresh_until: float = 0.0  # unix time; before it no revalidation happens
    retry_at: float = 0.0  # monotonic; earliest next refresh attempt after a failure
    last_error: str | None = None


class ManifestCache:
    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        stale_max_seconds: float = 86400.0,
        retry_seconds: float = 30.0,
        snapshots: SnapshotStore | None = None,
        wall_clock: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.stale_max_seconds = stale_max_seconds
        self.retry_seconds = retry_seconds
        self.snapshots = snapshots
        self._wall = wall_clock
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def invalidate(self, tenant_id: str) -> None:
        """Next get() refetches (keeps the entry as a stale fallback)."""
        entry = self._entries.get(tenant_id)
        if entry is not None:
            entry.fresh_until = 0.0
            entry.retry_at = 0.0

    def state(self, tenant_id: str) -> dict[str, Any]:
        entry = self._entries.get(tenant_id)
        if entry is None:
            return {"cached": False}
        return {
            "cached": True,
            "manifest_hash": entry.manifest.hash,
            "age_seconds": round(self._wall() - entry.fetched_at, 1),
            "stale": entry.manifest.stale,
            "last_error": entry.last_error,
            "tool_count": len(entry.manifest.tools),
        }

    async def get(self, tenant_id: str, fetch: FetchFn) -> Manifest | None:
        entry = self._entries.get(tenant_id)
        if entry is not None and self._usable_without_refresh(entry):
            return entry.manifest
        lock = self._locks.setdefault(tenant_id, asyncio.Lock())
        async with lock:
            entry = self._entries.get(tenant_id)
            if entry is not None and self._usable_without_refresh(entry):
                return entry.manifest
            return await self._refresh(tenant_id, entry, fetch)

    def _usable_without_refresh(self, entry: _Entry) -> bool:
        fresh = self._wall() < entry.fresh_until
        backing_off = entry.last_error is not None and self._clock() < entry.retry_at
        return fresh or backing_off

    async def _refresh(self, tenant_id: str, entry: _Entry | None, fetch: FetchFn) -> Manifest | None:
        started = time.perf_counter()
        try:
            not_modified, raw = await fetch(entry.manifest.hash if entry and entry.manifest.hash else None)
            if not_modified and entry is not None:
                entry.fetched_at = self._wall()
                entry.fresh_until = entry.fetched_at + self.ttl_seconds
                entry.last_error = None
                entry.manifest = replace(entry.manifest, stale=False)
                return entry.manifest
            if raw is None:
                raise ValueError("empty manifest")
            manifest = Manifest.parse(raw)
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the last good copy
            return await self._fallback(tenant_id, entry, f"{type(exc).__name__}: {exc}")

        changed = entry is None or entry.manifest.hash != manifest.hash
        now = self._wall()
        self._entries[tenant_id] = _Entry(raw=raw, manifest=manifest, fetched_at=now, fresh_until=now + self.ttl_seconds)
        logger.info(
            "manifest.fetched tenant=%s hash=%s tools=%d changed=%s duration_ms=%.0f",
            tenant_id, manifest.hash, len(manifest.tools), changed, (time.perf_counter() - started) * 1000,
        )
        if changed and self.snapshots is not None:
            try:
                await self.snapshots.save(tenant_id, raw)
            except Exception:  # noqa: BLE001 - the snapshot is an optimization
                logger.exception("manifest.snapshot_save_failed tenant=%s", tenant_id)
        return manifest

    async def _fallback(self, tenant_id: str, entry: _Entry | None, error: str) -> Manifest | None:
        if entry is None and self.snapshots is not None:
            try:
                loaded = await self.snapshots.load(tenant_id)
            except Exception:  # noqa: BLE001
                logger.exception("manifest.snapshot_load_failed tenant=%s", tenant_id)
                loaded = None
            if loaded is not None:
                raw, fetched_at = loaded
                try:
                    entry = _Entry(raw=raw, manifest=Manifest.parse(raw), fetched_at=fetched_at)
                except ValueError:
                    entry = None
        if entry is None or self._wall() - entry.fetched_at > self.stale_max_seconds:
            logger.error("manifest.unavailable tenant=%s error=%s", tenant_id, error)
            self._entries.pop(tenant_id, None)
            return None
        entry.last_error = error
        entry.retry_at = self._clock() + self.retry_seconds
        entry.manifest = replace(entry.manifest, stale=True)
        self._entries[tenant_id] = entry
        logger.warning(
            "manifest.stale tenant=%s hash=%s age_seconds=%.0f error=%s",
            tenant_id, entry.manifest.hash, self._wall() - entry.fetched_at, error,
        )
        return entry.manifest
