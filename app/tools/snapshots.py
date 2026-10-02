# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Postgres-backed SnapshotStore for the manifest cache (table tenant_tool_manifests)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from database.models import TenantToolManifest
from database.session import AsyncSessionLocal


class DbSnapshotStore:
    """Uses its own short session: called from inside request handlers whose session
    must not be committed or rolled back by the cache."""

    async def load(self, tenant_id: str) -> tuple[dict[str, Any], float] | None:
        async with AsyncSessionLocal() as db:
            row = await db.get(TenantToolManifest, tenant_id)
            if row is None:
                return None
            return dict(row.manifest_json), row.fetched_at.timestamp()

    async def save(self, tenant_id: str, raw: dict[str, Any]) -> None:
        async with AsyncSessionLocal() as db:
            await db.merge(TenantToolManifest(
                tenant_id=tenant_id,
                manifest_hash=str(raw.get("manifest_hash") or ""),
                manifest_json=raw,
                fetched_at=datetime.now(timezone.utc),
            ))
            await db.commit()
