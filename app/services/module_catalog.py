# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Per-user CRM module catalog served by Coripo (GET ai/modules), with a static fallback.

The BE lists only modules the user can read that have an ai_schema, so validation
becomes "known to the catalog and ACL allows it". When the BE call fails the static
sets keep the agent working on its historic modules.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

STATIC_READ = {"meetings", "calls", "tasks", "notes", "contacts", "accounts", "leads", "opportunities", "opportunites",
               "quotes", "acm_invoices", "acm_orders", "acm_orders_lines", "products", "producttemplates"}
STATIC_WRITE = {"meetings", "calls", "tasks", "notes", "contacts", "accounts", "leads", "quotes"}
STATIC_RAG = {"contacts", "accounts", "meetings", "calls", "tasks", "notes", "leads", "opportunities", "opportunites",
              "quotes", "acm_invoices", "producttemplates"}


@dataclass
class ModuleCatalog:
    modules: list[dict[str, Any]] = field(default_factory=list)
    is_fallback: bool = False

    @classmethod
    def static_fallback(cls) -> "ModuleCatalog":
        rows = [{"name": name, "read": True, "write": name in STATIC_WRITE, "rag": name in STATIC_RAG, "line_module": None}
                for name in sorted(STATIC_READ)]
        return cls(rows, is_fallback=True)

    @property
    def is_empty(self) -> bool:
        return not self.modules

    def readable(self) -> set[str]:
        return {str(m["name"]).lower() for m in self.modules if m.get("name") and m.get("read", True)}

    def writable(self) -> set[str]:
        return {str(m["name"]).lower() for m in self.modules if m.get("name") and m.get("write")}

    def rag(self) -> set[str]:
        return {str(m["name"]).lower() for m in self.modules if m.get("name") and m.get("rag")}

    def line_module(self, name: str) -> str | None:
        wanted = str(name or "").lower()
        for m in self.modules:
            if str(m.get("name") or "").lower() == wanted:
                return m.get("line_module") or None
        return None

    def readable_names(self) -> list[str]:
        return [str(m["name"]) for m in self.modules if m.get("name") and m.get("read", True)]


_cache: dict[tuple[str, str], tuple[float, ModuleCatalog]] = {}
_warned_at: dict[tuple[str, str], float] = {}


def reset_module_catalog_cache() -> None:
    _cache.clear()
    _warned_at.clear()


async def get_module_catalog(crm_client: Any, *, tenant_id: str, user_id: str) -> ModuleCatalog:
    ttl = float(get_settings().tenant_cache_ttl_seconds or 60)
    key = (tenant_id, user_id)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        rows = await crm_client.get_ai_modules()
        catalog = ModuleCatalog(rows) if rows else ModuleCatalog.static_fallback()
    except Exception as exc:  # noqa: BLE001 - never block a turn on the catalog
        if now - _warned_at.get(key, 0.0) >= ttl:
            logger.warning("module catalog unavailable tenant=%s user=%s error=%s; using static sets", tenant_id, user_id, exc)
            _warned_at[key] = now
        catalog = ModuleCatalog.static_fallback()
    _cache[key] = (now, catalog)
    return catalog
