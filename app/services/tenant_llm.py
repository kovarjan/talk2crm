# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Per-tenant LLM endpoint/key/model with fallback to the system default.

A tenant row may carry ``llm_base_url`` / ``llm_api_key`` / ``llm_model`` /
``llm_title_model`` (plain text, edited in the DB). Empty means "use the Settings
default". The URL and key travel together: a tenant URL never receives the system
key, and a tenant key is ignored without its own URL. A configured endpoint that
fails is NOT replaced by the system default (a local-AI tenant must not leak to a
cloud model); the error surfaces instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from database.models import Tenant


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    title_model: str
    custom_endpoint: bool = False


# Keyed by tenant_id; entries expire after Settings.tenant_cache_ttl_seconds.
_LLM_CONFIG_CACHE: dict[str, tuple[float, LLMConfig]] = {}


def invalidate_llm_config_cache(tenant_id: str | None = None) -> None:
    if tenant_id is None:
        _LLM_CONFIG_CACHE.clear()
    else:
        _LLM_CONFIG_CACHE.pop(tenant_id, None)


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def default_llm_config(settings: Settings | None = None) -> LLMConfig:
    settings = settings or get_settings()
    return LLMConfig(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        title_model=settings.llm_title_model,
    )


def build_llm_config(
    settings: Settings,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    title_model: str | None = None,
) -> LLMConfig:
    base_url, api_key, model, title_model = map(_clean, (base_url, api_key, model, title_model))
    default = default_llm_config(settings)
    if base_url:
        effective_model = model or default.model
        return LLMConfig(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            model=effective_model,
            title_model=title_model or effective_model,
            custom_endpoint=True,
        )
    return LLMConfig(
        base_url=default.base_url,
        api_key=default.api_key,
        model=model or default.model,
        title_model=title_model or default.title_model,
    )


async def resolve_llm_config(tenant_id: str | None, *, db: AsyncSession | None = None) -> LLMConfig:
    """Effective LLM config for ``tenant_id`` (system default when unset or no tenant)."""
    settings = get_settings()
    if not tenant_id:
        return default_llm_config(settings)

    ttl = settings.tenant_cache_ttl_seconds
    cached = _LLM_CONFIG_CACHE.get(tenant_id) if ttl > 0 else None
    if cached and cached[0] > time.monotonic():
        return cached[1]

    stmt = select(
        Tenant.llm_base_url, Tenant.llm_api_key, Tenant.llm_model, Tenant.llm_title_model
    ).where(Tenant.id == tenant_id)
    if db is not None:
        row = (await db.execute(stmt)).one_or_none()
    else:
        from database.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            row = (await session.execute(stmt)).one_or_none()

    config = default_llm_config(settings) if row is None else build_llm_config(
        settings, base_url=row[0], api_key=row[1], model=row[2], title_model=row[3]
    )
    if ttl > 0:
        _LLM_CONFIG_CACHE[tenant_id] = (time.monotonic() + ttl, config)
    return config
