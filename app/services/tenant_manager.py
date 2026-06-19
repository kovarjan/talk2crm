from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import decrypt_tenant_secret
from database.models import Tenant


logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class TenantCredentials:
    tenant_id: str
    crm_base_url: str
    crm_token: str


# Resolved credentials cached across requests; entries expire after
# Settings.tenant_cache_ttl_seconds. Keyed by tenant_id.
_CREDENTIALS_CACHE: dict[str, tuple[float, TenantCredentials]] = {}


def invalidate_tenant_cache(tenant_id: str | None = None) -> None:
    if tenant_id is None:
        _CREDENTIALS_CACHE.clear()
    else:
        _CREDENTIALS_CACHE.pop(tenant_id, None)


class TenantManager:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    @staticmethod
    def _is_uuid_like(value: str | None) -> bool:
        return bool(value and _UUID_RE.match(value.strip()))

    @staticmethod
    def _parse_token_json(token: str) -> dict | None:
        raw = str(token or "").strip()
        if not raw.startswith("{"):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _resolve_effective_crm_token(self, token: str) -> str:
        fallback = str(self.settings.coripo_test_token or "").strip()
        if not fallback:
            return token
        parsed = self._parse_token_json(token)
        has_static_sid = self._is_uuid_like(token)
        has_hmac_secret = False
        if parsed is not None:
            sid_value = str(parsed.get("session_id") or parsed.get("sid") or "").strip()
            has_static_sid = has_static_sid or bool(sid_value)
            has_hmac_secret = bool(str(parsed.get("hmac_secret") or "").strip())
        else:
            # Plain non-JSON token is interpreted by CoripoClient as HMAC secret.
            has_hmac_secret = not has_static_sid and bool(str(token or "").strip())

        if has_hmac_secret and not has_static_sid:
            return token

        logger.warning(
            "Tenant token is SID-style or missing HMAC secret; using CORIPO_TEST_TOKEN fallback in environment=%s",
            self.settings.environment,
        )
        return fallback

    async def get_tenant(self, tenant_id: str) -> Tenant:
        stmt = select(Tenant).where(Tenant.id == tenant_id, Tenant.active.is_(True))
        result = await self.db.execute(stmt)
        tenant = result.scalar_one_or_none()
        if tenant is None:
            raise HTTPException(status_code=401, detail="Unknown or inactive tenant")
        return tenant

    def _cached_credentials(self, tenant_id: str) -> TenantCredentials | None:
        if self.settings.tenant_cache_ttl_seconds <= 0:
            return None
        cached = _CREDENTIALS_CACHE.get(tenant_id)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        return None

    async def get_credentials(self, tenant_id: str) -> TenantCredentials:
        cached = self._cached_credentials(tenant_id)
        if cached is not None:
            return cached

        tenant = await self.get_tenant(tenant_id)
        token = decrypt_tenant_secret(tenant.crm_token_encrypted)
        token = self._resolve_effective_crm_token(token)
        credentials = TenantCredentials(
            tenant_id=tenant.id,
            crm_base_url=tenant.crm_base_url.rstrip("/"),
            crm_token=token,
        )
        ttl = self.settings.tenant_cache_ttl_seconds
        if ttl > 0:
            _CREDENTIALS_CACHE[tenant_id] = (time.monotonic() + ttl, credentials)
        return credentials

    async def assert_user_access(self, tenant_id: str, user_id: str) -> None:
        # NOTE: User-level authorization is intentionally NOT enforced here.
        # This service trusts that the caller (upstream gateway or HMAC auth in
        # app/api/dependencies.py) has already verified identity. The only check
        # performed is that the tenant exists and is active.
        #
        # DO NOT expose this API directly to the internet without implementing
        # proper user-level checks here first.
        _ = user_id
        if self._cached_credentials(tenant_id) is not None:
            # Credentials are only cached after a successful active-tenant lookup.
            return
        await self.get_tenant(tenant_id)
