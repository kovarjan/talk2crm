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
from app.core.crm_token import build_crm_token, parse_crm_token
from app.core.security import decrypt_tenant_secret
from app.services.tenant_llm import invalidate_llm_config_cache
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
    invalidate_llm_config_cache(tenant_id)
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

    @classmethod
    def _normalize_token(cls, tenant_id: str, token: str) -> str:
        """Make the stored token self-describing: a legacy plain-secret token or a
        JSON token without ``hmac_key_id`` gets the tenant id as its key id, so the
        key id never has to come from process-wide env config."""
        raw = str(token or "").strip()
        if not raw or cls._is_uuid_like(raw):
            return raw
        parsed = parse_crm_token(raw)
        if parsed:
            if parsed.get("hmac_secret") and not str(parsed.get("hmac_key_id") or "").strip():
                return json.dumps({**parsed, "hmac_key_id": tenant_id}, separators=(",", ":"))
            return raw
        if raw.startswith("{"):
            return raw
        # Plain non-JSON token is an HMAC secret.
        return build_crm_token(tenant_id, raw)

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
        token = self._normalize_token(tenant.id, token)
        credentials = TenantCredentials(
            tenant_id=tenant.id,
            crm_base_url=tenant.crm_base_url.rstrip("/"),
            crm_token=token,
        )
        ttl = self.settings.tenant_cache_ttl_seconds
        if ttl > 0:
            _CREDENTIALS_CACHE[tenant_id] = (time.monotonic() + ttl, credentials)
        return credentials

    async def get_hmac_secret(self, tenant_id: str, key_id: str) -> str | None:
        """Secret for ``key_id`` iff it is this tenant's own key (inbound HMAC auth).

        Unknown/inactive tenant, undecryptable token or a different key id all yield
        None, so the caller just reports an invalid signature.
        """
        try:
            credentials = await self.get_credentials(tenant_id)
        except HTTPException:
            return None
        except ValueError:
            logger.error("Cannot decrypt CRM token for tenant=%s (wrong TENANT_SECRET_KEY?)", tenant_id)
            return None
        config = parse_crm_token(credentials.crm_token)
        secret = str(config.get("hmac_secret") or "").strip()
        if not secret or str(config.get("hmac_key_id") or "").strip() != key_id:
            return None
        return secret

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
