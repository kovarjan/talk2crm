from __future__ import annotations

from dataclasses import dataclass
import json
import re

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
            # Plain non-JSON token is interpreted by SugarClient as HMAC secret.
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

    async def get_credentials(self, tenant_id: str) -> TenantCredentials:
        tenant = await self.get_tenant(tenant_id)
        token = decrypt_tenant_secret(tenant.crm_token_encrypted)
        token = self._resolve_effective_crm_token(token)
        return TenantCredentials(
            tenant_id=tenant.id,
            crm_base_url=tenant.crm_base_url.rstrip("/"),
            crm_token=token,
        )

    async def assert_user_access(self, tenant_id: str, user_id: str) -> None:
        # Placeholder policy hook for tenant-specific access checks.
        # Current model trusts upstream identity and only enforces tenant existence.
        _ = user_id
        await self.get_tenant(tenant_id)
