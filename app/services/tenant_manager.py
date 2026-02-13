from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decrypt_tenant_secret
from database.models import Tenant


@dataclass
class TenantCredentials:
    tenant_id: str
    crm_base_url: str
    crm_token: str


class TenantManager:
    def __init__(self, db: AsyncSession):
        self.db = db

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
