from __future__ import annotations

from typing import TypedDict

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_hmac_request
from app.services.tenant_manager import TenantManager
from database.session import get_db


class TenantContext(TypedDict):
    tenant_id: str
    user_id: str
    user_name: str | None


async def get_tenant_context(
    request: Request,
    x_tenant: str = Header(..., alias="X-Tenant"),
    x_user_id: str = Header(..., alias="X-User-Id"),
    x_user_name: str | None = Header(default=None, alias="X-User-Name"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_timestamp: str | None = Header(default=None, alias="X-Timestamp"),
    x_nonce: str | None = Header(default=None, alias="X-Nonce"),
    db: AsyncSession = Depends(get_db),
) -> TenantContext:
    auth_value = authorization or ""
    if auth_value.lower().startswith("hmac"):
        body = await request.body()
        ok = verify_hmac_request(
            method=request.method,
            path=request.url.path,
            body=body,
            authorization=authorization,
            timestamp=x_timestamp,
            nonce=x_nonce,
        )
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    tenant_manager = TenantManager(db)
    await tenant_manager.assert_user_access(tenant_id=x_tenant, user_id=x_user_id)

    return {"tenant_id": x_tenant, "user_id": x_user_id, "user_name": x_user_name}
