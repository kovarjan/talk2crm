from __future__ import annotations

from typing import TypedDict

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import parse_hmac_authorization, verify_hmac_request_async
from app.services.tenant_manager import TenantManager
from database.session import get_db


logger = get_logger(__name__)


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
    x_original_uri: str | None = Header(default=None, alias="X-Original-URI"),
    x_rewrite_url: str | None = Header(default=None, alias="X-Rewrite-URL"),
    x_forwarded_uri: str | None = Header(default=None, alias="X-Forwarded-Uri"),
    db: AsyncSession = Depends(get_db),
) -> TenantContext:
    auth_value = authorization or ""
    has_hmac_header = auth_value.lower().startswith("hmac")
    if get_settings().require_hmac and not has_hmac_header:
        raise HTTPException(status_code=401, detail="HMAC authentication required")
    tenant_manager = TenantManager(db)
    if has_hmac_header:
        body = await request.body()
        # The signing secret is the claimed tenant's own key from the DB, so a key
        # can only ever act for its tenant (no global key table, no env config).
        parsed_header = parse_hmac_authorization(auth_value)
        secret = (
            await tenant_manager.get_hmac_secret(x_tenant, parsed_header.key_id)
            if parsed_header
            else None
        )
        keys = {parsed_header.key_id: secret} if parsed_header and secret else {}
        try:
            ok = await verify_hmac_request_async(
                method=request.method,
                path=request.url.path,
                original_path=x_original_uri or x_rewrite_url or x_forwarded_uri,
                body=body,
                authorization=authorization,
                timestamp=x_timestamp,
                nonce=x_nonce,
                query=request.url.query,
                tenant=x_tenant,
                user_id=x_user_id,
                keys=keys,
            )
        except Exception:
            logger.exception("HMAC nonce store unavailable")
            raise HTTPException(status_code=503, detail="Authentication temporarily unavailable")
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    await tenant_manager.assert_user_access(tenant_id=x_tenant, user_id=x_user_id)

    return {"tenant_id": x_tenant, "user_id": x_user_id, "user_name": x_user_name}


async def get_stream_token_context(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_tenant: str | None = Header(default=None, alias="X-Tenant"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_name: str | None = Header(default=None, alias="X-User-Name"),
    db: AsyncSession = Depends(get_db),
) -> TenantContext:
    """Auth dependency for stream endpoints. Accepts Bearer stream tokens.
    Falls back to plain X-Tenant/X-User-Id headers when require_hmac is False (dev)."""
    settings = get_settings()
    auth_value = authorization or ""

    if auth_value.lower().startswith("bearer "):
        token = auth_value[len("bearer "):].strip()
        try:
            from app.core.stream_token import peek_stream_token_claims, validate_stream_token_async
            claims = peek_stream_token_claims(token)
            key_id = str(claims.get("key_id", ""))
            secret = await TenantManager(db).get_hmac_secret(str(claims.get("tenant", "")), key_id)
            identity = await validate_stream_token_async(
                token, hmac_keys={key_id: secret} if secret else {}
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid stream token: {exc}")
        except Exception:
            logger.exception("Stream token nonce store unavailable")
            raise HTTPException(status_code=503, detail="Authentication temporarily unavailable")
        tenant_id = identity["tenant"]
        user_id = identity["user_id"]
        user_name = identity["user_name"] or None
    elif x_tenant and x_user_id and not settings.require_hmac:
        tenant_id = x_tenant
        user_id = x_user_id
        user_name = x_user_name
    else:
        raise HTTPException(
            status_code=401,
            detail="Authorization: Bearer <stream-token> required",
        )

    tenant_manager = TenantManager(db)
    await tenant_manager.assert_user_access(tenant_id=tenant_id, user_id=user_id)
    return {"tenant_id": tenant_id, "user_id": user_id, "user_name": user_name}


async def get_skills_admin_context(
    ctx: TenantContext = Depends(get_tenant_context),
) -> TenantContext:
    """Tenant-scope skill management: user must be on SKILLS_ADMIN_USER_IDS
    (entries are ``user`` or ``tenant:user``). Empty list denies everyone."""
    admins = get_settings().skills_admin_ids
    if ctx["user_id"] in admins or f"{ctx['tenant_id']}:{ctx['user_id']}" in admins:
        return ctx
    raise HTTPException(status_code=403, detail="Skills administrator role required")
