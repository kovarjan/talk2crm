"""Per-tenant HMAC keys live in the tenant's encrypted DB token (no env key table)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import HTTPException

from app.core.crm_token import build_crm_token, parse_crm_token
from app.services.tenant_manager import TenantCredentials, TenantManager


def _manager(token: str | None = None, *, missing: bool = False) -> TenantManager:
    manager = TenantManager.__new__(TenantManager)
    if missing:
        manager.get_credentials = AsyncMock(side_effect=HTTPException(status_code=401))
    else:
        manager.get_credentials = AsyncMock(
            return_value=TenantCredentials("acme", "https://rest-acme/public", token or "")
        )
    return manager


def test_secret_returned_only_for_the_tenants_own_key() -> None:
    manager = _manager(build_crm_token("acme", "s3cret"))
    assert asyncio.run(manager.get_hmac_secret("acme", "acme")) == "s3cret"
    assert asyncio.run(manager.get_hmac_secret("acme", "other-key")) is None


def test_unknown_tenant_or_token_without_secret_yields_none() -> None:
    assert asyncio.run(_manager(missing=True).get_hmac_secret("nope", "nope")) is None
    sid = "11111111-2222-3333-4444-555555555555"
    assert asyncio.run(_manager(sid).get_hmac_secret("acme", "acme")) is None


def test_undecryptable_token_yields_none() -> None:
    manager = _manager()
    manager.get_credentials = AsyncMock(side_effect=ValueError("bad key"))
    assert asyncio.run(manager.get_hmac_secret("acme", "acme")) is None


def test_normalize_gives_legacy_tokens_the_tenant_id_as_key_id() -> None:
    norm = TenantManager._normalize_token
    assert parse_crm_token(norm("acme", "plain-secret")) == {
        "mode": "coripo_public", "hmac_key_id": "acme", "hmac_secret": "plain-secret",
    }
    no_kid = json.dumps({"mode": "coripo_public", "hmac_secret": "x"})
    assert parse_crm_token(norm("acme", no_kid))["hmac_key_id"] == "acme"
    explicit = build_crm_token("kid-1", "x")
    assert norm("acme", explicit) == explicit
    sid = "11111111-2222-3333-4444-555555555555"
    assert norm("acme", sid) == sid


def test_env_key_table_is_gone() -> None:
    from app.core.config import Settings

    fields = Settings.model_fields
    assert not {"hmac_keys_json", "hmac_key_tenants_json", "coripo_hmac_key_id", "coripo_test_token"} & set(fields)


# --- request path: get_tenant_context -------------------------------------------

def _signed_headers(secret: str, key_id: str, *, path: str = "/process-input/", body: bytes = b"{}"):
    import base64
    import hashlib
    import hmac as _hmac
    import time
    import uuid

    ts, nonce = str(int(time.time())), uuid.uuid4().hex
    sign_input = "\n".join(["POST", path, ts, nonce, body.decode()])
    sig = base64.b64encode(_hmac.new(secret.encode(), sign_input.encode(), hashlib.sha256).digest()).decode()
    return {"authorization": f"HMAC keyId={key_id}, signature={sig}", "x_timestamp": ts, "x_nonce": nonce}


def _call_context(tenant: str, headers: dict, tokens: dict[str, str]):
    from types import SimpleNamespace

    from app.api import dependencies

    request = SimpleNamespace(
        method="POST", url=SimpleNamespace(path="/process-input/", query=""),
        body=AsyncMock(return_value=b"{}"),
    )

    async def fake_credentials(self, tenant_id):
        if tenant_id not in tokens:
            raise HTTPException(status_code=401, detail="Unknown or inactive tenant")
        return TenantCredentials(tenant_id, "https://rest/public", tokens[tenant_id])

    settings = SimpleNamespace(require_hmac=True, hmac_max_skew_seconds=300, hmac_bind_identity=False,
                               hmac_nonce_store="memory", environment="dev", tenant_cache_ttl_seconds=0)
    with patch.object(TenantManager, "get_credentials", fake_credentials), \
            patch.object(TenantManager, "assert_user_access", AsyncMock()), \
            patch.object(dependencies, "get_settings", lambda: settings), \
            patch("app.core.security.get_settings", lambda: settings), \
            patch("app.core.nonce_store.get_settings", lambda: settings):
        return asyncio.run(dependencies.get_tenant_context(
            request, x_tenant=tenant, x_user_id="1", x_user_name=None, x_original_uri=None,
            x_rewrite_url=None, x_forwarded_uri=None, db=None, **headers,
        ))


def test_request_signed_with_tenants_own_key_is_accepted() -> None:
    tokens = {"acme": build_crm_token("acme", "acme-secret"), "globex": build_crm_token("globex", "globex-secret")}
    ctx = _call_context("acme", _signed_headers("acme-secret", "acme"), tokens)
    assert ctx["tenant_id"] == "acme"


def test_one_tenants_key_cannot_act_for_another_tenant() -> None:
    import pytest

    tokens = {"acme": build_crm_token("acme", "acme-secret"), "globex": build_crm_token("globex", "globex-secret")}
    with pytest.raises(HTTPException) as exc:
        _call_context("globex", _signed_headers("acme-secret", "acme"), tokens)
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException):
        _call_context("globex", _signed_headers("wrong", "globex"), tokens)
    with pytest.raises(HTTPException):
        _call_context("ghost", _signed_headers("acme-secret", "acme"), tokens)
