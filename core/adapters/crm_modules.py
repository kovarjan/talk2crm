import uuid
from typing import Any, Dict

from core.adapters.crm_direct import _request, CrmDirectError
from core.clients.client_loader import get_rest_client_config


def fetch_modules(tenant: str, user_id: str, user_name: str | None = None, device: str = "desktop") -> Dict[str, Any]:
    cfg = get_rest_client_config(tenant)
    path = f"/ai/v1/modules?user_id={user_id}&user_name={user_name or ''}&device={device}"
    return _request(
        "GET",
        cfg.rest_url,
        path,
        payload=None,
        key_id=cfg.rest_hmac_key_id,
        secret=cfg.rest_hmac_secret,
        user_id=user_id,
        user_name=user_name,
        tenant=tenant,
        user_id_header=cfg.rest_user_id_header,
        user_name_header=cfg.rest_user_name_header,
        tenant_header=cfg.rest_tenant_header,
    )
