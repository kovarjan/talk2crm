from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RestClientConfig:
    name: str
    rest_url: str
    rest_hmac_key_id: str
    rest_hmac_secret: str
    rest_user_id_header: str = "X-User-Id"
    rest_user_name_header: str = "X-User-Name"
    rest_tenant_header: str = "X-Tenant"


def _load_clients(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("clients_config.json must be a non-empty JSON array")
    out: Dict[str, dict] = {}
    for item in data:
        name = item.get("name")
        if not name:
            raise ValueError("Client config entry missing 'name'")
        out[name] = item
    return out


def get_rest_client_config(client_name: str, config_path: str = "core/clients/clients_config.json") -> RestClientConfig:
    configs = _load_clients(config_path)
    item = configs.get(client_name)
    if not item:
        raise ValueError(f"Client '{client_name}' not found in config")

    rest_url = item.get("rest_url") or item.get("rest_base_url")
    rest_key = item.get("rest_hmac_key_id")
    rest_secret = item.get("rest_hmac_secret")
    if not all([rest_url, rest_key, rest_secret]):
        raise ValueError(f"Missing rest_url/rest_hmac_key_id/rest_hmac_secret for client '{client_name}'")

    return RestClientConfig(
        name=item["name"],
        rest_url=rest_url,
        rest_hmac_key_id=rest_key,
        rest_hmac_secret=rest_secret,
        rest_user_id_header=item.get("rest_user_id_header", "X-User-Id"),
        rest_user_name_header=item.get("rest_user_name_header", "X-User-Name"),
        rest_tenant_header=item.get("rest_tenant_header", "X-Tenant"),
    )
