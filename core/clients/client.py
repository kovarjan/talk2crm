from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import Dict
import json, uuid, base64, hashlib, hmac
from datetime import datetime, timezone
import requests

@dataclass(frozen=True)
class ClientConfig:
    name: str
    api_key_id: str
    api_key: str
    api_url: str
    api_version: str = "v1"  # default if not in JSON

class Client:
    api_path = "/ai/"

    def __init__(self, client_name: str, config_path: str = "core/clients/clients_config.json") -> None:
        self._configs: Dict[str, ClientConfig] = {}
        self._load_config(config_path)
        self.client_name = client_name
        self._config = self.get_client_config(client_name)
        if client_name not in self._configs:
            raise ValueError(f"Client '{client_name}' not found in config")

    def get_client_config(self, name: str) -> ClientConfig:
        cfg = self._configs.get(name)
        if not cfg:
            raise ValueError(f"Client config '{name}' not found")
        return cfg

    def get_base_url(self) -> str:
        # e.g. "https://rest-ai.coripo.app/ai/v1/"
        base = self._config.api_url.rstrip("/")
        return f"{base}{self.api_path}{self._config.api_version}/"

    def get_api_key_id(self) -> str:
        return self._config.api_key_id

    def get_api_key(self) -> str:
        return self._config.api_key

    def _load_config(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            raise ValueError("Config must be a non-empty JSON array of client configs")

        for item in data:
            name = item.get("name")
            api_key_id = item.get("api_key_id")
            api_key = item.get("api_key")
            api_url = item.get("api_url")
            api_version = item.get("api_version", "v1")

            if not all([name, api_key_id, api_key, api_url]):
                raise ValueError(f"Incomplete client config entry: {item}")

            cfg = ClientConfig(
                name=name,
                api_key_id=api_key_id,
                api_key=api_key,
                api_url=api_url,
                api_version=api_version,
            )
            self._configs[name] = cfg

    def _sign(self, secret: str, ts: str, nonce: str, body: bytes) -> str:
        base = f"{ts}|{nonce}|".encode("utf-8") + body
        mac = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).digest()
        import base64 as b64
        return b64.b64encode(mac).decode("ascii")

    def get(self, path: str, params: dict | None = None) -> dict:
        """HMAC-signed GET for export endpoints."""
        cfg = self._config
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        nonce = str(uuid.uuid4())
        body = b""  # GET has empty body but we still sign it
        sig = self._sign(cfg.api_key, ts, nonce, body)

        headers = {
            "Authorization": f"HMAC keyId={cfg.api_key_id}, signature={sig}",
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "X-Request-Id": str(uuid.uuid4()),
            "Accept": "application/json",
        }
        base = cfg.api_url.rstrip("/")
        url = base + (path if path.startswith("/") else "/" + path)

        r = requests.get(url, headers=headers, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()
