from __future__ import annotations

import json
import os
from dataclasses import dataclass

# ---------------- Config types ----------------

@dataclass(frozen=True)
class ClientConfig:
    name: str
    api_key_id: str
    api_key: str
    api_url: str
    api_version: str


# ---------------- Client manager ----------------

class Client:
    """
    Connection manager for CRM AI Gateways defined in clients_config.json.

    Example:
        mgr = Client("clients_config.json")
        cmd = {
            "actor": {"id": "c7b6e1a2-1b23-44a0-9a31-0ad8f9c5d001", "username": "jkovar"},
            "action": "create",
            "module": "meetings",
            "parameters": {...},
            "metadata": {...}
        }
        resp = mgr.send_command("ai", cmd)  # returns JSON dict
    """

    api_path = "/ai/"

    def __init__(self, client_name: str, config_path: str = "clients_config.json") -> None:
        self._configs: dict[str, ClientConfig] = {}
        self._load_config(config_path)
        self.client_name = client_name
        self._config = self.get_client_config(client_name)

    def get_client_config(self, name: str) -> ClientConfig:
        cfg = self._configs.get(name)
        if not cfg:
            raise ValueError(f"Client config '{name}' not found")
        return cfg
    
    def get_base_url(self) -> str:
        # e.g. "https://rest-ai.coripo.app/ai/v1/"
        return self._config.api_url.rstrip("/") + self.api_path + self._config.api_version + "/"

    def get_api_key_id(self) -> str:
        return self._config.api_key_id

    def get_api_key(self) -> str:
        return self._config.api_key
    

    # ---------- Internals ----------

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

            if not all([name, api_key_id, api_key, api_url]):
                raise ValueError(f"Incomplete client config entry: {item}")

            cfg = ClientConfig(
                name=name,
                api_key_id=api_key_id,
                api_key=api_key,
                api_url=api_url,
                path=item.get("path", "/ai/v1/commands"),
                timeout=float(item.get("timeout", 15.0)),
            )
            self._configs[name] = cfg

