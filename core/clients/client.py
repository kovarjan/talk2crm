from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import Dict

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
