from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "SugarVoice AI Bridge"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "dev"
    log_format: str = "pretty"
    log_file_enabled: bool = False
    log_file_path: str = "./logs/talk2api2.log"
    log_file_format: str = "json"
    log_trace_enabled: bool = True
    log_trace_max_chars: int = 1200
    log_trace_history_messages: int = 10

    database_url: str = "sqlite+aiosqlite:///./sugar_voice_bridge.db"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:latest"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.1

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "tenant_knowledge"
    rag_embedding_size: int = 384
    rag_upsert_batch_size: int = 200
    qdrant_allow_local_fallback: bool = True
    qdrant_local_path: str = "./.qdrant_storage_local"

    cache_dir: str = "./cache"

    whisper_model_size: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_allow_cpu_fallback: bool = True

    tts_provider: str = "edge-tts"
    tts_voice: str = "en-US-AriaNeural"

    crm_mode: str = "on"
    crm_timeout_seconds: float = 25.0
    coripo_hmac_key_id: str = "acmark-ai"

    hmac_keys_json: str = Field(default="{}")
    hmac_max_skew_seconds: int = 300

    tenant_secret_key: str = (
        "CHANGE_ME_WITH_32_BYTE_URLSAFE_BASE64_FOR_FERNET"
    )

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def parse_cors_allow_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def cache_audio_dir(self) -> Path:
        return Path(self.cache_dir) / "audio"

    @property
    def hmac_keys(self) -> dict[str, str]:
        try:
            parsed = json.loads(self.hmac_keys_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_audio_dir.mkdir(parents=True, exist_ok=True)
    return settings
