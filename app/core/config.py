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

    app_name: str = "tak2crm"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "dev"
    log_format: str = "pretty"
    log_force_color: bool = True
    log_file_enabled: bool = False
    log_file_path: str = "./logs/tak2crm.log"
    log_file_format: str = "json"
    log_trace_enabled: bool = True
    log_trace_max_chars: int = 1200
    log_trace_history_messages: int = 10
    tool_call_logging: bool = False

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/talk2crm"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5"
    llm_title_model: str = "llama3"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.1
    llm_context_window_tokens: int = 32768

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "tenant_knowledge"
    rag_embedding_model: str = "qwen3-embedding:4b"
    rag_embedding_base_url: str = "http://192.168.8.16:4000/v1"
    rag_embedding_api_key: str | None = llm_api_key
    rag_embedding_size: int = 2560
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
    coripo_test_token: str = ""
    engine_v2_enabled: bool = True
    resolver_v2_enabled: bool = True
    temporal_v2_enabled: bool = True
    read_service_v2_enabled: bool = True
    write_service_v2_enabled: bool = True

    # Resolver thresholds stay config-driven until calibrated from runtime telemetry.
    resolver_read_confidence_threshold: float = 0.70
    resolver_mutation_confidence_threshold: float = 0.70
    resolver_ambiguity_gap_threshold: float = 0.08
    resolver_strict_mutation_confirmation: bool = True

    quick_action_enabled: bool = True
    quick_action_fallback_on_no_candidates: bool = True
    quick_action_fallback_on_ambiguous: bool = True
    quick_action_fallback_on_exception: bool = True
    quick_action_min_confidence: float = 1.0

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

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use PostgreSQL with asyncpg "
                "(postgresql+asyncpg://...)"
            )
        return normalized

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

    @property
    def resolver_v2_active(self) -> bool:
        return bool(self.engine_v2_enabled and self.resolver_v2_enabled)

    @property
    def temporal_v2_active(self) -> bool:
        return bool(self.engine_v2_enabled and self.temporal_v2_enabled)

    @property
    def read_service_v2_active(self) -> bool:
        return bool(self.engine_v2_enabled and self.read_service_v2_enabled)

    @property
    def write_service_v2_active(self) -> bool:
        return bool(self.engine_v2_enabled and self.write_service_v2_enabled)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_audio_dir.mkdir(parents=True, exist_ok=True)
    return settings
