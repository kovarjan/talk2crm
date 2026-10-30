from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "talk2crm"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "dev"
    log_format: str = "pretty"
    log_force_color: bool = True
    log_file_enabled: bool = False
    log_file_path: str = "./logs/talk2crm.log"
    log_file_format: str = "json"
    # CRM wire logs and LLM traces contain message bodies (PII). Off by default;
    # when enabled, payloads are truncated to log_trace_max_chars.
    crm_wire_log_enabled: bool = False
    crm_wire_log_path: str = "./logs/crm_wire.log"
    crm_wire_log_format: str = "pretty"
    log_trace_enabled: bool = False
    # Debug only: keep full message bodies / CRM payloads in traces and wire logs.
    log_include_payloads: bool = False
    log_trace_max_chars: int = 1200
    log_trace_history_messages: int = 10
    tool_call_logging: bool = False

    # Apply Alembic migrations in the app lifespan. Set false when a separate
    # one-shot step (docker-compose.prod.yml `migrate`) runs them.
    run_migrations_on_startup: bool = True

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/talk2crm"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5"
    llm_title_model: str = "llama3"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.1
    # Per-request timeout and retry budget for the chat LLM, and the overall
    # wall-clock deadline for one agent turn (all iterations together).
    llm_request_timeout_seconds: float = 60.0
    llm_max_retries: int = 1
    # Tool-call rounds per chat turn before the agent must answer from what it has.
    agent_max_iterations: int = 10
    agent_turn_timeout_seconds: float = 180.0
    llm_context_window_tokens: int = 32768
    llm_system_prompt_prefix: str = "/nothink"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "tenant_knowledge"
    rag_embedding_model: str = "qwen3-embedding:4b"
    rag_embedding_base_url: str = "http://localhost:11434/v1"
    rag_embedding_api_key: str | None = None
    rag_embedding_size: int = 2560
    rag_upsert_batch_size: int = 200
    rag_entity_min_score: float = 60.0
    rag_entity_stopwords_extra: str = ""
    qdrant_allow_local_fallback: bool = True
    qdrant_local_path: str = "./.qdrant_storage_local"

    cache_dir: str = "./cache"

    whisper_model_size: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_allow_cpu_fallback: bool = True

    tts_provider: str = "edge-tts"
    tts_voice: str = "en-US-AriaNeural"
    # Generated TTS responses older than this are deleted by the hourly cleanup
    # task (0 disables cleanup).
    audio_cache_max_age_hours: float = 24.0

    crm_mode: str = "on"
    crm_timeout_seconds: float = 25.0
    # ai_schema walks layouts, vardefs and line items (quotes: product bundles); allow more than a plain list.
    crm_schema_timeout_seconds: float = 60.0

    # Web search via a self-hosted SearXNG instance (docker-compose service
    # `searxng`). Set web_search_enabled=False to hide the tool from the agent.
    web_search_enabled: bool = True
    searxng_url: str = "http://searxng:8080"
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 12.0

    # Companion tool: lets the agent open a URL web_search_tool surfaced and
    # read its text. Same enable flag family, independent timeout/size cap.
    web_fetch_enabled: bool = True
    web_fetch_max_chars: int = 6000
    web_fetch_timeout_seconds: float = 10.0

    # How long an HMAC-derived Coripo SID is reused across requests (0 disables
    # the cache and restores one hmac-login per request).
    coripo_sid_cache_ttl_seconds: float = 600.0
    # How long resolved tenant credentials are served from memory before the
    # DB row is consulted again (0 disables the cache). Deactivating a tenant
    # can take up to this long to propagate.
    tenant_cache_ttl_seconds: float = 60.0
    # AI tool registry (app/tools): CRM tools come from each tenant's GET ai/tools manifest.
    # Manifests are revalidated (ETag) after the TTL; if the CRM is unreachable the last
    # good copy is served for up to stale_max. Concurrency/breaker limits are per tenant
    # and per worker process.
    tool_manifest_ttl_seconds: float = 300.0
    tool_manifest_stale_max_seconds: float = 86400.0
    tool_manifest_retry_seconds: float = 30.0
    tool_tenant_max_concurrency: int = 8
    tool_breaker_failure_threshold: int = 5
    tool_breaker_reset_seconds: float = 30.0
    engine_v2_enabled: bool = True
    read_service_v2_enabled: bool = True
    aggregate_tools_enabled: bool = True

    # Skills system
    skills_enabled: bool = Field(
        default=True,
        description="Master switch for the entire skills system.",
    )
    skills_modification_enabled: bool = Field(
        default=True,
        description=(
            "Allow creating/updating skill overlays. "
            "Set to False to freeze the current skill set — skills are still loaded and used."
        ),
    )
    skills_capture_enabled: bool = Field(
        default=True,
        description=(
            "Enable automatic correction capture from user messages. "
            "Requires skills_modification_enabled=True to have any effect."
        ),
    )
    skills_max_per_context: int = Field(
        default=20,
        description="Maximum number of skill overlays injected per request.",
    )
    skills_min_confidence: float = Field(
        default=0.5,
        description="Minimum confidence for auto-captured skills to be saved.",
    )


    # Comma/JSON list of user ids ("user" or "tenant:user") allowed to
    # create/approve/deactivate tenant-scope skills. Empty = nobody (deny).
    skills_admin_user_ids: str = ""

    # Audio upload limits (process-audio / transcribe endpoints).
    audio_max_upload_bytes: int = 25 * 1024 * 1024
    audio_max_concurrent_transcriptions: int = 2

    # Per-tenant HMAC key id + secret are NOT configured here: they live in each
    # tenant's encrypted CRM token in the DB (scripts/onboard_tenant.py).
    hmac_max_skew_seconds: int = 300
    # When true the signed string also covers the query string, X-Tenant and
    # X-User-Id (see app/core/security.py). Clients MUST sign the same fields,
    # so gateway and CRM client have to be upgraded together before enabling.
    hmac_bind_identity: bool = False
    # Replay-protection store: "database" (shared by all workers), "memory"
    # (per process) or "auto" (database outside dev/test environments).
    hmac_nonce_store: str = "auto"
    # When true, every protected request must carry a valid HMAC Authorization
    # header — tenant/user headers alone are rejected. Keep enabled in any
    # deployment where the API is not behind a trusted authenticating gateway.
    require_hmac: bool = False

    tenant_secret_key: str = (
        "CHANGE_ME_WITH_32_BYTE_URLSAFE_BASE64_FOR_FERNET"
    )

    # Defaults to "*" for dev only; production startup requires explicit origins.
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Browsers reject credentialed CORS with a wildcard origin, so enabling this
    # only takes effect when cors_allow_origins lists explicit origins.
    cors_allow_credentials: bool = False
    # Optional regex (fully matched against the Origin header) for origin families,
    # e.g. https://([a-z0-9-]+\.)*coripo\.(app|cloud|dev); used next to the explicit list.
    cors_allow_origin_regex: str | None = None

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

    @field_validator("tenant_secret_key")
    @classmethod
    def validate_tenant_secret_key(cls, value: str) -> str:
        placeholder = "CHANGE_ME_WITH_32_BYTE_URLSAFE_BASE64_FOR_FERNET"
        if value.strip() == placeholder:
            raise ValueError(
                "TENANT_SECRET_KEY is still the placeholder value. "
                "Generate a real key: python -c "
                "\"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        return value

    @field_validator("crm_mode")
    @classmethod
    def validate_crm_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"on", "off"}:
            raise ValueError(
                f"CRM_MODE must be 'on' or 'off', got {value!r}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if not self.is_production:
            return self
        problems: list[str] = []
        if not self.require_hmac:
            problems.append("REQUIRE_HMAC must be true")
        if "*" in self.cors_allow_origins:
            problems.append("CORS_ALLOW_ORIGINS must list explicit origins (not '*')")
        if problems:
            raise ValueError("Unsafe production configuration: " + "; ".join(problems))
        # Silent local-disk Qdrant fallback hides outages in production unless
        # the operator opted in explicitly.
        if "qdrant_allow_local_fallback" not in self.model_fields_set:
            self.qdrant_allow_local_fallback = False
        return self

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"production", "prod"}

    @property
    def skills_admin_ids(self) -> set[str]:
        raw = (self.skills_admin_user_ids or "").strip()
        if not raw:
            return set()
        if raw.startswith("["):
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                return set()
        else:
            items = raw.split(",")
        return {str(item).strip() for item in items if str(item).strip()}

    @property
    def cache_audio_dir(self) -> Path:
        return Path(self.cache_dir) / "audio"

    @property
    def read_service_v2_active(self) -> bool:
        return bool(self.engine_v2_enabled and self.read_service_v2_enabled)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_audio_dir.mkdir(parents=True, exist_ok=True)
    return settings
