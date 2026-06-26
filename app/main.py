# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.api.endpoints import router as api_router
from app.core.audio import cleanup_audio_cache
from app.core.config import get_settings
from app.core.logging import get_logger, request_context_middleware, setup_logging
from app.core.http import aclose_shared_http_client
from database.session import init_db


settings = get_settings()
setup_logging(
    settings.debug,
    log_format=settings.log_format,
    log_force_color=settings.log_force_color,
    log_file_enabled=settings.log_file_enabled,
    log_file_path=settings.log_file_path,
    log_file_format=settings.log_file_format,
    crm_wire_log_enabled=settings.crm_wire_log_enabled,
    crm_wire_log_path=settings.crm_wire_log_path,
    crm_wire_log_format=settings.crm_wire_log_format,
    log_trace_enabled=settings.log_trace_enabled,
    log_trace_max_chars=settings.log_trace_max_chars,
    log_trace_history_messages=settings.log_trace_history_messages,
)


logger = get_logger(__name__)


async def _audio_cache_cleanup_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(cleanup_audio_cache, settings.audio_cache_max_age_hours)
        except Exception:
            logger.exception("Audio cache cleanup failed")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    cleanup_task: asyncio.Task | None = None
    if settings.audio_cache_max_age_hours > 0:
        cleanup_task = asyncio.create_task(_audio_cache_cleanup_loop())
    yield
    if cleanup_task is not None:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
    await aclose_shared_http_client()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials and settings.cors_allow_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(request_context_middleware)


@app.get("/openapi.json", include_in_schema=False)
async def openapi_json():
    return app.openapi()


@app.get("/swagger", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=(
            "Multi-tenant AI voice bridge for Coripo CRM. "
            "Supports text/audio processing, chat lifecycle, and tenant-scoped search."
        ),
        routes=app.routes,
    )
    return app.openapi_schema


app.openapi = custom_openapi
app.include_router(api_router)
