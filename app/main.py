from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.api.endpoints import router as api_router
from app.core.config import get_settings
from app.core.logging import request_context_middleware, setup_logging
from database.session import init_db


settings = get_settings()
setup_logging(
    settings.debug,
    log_format=settings.log_format,
    log_force_color=settings.log_force_color,
    log_file_enabled=settings.log_file_enabled,
    log_file_path=settings.log_file_path,
    log_file_format=settings.log_file_format,
    log_trace_enabled=settings.log_trace_enabled,
    log_trace_max_chars=settings.log_trace_max_chars,
    log_trace_history_messages=settings.log_trace_history_messages,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


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
    allow_credentials=True,
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
