from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(debug: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    logger = get_logger("api.request")
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "%s %s -> 500 (%sms)",
            request.method,
            request.url.path,
            duration_ms,
        )
        request_id_ctx.reset(token)
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "%s %s -> %s (%sms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    request_id_ctx.reset(token)
    return response
