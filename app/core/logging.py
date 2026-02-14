import logging
import json
import time
import sys
import uuid
from typing import Callable
from fastapi import Request, Response
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Base log object
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        standard_attrs = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in log_obj and not key.startswith("_"):
                log_obj[key] = value

        return json.dumps(log_obj)

def setup_logging(debug: bool = False):
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)
    
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Silence the noise
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # We handle access logs ourselves now


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    
    logger = logging.getLogger("api.access")

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        
        logger.info(
            "Request finished", 
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,  # Number, not string!
                "ip": request.client.host
            }
        )
        
        response.headers["X-Request-Id"] = request_id
        return response

    except Exception as e:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "Request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms
            }
        )
        raise
    finally:
        request_id_ctx.reset(token)
