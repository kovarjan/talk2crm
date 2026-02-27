from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from fastapi import Request, Response
from rich.logging import RichHandler
from rich.markup import escape


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_TRACE_ENABLED = True
_TRACE_MAX_CHARS = 1200
_TRACE_HISTORY_MESSAGES = 10


_STANDARD_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


def _safe_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return f"{text[:max_chars]}... (truncated {hidden} chars)"


def _preview_value(value: Any, max_chars: int, *, compact: bool = False) -> str:
    if isinstance(value, str):
        rendered = value
    elif isinstance(value, (dict, list, tuple)):
        rendered = _safe_json_dumps(value, indent=None if compact else 2)
    else:
        rendered = str(value)
    return _truncate_text(rendered, max_chars)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in log_obj and not key.startswith("_"):
                log_obj[key] = value

        return json.dumps(log_obj, ensure_ascii=False, default=str)


class PrettyConsoleFormatter(logging.Formatter):
    _LEVEL_COLOR = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }
    _ROLE_COLOR = {
        "user": "green",
        "assistant": "cyan",
        "system": "magenta",
    }

    def __init__(self, *, max_chars: int, history_messages: int) -> None:
        super().__init__("%(message)s", "%H:%M:%S")
        self.max_chars = max_chars
        self.history_messages = max(1, history_messages)

    def format(self, record: logging.LogRecord) -> str:
        level_color = self._LEVEL_COLOR.get(record.levelname, "white")
        prefix = (
            f"[bright_black]{self.formatTime(record, self.datefmt)}[/] "
            f"[bold {level_color}]{record.levelname:<8}[/] "
            f"[cyan]{escape(record.name)}[/] "
            f"[bright_black]req={escape(request_id_ctx.get())}[/] "
            f"{escape(record.getMessage())}"
        )

        extras = self._extract_extras(record)
        trace = extras.pop("trace", None)
        lines = [prefix]
        lines.extend(self._format_extras(extras))
        if isinstance(trace, dict):
            lines.extend(self._format_trace(trace))
        return "\n".join(lines)

    def _extract_extras(self, record: logging.LogRecord) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                out[key] = value
        return out

    def _format_extras(self, extras: dict[str, Any]) -> list[str]:
        if not extras:
            return []
        lines: list[str] = []
        for key in sorted(extras.keys()):
            rendered = _preview_value(extras[key], self.max_chars, compact=True)
            lines.append(f"  [bright_black]|[/] [bold cyan]{escape(key)}[/]: {escape(rendered)}")
        return lines

    def _format_trace(self, trace: dict[str, Any]) -> list[str]:
        lines = [
            "  [bold magenta]================ LLM TRACE ================[/]",
        ]

        for section_key, section_title in [
            ("session", "Session"),
            ("request", "Request"),
            ("context", "Context"),
            ("tool_calls", "Tool Calls"),
            ("outcome", "Outcome"),
            ("resources", "Resources"),
            ("chat_history", "Chat History"),
        ]:
            section_value = trace.get(section_key)
            if section_value in (None, "", [], {}):
                continue
            lines.append(f"  [bold white]{section_title}[/]")
            if section_key == "tool_calls" and isinstance(section_value, list):
                lines.extend(self._format_tool_calls(section_value))
                continue
            if section_key == "chat_history" and isinstance(section_value, list):
                lines.extend(self._format_chat_history(section_value))
                continue
            rendered = _preview_value(section_value, self.max_chars, compact=False)
            for row in rendered.splitlines():
                lines.append(f"    {escape(row)}")
        lines.append("  [bold magenta]=========================================[/]")
        return lines

    def _format_tool_calls(self, tool_calls: list[Any]) -> list[str]:
        lines: list[str] = []
        if not tool_calls:
            return ["    none"]
        for idx, call in enumerate(tool_calls, start=1):
            if not isinstance(call, dict):
                lines.append(f"    {idx}. {escape(_preview_value(call, self.max_chars, compact=True))}")
                continue
            tool = escape(str(call.get("tool") or "unknown"))
            tool_input = escape(_preview_value(call.get("tool_input"), self.max_chars // 2))
            observation = escape(_preview_value(call.get("observation"), self.max_chars // 2))
            lines.append(f"    {idx}. [bold yellow]{tool}[/]")
            lines.append(f"       input: {tool_input}")
            lines.append(f"       output: {observation}")
        return lines

    def _format_chat_history(self, messages: list[Any]) -> list[str]:
        lines: list[str] = []
        max_items = min(len(messages), self.history_messages)
        for idx, item in enumerate(messages[:max_items], start=1):
            if not isinstance(item, dict):
                lines.append(f"    {idx}. {escape(_preview_value(item, self.max_chars // 2, compact=True))}")
                continue
            role = str(item.get("role") or "unknown").strip().lower()
            role_color = self._ROLE_COLOR.get(role, "white")
            created = str(item.get("created_at") or "-")
            content = escape(_preview_value(item.get("content") or "", self.max_chars // 3, compact=True))
            lines.append(
                f"    {idx}. [{role_color}]{escape(role)}[/] "
                f"[bright_black]at={escape(created)}[/] {content}"
            )
        remaining = len(messages) - max_items
        if remaining > 0:
            lines.append(f"    ... {remaining} more messages omitted")
        return lines


class PrettyFileFormatter(logging.Formatter):
    def __init__(self, *, max_chars: int, history_messages: int) -> None:
        super().__init__("%(message)s", "%Y-%m-%d %H:%M:%S")
        self.max_chars = max_chars
        self.history_messages = max(1, history_messages)

    def format(self, record: logging.LogRecord) -> str:
        line = (
            f"{self.formatTime(record, self.datefmt)} {record.levelname:<8} "
            f"{record.name} req={request_id_ctx.get()} {record.getMessage()}"
        )

        extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                extras[key] = value

        trace = extras.pop("trace", None)
        rows = [line]
        for key in sorted(extras.keys()):
            rows.append(f"  - {key}: {_preview_value(extras[key], self.max_chars, compact=True)}")

        if isinstance(trace, dict):
            rows.append("  ======== LLM TRACE ========")
            for section in ("session", "request", "context", "tool_calls", "outcome", "resources"):
                value = trace.get(section)
                if value in (None, "", [], {}):
                    continue
                rows.append(f"  {section}:")
                rendered = _preview_value(value, self.max_chars, compact=False)
                for part in rendered.splitlines():
                    rows.append(f"    {part}")
            history = trace.get("chat_history")
            if isinstance(history, list) and history:
                rows.append("  chat_history:")
                max_items = min(len(history), self.history_messages)
                for item in history[:max_items]:
                    rows.append(f"    {_preview_value(item, self.max_chars // 2, compact=True)}")
                remaining = len(history) - max_items
                if remaining > 0:
                    rows.append(f"    ... {remaining} more messages omitted")
            rows.append("  ===========================")
        return "\n".join(rows)


def setup_logging(
    debug: bool = False,
    *,
    log_format: str = "pretty",
    log_file_enabled: bool = False,
    log_file_path: str = "./logs/talk2api2.log",
    log_file_format: str = "json",
    log_trace_enabled: bool = True,
    log_trace_max_chars: int = 1200,
    log_trace_history_messages: int = 10,
) -> None:
    global _TRACE_ENABLED
    global _TRACE_MAX_CHARS
    global _TRACE_HISTORY_MESSAGES

    _TRACE_ENABLED = bool(log_trace_enabled)
    _TRACE_MAX_CHARS = max(200, int(log_trace_max_chars))
    _TRACE_HISTORY_MESSAGES = max(1, int(log_trace_history_messages))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if (log_format or "").strip().lower() == "json":
        console_handler: logging.Handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
        )
        console_handler.setFormatter(
            PrettyConsoleFormatter(
                max_chars=_TRACE_MAX_CHARS,
                history_messages=_TRACE_HISTORY_MESSAGES,
            )
        )
    root_logger.addHandler(console_handler)

    if log_file_enabled:
        file_path = Path(log_file_path).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        if (log_file_format or "").strip().lower() == "pretty":
            file_handler.setFormatter(
                PrettyFileFormatter(
                    max_chars=_TRACE_MAX_CHARS,
                    history_messages=_TRACE_HISTORY_MESSAGES,
                )
            )
        else:
            file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def summarize_chat_history(
    history: list[dict[str, Any]],
    *,
    max_chars: int = 500,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        summary.append(
            {
                "role": message.get("role"),
                "created_at": message.get("created_at"),
                "content": _preview_value(message.get("content") or "", max_chars, compact=True),
            }
        )
    return summary


def log_llm_trace(
    logger: logging.Logger,
    *,
    tenant_id: str,
    user_id: str,
    chat_id: str,
    request: dict[str, Any],
    context: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]] | None,
    outcome: dict[str, Any],
    resources: dict[str, Any] | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> None:
    if not _TRACE_ENABLED:
        return

    payload = {
        "session": {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "chat_id": chat_id,
            "request_id": request_id_ctx.get(),
        },
        "request": request,
        "context": context or {},
        "tool_calls": tool_calls or [],
        "outcome": outcome,
        "resources": resources or {},
        "chat_history": summarize_chat_history(chat_history or []),
    }

    logger.info("LLM interaction trace", extra={"trace": payload})


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
                "duration_ms": duration_ms,
                "ip": request.client.host if request.client else "-",
            },
        )

        response.headers["X-Request-Id"] = request_id
        return response

    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "Request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        raise
    finally:
        request_id_ctx.reset(token)
