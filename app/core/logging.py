from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar, cast

from fastapi import Request, Response
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.pretty import pretty_repr


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_TRACE_ENABLED = True
_TRACE_MAX_CHARS = 1200
_TRACE_HISTORY_MESSAGES = 10
_HEALTH_PATHS = ("/ping", "/ping/", "/health", "/health/")

_STANDARD_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

_F = TypeVar("_F", bound=Callable[..., Any])

LLM_STEP_USER_INPUT = "USER INPUT"
LLM_STEP_AGENT_ACTION = "AGENT THINKING/ACTION"
LLM_STEP_TOOL_RESULT = "TOOL RESULT"
LLM_STEP_FINAL_RESPONSE = "FINAL RESPONSE"


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return f"{text[:max_chars]}... (truncated {hidden} chars)"


def _safe_json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)


def _maybe_decode_unicode_escapes(value: str) -> str:
    if "\\u" not in value:
        return value
    try:
        decoded = value.encode("utf-8").decode("unicode_escape")
        return decoded
    except Exception:
        return value


def _parse_jsonish_string(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return value

    should_attempt = (
        raw.startswith("{")
        or raw.startswith("[")
        or raw.startswith('"{"')
        or raw.startswith('"["')
        or '\\"' in raw
        or "\\u" in raw
    )
    if not should_attempt:
        return _maybe_decode_unicode_escapes(value)

    candidate = raw
    for _ in range(4):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            decoded = _maybe_decode_unicode_escapes(candidate)
            if decoded != candidate:
                candidate = decoded
                continue
            return decoded

        if isinstance(parsed, str):
            candidate = parsed.strip()
            continue
        return _normalize_for_display(parsed)

    return _maybe_decode_unicode_escapes(candidate)


def _normalize_for_display(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_for_display(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_display(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_for_display(v) for v in value)
    if isinstance(value, str):
        return _parse_jsonish_string(value)
    return value


def _render_pretty_value(value: Any, *, compact: bool = False) -> str:
    normalized = _normalize_for_display(value)
    if isinstance(normalized, (dict, list, tuple)):
        return _safe_json_dumps(normalized, indent=None if compact else 2)
    if isinstance(normalized, str):
        return normalized
    return pretty_repr(normalized)


def _preview_value(value: Any, max_chars: int, *, compact: bool = False) -> str:
    rendered = _render_pretty_value(value, compact=compact)
    return _truncate_text(rendered, max_chars)


class HealthcheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name not in {"uvicorn.access", "api.access"}:
            return True

        path = self._extract_path(record)
        if not path:
            return True

        normalized = path.split("?", 1)[0].strip()
        if normalized.startswith("/health"):
            return False
        if normalized in _HEALTH_PATHS:
            return False
        return True

    @staticmethod
    def _extract_path(record: logging.LogRecord) -> str:
        direct = getattr(record, "path", None)
        if isinstance(direct, str) and direct:
            return direct

        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            possible = args[2]
            if possible.startswith("/"):
                return possible

        msg = record.getMessage()
        match = re.search(r'"\w+\s+(/[^ ]*)\s+HTTP/', msg)
        if match:
            return match.group(1)
        return ""


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload and not key.startswith("_"):
                payload[key] = _normalize_for_display(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


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
        llm_step = extras.pop("llm_step", None)

        lines = [prefix]
        if isinstance(llm_step, dict):
            lines.extend(self._format_llm_step(llm_step))
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

    def _format_llm_step(self, llm_step: dict[str, Any]) -> list[str]:
        step = str(llm_step.get("step") or "").strip() or "LLM STEP"
        payload = llm_step.get("payload")
        meta = llm_step.get("meta")

        lines = [f"  [bold magenta][{escape(step)}][/bold magenta]"]
        if payload not in (None, "", [], {}):
            rendered = _preview_value(payload, self.max_chars, compact=False)
            for row in rendered.splitlines():
                lines.append(f"    {escape(row)}")
        if meta not in (None, "", [], {}):
            lines.append("    [bold white]meta[/]")
            rendered = _preview_value(meta, self.max_chars, compact=True)
            lines.append(f"      {escape(rendered)}")
        return lines

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
            if section_key == "outcome" and isinstance(section_value, dict):
                final_answer = section_value.get("message_to_user") or section_value.get("final_answer")
                if isinstance(final_answer, str) and final_answer.strip():
                    lines.append("    final_answer:")
                    for row in final_answer.strip().splitlines():
                        lines.append(f"      {escape(row)}")
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
                extras[key] = _normalize_for_display(value)
        rows = [line]
        for key in sorted(extras.keys()):
            rows.append(f"  - {key}: {_preview_value(extras[key], self.max_chars, compact=True)}")
        return "\n".join(rows)


def setup_logging(
    debug: bool = False,
    *,
    log_format: str = "pretty",
    log_force_color: bool = True,
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

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    if (log_format or "").strip().lower() == "json":
        console_handler: logging.Handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
    else:
        rich_console = Console(
            force_terminal=bool(log_force_color),
            color_system="truecolor" if bool(log_force_color) else "auto",
            soft_wrap=True,
        )
        console_handler = RichHandler(
            console=rich_console,
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
    root.addHandler(console_handler)

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
        root.addHandler(file_handler)

    # 1) Silence network noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)

    # 2) Filter /ping and /health from access logs (uvicorn + api.access)
    health_filter = HealthcheckAccessFilter()
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(logging.INFO)
    uvicorn_access.addFilter(health_filter)
    for handler in uvicorn_access.handlers:
        handler.addFilter(health_filter)

    api_access = logging.getLogger("api.access")
    api_access.addFilter(health_filter)
    for handler in api_access.handlers:
        handler.addFilter(health_filter)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def summarize_chat_history(
    history: list[dict[str, Any]],
    *,
    max_chars: int = 500,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        out.append(
            {
                "role": message.get("role"),
                "created_at": message.get("created_at"),
                "content": _preview_value(message.get("content") or "", max_chars, compact=True),
            }
        )
    return out


def log_llm_step(
    logger: logging.Logger,
    step: str,
    payload: Any,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    logger.info(
        f"[{step}]",
        extra={
            "llm_step": {
                "step": step,
                "payload": _normalize_for_display(payload),
                "meta": _normalize_for_display(meta or {}),
            }
        },
    )


class LLMTraceTemplate:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def user_input(self, payload: Any, *, meta: dict[str, Any] | None = None) -> None:
        log_llm_step(self.logger, LLM_STEP_USER_INPUT, payload, meta=meta)

    def agent_action(self, payload: Any, *, meta: dict[str, Any] | None = None) -> None:
        log_llm_step(self.logger, LLM_STEP_AGENT_ACTION, payload, meta=meta)

    def tool_result(self, payload: Any, *, meta: dict[str, Any] | None = None) -> None:
        log_llm_step(self.logger, LLM_STEP_TOOL_RESULT, payload, meta=meta)

    def final_response(self, payload: Any, *, meta: dict[str, Any] | None = None) -> None:
        log_llm_step(self.logger, LLM_STEP_FINAL_RESPONSE, payload, meta=meta)


def trace_llm_loop(logger_or_name: logging.Logger | str) -> Callable[[_F], _F]:
    logger = logger_or_name if isinstance(logger_or_name, logging.Logger) else get_logger(logger_or_name)

    def decorator(func: _F) -> _F:
        tracer = LLMTraceTemplate(logger)

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                user_payload = kwargs.get("input_text") or kwargs.get("user_input")
                if user_payload not in (None, ""):
                    tracer.user_input(user_payload, meta={"function": func.__name__})
                tracer.agent_action({"function": func.__name__, "event": "start"})

                result = await cast(Callable[..., Awaitable[Any]], func)(*args, **kwargs)

                if isinstance(result, dict):
                    steps = result.get("intermediate_steps")
                    if steps:
                        tracer.tool_result(steps)
                    tracer.final_response(result.get("message_to_user") or result.get("output") or result)
                else:
                    tracer.final_response(result)
                return result

            return cast(_F, wraps(func)(async_wrapper))

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            user_payload = kwargs.get("input_text") or kwargs.get("user_input")
            if user_payload not in (None, ""):
                tracer.user_input(user_payload, meta={"function": func.__name__})
            tracer.agent_action({"function": func.__name__, "event": "start"})

            result = cast(Callable[..., Any], func)(*args, **kwargs)
            if isinstance(result, dict):
                steps = result.get("intermediate_steps")
                if steps:
                    tracer.tool_result(steps)
                tracer.final_response(result.get("message_to_user") or result.get("output") or result)
            else:
                tracer.final_response(result)
            return result

        return cast(_F, wraps(func)(sync_wrapper))

    return decorator


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
        "request": _normalize_for_display(request),
        "context": _normalize_for_display(context or {}),
        "tool_calls": _normalize_for_display(tool_calls or []),
        "outcome": _normalize_for_display(outcome),
        "resources": _normalize_for_display(resources or {}),
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
