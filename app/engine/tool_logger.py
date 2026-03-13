# app/engine/tool_logger.py
"""
Optional structured logging for LangChain tool calls.
Enable via settings: tool_call_logging = true
Log level is DEBUG so it's silent in production unless explicitly turned on.

Each log line is a single JSON object on stdout/the log stream, tagged with
"tool_call" so it can be filtered in Dozzle / any log aggregator.
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tool_call")


def _truncate(value: Any, max_chars: int = 2000) -> Any:
    """Truncate large strings/dicts so logs stay readable."""
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"
    if isinstance(value, dict):
        serialized = json.dumps(value, ensure_ascii=False)
        if len(serialized) > max_chars:
            return {"_truncated": True, "preview": serialized[:max_chars]}
    return value


class ToolCallLogger:
    """
    Context-manager / helper used inside each @tool function.

    Usage:
        async with ToolCallLogger("crm_query_tool", tenant_id, user_id,
                                   inputs={"module": module, "search": search}) as tcl:
            result = await do_work()
            tcl.set_output(result)
        return result
    """

    def __init__(
        self,
        tool_name: str,
        tenant_id: str,
        user_id: str,
        inputs: dict[str, Any],
    ):
        self.tool_name = tool_name
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.inputs = inputs
        self._output: Any = None
        self._error: str | None = None
        self._start: float = 0.0
        self._enabled = get_settings().tool_call_logging

    def set_output(self, value: Any) -> None:
        self._output = value

    def set_error(self, error: Exception) -> None:
        self._error = str(error)

    async def __aenter__(self) -> "ToolCallLogger":
        self._start = time.monotonic()
        if self._enabled:
            logger.debug(
                "tool_call_start",
                extra={
                    "event": "tool_call_start",
                    "tool": self.tool_name,
                    "tenant_id": self.tenant_id,
                    "user_id": self.user_id,
                    "inputs": _truncate(self.inputs),
                },
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._enabled:
            return
        elapsed_ms = round((time.monotonic() - self._start) * 1000, 1)
        if exc_type is not None:
            self._error = str(exc_val)

        logger.debug(
            "tool_call_end",
            extra={
                "event": "tool_call_end",
                "tool": self.tool_name,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "elapsed_ms": elapsed_ms,
                "status": "error" if self._error else "ok",
                "error": self._error,
                "output_preview": _truncate(self._output),
            },
        )
