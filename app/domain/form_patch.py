# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Wire contract for values the agent proposes for the CRM form open in the FE.

The FE shows a patch as a pending preview and applies it into the open form on
click; the user's normal Save persists it. The gateway never writes it.
"""
from __future__ import annotations

from typing import Any

MAX_SOURCES = 10


def build_form_patch(
    *,
    module: str,
    record_id: str | None,
    fields: dict[str, Any],
    schema: dict[str, Any],
    sources: list[str] | None = None,
    message: str = "",
) -> dict[str, Any]:
    sections = schema.get("sections") if isinstance(schema, dict) else None
    return {
        "module": module,
        "record": record_id or None,
        "fields": dict(fields or {}),
        "lines": {"mode": "append", "rows": []},
        "invitees": {},
        "sources": [str(url) for url in (sources or [])][:MAX_SOURCES],
        "message": str(message or ""),
        "schema": {"sections": list(sections or [])},
    }
