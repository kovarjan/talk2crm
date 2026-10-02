# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Renders a manifest tool for the text-based agent prompt, in the same style as the
hand-written gateway tool blocks:

    crm_query_tool(module: str, filters: list=[], limit: int=20)
       — first description line
       — second description line
"""
from __future__ import annotations

import json
from typing import Any

from app.tools.contracts import ToolSpec

_TYPE_NAMES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list", "object": "dict", "null": "null"}


def _type_label(schema: dict[str, Any]) -> str:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return "|".join(json.dumps(v, ensure_ascii=False) for v in enum)
    raw = schema.get("type")
    types = raw if isinstance(raw, list) else [raw] if raw else []
    return "|".join(_TYPE_NAMES.get(str(t), str(t)) for t in types) or "any"


def signature(spec: ToolSpec) -> str:
    schema = spec.input_schema
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") or [])
    params: list[str] = []
    # Required parameters first, then optional ones, each in declaration order.
    for name in sorted(properties, key=lambda n: n not in required):
        prop = properties[name] if isinstance(properties[name], dict) else {}
        label = f"{name}: {_type_label(prop)}"
        if name not in required:
            default = prop.get("default")
            if default == [] and "object" in (prop.get("type") if isinstance(prop.get("type"), list) else [prop.get("type")]):
                default = {}  # PHP encodes an empty-object default as []
            label += "=" + (json.dumps(default, ensure_ascii=False) if default is not None else "null")
        params.append(label)
    return f"{spec.name}({', '.join(params)})"


def render_block(spec: ToolSpec) -> str:
    lines = [line.strip() for line in spec.description.splitlines() if line.strip()]
    body = "\n".join(f"   — {line.lstrip('— ').strip()}" for line in lines)
    return f"{signature(spec)}\n{body}" if body else signature(spec)
