# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Argument validation against a tool's inputSchema, before anything leaves the gateway.

The CRM validates again (it is authoritative); failing here first saves a round trip and
gives the model the same JSON-pointer error details either way. Common LLM slips are
coerced first: JSON-in-a-string for array/object arguments, "20" for integers, "true"
for booleans, and null for optional non-nullable arguments (treated as "not given").
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import json_repair
from jsonschema import Draft202012Validator


@lru_cache(maxsize=512)
def _validator(schema_key: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads(schema_key))


def _types(schema: dict[str, Any]) -> set[str]:
    value = schema.get("type")
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)} if value else set()


def _coerce_value(value: Any, schema: dict[str, Any]) -> Any:
    types = _types(schema)
    if isinstance(value, str):
        text = value.strip()
        if types & {"array", "object"} and "string" not in types and text[:1] in "[{":
            try:
                return json_repair.loads(text)
            except Exception:  # noqa: BLE001 - leave it to the validator to report
                return value
        if "integer" in types and "string" not in types and text.lstrip("-").isdigit():
            return int(text)
        if "number" in types and "string" not in types:
            try:
                return float(text.replace(",", "."))
            except ValueError:
                return value
        if "boolean" in types and "string" not in types and text.lower() in {"true", "false"}:
            return text.lower() == "true"
    return value


def coerce_arguments(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") or [])
    out: dict[str, Any] = {}
    for name, value in (args or {}).items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            out[name] = value
            continue
        if value is None and name not in required and "null" not in _types(prop):
            continue
        out[name] = _coerce_value(value, prop)
    return out


def validate_arguments(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Returns (coerced args, errors[{path, problem}]); errors empty when valid."""
    coerced = coerce_arguments(args if isinstance(args, dict) else {}, schema)
    validator = _validator(json.dumps(schema, sort_keys=True))
    errors = [
        {"path": "/" + "/".join(str(p) for p in error.absolute_path), "problem": error.message}
        for error in sorted(validator.iter_errors(coerced), key=lambda e: [str(p) for p in e.absolute_path])
    ]
    return coerced, errors
