# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Data shapes of the tool layer, independent of where a tool runs.

A tool is either published by the tenant's Coripo (``source="crm"``, coripo-tools/1
manifest, MCP-shaped) or implemented here in the gateway (``source="gateway"``).
Both look the same to the agent: a ToolSpec plus an invoke returning a ToolCallResult.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

PROTOCOL = "coripo-tools/1"
DEFAULT_CAPABILITY = "crm"

ToolSource = Literal["crm", "gateway"]


@dataclass(frozen=True)
class ToolAnnotations:
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    requires_confirmation: bool = False
    capability: str = DEFAULT_CAPABILITY
    timeout_seconds: float = 15.0

    @property
    def needs_confirmation(self) -> bool:
        # Destructive tools are always confirmed — a floor the gateway enforces itself.
        return not self.read_only and (self.requires_confirmation or self.destructive)

    @classmethod
    def from_manifest(cls, raw: dict[str, Any] | None) -> "ToolAnnotations":
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            read_only=bool(raw.get("readOnlyHint", False)),
            destructive=bool(raw.get("destructiveHint", False)),
            idempotent=bool(raw.get("idempotentHint", False)),
            requires_confirmation=bool(raw.get("requiresConfirmation", False)),
            capability=str(raw.get("capability") or DEFAULT_CAPABILITY),
            timeout_seconds=float(raw.get("timeoutSeconds") or (15 if raw.get("readOnlyHint") else 30)),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: ToolAnnotations
    source: ToolSource
    version: str = "0.0.0"
    title: str = ""

    @classmethod
    def from_manifest(cls, raw: dict[str, Any]) -> "ToolSpec":
        schema = raw.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"tool {raw.get('name')!r}: inputSchema must be an object schema")
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description") or ""),
            input_schema=schema,
            annotations=ToolAnnotations.from_manifest(raw.get("annotations")),
            source="crm",
            version=str(raw.get("version") or "0.0.0"),
            title=str(raw.get("title") or ""),
        )


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    title: str = ""
    prompt: str = ""
    default_on: bool = True


@dataclass(frozen=True)
class Manifest:
    """A tenant's published tools. ``stale`` = served from an older copy because the CRM
    could not be reached; the tools still work if the CRM comes back."""

    protocol: str
    hash: str
    tools: tuple[ToolSpec, ...]
    capabilities: tuple[CapabilitySpec, ...]
    stale: bool = False

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Manifest":
        protocol = str(raw.get("protocol") or "")
        if protocol != PROTOCOL:
            raise ValueError(f"unsupported tool protocol {protocol!r} (expected {PROTOCOL})")
        tools: list[ToolSpec] = []
        for item in raw.get("tools") or []:
            if isinstance(item, dict) and item.get("name"):
                tools.append(ToolSpec.from_manifest(item))
        capabilities = tuple(
            CapabilitySpec(
                id=str(c["id"]),
                title=str(c.get("title") or ""),
                prompt=str(c.get("prompt") or ""),
                default_on=bool(c.get("default_on", True)),
            )
            for c in raw.get("capabilities") or []
            if isinstance(c, dict) and c.get("id")
        )
        return cls(protocol=protocol, hash=str(raw.get("manifest_hash") or ""), tools=tuple(tools), capabilities=capabilities)


@dataclass
class ToolCallResult:
    """MCP CallToolResult: text for the model, structured data for code, isError, meta."""

    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def error_code(self) -> str | None:
        if not self.is_error:
            return None
        error = self.structured.get("error") if isinstance(self.structured.get("error"), dict) else {}
        return str(error.get("code") or "internal")

    @property
    def confirmation(self) -> dict[str, Any] | None:
        value = self.meta.get("confirmation")
        return value if isinstance(value, dict) and value.get("required") else None

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> "ToolCallResult":
        texts = [str(c.get("text") or "") for c in raw.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
        structured = raw.get("structuredContent")
        return cls(
            text="\n".join(texts),
            structured=structured if isinstance(structured, dict) else {},
            is_error=bool(raw.get("isError")),
            meta=raw.get("meta") if isinstance(raw.get("meta"), dict) else {},
        )

    @classmethod
    def error(cls, code: str, message: str, details: Any = None) -> "ToolCallResult":
        return cls(
            text=message,
            structured={"status": "error", "error": {"code": code, "message": message, "details": details or []}},
            is_error=True,
        )


@dataclass(frozen=True)
class CallContext:
    tenant_id: str
    user_id: str
    request_context: dict[str, Any] | None = None

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex[:16]
