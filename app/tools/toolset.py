# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""ToolSet: the tools one agent turn may use, and the only way the agent calls them.

Built per turn from
  - the tenant's CRM manifest (cached, see manifest_cache) — tools run in Coripo;
  - the gateway's native tools (RAG, web, form extraction, briefing, …).
Filtered by the turn's enabled capabilities. The agent sees one flat namespace; native
names are reserved, so a CRM tool that collides with one is dropped (and logged).

CRM tool call: validate (+coerce) → interceptors (e.g. open-form redirect) →
preview (confirmation-gated tools) or execute → observation dict (see observation.py).
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.engine.capabilities import TOOL_PROMPT_BLOCKS, TOOL_PROMPT_ORDER, prompt_blocks_for, resolve_capabilities
from app.tools.contracts import CallContext, CapabilitySpec, Manifest, ToolCallResult, ToolSpec
from app.tools.crm_provider import MODE_EXECUTE, MODE_PREVIEW, CoripoToolTransport, CrmToolProvider
from app.tools.manifest_cache import ManifestCache
from app.tools.observation import error_observation, to_observation
from app.tools.prompt import render_block
from app.tools.resilience import TenantGuards
from app.tools.validation import validate_arguments

logger = get_logger(__name__)

# Every tool the gateway implements itself, enabled this turn or not: a CRM manifest tool
# must never shadow one of them.
RESERVED_NATIVE_NAMES: frozenset[str] = frozenset(TOOL_PROMPT_BLOCKS) | {"crm_aggregate_tool", "math_tool"}

# (spec, args) -> observation JSON string to use instead of calling the tool, or None.
Interceptor = Callable[[ToolSpec, dict[str, Any]], Awaitable[str | None]]

_cache: ManifestCache | None = None
_guards: TenantGuards | None = None


def get_manifest_cache() -> ManifestCache:
    """Process-wide manifest cache (one per worker)."""
    global _cache
    if _cache is None:
        from app.tools.snapshots import DbSnapshotStore

        settings = get_settings()
        _cache = ManifestCache(
            ttl_seconds=settings.tool_manifest_ttl_seconds,
            stale_max_seconds=settings.tool_manifest_stale_max_seconds,
            retry_seconds=settings.tool_manifest_retry_seconds,
            snapshots=DbSnapshotStore(),
        )
    return _cache


def get_tenant_guards() -> TenantGuards:
    global _guards
    if _guards is None:
        settings = get_settings()
        _guards = TenantGuards(
            max_concurrency=settings.tool_tenant_max_concurrency,
            failure_threshold=settings.tool_breaker_failure_threshold,
            reset_seconds=settings.tool_breaker_reset_seconds,
        )
    return _guards


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass
class AgentTool:
    """What run_agent() calls: a name and an async ainvoke(args) -> observation text."""

    name: str
    _invoke: Callable[[dict[str, Any]], Awaitable[str]]

    async def ainvoke(self, args: dict[str, Any]) -> str:
        return await self._invoke(args)


@dataclass
class ToolSet:
    ctx: CallContext
    enabled_capabilities: set[str]
    unknown_capabilities: list[str]
    crm_specs: dict[str, ToolSpec]
    native_tools: dict[str, Any]
    manifest: Manifest | None
    provider: CrmToolProvider | None
    extra_capabilities: dict[str, CapabilitySpec] = field(default_factory=dict)
    interceptors: list[Interceptor] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build

    @classmethod
    async def build(
        cls,
        *,
        tenant_id: str,
        user_id: str,
        request_context: dict[str, Any] | None,
        crm_client: Any,
        native_tools: Callable[[set[str] | None], list[Any]],
        capabilities: set[str] | None = None,
        provider: CrmToolProvider | None = None,
    ) -> "ToolSet":
        """``capabilities=None`` resolves them from ``request_context`` (chat turns);
        pass an explicit set for purpose-built helpers. ``native_tools(enabled)`` builds
        the gateway's own tools for those capabilities."""
        if provider is None:
            provider = CrmToolProvider(
                tenant_id=tenant_id,
                transport=CoripoToolTransport(crm_client),
                cache=get_manifest_cache(),
                guards=get_tenant_guards(),
            )
        manifest = await provider.manifest()
        extra = {c.id: c for c in manifest.capabilities} if manifest else {}
        if capabilities is None:
            enabled, unknown = resolve_capabilities(request_context, extra)
        else:
            enabled, unknown = set(capabilities) | {"crm"}, []

        natives = {str(getattr(t, "name", "")): t for t in native_tools(enabled) if getattr(t, "name", "")}
        crm_specs: dict[str, ToolSpec] = {}
        dropped: list[str] = []
        for spec in manifest.tools if manifest else ():
            if spec.name in natives or spec.name in RESERVED_NATIVE_NAMES:
                logger.warning("tool.name_conflict tenant=%s tool=%s (reserved by the gateway)", tenant_id, spec.name)
                dropped.append(spec.name)
                continue
            if spec.annotations.capability in enabled:
                crm_specs[spec.name] = spec

        toolset = cls(
            ctx=CallContext(tenant_id=tenant_id, user_id=user_id, request_context=request_context),
            enabled_capabilities=enabled,
            unknown_capabilities=unknown,
            crm_specs=crm_specs,
            native_tools=natives,
            manifest=manifest,
            provider=provider,
            extra_capabilities=extra,
            dropped=dropped,
        )
        if "form" in enabled:
            toolset.interceptors.append(open_form_redirect(request_context, crm_client))
        return toolset

    # ------------------------------------------------------------------ agent surface

    @property
    def crm_available(self) -> bool:
        return self.manifest is not None

    def names(self) -> list[str]:
        order = {name: i for i, name in enumerate(TOOL_PROMPT_ORDER)}
        return sorted([*self.crm_specs, *self.native_tools], key=lambda n: (order.get(n, len(order)), n))

    def is_read_only(self, name: str) -> bool:
        """Native tools never write to the CRM; CRM tools say so in their annotations."""
        spec = self.crm_specs.get(name)
        return spec.annotations.read_only if spec is not None else name in self.native_tools

    def agent_tools(self) -> list[AgentTool]:
        return [AgentTool(name, self._invoker(name)) for name in self.names()]

    def prompt_blocks(self) -> list[str]:
        """Tool descriptions for the system prompt: CRM tools rendered from their manifest,
        native tools from their hand-written blocks."""
        blocks: list[str] = []
        for name in self.names():
            if name in self.crm_specs:
                blocks.append(render_block(self.crm_specs[name]))
            elif name in TOOL_PROMPT_BLOCKS:
                blocks.append(TOOL_PROMPT_BLOCKS[name].replace("{{", "{").replace("}}", "}"))
        return blocks

    def capability_prompt(self) -> str:
        return prompt_blocks_for(self.enabled_capabilities, self.extra_capabilities)

    def _invoker(self, name: str) -> Callable[[dict[str, Any]], Awaitable[str]]:
        async def invoke(args: dict[str, Any]) -> str:
            return await self.invoke(name, args)

        return invoke

    async def invoke(self, name: str, args: dict[str, Any]) -> str:
        if name in self.crm_specs:
            return _dumps(await self._invoke_crm(self.crm_specs[name], args if isinstance(args, dict) else {}))
        tool = self.native_tools.get(name)
        if tool is None:
            return _dumps({"status": "error", "message": f"Nástroj '{name}' neexistuje. Dostupné: {self.names()}"})
        return str(await tool.ainvoke(args))

    async def _invoke_crm(self, spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        args, errors = validate_arguments(args, spec.input_schema)
        if errors:
            summary = "; ".join(f"{e['path']}: {e['problem']}" for e in errors[:5])
            return error_observation(ToolCallResult.error("invalid_arguments", f"Neplatné argumenty pro {spec.name}: {summary}", errors))
        for interceptor in self.interceptors:
            redirected = await interceptor(spec, args)
            if redirected is not None:
                return json.loads(redirected)
        assert self.provider is not None
        mode = MODE_PREVIEW if spec.annotations.needs_confirmation else MODE_EXECUTE
        result = await self.provider.call(spec, args, self.ctx, mode=mode)
        return to_observation(spec, args, result)

    # ------------------------------------------------------------------ confirmation

    async def execute_pending(self, pending: dict[str, Any]) -> dict[str, Any]:
        """Run a confirmed pending tool call with its token (no LLM involved)."""
        spec = self._pending_spec(pending)
        if spec is None:
            return {"status": "error", "message": "Navrhovanou akci už nelze provést (nástroj není k dispozici)."}
        args = pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {}
        result = await self.provider.call(spec, args, self.ctx, mode=MODE_EXECUTE, confirmation_token=str(pending.get("confirmation_token") or ""))
        return to_observation(spec, args, result)

    async def repreview_pending(self, pending: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Preview a pending tool call again after the user edited it (new token)."""
        spec = self._pending_spec(pending)
        if spec is None:
            return {"status": "error", "message": "Navrhovanou akci už nelze upravit (nástroj není k dispozici)."}
        args = dict(pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {})
        args["data_json"] = data
        result = await self.provider.call(spec, args, self.ctx, mode=MODE_PREVIEW)
        return to_observation(spec, args, result)

    def _pending_spec(self, pending: dict[str, Any]) -> ToolSpec | None:
        name = str(pending.get("tool") or "")
        spec = self.crm_specs.get(name)
        if spec is None and self.manifest is not None:
            spec = next((s for s in self.manifest.tools if s.name == name), None)
        return spec if spec is not None and self.provider is not None else None

    # ------------------------------------------------------------------ diagnostics

    def describe(self) -> dict[str, Any]:
        return {
            "tenant_id": self.ctx.tenant_id,
            "crm_available": self.crm_available,
            "manifest_hash": self.manifest.hash if self.manifest else None,
            "manifest_stale": self.manifest.stale if self.manifest else None,
            "capabilities": sorted(self.enabled_capabilities),
            "dropped_tools": self.dropped,
            "tools": [
                {
                    "name": name,
                    "source": "crm" if name in self.crm_specs else "gateway",
                    "version": self.crm_specs[name].version if name in self.crm_specs else None,
                    "capability": self.crm_specs[name].annotations.capability if name in self.crm_specs else None,
                    "needs_confirmation": self.crm_specs[name].annotations.needs_confirmation if name in self.crm_specs else False,
                }
                for name in self.names()
            ],
        }


def open_form_redirect(request_context: dict[str, Any] | None, crm_client: Any) -> Interceptor:
    """With the form capability on, a crm_action_tool call aimed at the record open in the
    form becomes a live form_patch the user applies in the FE — never a CRM write."""

    async def intercept(spec: ToolSpec, args: dict[str, Any]) -> str | None:
        if spec.name != "crm_action_tool":
            return None
        from app.engine.form_tools import action_as_form_patch, targets_open_form

        data = dict(args.get("data_json") if isinstance(args.get("data_json"), dict) else {})
        if args.get("record_id") and not data.get("id"):
            data["id"] = args["record_id"]  # record_id is the legacy spelling of data_json.id
        module = str(args.get("module") or "")
        action = str(args.get("action") or "")
        if not targets_open_form(request_context=request_context, module=module, action=action, data=data):
            return None
        ctx = request_context or {}
        open_record = str(ctx.get("record_id") or ctx.get("record") or "").strip() or None
        return await action_as_form_patch(module=module, record_id=open_record, data=data, crm_client=crm_client)

    return intercept


async def build_turn_toolset(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    request_context: dict[str, Any] | None,
    crm_client: Any,
    rag_service: Any,
    module_catalog: Any = None,
    capabilities: set[str] | None = None,
) -> ToolSet:
    """ToolSet for one request: tenant CRM tools + the gateway's native tools.

    ``capabilities=None`` resolves them from ``request_context`` (chat turns); helpers
    with a fixed purpose pass an explicit set (``crm`` is always added).
    """
    from app.engine.tools import build_native_tools

    def natives(enabled: set[str] | None) -> list[Any]:
        return build_native_tools(
            tenant_id=tenant_id,
            user_id=user_id,
            input_text=input_text,
            request_context=request_context,
            crm_client=crm_client,
            rag_service=rag_service,
            capabilities=enabled,
            module_catalog=module_catalog,
        )

    toolset = await ToolSet.build(
        tenant_id=tenant_id,
        user_id=user_id,
        request_context=request_context,
        crm_client=crm_client,
        native_tools=natives,
        capabilities=capabilities,
    )
    if {"crm_query_tool", "crm_action_tool"} <= set(toolset.crm_specs):
        from app.engine.contact_tools import NAME, build_contact_tool, contact_form_misroute, related_contacts_response

        toolset.native_tools[NAME] = build_contact_tool(toolset)
        async def guard_contact_description(spec: ToolSpec, args: dict[str, Any]) -> str | None:
            data = args.get("data_json") or {}
            fields = data.get("fields") or {}
            if (spec.name == "crm_action_tool" and str(args.get("module") or "").lower() == "accounts"
                    and "description" in fields and contact_form_misroute(str(fields["description"]), user_input=input_text)):
                return _dumps(related_contacts_response())
            return None

        toolset.interceptors.insert(0, guard_contact_description)
    return toolset
