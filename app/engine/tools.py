from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.tools import tool

from app.core.config import get_settings
from app.engine.adjustments import ModuleAdjustmentEngine
from app.engine.rag import TenantRAGService
from app.services.crm_client import SugarClient


def build_tools(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    request_context: dict[str, Any] | None,
    crm_client: SugarClient,
    rag_service: TenantRAGService | None,
    action_confirmation: bool = False,
) -> list:
    settings = get_settings()
    adjustment_engine = ModuleAdjustmentEngine(
        tenant_id=tenant_id,
        user_id=user_id,
        crm_client=crm_client,
        input_text=input_text,
        request_context=request_context,
    )

    @tool("crm_action_tool")
    async def crm_action_tool(module: str, action: str, data_json: str = "{}") -> str:
        """Execute a SugarCRM module action using module/action/data JSON."""
        if settings.crm_mode.lower() == "off":
            return json.dumps(
                {
                    "status": "crm-disabled",
                    "message": "CRM mode is off, action skipped",
                }
            )

        try:
            data: dict[str, Any] = json.loads(data_json) if data_json else {}
            if not isinstance(data, dict):
                raise ValueError("data_json must decode to an object")
        except Exception as exc:
            return json.dumps({"error": f"Invalid data_json: {exc}"})

        normalized_action = (action or "").strip().lower()
        mutating_actions = {"create", "update", "patch", "delete"}
        adjustment = await adjustment_engine.apply(
            module=module,
            action=normalized_action,
            data=data,
        )
        data = adjustment.data

        if normalized_action in mutating_actions and not action_confirmation:
            return json.dumps(
                {
                    "status": "confirmation_required",
                    "message": (
                        "Akce mění CRM data a vyžaduje explicitní potvrzení uživatele. "
                        "Pro provedení zopakujte požadavek s context.confirm_action=true."
                    ),
                    "pending_action": {
                        "module": module,
                        "action": normalized_action,
                        "data": data,
                    },
                    "adjustments": adjustment.notes,
                },
                ensure_ascii=True,
            )

        data.setdefault("requested_by_user_id", user_id)
        try:
            result = await crm_client.execute_module_action(
                module=module,
                action=action,
                data=data,
            )
            if normalized_action in mutating_actions and rag_service is not None:
                try:
                    ingested = await rag_service.ingest_from_crm(
                        tenant_id=tenant_id,
                        crm_client=crm_client,
                        module=module,
                    )
                    if isinstance(result, dict):
                        result["_rag_ingested"] = ingested
                except Exception:
                    if isinstance(result, dict):
                        result["_rag_ingest_error"] = "failed"
            if adjustment.notes and isinstance(result, dict):
                result["_adjustments"] = adjustment.notes
            return json.dumps(result, ensure_ascii=True)
        except httpx.HTTPStatusError as exc:
            body_preview = ""
            try:
                body_preview = exc.response.text[:1000]
            except Exception:
                body_preview = ""
            return json.dumps(
                {
                    "status": "crm_http_error",
                    "http_status": exc.response.status_code,
                    "url": str(exc.request.url),
                    "message": (
                        "CRM rejected the action. Confirm field mapping/required "
                        "values for this module."
                    ),
                    "response_body": body_preview,
                    "failed_action": {
                        "module": module,
                        "action": normalized_action,
                        "data": data,
                    },
                },
                ensure_ascii=True,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "crm_action_error",
                    "message": str(exc),
                    "failed_action": {
                        "module": module,
                        "action": normalized_action,
                        "data": data,
                    },
                },
                ensure_ascii=True,
            )

    @tool("rag_search_tool")
    def rag_search_tool(query: str, limit: int = 5) -> str:
        """Search tenant-scoped knowledge in Qdrant using semantic retrieval."""
        if rag_service is None:
            return json.dumps(
                {"warning": "RAG unavailable", "results": []},
                ensure_ascii=True,
            )
        results = rag_service.search(tenant_id=tenant_id, query=query, limit=limit)
        return json.dumps(results, ensure_ascii=True)

    @tool("crm_search_tool")
    async def crm_search_tool(query: str, scope: str = "all") -> str:
        """Run SugarCRM global search for contacts/accounts/meetings/all."""
        result = await crm_client.generic_search(query=query, scope=scope)
        return json.dumps(result, ensure_ascii=True)

    return [crm_action_tool, rag_search_tool, crm_search_tool]
