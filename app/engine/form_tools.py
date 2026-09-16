# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Agent tools of the ``form`` capability.

Both tools fetch the module's live ai_schema, run the existing read-only
helpers (extract_fields / research_company) and return a form_patch. They
never write to the CRM; the FE applies the patch into the open form.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.domain.form_patch import build_form_patch
from app.engine.company_research import research_company
from app.engine.extraction import extract_fields

logger = logging.getLogger(__name__)

SUBJECT_LABELS = {
    "accounts": "Firma",
    "leads": "Firma",
    "producttemplates": "Produkt",
    "products": "Produkt",
}


class ProposeFormFieldsArgs(BaseModel):
    module: str = Field(description="CRM modul otevřeného formuláře, např. Contacts")
    record_id: str | None = Field(default=None, description="ID záznamu, null pro nový záznam")
    instructions: str = Field(description="Co doplnit a z jakého textu vycházet")


class ResearchRecordArgs(BaseModel):
    module: str = Field(description="CRM modul otevřeného formuláře, např. Accounts")
    record_id: str | None = Field(default=None)
    subject: str = Field(description="Název firmy nebo produktu tak, jak je ve formuláři")
    hint: str | None = Field(default=None, description="Upřesnění od uživatele")


def _current_values(request_context: dict[str, Any] | None) -> dict[str, Any]:
    form = (request_context or {}).get("form")
    values = form.get("values") if isinstance(form, dict) else None
    return dict(values) if isinstance(values, dict) else {}


def _unavailable(module: str, exc: Exception) -> str:
    logger.warning("ai_schema unavailable module=%s error=%s", module, exc)
    return json.dumps(
        {"status": "form_unavailable", "message": f"Formulář modulu {module} není dostupný pro doplnění ({exc})."},
        ensure_ascii=False,
    )


def build_form_tools(
    *,
    tenant_id: str,
    user_id: str,
    request_context: dict[str, Any] | None,
    crm_client: Any,
    rag_service: Any | None,
    db: Any | None = None,
) -> list:
    current_values = _current_values(request_context)

    @tool("propose_form_fields_tool", args_schema=ProposeFormFieldsArgs)
    async def propose_form_fields_tool(module: str, record_id: str | None = None, instructions: str = "") -> str:
        """Připraví hodnoty polí otevřeného formuláře z textu. Nikdy nezapisuje do CRM."""
        try:
            schema = await crm_client.get_ai_schema(module, record_id)
        except Exception as exc:  # noqa: BLE001 - any failure means "no form"
            return _unavailable(module, exc)
        result = await extract_fields(
            tenant_id=tenant_id,
            user_id=user_id,
            module=module,
            record_id=record_id,
            field_schema=schema,
            current_values=current_values,
            messages=[{"role": "user", "content": instructions}],
            crm_client=crm_client,
            rag_service=rag_service,
            db=db,
        )
        patch = build_form_patch(
            module=module, record_id=record_id, fields=result.get("fields") or {},
            schema=schema, message=result.get("message") or "",
        )
        return json.dumps({"status": "ok", "form_patch": patch, "message_to_user": patch["message"]}, ensure_ascii=False)

    @tool("research_record_tool", args_schema=ResearchRecordArgs)
    async def research_record_tool(module: str, record_id: str | None = None, subject: str = "", hint: str | None = None) -> str:
        """Dohledá veřejné informace k subjektu a připraví hodnoty polí otevřeného formuláře. Nikdy nezapisuje do CRM."""
        try:
            schema = await crm_client.get_ai_schema(module, record_id)
        except Exception as exc:  # noqa: BLE001
            return _unavailable(module, exc)
        result = await research_company(
            tenant_id=tenant_id,
            user_id=user_id,
            record_id=record_id,
            field_schema=schema,
            current_values=current_values,
            subject=subject,
            subject_label=SUBJECT_LABELS.get(module.lower(), "Subjekt"),
            hint=hint,
            crm_client=crm_client,
            rag_service=rag_service,
            db=db,
        )
        patch = build_form_patch(
            module=module, record_id=record_id, fields=result.get("fields") or {},
            schema=schema, sources=result.get("sources") or [], message=result.get("message") or "",
        )
        return json.dumps({"status": "ok", "form_patch": patch, "message_to_user": patch["message"]}, ensure_ascii=False)

    return [propose_form_fields_tool, research_record_tool]
