from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.adapters.crm_direct import (
    execute_direct_command,
    get_module_template,
    get_quickform_template,
)
from core.adapters.crm_modules import fetch_modules


@dataclass(frozen=True)
class CrmRequestContext:
    tenant: str
    user_id: str
    user_name: Optional[str] = None


class CRMService:
    """
    Unified CRM facade used by tools/pipelines.

    This keeps backend details (Coripo direct REST vs Sugar v4.1) hidden behind
    one small API so adding new modes does not require touching every tool.
    """

    def list_records(self, ctx: CrmRequestContext, module: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return execute_direct_command(
            {"action": "list", "module": module, "parameters": parameters or {}},
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def execute_command(self, ctx: CrmRequestContext, command: Dict[str, Any]) -> Dict[str, Any]:
        return execute_direct_command(
            command,
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def get_record(self, ctx: CrmRequestContext, module: str, record_id: str) -> Dict[str, Any]:
        return execute_direct_command(
            {"action": "detail", "module": module, "updateId": record_id},
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def create_record(self, ctx: CrmRequestContext, module: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        return execute_direct_command(
            {"action": "create", "module": module, "parameters": fields},
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def update_record(self, ctx: CrmRequestContext, module: str, record_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        return execute_direct_command(
            {"action": "update", "module": module, "updateId": record_id, "parameters": fields},
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def delete_record(self, ctx: CrmRequestContext, module: str, record_id: str) -> Dict[str, Any]:
        return execute_direct_command(
            {"action": "delete", "module": module, "updateId": record_id},
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def get_template(self, ctx: CrmRequestContext, module: str) -> Dict[str, Any]:
        return get_module_template(
            module,
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def get_quickform(self, ctx: CrmRequestContext, module: str) -> Dict[str, Any]:
        return get_quickform_template(
            module,
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
        )

    def get_modules(self, ctx: CrmRequestContext, device: str = "desktop") -> Dict[str, Any]:
        return fetch_modules(
            tenant=ctx.tenant,
            user_id=ctx.user_id,
            user_name=ctx.user_name,
            device=device,
        )
