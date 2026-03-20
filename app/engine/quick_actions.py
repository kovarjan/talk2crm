from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.nlu.command_parser import CommandParser
from app.services.crm_write_service import CRMWriteService

if TYPE_CHECKING:
    from app.services.crm_client import SugarClient


async def try_handle_quick_action(
    *,
    input_text: str,
    crm_client: "SugarClient",
    user_id: str,
    action_confirmation: bool,
) -> dict[str, Any] | None:
    parser = CommandParser()
    command = parser.parse(input_text)
    if command is None:
        return None

    if command.intent not in {"create_contact", "create_meeting"}:
        return None

    write_service = CRMWriteService(
        tenant_id="quick-action",
        user_id=user_id,
        input_text=input_text,
        request_context={},
        crm_client=crm_client,
        rag_service=None,
        action_confirmation=action_confirmation,
    )
    return await write_service.execute_quick_command(command)
