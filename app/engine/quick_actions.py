from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.nlu.command_parser import CommandParser
from app.services.crm_write_service import CRMWriteService

if TYPE_CHECKING:
    from app.services.crm_client import SugarClient


@dataclass
class QuickActionResult:
    data: dict[str, Any]
    should_fallback: bool


async def try_handle_quick_action(
    *,
    input_text: str,
    crm_client: "SugarClient",
    user_id: str,
    action_confirmation: bool,
) -> QuickActionResult | None:
    """Parse input as a quick action and execute it.

    Returns:
        None              — no pattern matched; caller should go straight to agent.
        QuickActionResult — pattern matched.
                            should_fallback=True  → matched but agent should handle it.
                            should_fallback=False → result is ready to return to the user.
    """
    cfg = get_settings()

    parser = CommandParser()
    command = parser.parse(input_text)
    if command is None:
        return None

    if command.intent not in {"create_contact", "create_meeting"}:
        return None

    if command.confidence < cfg.quick_action_min_confidence:
        return QuickActionResult(data={}, should_fallback=True)

    try:
        write_service = CRMWriteService(
            tenant_id="quick-action",
            user_id=user_id,
            input_text=input_text,
            request_context={},
            crm_client=crm_client,
            rag_service=None,
            action_confirmation=action_confirmation,
        )
        result = await write_service.execute_quick_command(command)
    except Exception:
        if cfg.quick_action_fallback_on_exception:
            return QuickActionResult(data={}, should_fallback=True)
        raise

    if result is None:
        return None

    if result.get("status") == "resolution_required":
        candidates = (result.get("resolution") or {}).get("candidates") or []
        if not candidates and cfg.quick_action_fallback_on_no_candidates:
            return QuickActionResult(data=result, should_fallback=True)
        if candidates and cfg.quick_action_fallback_on_ambiguous:
            return QuickActionResult(data=result, should_fallback=True)

    return QuickActionResult(data=result, should_fallback=False)
