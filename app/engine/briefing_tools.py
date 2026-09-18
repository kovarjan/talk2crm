"""Daily briefing tool; identity is captured from the authenticated request."""
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.services.briefing_service import get_briefing
from database.session import AsyncSessionLocal


class BriefingArgs(BaseModel):
    refresh: bool = False
    closing_days: int = Field(default=14, ge=1, le=90)


def build_briefing_tools(*, tenant_id, user_id, crm_client, request_context=None):
    @tool("daily_briefing_tool", args_schema=BriefingArgs)
    async def daily_briefing_tool(refresh: bool = False, closing_days: int = 14) -> str:
        """Můj den: dnešní schůzky, hovory, úkoly, nabídky a další důležité záznamy přihlášeného uživatele."""
        zone = str((request_context or {}).get("timezone") or "Europe/Prague")
        try:
            ZoneInfo(zone)
        except (ValueError, ZoneInfoNotFoundError):
            zone = "Europe/Prague"
        async with AsyncSessionLocal() as db:
            result = await get_briefing(db=db, crm_client=crm_client, tenant_id=tenant_id, user_id=user_id,
                                        timezone_name=zone, days=closing_days, refresh=refresh)
        return json.dumps({"status": "ok", **result, "message_to_user": result["message"]}, ensure_ascii=False)
    return [daily_briefing_tool]
