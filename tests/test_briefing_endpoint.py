from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints import daily_briefing_endpoint
from app.api.models import ChatMessageItem
from app.services.chat_service import chat_message_item_to_agent_history
from app.presentation.agent_result import normalize_agent_result_for_ui


@pytest.mark.asyncio
async def test_endpoint_uses_authenticated_identity_and_returns_same_chat():
    result = {"chat_id": "briefing-1", "message": "Today", "blocks": [], "cards": []}
    db = SimpleNamespace()
    with (patch("app.api.endpoints.TenantManager") as manager,
          patch("app.api.endpoints.CoripoClient") as client,
          patch("app.services.briefing_service.get_briefing", AsyncMock(return_value=result)) as get):
        manager.return_value.get_credentials = AsyncMock(return_value=SimpleNamespace(crm_base_url="http://crm", crm_token="secret"))
        response = await daily_briefing_endpoint(refresh=True, timezone_name="Europe/Prague", closing_days=7,
                                                ctx={"tenant_id": "t", "user_id": "u", "user_name": "Name"}, db=db)
        assert response.response == result
        assert get.call_args.kwargs["user_id"] == "u"
        assert get.call_args.kwargs["tenant_id"] == "t"
        assert get.call_args.kwargs["refresh"] is True
        assert get.call_args.kwargs["days"] == 7
        client.assert_called_once_with("http://crm", "secret", user_id="u", user_name="Name")


@pytest.mark.parametrize("zone,days", [("invalid", 14), ("UTC", 0), ("UTC", 91)])
@pytest.mark.asyncio
async def test_endpoint_rejects_invalid_bounds_before_crm(zone, days):
    with pytest.raises(HTTPException) as error:
        await daily_briefing_endpoint(timezone_name=zone, closing_days=days, ctx={}, db=None)
    assert error.value.status_code == 422


def test_saved_briefing_cards_and_all_sections_are_available_to_followup_chat():
    message = ChatMessageItem(role="assistant", content="Today", metadata={"briefing": {"blocks": [
        {"title": "Quotes", "status": "ok", "records": [{"name": "Offer", "date": "2026-09-18", "url": "/#detail/Quotes/q1"}]},
    ]}})
    history = chat_message_item_to_agent_history(message)
    assert "Quotes" in history["content"] and "/#detail/Quotes/q1" in history["content"]


def test_tool_cards_survive_agent_normalization():
    cards = [{"type": "table", "title": "Quotes", "rows": []}]
    normalized = normalize_agent_result_for_ui({"output": "<answer>Today</answer>", "intermediate_steps": [
        {"tool": "daily_briefing_tool", "observation": {"status": "ok", "cards": cards, "message_to_user": "Today"}},
    ]})
    assert normalized["cards"] == cards
