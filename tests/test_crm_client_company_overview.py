from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.crm_client import CoripoClient


def test_company_overview_uses_detail_post_and_unwraps_message_data() -> None:
    client = CoripoClient(
        base_url="http://localhost:2000/public",
        token="11111111-2222-3333-4444-555555555555",
    )

    captured: dict[str, Any] = {}

    async def fake_coripo_request(
        self: CoripoClient,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        require_sid: bool = True,
    ) -> dict[str, Any]:
        captured.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
                "require_sid": require_sid,
            }
        )
        return {
            "status": True,
            "message": {
                "text": "detailViewData",
                "data": {
                    "module": "Accounts",
                    "record": {"id": "7d956317-c1ba-0118-1c91-61545f91b878", "name": "365.bank"},
                    "activities_summary": {"calls": "2"},
                    "activities": {"calls": "2"},
                    "related_records": {"Contacts": []},
                },
            },
        }

    client._coripo_request = types.MethodType(fake_coripo_request, client)

    result = asyncio.run(
        client.execute_module_action(
            module="Accounts",
            action="company_overview",
            data={"id": "7d956317-c1ba-0118-1c91-61545f91b878"},
        )
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "detail/Accounts/7d956317-c1ba-0118-1c91-61545f91b878"
    assert captured["json_body"] == {"AiRequest": True}
    assert captured["require_sid"] is True
    assert result["module"] == "Accounts"
    assert result["record"]["name"] == "365.bank"


def test_company_overview_requires_id() -> None:
    client = CoripoClient(
        base_url="http://localhost:2000/public",
        token="11111111-2222-3333-4444-555555555555",
    )

    with pytest.raises(ValueError, match="company_overview action requires 'id' in data"):
        asyncio.run(
            client.execute_module_action(
                module="Accounts",
                action="company_overview",
                data={},
            )
        )
