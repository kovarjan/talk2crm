from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.dependencies import get_tenant_context
from app.main import app  # FastAPI app instance
from database.session import get_db


@pytest.mark.asyncio
async def test_extract_fields_endpoint_returns_agent_result() -> None:
    fake_result = {"message": "Vyplněno.", "fields": {"first_name": "Jan"}}

    app.dependency_overrides[get_tenant_context] = lambda: {"tenant_id": "t1", "user_id": "u1", "user_name": None}
    app.dependency_overrides[get_db] = lambda: None
    try:
        with (
            patch("app.api.endpoints.extract_fields", new=AsyncMock(return_value=fake_result)) as mock_extract,
            patch("app.api.endpoints.TenantManager") as mock_tenant_manager_cls,
            patch("app.api.endpoints.CoripoClient") as mock_crm_client_cls,
            patch("app.api.endpoints.get_rag_service", return_value=None),
        ):
            mock_tenant_manager = mock_tenant_manager_cls.return_value
            mock_tenant_manager.get_credentials = AsyncMock(
                return_value=type("Creds", (), {"crm_base_url": "http://crm.test", "crm_token": "tok"})()
            )

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/extract-fields/",
                    json={
                        "module": "Contacts",
                        "record_id": None,
                        "field_schema": {"module": "Contacts", "sections": []},
                        "current_values": {},
                        "messages": [{"role": "user", "content": "Jan Novák"}],
                    },
                    headers={"X-Tenant": "t1", "X-User-Id": "u1"},
                )
    finally:
        app.dependency_overrides.pop(get_tenant_context, None)
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["response"] == fake_result
    assert mock_extract.called
    assert mock_crm_client_cls.called
