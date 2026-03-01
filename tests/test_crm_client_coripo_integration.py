from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.crm_client import SugarClient


def _to_pretty_json(value: Any, *, limit: int = 2500) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        rendered = repr(value)
    # if len(rendered) > limit:
    #     return rendered[:limit] + f"\n... <truncated {len(rendered) - limit} chars>"
    return rendered


def _attach_client_trace(client: SugarClient) -> None:
    trace_raw = str(os.getenv("CORIPO_TEST_TRACE_RAW", "")).strip().lower() in {"1", "true", "yes"}
    original_action = client.execute_module_action

    async def traced_action(
        self: SugarClient,
        module: str,
        action: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        print("\n=== CRM ACTION ===")
        print(f"module: {module}")
        print(f"action: {action}")
        print(f"base_url: {self._coripo_api_base()}")
        print(f"request_keys: {sorted((data or {}).keys())}")
        print(f"sid_before: {'yes' if bool(self._coripo_session_id) else 'no'}")
        try:
            result = await original_action(module, action, data)
        except Exception as exc:
            print("=== CRM ERROR ===")
            print(f"type: {type(exc).__name__}")
            print(f"message: {exc}")
            raise

        print("=== CRM RESULT (SANITIZED) ===")
        print(f"sid_after: {'yes' if bool(self._coripo_session_id) else 'no'}")
        if isinstance(result, dict):
            print(f"result_keys: {sorted(result.keys())}")
            if isinstance(result.get("records"), list):
                print(f"records_count: {len(result['records'])}")
                if result["records"]:
                    print("first_record:")
                    print(_to_pretty_json(result["records"][0], limit=1200))
            print(f"column_fields: {len(result.get('column_fields') or [])}")
            print(f"def_fields: {len(result.get('def_fields') or [])}")
        print(_to_pretty_json(result))
        return result

    client.execute_module_action = types.MethodType(traced_action, client)

    if trace_raw:
        original_raw = client._coripo_request

        async def traced_raw(
            self: SugarClient,
            method: str,
            path: str,
            *,
            params: dict[str, Any] | None = None,
            json_body: dict[str, Any] | None = None,
            require_sid: bool = True,
        ) -> dict[str, Any]:
            print("\n--- RAW HTTP REQUEST ---")
            print(f"method: {method} path: /{path.lstrip('/')} require_sid={require_sid}")
            if params:
                print(_to_pretty_json(params, limit=1200))
            if json_body is not None:
                print(_to_pretty_json(json_body, limit=1200))
            result = await original_raw(
                method,
                path,
                params=params,
                json_body=json_body,
                require_sid=require_sid,
            )
            print("--- RAW HTTP RESPONSE ---")
            print(_to_pretty_json(result, limit=1200))
            return result

        client._coripo_request = types.MethodType(traced_raw, client)


def _build_client_from_env() -> SugarClient:
    if os.getenv("CORIPO_TEST_BASE_URL"):
        base_url = str(os.getenv("CORIPO_TEST_BASE_URL")).strip()
    else:
        # In Docker, localhost is the container itself; prefer host gateway.
        in_container = Path("/.dockerenv").exists()
        base_url = (
            "http://host.docker.internal:2000/public"
            if in_container
            else "http://localhost:2000/public"
        )
    token = (
        os.getenv("CORIPO_TEST_TOKEN")
        or os.getenv("CRM_TEST_TOKEN")
        or os.getenv("CRM_TOKEN")
        or ""
    ).strip()
    user_id = (os.getenv("CORIPO_TEST_USER_ID") or os.getenv("CRM_TEST_USER_ID") or "").strip() or None
    user_name = (os.getenv("CORIPO_TEST_USER_NAME") or os.getenv("CRM_TEST_USER_NAME") or "").strip() or None

    if not token:
        pytest.skip(
            "Set CORIPO_TEST_TOKEN (or CRM_TEST_TOKEN/CRM_TOKEN) to run real Coripo integration tests."
        )

    # Coripo HMAC auth resolves user by ID first; if ID is not a CRM UUID, it may
    # require X-User-Name fallback (for example numeric IDs like '28').
    if user_id and "-" not in user_id and not user_name:
        pytest.skip(
            "CORIPO_TEST_USER_ID appears non-UUID; set CORIPO_TEST_USER_NAME as well "
            "(for example 'jkovar') so HMAC user fallback can resolve."
        )

    client = SugarClient(
        base_url=base_url,
        token=token,
        user_id=user_id,
        user_name=user_name,
    )
    _attach_client_trace(client)
    return client


def _run_or_skip_connect_error(client: SugarClient, module: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return asyncio.run(client.execute_module_action(module, "list", payload))
    except ValueError as exc:
        pytest.skip(f"Auth is not configured for checksid/HMAC flow: {exc}")
    except httpx.ConnectError as exc:
        pytest.skip(
            "CRM endpoint is not reachable from this environment "
            f"(base_url={client._coripo_api_base()}). "
            "Set CORIPO_TEST_BASE_URL to a reachable URL (for Docker often "
            "'http://host.docker.internal:2000/public' or service DNS). "
            f"Original error: {exc}"
        )
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            pytest.skip(
                "Coripo HMAC auth rejected credentials (401 on checksid). "
                "Verify: CORIPO_TEST_TOKEN secret, keyId (default acmark-ai), "
                "CORIPO_TEST_USER_ID, and CORIPO_TEST_USER_NAME. "
                f"Response body: {exc.response.text[:400]}"
            )
        raise


def _assert_records_result(result: dict[str, Any]) -> None:
    assert isinstance(result, dict)
    assert "records" in result
    assert isinstance(result["records"], list)
    if "count" in result:
        assert isinstance(result["count"], int)
        assert result["count"] == len(result["records"])

    # Sanitized payload should not expose Coripo envelope noise.
    assert "message" not in result
    assert "data" not in result
    assert "defs" not in result
    assert "field_defs" not in result
    assert "rows" not in result
    assert "def" not in result
    assert "menu" not in result

    if result["records"]:
        first = result["records"][0]
        assert isinstance(first, dict)
        assert isinstance(first.get("id"), str)
        assert first["id"].strip() != ""


def test_coripo_accounts_list_with_text_filter_real_data() -> None:
    client = _build_client_from_env()
    query = os.getenv("CORIPO_TEST_QUERY", "zliner")

    payload = {
        "limit": 25,
        "offset": 0,
        "filter": {
            "operator": "and",
            "operands": [
                {
                    "operator": "and",
                    "operands": [
                        {
                            "field": "*",
                            "fieldModule": None,
                            "fieldRel": None,
                            "type": "cont",
                            "value": query,
                            "relationField": None,
                        }
                    ],
                }
            ],
        },
        "order": [
            {
                "field": "billing_address_street",
                "sort": "ASC",
                "module": "Accounts",
            }
        ],
    }

    result = _run_or_skip_connect_error(client, "Accounts", payload)

    print(f"Accounts filter '{query}' returned {len(result.get('records', []))} records")
    _assert_records_result(result)


def test_coripo_contacts_complex_relate_filter_real_data() -> None:
    client = _build_client_from_env()

    payload = {
        "limit": 25,
        "offset": 0,
        "columns": [
            {"field": "name", "module": "Contacts", "width": "20%", "function": None},
            {"field": "title", "module": "Contacts", "width": "15%", "function": None},
            {"field": "account_name", "module": "Contacts", "width": "25%", "function": None},
            {"field": "phone_mobile", "module": "Contacts", "width": "10%", "function": None},
            {"field": "phone_work", "module": "Contacts", "width": "15%", "function": None},
            {"field": "primary_address_street", "module": "Contacts", "width": "10%", "function": None},
            {"field": "primary_address_city", "module": "Contacts", "width": "10%", "function": None},
            {"field": "assigned_user_name", "module": "Contacts", "width": "10%", "function": None},
        ],
        "order": [],
        "groupBy": [],
        "function": {},
        "alterName": {},
        "groupByDate": [],
        "filter": {
            "operator": "and",
            "operands": [
                {
                    "operator": "and",
                    "operands": [
                        {
                            "module": "Accounts",
                            "type": "relate",
                            "name": "account_name",
                            "relationship": ["accounts"],
                            "filter": {
                                "operator": "or",
                                "operands": [
                                    {
                                        "field": "billing_address_city",
                                        "type": "eq",
                                        "value": "Brno",
                                    },
                                    {
                                        "field": "billing_address_city",
                                        "type": "eq",
                                        "value": "Ostrava",
                                    },
                                ],
                            },
                        }
                    ],
                },
                {
                    "field": "email",
                    "type": "nnull",
                    "value": None,
                },
            ],
        },
        "savedSearch": True,
    }

    result = _run_or_skip_connect_error(client, "Contacts", payload)

    print(f"Contacts complex filter returned {len(result.get('records', []))} records")
    _assert_records_result(result)
